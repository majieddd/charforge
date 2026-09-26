"""Krea 2 against Qwen-Image 2.1 on the pipeline's own reference prompts, side by side.

    python tools/compare_image_models.py --names mara,juno3,rowan,wren,aoi,pip --out results/v3/image_models

For each character with a recorded prompt (work/<name>/style.json), the reference image the
`reference` stage would make - same prompt builder (charforge.reference_prompt), same size and
seed - once with Krea 2 Turbo (8 steps) and once with Qwen-Image 2.1 (30 steps, transparent).
All of one model first, then the other, with ComfyUI's models dropped in between (the two do not
fit in a 24 GB Mac together).

Measured, per image (the numbers are in <out>/compare.json, the pictures in <out>/sheet.png):
  seconds      wall time for the job
  in_frame     the figure clears every edge of the image (head and feet not cut off)
  arm_gap      how far the wrists hang from the hips, in torso lengths (an A-pose clears the body;
               near 0 means the hands touch the thighs - the arms then fuse in the 3D model)
  background   Krea: spread of the border's colour (0 = one flat grey); Qwen: share of the image's
               pixels that are neither clear nor opaque (a clean cut-out has almost none)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.argv, _argv = [sys.argv[0]], sys.argv                  # charforge parses nothing at import
import charforge  # noqa: E402
import comfy  # noqa: E402
sys.argv = _argv


def free():
    req = urllib.request.Request(f"{comfy.HOST}/free", data=json.dumps({"unload_models": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60).read()
    time.sleep(3)


def measures(rgba: np.ndarray, pose) -> dict:
    alpha = rgba[..., 3].astype(np.float32) / 255 if rgba.shape[2] == 4 else None
    rgb = rgba[..., :3]
    if alpha is not None and (alpha < 0.99).any():
        mask = alpha > 0.5
        bg = float(((alpha > 0.02) & (alpha < 0.98)).mean())
    else:
        sys.path.insert(0, str(ROOT / "pipeline"))
        from align import image_mask
        mask = image_mask(rgb)
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(np.float32)
        bg = float(border.std(0).mean() / 255)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    in_frame = bool(len(xs) and ys.min() > 2 and xs.min() > 2 and ys.max() < h - 3 and xs.max() < w - 3)
    out = {"in_frame": in_frame, "background": round(bg, 4), "height_px": int(np.ptp(ys)) if len(ys) else 0}
    kp = pose(rgb, mask)
    if kp is not None:
        sh, hip = (kp[5] + kp[6]) / 2, (kp[11] + kp[12]) / 2
        torso = float(np.linalg.norm(sh - hip)) + 1e-6
        # each wrist's sideways distance from its own hip, beyond the hip
        gap = [abs(kp[9][0] - kp[11][0]) / torso, abs(kp[10][0] - kp[12][0]) / torso]
        out["arm_gap"] = round(float(min(gap)), 3)
    return out


def pose_model():
    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(dev).eval()

    def run(rgb, mask):
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            return None
        box = [[[float(xs.min()), float(ys.min()), float(np.ptp(xs)), float(np.ptp(ys))]]]
        inp = proc(Image.fromarray(rgb), boxes=box, return_tensors="pt").to(dev)
        with torch.no_grad():
            res = proc.post_process_pose_estimation(model(**inp), boxes=box)[0][0]
        return res["keypoints"].cpu().numpy()
    return run


def main(a):
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    chars = []
    for n in a.names.split(","):
        rec = json.load(open(ROOT / "work" / n / "style.json"))
        if rec.get("prompt"):
            chars.append((n, rec["prompt"], rec.get("style", "realistic")))
    report = {}
    for model in ("krea2", "qwen21"):
        free()
        for n, desc, style in chars:
            full, neg = charforge.reference_prompt(desc, style, model)
            if model == "qwen21":
                wf = comfy.qwen21(full, width=832, height=1216, steps=a.qwen_steps, seed=a.seed, transparent=True,
                                  prefix=f"cmp_{n}_qwen")
            else:
                wf = comfy.krea2_t2i(full, neg, width=832, height=1216, steps=8, cfg=1.0, seed=a.seed, gguf=True)
            t = time.time()
            paths, _ = comfy.run(wf, out, prefix=f"{n}_{model}", timeout=3600)
            report.setdefault(n, {})[model] = {"seconds": round(time.time() - t, 1), "file": paths[0].name}
            print(f"[compare] {n} {model}: {time.time() - t:.0f} s", flush=True)
    free()
    pose = pose_model()
    rows = []
    for n, _, style in chars:
        tiles = []
        for model in ("krea2", "qwen21"):
            p = out / report[n][model]["file"]
            im = np.asarray(Image.open(p).convert("RGBA"))
            report[n][model].update(measures(im, pose))
            # the sheet shows a cut-out on a checker so its alpha is seen
            tile = Image.open(p).convert("RGBA")
            chk = np.indices((tile.height // 16 + 1, tile.width // 16 + 1)).sum(0) % 2
            chk = np.kron(chk, np.ones((16, 16)))[:tile.height, :tile.width]
            bg = Image.fromarray(np.uint8(np.stack([200 + 30 * chk] * 3 + [np.full_like(chk, 255)], -1)))
            tiles.append(Image.alpha_composite(bg, tile).convert("RGB").resize((416, 608)))
        row = Image.new("RGB", (832, 608))
        row.paste(tiles[0], (0, 0))
        row.paste(tiles[1], (416, 0))
        rows.append(row)
    sheet = Image.new("RGB", (832 * min(3, len(rows)), 608 * ((len(rows) + 2) // 3)), (255, 255, 255))
    for i, row in enumerate(rows):
        sheet.paste(row, ((i % 3) * 832, (i // 3) * 608))
    sheet.save(out / "sheet.png")
    json.dump(report, open(out / "compare.json", "w"), indent=1)
    for n in report:
        k, q = report[n]["krea2"], report[n]["qwen21"]
        print(f"[compare] {n:6s} krea2 {k['seconds']:5.0f}s frame {k['in_frame']!s:5} arms {k.get('arm_gap', '-')!s:5} | "
              f"qwen21 {q['seconds']:5.0f}s frame {q['in_frame']!s:5} arms {q.get('arm_gap', '-')!s:5} "
              f"edge {q['background']}", flush=True)
    print(f"[compare] -> {out / 'sheet.png'} (each pair: Krea 2 left, Qwen-Image 2.1 right)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="mara,juno3,rowan,wren,aoi,pip")
    ap.add_argument("--out", default="results/v3/image_models")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--qwen-steps", type=int, default=30)
    main(ap.parse_args())
