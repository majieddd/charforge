"""What the multiview pass changes, on characters already built (experiment E106, first pilot).

    python tools/multiview_ablation.py [--name aoi,bo] [--json research/data/e106_multiview.json]

The multiview stage repaints side and back renders of TRELLIS's first model with an image model and
runs TRELLIS again on the reference plus those views (mesh.glb from pass1.glb): a median 16 minutes of
a 41-minute build. Every character built with it keeps both models, so the two can be compared without
building anything. Each is rendered from the front, left, back and right (render_views.py, orthographic,
level), and measured:

  front IoU     the front silhouette on the reference picture's (the generate stage's own gate)
  cross IoU     the two models' silhouettes on each other at each side: how much the second pass
                changes the shape (1.0 = not at all)
  detail        the variance of the Laplacian of luminance inside each silhouette (eroded 3 px, so the
                outline does not count): how much surface detail the colour carries, per side

There is no ground truth for the side and the back - the repainted views are the second pass's own
input - so the pilot says how big the change is and in which direction the proxies move; whether it
is better is a human judgement (the sheets in work/<name>/qa/e106/).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.argv, _argv = [sys.argv[0]], sys.argv
import charforge  # noqa: E402
sys.argv = _argv
from align import image_mask, similarity  # noqa: E402

SIDES = ("front", "left", "back", "right")


def render(glb, out):
    if (out / "view_03.png").exists() and (out / "view_03.png").stat().st_mtime > glb.stat().st_mtime:
        return
    cmd = [charforge.blender_bin(), "-b", "-noaudio", "--python", str(ROOT / "blender" / "render_views.py"), "--",
           "--mesh", str(glb), "--out", str(out), "--views", "4", "--res", "512", "--elev", "0", "--ortho", "2.15"]
    with charforge.gpu(f"E106 renders of {glb.parent.name}/{glb.name}"):
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not (out / "view_03.png").exists():
        raise SystemExit("\n".join((r.stdout + r.stderr).splitlines()[-10:]))


def load(p):
    a = np.asarray(Image.open(p).convert("RGBA")).astype(np.float32)
    return a[..., :3] / 255.0, a[..., 3] > 127


def detail(rgb, mask):
    from scipy import ndimage
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    lap = ndimage.laplace(lum)
    inner = ndimage.binary_erosion(mask, iterations=3)
    return float(lap[inner].var()) if inner.any() else 0.0


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def one(name):
    w = ROOT / "work" / name
    p1, mv = w / "pass1.glb", w / "mesh.glb"
    if not (p1.exists() and mv.exists()):
        return None
    if p1.read_bytes() == mv.read_bytes():
        return {"same": True}
    d = w / "qa" / "e106"
    render(p1, d / "pass1")
    render(mv, d / "mesh")
    mask_p = w / "reference_mask.png"
    ref = (np.asarray(Image.open(mask_p).convert("L")) > 127 if mask_p.exists()
           else image_mask(np.asarray(Image.open(w / "reference.png").convert("RGB"))))
    row = {"same": False, "sides": {}}
    ims = {}
    for k, side in enumerate(SIDES):
        (r1, m1), (r2, m2) = load(d / "pass1" / f"view_{k:02d}.png"), load(d / "mesh" / f"view_{k:02d}.png")
        ims[side] = (d / "pass1" / f"view_{k:02d}.png", d / "mesh" / f"view_{k:02d}.png")
        row["sides"][side] = {"cross_iou": round(iou(m1, m2), 4), "detail_pass1": round(detail(r1, m1), 6),
                              "detail_mesh": round(detail(r2, m2), 6)}
    _, m1 = load(d / "pass1" / "view_00.png")
    _, m2 = load(d / "mesh" / "view_00.png")
    row["front_iou_pass1"] = round(float(similarity(m1, ref)[1]), 4)
    row["front_iou_mesh"] = round(float(similarity(m2, ref)[1]), 4)
    # the sheet: first pass above, second pass below, front / left / back / right
    tiles = [[Image.open(ims[s][i]).convert("RGBA") for s in SIDES] for i in (0, 1)]
    W, H = tiles[0][0].size
    sheet = Image.new("RGB", (W * 4, (H + 24) * 2), (238, 238, 240))
    dr = ImageDraw.Draw(sheet)
    for i, lab in enumerate(("first pass (pass1.glb)", "after multiview (mesh.glb)")):
        dr.text((8, i * (H + 24) + 6), f"{name}: {lab} - front, left, back, right", fill=(20, 20, 20))
        for j, t in enumerate(tiles[i]):
            sheet.paste(t, (j * W, i * (H + 24) + 24), t)
    sheet.save(d / "sheet.png")
    return row


def main(a):
    names = [n for n in a.name.split(",") if n] if a.name else sorted(
        p.parent.name for p in (ROOT / "work").glob("*/mesh.glb") if p.parent.name not in ("juno",))
    out = {}
    for n in names:
        row = one(n)
        if row is None:
            continue
        out[n] = row
        if row["same"]:
            print(f"[e106] {n:9s} multiview fell back to the first pass (same model)", flush=True)
            continue
        s = row["sides"]
        print(f"[e106] {n:9s} front IoU {row['front_iou_pass1']:.3f} -> {row['front_iou_mesh']:.3f}  cross IoU "
              + " ".join(f"{k} {v['cross_iou']:.3f}" for k, v in s.items())
              + "  detail back " + f"{s['back']['detail_pass1']:.4f} -> {s['back']['detail_mesh']:.4f}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps({"_about": "tools/multiview_ablation.py: pass1.glb against mesh.glb (E106 pilot)",
                                            "characters": out}, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", default="")
    ap.add_argument("--json", default=str(ROOT / "research" / "data" / "e106_multiview.json"))
    main(ap.parse_args())
