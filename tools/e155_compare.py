#!/usr/bin/env python3
"""E155 - Pixal3D (pixel-aligned TRELLIS.2, the Apple Silicon build, sv weights) against the TRELLIS.2 pass1 the build kept.

    ../.venv/bin/python tools/e155_compare.py --names mara,rowan,hana,pip,aoi,knight [--seed 7] [--force] [--generate-only]

For each character, from the same picture (work/<n>/reference.png with the cut-out mask as its alpha):
  trellis2  work/<n>/pass1.glb, the model the build kept (TRELLIS.2 with its own background remover)
  pixal3d   work/_e155/<n>/pixal3d.glb, one raw Pixal3D single-view run (--decim 0), brought into pass1's convention by
            blender/fix_pixal3d.py: work/_e155/<n>/pixal3d_pass1.glb (pipeline/pixal3d_generate.py)
Both are judged the same way:
  front_iou, back_face   the generate stage's own gates (charforge.front_iou, charforge.back_face)
  head close-up          framed on the face DWPose finds on the model's own front view (not a fixed box): centred on the
                         face, 2.6 face heights high; without a face found, on the neck (the narrowest slice of the top 30%)
  nme, nme_parts         DWPose landmark error of that close-up against the picture's face (tools/face_score.py)
  likeness               DINOv3 likeness of the face, eyes, nose and mouth (tools/face_likeness.py)
  raking                 the surface under a raking light, front and side (blender/render_raking.py)
  shape                  noise, lumps, symmetry and triangle quality of the surface (blender/model_quality.py, tools/model_quality.py)
Writes work/_e155/<n>/results.json. Attempts: "v1" (the first run) and "v2" (its Pixal3D side was judged on the raw,
unturned output: the turn never reached the mesh - see v2_note) are kept as they were; "v3" is the attempt the figures
in the lane report come from. A TRELLIS.2 figure of v2 is carried into v3 when its pass1.glb is older than v2 (the
model and the gates are the same); the Pixal3D side is judged again whenever its fixed mesh was made again.
"""
import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE))
import charforge  # noqa: E402  (this worktree's charforge: its work/ and gates)
from pipeline import pixal3d_generate as p3d  # noqa: E402

TPY = charforge.TRELLIS / ".venv" / "bin" / "python"          # DWPose and the back-face gate run here (onnxruntime)
PY = sys.executable                                            # DINOv3 likeness runs here (torch, transformers)
TOOLS = WORKTREE / "tools"
HEAD_FACE_MIN_CONF = 0.3
V2_NOTE = ("the Pixal3D sides of v1 and v2 were judged on the raw output, unturned: the 180-degree turn in "
           "blender/fix_pixal3d.py never reached the mesh (the object is in quaternion rotation mode, so "
           "rotation_euler was ignored). Their back-face, head and likeness figures are not valid; the TRELLIS.2 "
           "figures are.")


def face_frame_main(view: Path, out: Path):
    """Run with TPY: DWPose's face on a front view (its figure's own alpha), as a close-up's centre and height."""
    from PIL import Image
    sys.path.insert(0, str(TOOLS))
    from dwpose import DWPose, read_face
    im = Image.open(view).convert("RGBA")
    al = np.asarray(im)[..., 3] > 127
    rows = np.nonzero(al.any(1))[0]
    top, bot = int(rows.min()), int(rows.max())
    bg = Image.new("RGBA", im.size, (205, 205, 205, 255))
    bg.alpha_composite(im)
    pts, sc = read_face(DWPose(), np.asarray(bg.convert("RGB")), al)
    ok = np.isfinite(pts).all(1)
    res = {"conf": round(float(np.median(sc[ok])), 3) if ok.any() else 0.0, "points": int(ok.sum())}
    if ok.sum() >= 40:
        y0, y1 = float(pts[ok, 1].min()), float(pts[ok, 1].max())
        H = bot - top + 1
        res.update(z=round((bot - (y0 + y1) / 2) / H, 3),
                   span=round(min(0.35, max(0.08, 2.6 * (y1 - y0) / H)), 3))
    json.dump(res, open(out, "w"))


def neck_frame(glb: Path):
    """Without a face: the close-up on the head, from the model's own neck (the narrowest slice from 70% to 92% of the height)."""
    import trimesh
    m = trimesh.load(glb, force="mesh", process=False)
    V = np.asarray(m.vertices)
    y = V[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    H = y1 - y0
    best = None
    for f in np.arange(0.70, 0.92, 0.005):
        sl = V[(y >= y0 + f * H) & (y < y0 + (f + 0.01) * H)]
        if len(sl) < 20:
            continue
        w = float(np.ptp(sl[:, 0]))
        if best is None or w < best[0]:
            best = (w, f)
    neck = best[1] if best else 0.86
    return round((neck + 1.0) / 2, 3), round(min(0.35, 1.25 * (1.0 - neck)), 3)


def run(cmd, r, stage, gpu=None):
    """A subprocess under the GPU lock when it runs a model on the GPU; its output is in the character's logs."""
    if gpu:
        with charforge.gpu(gpu):
            r.sh(stage, cmd, keep=())
    else:
        r.sh(stage, cmd, keep=())


def judge(r, glb: Path, tag: str, pic: Path, mask: Path) -> dict:
    d = r.work / "gate" / tag
    res = {"glb": str(glb.relative_to(WORKTREE)) if glb.is_relative_to(WORKTREE) else str(glb)}
    res["front_iou"] = round(charforge.front_iou(r, glb, tag), 3)
    res["back_face"] = round(charforge.back_face(r, glb, tag), 3)
    view = d / "view_00.png"
    frame_f = d / "head_frame.json"
    run([TPY, Path(__file__), "--face-frame", view, "--out", frame_f], r, "e155_head_frame", gpu="E155 head frame")
    fr = json.load(open(frame_f))
    if "z" in fr and fr["conf"] >= HEAD_FACE_MIN_CONF:
        zc, span, how = fr["z"], fr["span"], "face"
    else:
        zc, span = neck_frame(glb)
        how = "neck"
    res["head"] = {"method": how, "z": zc, "span": span, "face_conf": fr["conf"]}
    r.bl("e155_head", "render_lit.py", "--mesh", glb, "--out", d / "head", "--az", "0", "--res", 1024,
         "--z", zc, "--span", span, keep=())
    shot = next(iter(sorted(d.glob("head_*0*.png"))), None) or next(iter(sorted(d.glob("head*.png"))), None)
    if shot is None:
        return res
    out = d / "face.json"
    run([TPY, TOOLS / "face_score.py", "--render", shot, "--picture", pic, "--mask", mask, "--out", out], r,
        "e155_face_score", gpu="E155 face score")
    S = json.load(open(out))
    res["nme"] = S["nme"]
    res["nme_parts"] = S["nme_parts"]
    res["face_conf_model"] = S["conf_model"]
    run([PY, TOOLS / "face_likeness.py", "--score", out], r, "e155_face_likeness", gpu="E155 likeness")
    S = json.load(open(out))
    res["likeness"] = S.get("likeness")
    r.bl("e155_raking", "render_raking.py", "--mesh", glb, "--out", d / "raking", "--az", "0,90", "--weld", "--res", 600,
         keep=())
    q = d / "quality.npz"
    run([charforge.blender_bin(), "-b", "-noaudio", "--python", WORKTREE / "blender" / "model_quality.py", "--",
         "--mesh", glb, "--height", "1.75", "--out", q], r, "e155_quality")
    run([PY, TOOLS / "model_quality.py", "--npz", q, "--out", d / "quality.json"], r, "e155_quality_shape")
    res["shape"] = json.load(open(d / "quality.json"))["shape"]
    return res


def tri_faces(glb: Path) -> int:
    import trimesh
    return int(len(trimesh.load(glb, force="mesh", process=False).faces))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--names", default="mara,rowan,hana,pip,aoi,knight")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true", help="judge again (the models are kept)")
    ap.add_argument("--refix", action="store_true", help="make the fixed Pixal3D meshes again from their raw runs")
    ap.add_argument("--generate-only", action="store_true", help="make the Pixal3D models, do not judge them")
    ap.add_argument("--summary", action="store_true", help="print the table from the results.json files")
    ap.add_argument("--face-frame", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.face_frame:
        return face_frame_main(a.face_frame, a.out)
    if a.summary:
        return summary([x.strip() for x in a.names.split(",") if x.strip()])
    for n in [x.strip() for x in a.names.split(",") if x.strip()]:
        r = charforge.Run(types.SimpleNamespace(name=n, seed=a.seed, keep_comfy_loaded=True))
        outd = WORKTREE / "work" / "_e155" / n
        outd.mkdir(parents=True, exist_ok=True)
        res_f = outd / "results.json"
        old = json.load(open(res_f)) if res_f.exists() else {}
        # the earlier attempts are kept as they were written
        res = {k: old[k] for k in ("v1", "v2", "v2_note") if k in old}
        res.setdefault("v2_note", V2_NOTE)
        pic, mask = r.path("reference.png"), r.path("reference_mask.png")
        raw, fixed = outd / "pixal3d.glb", outd / "pixal3d_pass1.glb"
        t0 = time.time()
        info = p3d.produce(pic, mask, raw, fixed, a.seed, refix=a.refix)
        res["pixal3d_run"] = info.get("run") or old.get("pixal3d_run")
        res["pixal3d_fix"] = info.get("fix")
        res["pixal3d_facing"] = {k: info[k] for k in ("turn", "face_front", "face_back") if k in info}
        res["pixal3d_faces"] = tri_faces(fixed)
        res["trellis2_faces"] = tri_faces(r.path("pass1.glb"))
        print(f"[e155] {n}: Pixal3D ready in {time.time() - t0:.0f}s, turned {info.get('turn')} deg, face front "
              f"{info.get('face_front')} back {info.get('face_back')}", flush=True)
        if a.generate_only:
            json.dump(res, open(res_f, "w"), indent=1)
            continue
        v3 = res.setdefault("v3", {k: val for k, val in (old.get("v3") or {}).items() if k == "trellis2"})
        if info.get("made") and "pixal3d" in (old.get("v3") or {}):
            # the figures of the fixed mesh before it was welded (E155 v3a): kept, not current
            res["v3_preweld"] = {"pixal3d": old["v3"]["pixal3d"]}
        if "pixal3d" not in v3 or a.force or info.get("made"):
            v3["pixal3d"] = judge(r, fixed, "e155_pixal3d", pic, mask)
            json.dump(res, open(res_f, "w"), indent=1)
        v2t = (old.get("v2") or {}).get("trellis2")
        carry = (v2t is not None and "trellis2" not in v3 and not a.force
                 and r.path("pass1.glb").stat().st_mtime < res_f.stat().st_mtime)
        if carry:
            v3["trellis2"] = dict(v2t, carried_from="v2")
            print(f"[e155] {n}: TRELLIS.2 figures carried from v2 (same pass1.glb, same gates)", flush=True)
        elif "trellis2" not in v3 or a.force:
            v3["trellis2"] = judge(r, r.path("pass1.glb"), "e155_trellis2", pic, mask)
        json.dump(res, open(res_f, "w"), indent=1)
        print(f"[e155] {n}: pixal3d  {json.dumps(v3['pixal3d'])[:500]}", flush=True)
        print(f"[e155] {n}: trellis2 {json.dumps(v3['trellis2'])[:500]}", flush=True)


def summary(names):
    """The table from each character's results.json: every attempt (v1, v2, v3) beside the others, TRELLIS.2 against Pixal3D."""
    rows = []
    for n in names:
        f = WORKTREE / "work" / "_e155" / n / "results.json"
        if not f.exists():
            continue
        R = json.load(open(f))
        for key in ("v1", "v2", "v3"):
            d = R.get(key) or {}
            for model in ("trellis2", "pixal3d"):
                M = d.get(model)
                if not M:
                    continue
                lk = M.get("likeness") or {}
                sh = M.get("shape") or {}
                rows.append((n, key, model, M.get("front_iou"), M.get("back_face"), M.get("nme"), lk.get("face"),
                             lk.get("eyes"), M.get("head", {}).get("method", "-"), M.get("face_conf_model"), sh))
    print(f"{'char':<8}{'att':<4}{'model':<9}{'IoU':>6}{'back':>7}{'NME':>8}{'face':>7}{'eyes':>7}{'head':>7}{'conf':>6}")
    for r_ in rows:
        f_ = lambda v, w, p: f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
        print(f"{r_[0]:<8}{r_[1]:<4}{r_[2]:<9}{f_(r_[3], 6, 3)}{f_(r_[4], 7, 3)}{f_(r_[5], 8, 4)}{f_(r_[6], 7, 3)}"
              f"{f_(r_[7], 7, 3)}{r_[8]:>7}{f_(r_[9], 6, 2)}")
    print("shape:")
    for r_ in rows:
        if r_[10]:
            print(f"  {r_[0]:<8}{r_[1]:<4}{r_[2]:<9}" + json.dumps(r_[10])[:300])


if __name__ == "__main__":
    main()
