"""The reference's head, cut out and framed for generating it on its own (E139).

    python pipeline/head_crop.py --work work/<name> --out-dir work/<name>/head

The head is found on the figure's silhouette (the reference's cut-out): from the top of the figure (hair, hat) down
to the neck, the narrowest row of the silhouette between the pose model's nose and its shoulders - a third of the
way below the nose at the least, so a collar that hides the neck (Cadet's armour) cannot put the cut across the
face, and above the shoulders, so long hair down the sides (Aoi's) cannot carry it to the waist. Without the pose
model, between a twelfth and a third of the figure's height below its top. The crop is
a square around the head and its hair or hat, a little of the shoulders below, faded out under the neck so the
generator makes a head and not a torso. Writes head_rgba.png (1024 px) and head_box.json: the crop's window of the
reference (x0, y0, x1, y1, px), the neck's row and the head's height in px.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--res", type=int, default=1024)
a = ap.parse_args()
w = Path(a.work)
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)

if (w / "reference_rgba.png").exists():
    rgba = Image.open(w / "reference_rgba.png").convert("RGBA")
    if (w / "reference_mask.png").exists():           # the cut-out the rest of the pipeline trusts
        rgba.putalpha(Image.open(w / "reference_mask.png").convert("L").resize(rgba.size))
else:
    rgba = Image.open(w / "reference.png").convert("RGBA")
    rgba.putalpha(Image.open(w / "reference_mask.png").convert("L").resize(rgba.size))
A = np.asarray(rgba.getchannel("A")) > 127
A = ndimage.binary_opening(A, iterations=2)
lab, n = ndimage.label(A)
if n > 1:                                              # the figure: the largest piece
    A = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
rows = np.nonzero(A.any(1))[0]
top, bot = int(rows.min()), int(rows.max())
H = bot - top
cols = [np.nonzero(A[r])[0] for r in range(A.shape[0])]


def run(r):
    """The row's run of figure that holds its middle: the head and neck, not a raised hand or a shoulder plate beside them."""
    c = cols[r]
    if len(c) == 0:
        return None
    mid = A.shape[1] / 2 if r <= top else (cols[top + 2].mean() if len(cols[top + 2]) else A.shape[1] / 2)
    segs = np.split(c, np.flatnonzero(np.diff(c) > 1) + 1)
    return min(segs, key=lambda s: abs(s.mean() - mid))


def width(r):
    s_ = run(r)
    return 0 if s_ is None else len(s_)


r0, r1 = top + int(0.08 * H), top + int(0.34 * H)
kp_note = "no pose"
try:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pose_gate import _pose
    rgb = np.asarray(Image.alpha_composite(Image.new("RGBA", rgba.size, (205, 205, 205, 255)), rgba).convert("RGB"))
    kp, sc = _pose()(rgb, A)
    if kp is not None and sc[0] > 0.3 and min(sc[5], sc[6]) > 0.3:
        nose_y, sh_y = float(kp[0][1]), float((kp[5][1] + kp[6][1]) / 2)
        if sh_y > nose_y + 5:
            r0, r1 = int(nose_y + 0.33 * (sh_y - nose_y)), int(sh_y - 0.05 * (sh_y - nose_y))
            kp_note = f"between the nose (row {nose_y:.0f}) and the shoulders (row {sh_y:.0f})"
except Exception as e:                                  # noqa: BLE001
    kp_note = f"no pose ({type(e).__name__})"
wd = np.array([width(r) for r in range(r0, r1)], float)
wd_s = ndimage.uniform_filter1d(wd, 9)
# the neck: the narrowest row before the shoulders widen; the rows past where the width doubles are the shoulders
widest_head = wd_s[: len(wd_s) // 2].max()
neck_i = int(np.argmin(np.where(np.arange(len(wd_s)) > 3, wd_s, np.inf))) if len(wd_s) > 4 else len(wd_s) // 2
neck = r0 + neck_i
head_h = neck - top
# the head's columns: its own run in each row above the neck (Cadet's pauldrons, level with his face, are beside it)
cs = np.concatenate([run(r) for r in range(top, neck) if run(r) is not None])
cx = float(np.median(cs))
half_w = float(np.quantile(np.abs(cs - cx), 0.98))
side = int(max(2 * half_w * 1.25, head_h * 1.35))
cy = top + head_h * 0.56
x0, y0 = int(round(cx - side / 2)), int(round(cy - side * 0.5))
x1, y1 = x0 + side, y0 + side
# above the neck only the head's own run of each row: a shoulder plate level with the face (Cadet's) is not the head's
keep = np.ones(A.shape, bool)
for r in range(top, neck):
    s_ = run(r)
    keep[r] = False
    if s_ is not None:
        keep[r, s_.min():s_.max() + 1] = True
alpha = np.asarray(rgba.getchannel("A"), np.float32) * keep
rgba = rgba.copy()
rgba.putalpha(Image.fromarray(alpha.astype(np.uint8)))
crop = rgba.crop((x0, y0, x1, y1))
al = np.asarray(crop.getchannel("A"), np.float32)
# faded out from just under the neck
nr = (neck - y0) / side
ramp = np.clip((np.arange(al.shape[0]) / al.shape[0] - (nr + 0.04)) / 0.08, 0, 1)[:, None]
crop.putalpha(Image.fromarray((al * (1 - ramp)).astype(np.uint8)))
crop = crop.resize((a.res, a.res), Image.LANCZOS)
crop.save(out / "head_rgba.png")
json.dump({"box": [x0, y0, x1, y1], "neck_row": int(neck), "top_row": top, "head_px": int(head_h),
           "figure_px": int(H)}, open(out / "head_box.json", "w"), indent=1)
print(f"[head] crop {x0},{y0},{x1},{y1} of the reference: head {head_h} px of a {H} px figure "
      f"({head_h / H:.2f}), neck at row {neck}, searched {kp_note} -> {out / 'head_rgba.png'}", flush=True)
