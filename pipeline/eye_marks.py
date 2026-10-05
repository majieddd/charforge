"""Where the model's eyes are, on blender/eye_fill.py's front render: DWPose's six points round each eye (E140).

    vendor/trellis2mlx/.venv/bin/python pipeline/eye_marks.py --render eyes/front.png --out eyes/marks.json \
        [--debug eyes/marks.png]

The render is the whole figure; DWPose is given a portrait crop round the face (read_face), so the face fills its
input. The 68 face points are COCO-WholeBody's 23-90; 36-41 and 42-47 are the eyes. An eye is
reported with its points' mean confidence and used by eye_fill.py above min_score.
A drawn face can defeat it: on Aoi's model (anime eyes as large cavities, the generator's paint cracked across
them) it scored 0.2, and those eyes are left as generated (see blender/eye_fill.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from dwpose import DWPose, read_face  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--render", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--debug", default=None)
ap.add_argument("--min-score", type=float, default=0.5)
ap.add_argument("--reference", default=None, help="also the picture's 68 points, for the texture stage's per-part "
                                                   "alignment (pipeline/project_texture.py --face-marks)")
ap.add_argument("--ref-mask", default=None)
a = ap.parse_args()


im = Image.open(a.render).convert("RGBA")
alpha = np.asarray(im)[..., 3] > 127
bg = Image.new("RGBA", im.size, (205, 205, 205, 255))
bg.alpha_composite(im)
rgb = np.asarray(bg.convert("RGB"))
dw = DWPose()
face, fs = read_face(dw, rgb, alpha)
out = {"min_score": a.min_score, "face_score": round(float(np.median(fs)), 3), "face": face.tolist(),
       "face_scores": [round(float(x), 3) for x in fs]}
for side, idx in (("left", range(36, 42)), ("right", range(42, 48))):
    out[side] = {"points": face[list(idx)].tolist(), "score": round(float(fs[list(idx)].mean()), 3)}
if a.reference:
    from scipy import ndimage
    ref = np.asarray(Image.open(a.reference).convert("RGB"))
    if a.ref_mask:
        rm = np.asarray(Image.open(a.ref_mask).convert("L").resize(ref.shape[1::-1])) > 127
        rm = ndimage.binary_fill_holes(rm)
    else:
        rm = np.ones(ref.shape[:2], bool)
    fr, sr = read_face(dw, ref, rm)
    out["picture"] = {"face": fr.tolist(), "face_scores": [round(float(x), 3) for x in sr],
                      "face_score": round(float(np.median(sr)), 3), "size": list(ref.shape[1::-1])}
    print(f"[eye_marks] the picture's face {out['picture']['face_score']:.2f}", flush=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"[eye_marks] face {out['face_score']:.2f}, eyes {out['left']['score']:.2f} / {out['right']['score']:.2f}",
      flush=True)
if a.debug:
    pts = face[np.isfinite(face).all(1)]
    w = float(np.ptp(pts[:, 0]))
    x0, y0 = pts.min(0) - 0.4 * w
    x1, y1 = pts.max(0) + 0.4 * w
    c = bg.convert("RGB").crop((int(x0), int(y0), int(x1), int(y1)))
    k = 600 / c.width
    c = c.resize((600, int(c.height * k)))
    d = ImageDraw.Draw(c)
    for i, (x, y) in enumerate(face):
        col = (230, 30, 30) if 36 <= i < 48 else (0, 160, 0)
        X, Y = (x - x0) * k, (y - y0) * k
        d.ellipse((X - 3, Y - 3, X + 3, Y + 3), outline=col, width=2)
    c.save(a.debug)
