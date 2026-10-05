"""How closely a model's face, rendered from the front, follows its picture: DWPose's 68 face points on both (E140).

    vendor/trellis2mlx/.venv/bin/python tools/face_score.py --render qa/face_review_parts/<n>/az0_rest.png \
        --picture work/<n>/reference.png [--mask work/<n>/reference_mask.png] --out qa/face_score.json

The model's points are fitted onto the picture's by one similarity (scale, turn, shift - the render's framing is
not the picture's), and what is left is the face's own error: the mean distance per part as a share of the
picture's eye spacing (outer corners, 36-45), the NME of face alignment. Also how surely each face reads as one
(DWPose's median confidence): a smeared or doubled feature reads less surely.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dwpose import DWPose, read_face  # noqa: E402

PARTS = {"jaw": range(0, 17), "brows": range(17, 27), "nose": range(27, 36), "eyes": range(36, 48),
         "mouth": range(48, 68)}

ap = argparse.ArgumentParser()
ap.add_argument("--render", required=True)
ap.add_argument("--picture", required=True)
ap.add_argument("--mask", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def load(path, mask=None):
    im = Image.open(path).convert("RGBA")
    al = np.asarray(im)[..., 3] > 127
    bg = Image.new("RGBA", im.size, (205, 205, 205, 255))
    bg.alpha_composite(im)
    if mask:
        al = np.asarray(Image.open(mask).convert("L").resize(im.size)) > 127
    if al.mean() > 0.98 or al.sum() < 100:
        al = np.ones_like(al)
    return np.asarray(bg.convert("RGB")), al


dw = DWPose()
rm, am = load(a.render)
rp, ap_ = load(a.picture, a.mask)
Lm, sm = read_face(dw, rm, am)
Lp, sp = read_face(dw, rp, ap_)
ok = np.isfinite(Lm).all(1) & np.isfinite(Lp).all(1)
A, B = Lm[ok], Lp[ok]
ma, mb = A.mean(0), B.mean(0)
U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma) / len(A))
d = np.sign(np.linalg.det(U @ Vt)) or 1.0
Rm = U @ np.diag([1.0, d]) @ Vt
c = (S * [1.0, d]).sum() / ((A - ma) ** 2).sum() * len(A)
fit = np.full_like(Lm, np.nan)
fit[ok] = c * (A - ma) @ Rm.T + mb
iod = float(np.linalg.norm(Lp[36] - Lp[45]))
err = np.linalg.norm(fit - Lp, axis=1) / iod
out = {"render": str(a.render), "picture": str(a.picture), "model_pts": Lm.tolist(), "picture_pts": Lp.tolist(),
       "conf_model": round(float(np.median(sm)), 3), "conf_picture": round(float(np.median(sp)), 3),
       "nme": round(float(np.nanmean(err)), 4),
       "nme_parts": {k: round(float(np.nanmean(err[list(v)])), 4) for k, v in PARTS.items()}}
json.dump(out, open(a.out, "w"), indent=1)
print(f"[face_score] conf {out['conf_model']:.2f} (picture {out['conf_picture']:.2f}), NME {out['nme']:.1%}: "
      + ", ".join(f"{k} {v:.1%}" for k, v in out["nme_parts"].items()), flush=True)
