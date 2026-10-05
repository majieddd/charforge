"""How alike a model's face and its picture look, part by part: DINOv3 features of each part, the render laid onto
the picture by their face points (tools/face_score.py's), compared by cosine (E140).

    .venv/bin/python tools/face_likeness.py --score qa/face_review.json

Landmark error misses what the eye sees: Mara's smeared, squinting v0.12 face had every point within 3% of the
picture's. Features from a self-supervised vision model (DINOv3 ViT-L/16, models/dinov3-vitl16-hf) are robust to the
render's lighting and framing but not to a smeared eye or a doubled lid. The render is warped onto the picture's
frame by the similarity through the two sets of points; then the whole face, the eyes, the nose and the mouth are cut
from both at boxes round the picture's points and compared (mean of the patch tokens and the class token).
Writes the likeness into the same JSON.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--score", required=True)
ap.add_argument("--debug", default=None, help="save the picture's and the render's crops side by side here")
a = ap.parse_args()
S = json.load(open(a.score))
Lm, Lp = np.array(S["model_pts"]), np.array(S["picture_pts"])
ok = np.isfinite(Lm).all(1) & np.isfinite(Lp).all(1)
A, B = Lm[ok], Lp[ok]
ma, mb = A.mean(0), B.mean(0)
U, Sv, Vt = np.linalg.svd((B - mb).T @ (A - ma) / len(A))
d = np.sign(np.linalg.det(U @ Vt)) or 1.0
Rm = U @ np.diag([1.0, d]) @ Vt
c = (Sv * [1.0, d]).sum() / ((A - ma) ** 2).sum() * len(A)
# picture px -> render px: the inverse of render -> picture (x -> c R (x - ma) + mb)
inv = np.linalg.inv(c * Rm)


def rgb(path):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (205, 205, 205, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")


pic, ren = rgb(S["picture"]), rgb(S["render"])
# the render resampled into the picture's frame
M = inv                                  # render = inv @ (pic - mb) + ma
off = ma - inv @ mb
ren_w = ren.transform(pic.size, Image.AFFINE, (M[0, 0], M[0, 1], off[0], M[1, 0], M[1, 1], off[1]), Image.BICUBIC)
model = AutoModel.from_pretrained(str(ROOT / "models" / "dinov3-vitl16-hf")).eval()
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def feats(im):
    x = (np.asarray(im.resize((224, 224), Image.BICUBIC), np.float32) / 255.0 - MEAN) / STD
    with torch.no_grad():
        h = model(pixel_values=torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float32)).last_hidden_state[0]
    return torch.cat([h[0], h[5:].mean(0)]).numpy()       # the class token, and the patches' mean (after 4 registers)


def box(idx, pad):
    P = Lp[idx]
    P = P[np.isfinite(P).all(1)]
    (x0, y0), (x1, y1) = P.min(0), P.max(0)
    s = max(x1 - x0, y1 - y0) * (1 + pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))


PARTS = {"face": (list(range(0, 27)), 0.25), "eyes": (list(range(36, 48)), 0.35), "nose": (list(range(27, 36)), 0.6),
         "mouth": (list(range(48, 68)), 0.6)}
like = {}
tiles = []
for k, (idx, pad) in PARTS.items():
    b = box(idx, pad)
    tiles += [pic.crop(b).resize((160, 160)), ren_w.crop(b).resize((160, 160))]
    f1, f2 = feats(pic.crop(b)), feats(ren_w.crop(b))
    like[k] = round(float(f1 @ f2 / np.linalg.norm(f1) / np.linalg.norm(f2)), 4)
if a.debug:
    g = Image.new("RGB", (160 * len(tiles), 160))
    for i, t in enumerate(tiles):
        g.paste(t, (160 * i, 0))
    g.save(a.debug)
S["likeness"] = like
json.dump(S, open(a.score, "w"), indent=1)
print("[face_likeness] " + ", ".join(f"{k} {v:.3f}" for k, v in like.items()), flush=True)
