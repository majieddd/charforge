"""The face, close up, from the reference: the source a sharp face texture is projected from.

    python face_detail.py --dir texproj --reference reference.png [--mask reference_mask.png]

A full-body reference gives the face about a hundred pixels (Juno's: ~95 across), while the
texture has room for ~270 texels there - so however well it is projected, a face painted from
the reference is soft, and the texture's resolution goes unused. This crops the face out of the
reference exactly where blender/uv_maps.py's close-up "face" camera looks: the front view's
render is laid over the reference by silhouette (the alignment project_texture.py uses), and the
face camera's frame is carried through the same transform. The crop is enlarged to the face
camera's resolution and written as face_src.png (with face_src_mask.png, the reference's own
foreground cut the same way).

The orchestrator then runs it through an image-to-image diffusion pass at low strength - the
enlargement fixes the layout and the colours, the diffusion puts back eyes, lashes, lips and
skin that a 5x enlargement cannot - and projects the result through the face camera, weighted
above the full-body view.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import image_mask, similarity  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="uv_maps.py's output (views.json with a 'face' view)")
ap.add_argument("--reference", required=True)
ap.add_argument("--mask", default=None, help="the reference's foreground (pipeline/foreground.py)")
a = ap.parse_args()

meta = json.load(open(os.path.join(a.dir, "views.json")))
vm = {v["tag"]: v for v in meta["views"]}
if not meta.get("face_frame") or "000" not in vm:
    raise SystemExit("[face_detail] views.json has no front view or no face frame - nothing to do")
v0, vf = vm["000"], meta["face_frame"]
rend = np.asarray(Image.open(os.path.join(a.dir, "view_000_albedo.png")).convert("RGBA"))
rmask = rend[..., 3] > 127
ref = Image.open(a.reference).convert("RGB")
img = np.asarray(ref)
if a.mask and os.path.exists(a.mask):
    imask_img = Image.open(a.mask).convert("L").resize(ref.size)
    imask = np.asarray(imask_img) > 127
else:
    imask = image_mask(img)
    imask_img = Image.fromarray((imask * 255).astype(np.uint8))
(s, tx, ty), iou = similarity(rmask, imask)

# the face camera's frame, in front-render pixels, then in reference pixels
half0 = v0["ortho_scale"] / 2
res0 = v0["res"]
cx, _, cz = vf["target"]
u = (cx / half0 * 0.5 + 0.5) * res0
w = (1.0 - (cz / half0 * 0.5 + 0.5)) * res0
half_px = vf["ortho_scale"] / 2 / half0 * res0 / 2
U, W, Hp = u * s + tx, w * s + ty, half_px * s
box = (int(round(U - Hp)), int(round(W - Hp)), int(round(U + Hp)), int(round(W + Hp)))


def crop(im, fill):
    """The box out of the image, padded past its edges with the fill."""
    out = Image.new(im.mode, (box[2] - box[0], box[3] - box[1]), fill)
    out.paste(im, (-box[0], -box[1]))
    return out


bg = tuple(int(c) for c in np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]), 0))
res = int(vf["res"])
src = crop(ref, bg).resize((res, res), Image.LANCZOS)
msk = crop(imask_img, 0).resize((res, res), Image.NEAREST)
src.save(os.path.join(a.dir, "face_src.png"))
msk.save(os.path.join(a.dir, "face_src_mask.png"))
json.dump({"box": box, "iou": float(iou), "source_px": box[2] - box[0], "res": res},
          open(os.path.join(a.dir, "face_src.json"), "w"), indent=1)
print(f"[face_detail] the face camera's frame is {box[2] - box[0]} px of the reference (front view IoU "
      f"{iou:.3f}); enlarged {res / max(box[2] - box[0], 1):.1f}x -> {os.path.join(a.dir, 'face_src.png')}",
      flush=True)
