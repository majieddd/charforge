"""Cut a character out of its background: writes the image's foreground mask as a PNG.

Texture projection aligns the mesh to each source image by silhouette, which needs the image's
own silhouette. A generated reference sits on a plain field and a colour threshold finds it;
a painting does not - Vex stands in a street at dusk, and the threshold found 1% of her. This
runs the same background remover the generation stage uses (rembg's u2net, already on disk for
TRELLIS), so it needs that stage's environment:

    vendor/trellis2mlx/.venv/bin/python pipeline/foreground.py --image reference.png --out mask.png
    vendor/trellis2mlx/.venv/bin/python pipeline/foreground.py --video move.mp4 --out masks.npz

--video cuts out every frame of a video (tools/motion_fidelity.py compares a character's silhouette
with a generated video's figure): the studio behind H3's figures is darker in the middle than at
the edges, and the colour threshold took half the background for the figure.
"""
import argparse

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=None)
ap.add_argument("--video", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="u2net")
a = ap.parse_args()

from rembg import new_session, remove  # noqa: E402

if a.video:
    import cv2
    session = new_session(a.model)
    cap = cv2.VideoCapture(a.video)
    masks = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cut = remove(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), session=session, only_mask=True)
        masks.append(np.asarray(cut.convert("L")) > 127)
    np.savez_compressed(a.out, masks=np.array(masks))
    print(f"[foreground] {a.video}: {len(masks)} frames, the figure {np.mean(masks):.1%} of each -> {a.out}", flush=True)
    raise SystemExit(0)
img = Image.open(a.image).convert("RGB")
cut = remove(img, session=new_session(a.model), only_mask=True)
m = np.asarray(cut.convert("L"))
Image.fromarray(((m > 127) * 255).astype(np.uint8)).save(a.out)
print(f"[foreground] {a.image}: {(m > 127).mean():.1%} of the image is the character -> {a.out}", flush=True)
