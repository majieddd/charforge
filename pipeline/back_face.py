"""Is there a face on the back of the head? DWPose's face points on a render of the model from behind (E145).

    vendor/trellis2mlx/.venv/bin/python pipeline/back_face.py --render gate/<tag>/back_180.png --out gate/<tag>/back_face.json

TRELLIS now and then models a head with a face on both sides - Vex's back of head had eyes, a nose and a mouth,
sculpted and painted - and nothing later in the pipeline looks at the back of the head. On the back of an ordinary
head DWPose's face points score 0.34-0.55 (it finds a ghost of a face in any hair); Vex's scored 0.97.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from dwpose import DWPose, read_face  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--render", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
im = Image.open(a.render).convert("RGBA")
al = np.asarray(im)[..., 3] > 127
bg = Image.new("RGBA", im.size, (205, 205, 205, 255))
bg.alpha_composite(im)
face, sc = read_face(DWPose(), np.asarray(bg.convert("RGB")), al)
res = {"face": round(float(np.median(sc)), 3), "eyes": round(float(np.mean(sc[36:48])), 3),
       "nose": round(float(np.mean(sc[27:36])), 3)}
json.dump(res, open(a.out, "w"))
print(f"[back_face] a face on the back of the head: {res['face']:.2f} (eyes {res['eyes']:.2f})", flush=True)
