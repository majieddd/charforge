"""Headless Hunyuan3D-2.1 shape generation, for the same-input comparison against TRELLIS.2.

The Mac port ships only a Gradio UI, so this drives its pipeline directly with the same
settings its interface uses. Run it with the port's own venv:

    vendor/hunyuan3d-2.1-mac-rocm/venv/bin/python pipeline/run_hunyuan.py \
        --image work/char01/ref_9_0.png --out work/char01/hunyuan_mesh.glb
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PORT = Path(__file__).resolve().parents[1] / "vendor" / "hunyuan3d-2.1-mac-rocm"
sys.path.insert(0, str(PORT))
sys.path.insert(0, str(PORT / "Hunyuan3D-2.1"))
# hy3dshape and hy3dpaint are nested one level deeper inside the repo
for _sub in ("hy3dshape", "hy3dpaint"):
    _p = PORT / "Hunyuan3D-2.1" / _sub
    if _p.is_dir():
        sys.path.insert(0, str(_p))
_CWD = Path.cwd()   # resolve user paths before chdir into the port dir

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=5.0)
ap.add_argument("--octree", type=int, default=192)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
a.image = str((_CWD / a.image).resolve()) if not Path(a.image).is_absolute() else a.image
a.out = str((_CWD / a.out).resolve()) if not Path(a.out).is_absolute() else a.out
os.chdir(PORT)

import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

import backend as backend_mod  # noqa: E402
import compat_patches  # noqa: E402

b = backend_mod.detect()
DEVICE = b.device
DTYPE = b.dtype
print(f"[hunyuan] backend: {backend_mod.describe(b)}", flush=True)

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402

compat_patches.patch_scheduler_timestep_lookup()

weights = PORT / "weights" / "Hunyuan3D-2.1"
t0 = time.perf_counter()
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    str(weights), subfolder="hunyuan3d-dit-v2-1", use_safetensors=False,
    variant="fp16", device=DEVICE, dtype=DTYPE)
if hasattr(pipe, "to"):
    try:
        pipe.to(DEVICE)           # note: Tencent's .to() returns None, call for effect only
    except Exception as e:
        print(f"[hunyuan] pipeline.to failed ({e}); moving components", flush=True)
        for attr in ("model", "vae", "conditioner"):
            o = getattr(pipe, attr, None)
            if o is not None and hasattr(o, "to"):
                o.to(DEVICE)
load_s = time.perf_counter() - t0
print(f"[hunyuan] pipeline loaded in {load_s:.0f}s", flush=True)

img = Image.open(a.image).convert("RGBA")
try:
    from rembg import new_session, remove
    img = remove(img, session=new_session("u2net"))
except Exception as e:
    print(f"[hunyuan] rembg unavailable ({e}); using the image as-is", flush=True)
w, h = img.size
side = max(w, h)
canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
canvas.paste(img, ((side - w) // 2, (side - h) // 2), img)
img = canvas.resize((512, 512), Image.LANCZOS)

gen = torch.Generator().manual_seed(a.seed)
t0 = time.perf_counter()
out = pipe(image=img, num_inference_steps=a.steps, guidance_scale=a.guidance,
           octree_resolution=a.octree, generator=gen)
gen_s = time.perf_counter() - t0
mesh = out[0] if isinstance(out, (list, tuple)) else out
if not isinstance(mesh, trimesh.Trimesh):
    mesh = mesh[0] if isinstance(mesh, (list, tuple)) else mesh

outp = Path(a.out)
outp.parent.mkdir(parents=True, exist_ok=True)
mesh.export(str(outp))
info = {"engine": "hunyuan3d-2.1 (shape only)", "device": str(DEVICE), "dtype": str(DTYPE),
        "steps": a.steps, "octree": a.octree, "guidance": a.guidance,
        "load_seconds": round(load_s, 1), "generate_seconds": round(gen_s, 1),
        "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
        "output": str(outp)}
json.dump(info, open(str(outp.with_suffix("")) + "_info.json", "w"), indent=2)
print("[hunyuan] " + json.dumps(info), flush=True)
