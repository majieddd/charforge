"""A character's 3D shape from its reference by Hunyuan3D 2.1, through vendor/hunyuan3d-2.1-mac-rocm (E104).

    vendor/hunyuan3d-2.1-mac-rocm/venv/bin/python tools/hunyuan_shape.py --name cadet [--octree 384] [--steps 50]

The same cut-out reference TRELLIS.2 was given, Hunyuan's own settings (50 flow steps, guidance 5), the
octree resolution raised from the app's 256 to the 384 the model was trained to decode. Writes
work/<name>/hy_shape/shape_o<octree>_s<seed>.glb - the raw surface, for blender/model_quality.py and a
side-by-side look against TRELLIS.2's mesh.glb.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT / "vendor" / "hunyuan3d-2.1-mac-rocm"
for p in (PORT, PORT / "Hunyuan3D-2.1", PORT / "Hunyuan3D-2.1" / "hy3dshape"):
    sys.path.insert(0, str(p))

import backend  # noqa: E402
import compat_patches  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--octree", type=int, default=384)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--guidance", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--image", default=None, help="an RGBA picture instead of the reference (a crop of the head, E139)")
ap.add_argument("--out", default=None, help="the GLB to write (default work/<name>/hy_shape/shape_o<octree>_s<seed>.glb)")
a = ap.parse_args()
w = ROOT / "work" / a.name
b = backend.detect()
if a.image:
    img = Image.open(a.image).convert("RGBA")
elif (w / "reference_rgba.png").exists():
    img = Image.open(w / "reference_rgba.png").convert("RGBA")
else:
    img = Image.open(w / "reference.png").convert("RGBA")
    img.putalpha(Image.open(w / "reference_mask.png").convert("L").resize(img.size))
side = max(img.size)
canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2), img)
img = canvas.resize((512, 512), Image.LANCZOS)

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402

compat_patches.patch_scheduler_timestep_lookup()
t0 = time.time()
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(str(PORT / "weights" / "Hunyuan3D-2.1"), subfolder="hunyuan3d-dit-v2-1",
                                                        use_safetensors=False, variant="fp16", device=b.device, dtype=b.dtype)
try:
    pipe.to(b.device)
except Exception:                                                       # noqa: BLE001
    for attr in ("model", "vae", "conditioner"):
        getattr(pipe, attr).to(b.device)
with torch.inference_mode():
    res = pipe(image=img, num_inference_steps=a.steps, guidance_scale=a.guidance, octree_resolution=a.octree,
               generator=torch.Generator(device="cpu").manual_seed(a.seed))
mesh = res[0] if isinstance(res, (list, tuple)) else res
if a.out:
    path = Path(a.out)
else:
    out = w / "hy_shape"
    out.mkdir(exist_ok=True)
    path = out / f"shape_o{a.octree}_s{a.seed}.glb"
mesh.export(str(path))
print(f"[hy_shape] {a.name}: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces in {time.time() - t0:.0f} s -> {path}", flush=True)
