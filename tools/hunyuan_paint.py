"""Texture a character's own mesh with Hunyuan3D-Paint 2.1, through vendor/hunyuan3d-2.1-mac-rocm (E134).

    vendor/hunyuan3d-2.1-mac-rocm/venv/bin/python tools/hunyuan_paint.py --name pip [--preset safe]

The mesh is the retopologised one (work/<name>/retopo.glb), not re-meshed, so the result lies on exactly
our surface; the conditioning image is the cut-out reference. Writes work/<name>/hy_paint/: the paint
pipeline's textured .obj with its own UV layout and its albedo and metallic-roughness maps. Moving them
onto our UV layout is blender/transfer_texture.py's job.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT / "vendor" / "hunyuan3d-2.1-mac-rocm"
sys.path.insert(0, str(PORT))

import backend  # noqa: E402
import paint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--preset", default="safe", choices=sorted(backend.PAINT_PRESETS))
ap.add_argument("--remesh", action="store_true", help="let the pipeline convert and decimate the mesh first (its tested path)")
ap.add_argument("--out", default="hy_paint", help="output folder inside work/<name>")
a = ap.parse_args()
w = ROOT / "work" / a.name
img = w / "reference_rgba.png" if (w / "reference_rgba.png").exists() else w / "reference.png"
b = backend.detect()
print(f"[hy_paint] {a.name}: {backend.describe(b)}; image {img.name}; preset {a.preset}", flush=True)
res = paint.texture_mesh(str(w / "retopo.glb"), str(img), b, preset=a.preset, use_remesh=a.remesh,
                         output_dir=str(w / a.out))
print("[hy_paint] done", {k: (str(v) if not isinstance(v, dict) else list(v)) for k, v in res.items()}, flush=True)
