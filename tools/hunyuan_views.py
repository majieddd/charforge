"""Side and back views of a character painted by Hunyuan3D-Paint 2.1's multi-view model, for our own projection (E134).

    vendor/hunyuan3d-2.1-mac-rocm/venv/bin/python tools/hunyuan_views.py --name pip [--preset safe]

Hunyuan3D-Paint renders the mesh's normals and positions from six cameras and generates an unlit albedo for
each, consistent with the reference picture and with each other. Its own bake back onto a UV atlas lost a
third to a half of the surface on our meshes (its depth test is tuned for 2048 px renders and a smooth
surface) and filled the holes in UV space, which made a patchwork; so only the views are kept here, and
pipeline/hy_views_to_mv.py fits each onto our cameras for the texture stage to project like any view.

Writes work/<name>/hy_views/: albedo_<azim>_<elev>.png (super-resolved to the render size), the matching
normal render (for its silhouette) and views.json with Hunyuan's camera model.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT / "vendor" / "hunyuan3d-2.1-mac-rocm"
sys.path.insert(0, str(PORT))

import backend  # noqa: E402
import paint  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--preset", default="safe", choices=sorted(backend.PAINT_PRESETS))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mesh", default=None, help="the mesh to paint (default work/<name>/retopo.glb)")
ap.add_argument("--image", default=None, help="the reference, RGBA (default the character's reference)")
ap.add_argument("--out", default=None, help="output folder (default work/<name>/hy_views)")
a = ap.parse_args()
w = ROOT / "work" / a.name
out = Path(a.out) if a.out else w / "hy_views"
out.mkdir(exist_ok=True)

# the reference with its own cut-out: the paint model wants the figure on white
if a.image:
    ref = Image.open(a.image).convert("RGBA")
elif (w / "reference_rgba.png").exists():
    ref = Image.open(w / "reference_rgba.png").convert("RGBA")
else:
    ref = Image.open(w / "reference.png").convert("RGB")
    m = Image.open(w / "reference_mask.png").convert("L").resize(ref.size)
    ref.putalpha(m)
ref = ref.resize((512, 512))
bg = Image.new("RGB", ref.size, (255, 255, 255))
bg.paste(ref, mask=ref.getchannel("A"))

b = backend.detect()
print(f"[hy_views] {a.name}: {backend.describe(b)}; preset {a.preset}", flush=True)
pipe = paint.load_pipeline(b, a.preset)
import textureGenPipeline as tgp  # noqa: E402  (importable once the pipeline has set its paths up)
import torch  # noqa: E402
import trimesh  # noqa: E402

torch.manual_seed(a.seed)
mesh = tgp.mesh_uv_wrap(trimesh.load(str(a.mesh or w / "retopo.glb"), force="mesh"))
pipe.render.load_mesh(mesh=mesh)
cfg = pipe.config
el, az, wt = pipe.view_processor.bake_view_selection(cfg.candidate_camera_elevs, cfg.candidate_camera_azims,
                                                     cfg.candidate_view_weights, cfg.max_selected_view_num)
normals = pipe.view_processor.render_normal_multiview(el, az, use_abs_coor=True)
positions = pipe.view_processor.render_position_multiview(el, az)
mv = pipe.models["multiview_model"]([bg], normals + positions, prompt="high quality",
                                     custom_view_size=cfg.resolution, resize_input=True)
views = []
for i, (e, z) in enumerate(zip(el, az)):
    tag = f"{int(z):03d}_{int(e):+03d}"
    alb = pipe.models["super_model"](mv["albedo"][i]).resize((cfg.render_size, cfg.render_size))
    alb.save(out / f"albedo_{tag}.png")
    normals[i].resize((cfg.render_size, cfg.render_size)).save(out / f"normal_{tag}.png")
    views.append({"azim": float(z), "elev": float(e), "weight": float(wt[i]), "tag": tag})
info = {"views": views, "camera": "orthographic", "ortho_scale": 1.2, "camera_distance": pipe.render.camera_distance,
        "resolution": cfg.resolution, "render_size": cfg.render_size, "preset": a.preset, "seed": a.seed,
        "mesh_scale": float(pipe.render.mesh_normalize_scale_factor)}
(out / "views.json").write_text(json.dumps(info, indent=1))
print(f"[hy_views] {len(views)} views -> {out}", flush=True)
