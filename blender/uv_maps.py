"""Where every texel is, and what each view camera sees - the inputs texture projection needs.

    blender -b --python uv_maps.py -- --mesh retopo.glb --out-dir texproj [--res 4096] \
        [--views 0,90,180] [--view-res 1024] [--ortho 2.15]

Writes, into --out-dir:
  uv_position.npy, uv_normal.npy  float16 (res, res, 3): world position and normal per texel
                                  (NaN where no face covers the texel), rows top-down like a PNG
  view_<az>_albedo.png            the mesh with its own baked albedo, unlit, from each camera
  view_<az>_position.npy          float32 (h, w, 3) world position seen at each pixel (NaN = none)
  views.json "face_frame"         with --face-joints, a front frame around the whole head (for
                                  pipeline/face_detail.py)
  views.json                      the normalisation and cameras (render_views.py's convention:
                                  bbox-centred, 2 units tall, orthographic, azimuth about +Z)

Both are made by Blender's own rasteriser through emission bakes and renders, so the texel and
pixel coverage is exactly what the shipped mesh will have.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--res", type=int, default=4096)
ap.add_argument("--views", default="0,90,180")
ap.add_argument("--view-res", type=int, default=1024)
ap.add_argument("--ortho", type=float, default=2.15)
ap.add_argument("--face-joints", default=None,
                help="joints_refined.json: adds a close-up 'face' camera framed chin to crown")
ap.add_argument("--face-frame", default=None, help="sdf.npz - the frame the joints are in")
ap.add_argument("--face-res", type=int, default=1024)
a = ap.parse_args(argv)
os.makedirs(a.out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
ob = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
bpy.context.view_layer.objects.active = ob
ob.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
me = ob.data
V = np.empty(len(me.vertices) * 3)
me.vertices.foreach_get("co", V)
V = V.reshape(-1, 3)
lo, hi = V.min(0), V.max(0)
ctr = (lo + hi) / 2
scale = 2.0 / float(hi[2] - lo[2])
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 1
sc.cycles.device = "CPU"
sc.render.bake.margin = 0

# the baked albedo, whichever image the material samples for base colour
mat = me.materials[0]
nt = mat.node_tree
bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
albedo_img = bsdf.inputs["Base Color"].links[0].from_node.image if bsdf.inputs["Base Color"].links else None
out_node = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
geo = nt.nodes.new("ShaderNodeNewGeometry")
emit = nt.nodes.new("ShaderNodeEmission")
target = nt.nodes.new("ShaderNodeTexImage")


def bake_attr(socket, name):
    img = bpy.data.images.new(name, a.res, a.res, alpha=True, float_buffer=True, is_data=True)
    img.pixels.foreach_set(np.full(a.res * a.res * 4, np.nan, np.float32))
    target.image = img
    for n in nt.nodes:
        n.select = False
    target.select = True
    nt.nodes.active = target
    nt.links.new(socket, emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])
    bpy.ops.object.bake(type="EMIT", use_clear=False, margin=0)
    px = np.empty(a.res * a.res * 4, np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(a.res, a.res, 4)[::-1, :, :3]       # Blender rows run bottom-up


pos = bake_attr(geo.outputs["Position"], "uvpos")
nrm = bake_attr(geo.outputs["Normal"], "uvnrm")
np.save(os.path.join(a.out_dir, "uv_position.npy"), pos.astype(np.float16))
np.save(os.path.join(a.out_dir, "uv_normal.npy"), nrm.astype(np.float16))
cov = float(np.isfinite(pos[..., 0]).mean())
print(f"[uv_maps] position/normal maps {a.res}^2, {cov:.1%} of texels covered", flush=True)

# ---- renders from each view camera -----------------------------------------------------------
# normalise like render_views.py so its cameras apply as they are
for v in me.vertices:
    v.co = (Vector(v.co) - Vector(ctr.tolist())) * scale
me.update()
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"
cam.data.ortho_scale = a.ortho
sc.render.resolution_x = sc.render.resolution_y = a.view_res
sc.render.film_transparent = True
sc.render.dither_intensity = 0.0
sc.view_settings.view_transform = "Standard"
# EEVEE, not Cycles: measured on an orthographic camera in Blender 5.2, a Cycles render lands
# ~5 pixels off where the camera projects (EEVEE: under one), which put every projected texel
# a centimetre off its own surface.
eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
sc.render.filter_size = 0.5                           # narrow: positions must barely blend
views = []
shots = [(f"{int(round(float(x))):03d}", float(x), (0.0, 0.0, 0.0), a.ortho, a.view_res)
         for x in a.views.split(",")]
if a.face_joints and a.face_frame and os.path.exists(a.face_joints) and os.path.exists(a.face_frame):
    # The face, close up: a full-body view gives a face a hundred pixels, and the projection can
    # only be as sharp as its source. This frames the whole head from the front (chin to crown,
    # with background around it); pipeline/face_detail.py cuts that frame out of the reference and
    # details it, and the front view reads the face from it.
    Pf = np.load(a.face_frame)["points"]
    flo, fhi = Pf.min(0), Pf.max(0)
    fk, fc = 2.0 / float(fhi[2] - flo[2]), (flo + fhi) / 2
    Jf = json.load(open(a.face_joints))["joints"]
    hj = (np.array(Jf["head"]) / fk + fc - ctr) * scale              # into this render's frame
    # the crown from the mesh itself (hair included), above the head joint: the whole head must be
    # in frame with background around it, or the image cannot be aligned by its silhouette
    Vn = (V - ctr) * scale
    near_ = np.linalg.norm(Vn[:, :2] - hj[:2], axis=1) < 0.12
    top_z = float(Vn[near_, 2].max()) if near_.any() else float(hj[2] + 0.12)
    span = max(top_z - float(hj[2]), 0.05)
    face_frame = {"target": [float(hj[0]), 0.0, float(hj[2] + 0.2 * span)], "ortho_scale": 2.3 * span,
                  "res": a.face_res}
else:
    face_frame = None
for tag, az, tgt, ortho, vres in shots:
    r = math.radians(az)
    cam.data.ortho_scale = ortho
    sc.render.resolution_x = sc.render.resolution_y = vres
    cam.location = (tgt[0] + 4.0 * math.sin(r), tgt[1] - 4.0 * math.cos(r), tgt[2])
    cam.rotation_euler = (Vector(tgt) - cam.location).to_track_quat("-Z", "Y").to_euler()
    # albedo, unlit
    if albedo_img is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = albedo_img
        nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGBA"
        sc.render.filepath = os.path.join(a.out_dir, f"view_{tag}_albedo.png")
        bpy.ops.render.render(write_still=True)
    # world position per pixel (in the ORIGINAL mesh frame: undo the normalisation). A render
    # clamps negative emission and applies the display transform, so the position goes out
    # offset into positive values and the view transform is Raw.
    add = nt.nodes.get("_pos_offset") or nt.nodes.new("ShaderNodeVectorMath")
    add.name = "_pos_offset"
    add.operation = "ADD"
    add.inputs[1].default_value = (4.0, 4.0, 4.0)
    nt.links.new(geo.outputs["Position"], add.inputs[0])
    nt.links.new(add.outputs["Vector"], emit.inputs["Color"])
    sc.view_settings.view_transform = "Raw"
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.color_mode = "RGBA"
    f = os.path.join(a.out_dir, f"view_{tag}_position.exr")
    sc.render.filepath = f
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(f)
    px = np.empty(im.size[0] * im.size[1] * 4, np.float32)
    im.pixels.foreach_get(px)
    px = px.reshape(im.size[1], im.size[0], 4)[::-1]
    # only fully covered pixels: at the silhouette the filter blends the position with the empty
    # background, which decodes to a point far outside the character
    P = np.where(px[..., 3:4] > 0.995, (px[..., :3] / np.maximum(px[..., 3:4], 1e-6) - 4.0) / scale + ctr,
                 np.nan).astype(np.float32)
    sc.view_settings.view_transform = "Standard"
    np.save(os.path.join(a.out_dir, f"view_{tag}_position.npy"), P)
    os.remove(f)
    views.append({"azimuth": az, "tag": tag, "cam_location": list(cam.location), "target": list(tgt),
                  "ortho_scale": ortho, "res": vres})
json.dump({"center": ctr.tolist(), "scale": scale, "views": views, "uv_res": a.res, "face_frame": face_frame},
          open(os.path.join(a.out_dir, "views.json"), "w"), indent=1)
print(f"[uv_maps] {len(views)} view renders -> {a.out_dir}", flush=True)
