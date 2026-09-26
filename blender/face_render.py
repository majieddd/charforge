"""Front renders of a rigged character for face landmarking: the whole body, and the head close up.

    blender -b --python face_render.py -- --blend rig_t.blend --out-dir face

Writes body_front.png (for the pose model, which wants a whole person), head_front.png (flat
albedo, where eyes and lips are found), head_front_pos.npy (the world position under every head
pixel, NaN where there is no surface) and face_render.json (both cameras). Rendered with EEVEE
and an unlit emission of the base colour, so what the detector sees is the texture itself.
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
ap.add_argument("--blend", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--res", type=int, default=1024)
a = ap.parse_args(argv)
os.makedirs(a.out_dir, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
rig.data.pose_position = "REST"
meshes = [o for o in sc.objects if o.type == "MESH" and o.find_armature() == rig]
for o in sc.objects:
    if o.type in ("LIGHT", "CAMERA"):
        o.hide_render = True
bpy.context.view_layer.update()

# unlit albedo: every material emits its own base colour
for o in meshes:
    for m in o.data.materials:
        nt = m.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if bsdf is None or out is None:
            continue
        em = nt.nodes.new("ShaderNodeEmission")
        src = bsdf.inputs["Base Color"].links[0].from_socket if bsdf.inputs["Base Color"].links else None
        if src is not None:
            nt.links.new(src, em.inputs["Color"])
        else:
            em.inputs["Color"].default_value = bsdf.inputs["Base Color"].default_value
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        m["_emit"] = em.name

eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
sc.render.resolution_x = sc.render.resolution_y = a.res
sc.render.film_transparent = True
sc.render.filter_size = 0.5
sc.view_settings.view_transform = "Standard"
w = bpy.data.worlds.new("w")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[1].default_value = 0.0
cam = bpy.data.objects.new("face_cam", bpy.data.cameras.new("face_cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"

P = np.vstack([np.array([(o.matrix_world @ v.co)[:] for v in o.data.vertices]) for o in meshes])
lo, hi = P.min(0), P.max(0)
H = float(hi[2] - lo[2])
bone_w = lambda n: np.array((rig.matrix_world @ rig.data.bones[n].head_local)[:])
head_j = bone_w("mixamorig:Head") if "mixamorig:Head" in rig.data.bones else bone_w("head")
top = float(hi[2])


def shoot(name, centre, ortho):
    cam.data.ortho_scale = ortho
    cam.location = Vector((centre[0], centre[1] - 4.0, centre[2]))
    cam.rotation_euler = (math.radians(90), 0, 0)
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.filepath = os.path.join(a.out_dir, f"{name}.png")
    bpy.ops.render.render(write_still=True)
    return {"centre": [float(c) for c in centre], "ortho_scale": ortho, "res": a.res}


cams = {}
body_c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])
cams["body_front"] = shoot("body_front", body_c, H * 1.08)
# the head from its joint to the top - at least 11% of the height (a head is 12-13% of an adult): knight2's
# head joint sat 12 cm under the top of his hair, and a frame built on it cut off his chin
hj = max(top - head_j[2], 0.11 * H)
head_c = np.array([head_j[0], head_j[1], top - hj / 2])
head_span = hj * 2.3
cams["head_front"] = shoot("head_front", head_c, head_span)

# world position under every head pixel: emission of (position - the frame's centre + 1), raw, from the same
# camera. EEVEE keeps the film in half floats: at position + 4 (values 4-6) a half float's step is 3.9 mm, and
# boyscout's eyelids and lips were read in 4 mm stairs; around 1 the step is 0.5-1 mm
OFF = Vector((1.0, 1.0, 1.0)) - Vector(head_c.tolist())
for o in meshes:
    for m in o.data.materials:
        nt = m.node_tree
        em = nt.nodes.get(m.get("_emit", ""))
        if em is None:
            continue
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        add = nt.nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        add.inputs[1].default_value = tuple(OFF)
        nt.links.new(geo.outputs["Position"], add.inputs[0])
        nt.links.new(add.outputs["Vector"], em.inputs["Color"])
sc.view_settings.view_transform = "Raw"
sc.render.image_settings.file_format = "OPEN_EXR"
sc.render.image_settings.color_depth = "32"
f = os.path.join(a.out_dir, "head_front_pos.exr")
cam.data.ortho_scale = head_span
sc.render.filepath = f
bpy.ops.render.render(write_still=True)
im = bpy.data.images.load(f)
px = np.empty(im.size[0] * im.size[1] * 4, np.float32)
im.pixels.foreach_get(px)
px = px.reshape(im.size[1], im.size[0], 4)[::-1]
pos = np.where(px[..., 3:4] > 0.995, px[..., :3] / np.maximum(px[..., 3:4], 1e-6) - np.array(OFF[:]), np.nan).astype(np.float32)
np.save(os.path.join(a.out_dir, "head_front_pos.npy"), pos)
os.remove(f)
json.dump({"cameras": cams, "height": H, "head_joint": head_j.tolist(), "top": top},
          open(os.path.join(a.out_dir, "face_render.json"), "w"), indent=1)
print(f"[face] renders -> {a.out_dir}", flush=True)
