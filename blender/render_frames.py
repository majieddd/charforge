"""Frames of an animated glTF or FBX from one side, framed on the whole clip - previews of a move.

    blender -b -noaudio --python render_frames.py -- --file move.glb --out prefix --frames 0,12,24,36,47 \
        [--az 25] [--res 360 480]

Writes <prefix>_<frame>.png. The camera is orthographic, level, `az` degrees round from the front, and
does not move during the clip, so the frames compare directly.
"""
import argparse
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", default="0,12,24,36,47")
ap.add_argument("--az", type=float, default=25.0)
ap.add_argument("--res", type=int, nargs=2, default=(360, 480))
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
if a.file.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=a.file)
else:
    bpy.ops.import_scene.gltf(filepath=a.file)
sc = bpy.context.scene
# Blender's glTF importer adds a bone-display Icosphere in "glTF_not_exported": not part of the file,
# and framing on it left the figure in the top half of every preview
meshes = [o for o in sc.objects if o.type == "MESH"
          and not any(c.name == "glTF_not_exported" for c in o.users_collection)]
for o in sc.objects:
    if o.type == "MESH" and o not in meshes:
        o.hide_render = True
frames = [int(x) for x in a.frames.split(",")]
eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = a.res
w = bpy.data.worlds.new("w")
sc.world = w
w.color = (0.75, 0.75, 0.77)
for nm, rot, e in (("sun", (50, 0, 35), 3.5), ("fill", (60, 0, 215), 1.5)):
    lt = bpy.data.objects.new(nm, bpy.data.lights.new(nm, "SUN"))
    sc.collection.objects.link(lt)
    lt.data.energy = e
    lt.rotation_euler = tuple(math.radians(x) for x in rot)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"
pts = []
for f in frames:
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        pts += [tuple(o.matrix_world @ me.vertices[i].co) for i in range(0, len(me.vertices), 40)]
        ev.to_mesh_clear()
P = np.array(pts)
lo, hi = P.min(0), P.max(0)
ctr = (lo + hi) / 2
size = float(max(hi[2] - lo[2], (hi[0] - lo[0]) * a.res[1] / a.res[0]))
cam.data.ortho_scale = size * 1.08
r = math.radians(a.az)
cam.location = Vector((ctr[0] + math.sin(r) * size * 3, ctr[1] - math.cos(r) * size * 3, ctr[2]))
cam.rotation_euler = (Vector(ctr.tolist()) - cam.location).to_track_quat("-Z", "Y").to_euler()
for f in frames:
    sc.frame_set(f)
    sc.render.filepath = f"{a.out}_{f:03d}.png"
    bpy.ops.render.render(write_still=True)
print(f"[frames] {len(frames)} frames -> {a.out}_*.png", flush=True)
