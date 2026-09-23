"""Render a character's roster thumbnail: head to toe, transparent background, first idle frame.

The playground's thumbnails used to be rendered and cropped by hand in a scratch script. That
is how a roster drifts from the characters it lists, so it is a build step now.

Run: blender -b -noaudio --python thumbnail.py -- --blend final.blend --out thumb.png [--res 420]
"""
import argparse
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--clip", default="idle")
ap.add_argument("--res", type=int, default=420)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
act = bpy.data.actions.get(a.clip)
if act:
    rig.animation_data_create()
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    sc.frame_set(int(act.frame_range[0]))
dg = bpy.context.evaluated_depsgraph_get()
pts = []
for o in sc.objects:
    if o.type != "MESH":
        continue
    ev = o.evaluated_get(dg)
    m = ev.to_mesh()
    V = np.empty(len(m.vertices) * 3)
    m.vertices.foreach_get("co", V)
    mw = np.array(o.matrix_world)
    pts.append(V.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
    ev.to_mesh_clear()
P = np.vstack(pts)
lo, hi = P.min(0), P.max(0)
ctr = Vector(((lo + hi) / 2).tolist())
h = float(hi[2] - lo[2])

cam = bpy.data.objects.new("thumb_cam", bpy.data.cameras.new("thumb_cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"
cam.data.ortho_scale = h * 1.06
r = math.radians(22)
cam.location = ctr + Vector((math.sin(r) * 4, -math.cos(r) * 4, 0.0))
cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
w = bpy.data.worlds.new("thumb_world")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[1].default_value = 0.55
for name, energy, colour, rot in (("key", 4.2, (1, 1, 1), (58, 0, 28)),
                                  ("rim", 2.0, (0.65, 0.76, 1.0), (70, 0, 200))):
    L = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
    sc.collection.objects.link(L)
    L.data.energy, L.data.color = energy, colour
    L.rotation_euler = tuple(math.radians(x) for x in rot)
eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
sc.view_settings.view_transform = "Standard"
sc.render.resolution_x = a.res
sc.render.resolution_y = int(a.res * 1.34)
sc.render.filepath = a.out
bpy.ops.render.render(write_still=True)
print(f"[thumb] -> {a.out}", flush=True)
