"""Render a contact sheet of poses from each animation clip, for visual inspection.

Run: blender -b -noaudio --python render_anim.py -- --blend animated.blend --out dir [--frames 4]
"""
import argparse
import math
import os
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=4)
ap.add_argument("--res", type=int, default=420)
ap.add_argument("--az", type=float, default=25.0)
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
mesh = next(o for o in sc.objects if o.type == "MESH")

engines = [i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else engines[0]
sc.render.resolution_x = sc.render.resolution_y = a.res
sc.render.film_transparent = False
sc.render.image_settings.file_format = "PNG"
if sc.render.engine == "BLENDER_EEVEE_NEXT":
    sc.eevee.taa_render_samples = 24

for name, loc, energy in (("key", (3, -4, 4), 1200), ("fill", (-4, -2, 1.5), 500), ("rim", (0, 5, 3), 800)):
    lt = bpy.data.lights.new(name, type="AREA")
    lt.energy, lt.size = energy, 6
    ob = bpy.data.objects.new(name, lt)
    ob.location = loc
    ob.rotation_euler = (mathutils.Vector((0, 0, 0.2)) - mathutils.Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(ob)
w = bpy.data.worlds.new("w")
w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.045, 0.05, 0.065, 1)
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
sc.world = w

cd = bpy.data.cameras.new("cam")
cd.type = "ORTHO"
cd.ortho_scale = 2.5
cam = bpy.data.objects.new("cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam
az = math.radians(a.az)
cam.location = (4 * math.sin(az), -4 * math.cos(az), 0.35)
cam.rotation_euler = (mathutils.Vector((0, 0, 0.0)) - mathutils.Vector(cam.location)).to_track_quat("-Z", "Y").to_euler()

for act in bpy.data.actions:
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and len(act.slots):
        rig.animation_data.action_slot = act.slots[0]
    for pb in rig.pose.bones:
        pb.rotation_euler = (0, 0, 0)
    f0, f1 = [int(x) for x in act.frame_range]
    for i in range(a.frames):
        f = int(f0 + (f1 - f0) * i / max(1, a.frames - 1))
        sc.frame_set(f)
        sc.render.filepath = os.path.join(a.out, f"{act.name}_{i}.png")
        bpy.ops.render.render(write_still=True)
    print(f"[render_anim] {act.name}: {a.frames} frames", flush=True)
print("[render_anim] done ->", a.out)
