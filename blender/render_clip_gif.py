"""Render clips side-on over a floor that moves at each clip's ground speed - the README animation.

An in-place clip rendered over a still floor hides the one thing worth checking: whether a
planted foot stays planted. Here the floor is a checkerboard carried backwards at the speed the
manifest states, the way a character controller moves the world past the character, so a foot
that skates visibly slides across the tiles.

Run: blender -b -noaudio --python render_clip_gif.py -- --blend final.blend --report retarget.json
         --out frames/ [--clips walk jog run jump] [--fps 20] [--res 420]
Frames land in <out>/<clip>_NNN.png; tools assemble them into a GIF.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--report", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--clips", nargs="*", default=["walk", "jog", "run", "jump"])
ap.add_argument("--fps", type=float, default=20.0)
ap.add_argument("--res", type=int, default=420)
a = ap.parse_args(argv)

rep = json.load(open(a.report))
bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
src_fps = sc.render.fps / sc.render.fps_base
os.makedirs(a.out, exist_ok=True)

eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = int(a.res * 1.33), a.res
sc.render.film_transparent = False
sc.view_settings.view_transform = "Standard"
w = bpy.data.worlds.new("w")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.055, 0.066, 0.086, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
for name, energy, rot, col in (("key", 3.8, (52, 0, 35), (1, 0.97, 0.92)),
                               ("rim", 2.4, (65, 0, 200), (0.62, 0.74, 1.0))):
    L = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
    sc.collection.objects.link(L)
    L.data.energy, L.data.color = energy, col
    L.rotation_euler = tuple(math.radians(x) for x in rot)

# a checkerboard floor, 0.5 m tiles, moved per frame
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
floor = bpy.context.active_object
mat = bpy.data.materials.new("floor")
mat.use_nodes = True
nt = mat.node_tree
chk = nt.nodes.new("ShaderNodeTexChecker")
chk.inputs["Scale"].default_value = 40.0            # 40 m plane / 40 = 1 m pairs, 0.5 m tiles
chk.inputs["Color1"].default_value = (0.20, 0.22, 0.26, 1)
chk.inputs["Color2"].default_value = (0.13, 0.145, 0.17, 1)
bsdf = nt.nodes["Principled BSDF"]
bsdf.inputs["Roughness"].default_value = 0.9
nt.links.new(chk.outputs["Color"], bsdf.inputs["Base Color"])
floor.data.materials.append(mat)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.lens = 50
cam.location = Vector((5.2, -0.2, 1.05))
cam.rotation_euler = (math.radians(88), 0, math.radians(90))

for clip in a.clips:
    act = bpy.data.actions.get(clip)
    if act is None:
        continue
    rig.animation_data_create()
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    f0, f1 = (int(x) for x in act.frame_range)
    v = (rep.get(clip) or {}).get("stride_speed_mps") or 0.0
    dur = (f1 - f0) / src_fps
    n = max(1, int(round(dur * a.fps)))
    for k in range(n):
        t = k / a.fps
        f = f0 + t * src_fps
        sc.frame_set(int(f), subframe=f - int(f))
        floor.location.y = (v * t) % 1.0              # the world moves back past the character
        sc.render.filepath = os.path.join(a.out, f"{clip}_{k:03d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"[gif] {clip}: {n} frames at {v:.2f} m/s", flush=True)
