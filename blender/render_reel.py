"""Render several clips back to back from one fixed camera, as one frame sequence - a short reel of a
character for a page or a review.

    blender -b -noaudio --python blender/render_reel.py -- --blend work/bo/final.blend \
        --clips idle:1.0,wave,walk:2,jump --out reel/bo [--res 480] [--fps 30] [--az 25]

Each clip is `name` (played once), `name:N` (N loops) or `name:S` with a decimal point (S seconds).
The camera frames the character over every clip it will play, so nothing leaves the frame, and
does not move between clips. Frames are PNGs named 00000.png...; `ffmpeg -framerate 30 -i %05d.png`
makes the video.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--clips", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=480)
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--az", type=float, default=25.0)
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
meshes = [o for o in sc.objects if o.type == "MESH" and o.visible_get()
          and any(m.type == "ARMATURE" for m in o.modifiers)]
for o in sc.objects:
    if o.type == "MESH" and o not in meshes:
        o.hide_render = True
scene_fps = sc.render.fps / sc.render.fps_base


def use(name):
    act = bpy.data.actions.get(name)
    if act is None:
        raise SystemExit(f"[reel] no clip {name!r}; clips: {[x.name for x in bpy.data.actions]}")
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    return act


# the frames to render: (clip, scene frame) in order, at the reel's frame rate
plan = []
for spec in a.clips.split(","):
    name, _, n = spec.partition(":")
    act = use(name)
    f0, f1 = act.frame_range
    length = (f1 - f0) / scene_fps
    secs = float(n) if "." in n else length * (int(n) if n else 1)
    for k in range(int(round(secs * a.fps))):
        t = (k / a.fps) % length if length > 0 else 0.0
        plan.append((name, f0 + t * scene_fps))

# one camera for every frame: the bounds of the whole reel
pts = []
for name, fr in plan[::6]:
    use(name)
    sc.frame_set(int(fr), subframe=fr - int(fr))
    dg = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
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
size = float(max(hi[2] - lo[2], (hi[0] - lo[0]) * 1.1, (hi[1] - lo[1]) * 1.1))
cam = bpy.data.objects.new("reel_cam", bpy.data.cameras.new("reel_cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.lens = 60
dist = size * 2.05
r = math.radians(a.az)
cam.location = ctr + Vector((math.sin(r) * dist, -math.cos(r) * dist, size * 0.18))
cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()

engines = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
sc.render.resolution_x = sc.render.resolution_y = a.res
sc.render.film_transparent = False
w = bpy.data.worlds.new("reel_world")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.105, 0.125, 0.15, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
for n, rot, e in (("key", (50, 0, 35), 3.2), ("fill", (60, 0, -60), 1.2), ("rim", (60, 0, 180), 1.8)):
    lt = bpy.data.objects.new(n, bpy.data.lights.new(n, "SUN"))
    sc.collection.objects.link(lt)
    lt.data.energy = e
    lt.rotation_euler = tuple(math.radians(x) for x in rot)
# a floor, so the jump reads as a jump
bpy.ops.mesh.primitive_plane_add(size=size * 6, location=(ctr.x, ctr.y, float(lo[2]) - 0.002))
floor = bpy.context.active_object
fm = bpy.data.materials.new("floor")
fm.use_nodes = True
fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.16, 0.19, 0.23, 1)
fm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
floor.data.materials.append(fm)

for k, (name, fr) in enumerate(plan):
    use(name)
    sc.frame_set(int(fr), subframe=fr - int(fr))
    sc.render.filepath = os.path.join(a.out, f"{k:05d}.png")
    bpy.ops.render.render(write_still=True)
print(f"[reel] {len(plan)} frames ({len(plan) / a.fps:.1f} s at {a.fps:g} fps) -> {a.out}", flush=True)
