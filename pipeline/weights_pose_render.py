"""Render a rigged character held in a pose, for the skin-weight trial (E161): the fins and wings are visible in a plain shaded
view, where the audit's flagged-face renders show only where the surface is stretched.

    blender -b -noaudio --python pipeline/weights_pose_render.py -- --blend work/pip__ctl/qa/stress.blend \
        --out renders/pip_ctl --tag ctl --shots stress_arms_overhead:1,stress_arms_forward:1 --views 0,90

--shots is action:frame pairs (an action is a clip or a stress pose of the blend; the frame is the one to hold). --views are
azimuths in degrees: 0 looks at the character from the front (the figure faces -Y), 90 from its side. Orthographic, one
light setup, the same framing on every blend of a comparison: the frame is fitted to the figure's evaluated bounds at the
held pose, so two builds of one pose line up. Workbench engine (no GPU shading model to differ between builds)."""
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
ap.add_argument("--out", required=True, help="folder for the PNGs (created)")
ap.add_argument("--tag", required=True, help="prefix for the file names, e.g. ctl or surf")
ap.add_argument("--shots", required=True, help="action:frame,... e.g. wave:12,stress_arms_overhead:1")
ap.add_argument("--views", default="0,90", help="azimuths in degrees, 0 = front")
ap.add_argument("--res", type=int, default=560)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if rig is None:
    raise SystemExit("[pose] no armature in the blend")
meshes = [o for o in bpy.data.objects if o.type == "MESH" and not o.hide_render and o.visible_get()]
if not meshes:
    raise SystemExit("[pose] no visible mesh in the blend")
os.makedirs(a.out, exist_ok=True)

sc.render.engine = "BLENDER_WORKBENCH"
sc.render.resolution_x = sc.render.resolution_y = a.res
sc.render.image_settings.file_format = "PNG"
sh = sc.display.shading
sh.light = "STUDIO"
sh.color_type = "SINGLE"
sh.single_color = (0.74, 0.76, 0.80)
if sc.world is None:
    sc.world = bpy.data.worlds.new("pose_world")
sc.world.color = (1.0, 1.0, 1.0)
sc.render.film_transparent = False

cam = bpy.data.objects.get("pose_cam") or bpy.data.objects.new("pose_cam", bpy.data.cameras.new("pose_cam"))
if cam.name not in sc.collection.objects:
    sc.collection.objects.link(cam)
cam.data.type = "ORTHO"
sc.camera = cam


def set_action(name):
    act = bpy.data.actions.get(name)
    if act is None:
        raise SystemExit(f"[pose] no action {name!r} in {a.blend}")
    if rig.animation_data is None:
        rig.animation_data_create()
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]


def bounds():
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in meshes:
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        co = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
        oe.to_mesh_clear()
        M = np.array(o.matrix_world.to_4x4()).reshape(4, 4)
        pts.append(co @ M[:3, :3].T + M[:3, 3])
    P = np.concatenate(pts)
    return P.min(0), P.max(0)


for item in a.shots.split(","):
    action, frame = item.rsplit(":", 1)
    set_action(action)
    sc.frame_set(int(frame))
    lo, hi = bounds()
    ctr = Vector(((lo + hi) / 2).tolist())
    size = float(max(hi - lo))
    cam.data.ortho_scale = size * 1.12
    for az in (float(v) for v in a.views.split(",")):
        r = math.radians(az)
        cam.location = Vector((ctr.x + math.sin(r) * size * 3, ctr.y - math.cos(r) * size * 3, ctr.z))
        cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(a.out, f"{a.tag}_{action}_f{int(frame)}_az{int(az)}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[pose] {sc.render.filepath}", flush=True)
