"""Render a character's clip frame for frame against a video: same timing, the video's camera angle,
transparent background - the right-hand side of a pixel comparison (tools/motion_fidelity.py).

    blender -b -noaudio --python render_match.py -- --blend final.blend --clip punch_combo \
        --frames 124 --fps 24 --yaw 20 --out renders/ [--res 512 768] [--pose rest|clip]

Frame i of the output is the clip at time i / fps - the video's frame i, since a clip made from a
video keeps the video's timing. The camera is orthographic, level, at `yaw` degrees round from the
character's front (the same convention as render_clip_seq.py and the fit's preview yaw), framing
the clip's whole extent so every frame shares one scale; tools/motion_fidelity.py then finds the
scale and offset that lay the renders over the video. Lit like a plain studio, as the videos are,
so a pose model reads both the same way.

--pose rest renders frame 0 only, in the rig's rest pose: with --yaw 0 the image to hold against
the reference picture.
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
ap.add_argument("--clip", default=None)
ap.add_argument("--frames", type=int, default=124)
ap.add_argument("--fps", type=float, default=24.0)
ap.add_argument("--yaw", type=float, default=0.0)
ap.add_argument("--res", type=int, nargs=2, default=(512, 768))
ap.add_argument("--out", required=True)
ap.add_argument("--pose", choices=("clip", "rest"), default="clip")
ap.add_argument("--every", type=int, default=1, help="render every n-th frame (files keep their frame index)")
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
meshes = [o for o in sc.objects if o.type == "MESH" and o.visible_get()]
if a.pose == "rest":
    rig.data.pose_position = "REST"
    frames = [sc.frame_current]
else:
    act = bpy.data.actions.get(a.clip)
    if act is None:
        raise SystemExit(f"[match] no clip {a.clip!r} in {a.blend}")
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    f0, f1 = act.frame_range
    scene_fps = sc.render.fps / sc.render.fps_base
    frames = [min(f0 + i * scene_fps / a.fps, f1) for i in range(a.frames)]

# the extent over every frame, so one scale holds for the whole clip
pts = []
for f in frames[::max(1, len(frames) // 24)]:
    sc.frame_set(int(math.floor(f)), subframe=float(f - math.floor(f)))
    dg = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        V = np.empty(len(m.vertices) * 3)
        m.vertices.foreach_get("co", V)
        M = np.array(o.matrix_world)
        pts.append(V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        ev.to_mesh_clear()
P = np.vstack(pts)
lo, hi = P.min(0), P.max(0)
ctr = Vector(((lo + hi) / 2).tolist())
r = math.radians(a.yaw)
side = Vector((math.cos(r), math.sin(r), 0.0))                      # the image's horizontal axis
span_w = float(np.ptp(P @ np.array([side.x, side.y, 0.0])))
span_h = float(hi[2] - lo[2])
aspect = a.res[0] / a.res[1]
ortho = 1.12 * max(span_h, span_w / aspect)

cam_d = bpy.data.cameras.new("match")
cam_d.type = "ORTHO"
cam_d.sensor_fit = "VERTICAL"                                      # ortho_scale = the visible height
cam_d.ortho_scale = ortho
cam = bpy.data.objects.new("match", cam_d)
sc.collection.objects.link(cam)
sc.camera = cam
cam.location = ctr + Vector((math.sin(r) * 6.0, -math.cos(r) * 6.0, 0.0))
cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()

world = bpy.data.worlds.new("studio")
sc.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.8, 0.8, 0.8, 1)
bg.inputs[1].default_value = 0.9
key = bpy.data.objects.new("key", bpy.data.lights.new("key", "SUN"))
sc.collection.objects.link(key)
key.data.energy = 3.0
key.rotation_euler = (math.radians(50), 0, r + math.radians(25))

eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = a.res
sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
names = {t.name for t in sc.view_settings.bl_rna.properties["view_transform"].enum_items}
sc.view_settings.view_transform = "Standard" if "Standard" in names else sc.view_settings.view_transform

for i, f in enumerate(frames):
    if i % a.every:
        continue
    sc.frame_set(int(math.floor(f)), subframe=float(f - math.floor(f)))
    sc.render.filepath = os.path.join(a.out, f"{i:04d}.png")
    bpy.ops.render.render(write_still=True)
with open(os.path.join(a.out, "camera.txt"), "w") as fh:
    fh.write(f"yaw {a.yaw}\northo_scale {cam_d.ortho_scale}\ncentre {tuple(ctr)}\nres {a.res[0]} {a.res[1]}\n")
print(f"[match] {len(frames)} frames of {a.clip or 'the rest pose'} at yaw {a.yaw:g} -> {a.out}", flush=True)
