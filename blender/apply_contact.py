"""Write the contact solve's turns (pipeline/contact_solve.py, E151) into the rig's clips.
    blender -b -noaudio --python apply_contact.py -- --blend final.blend --dir <solve dir> [--out out.blend] [--clips a,b]
Each shoulder, elbow and wrist key gets its turn d added in the bone's own frame (basis' = basis R(d)), keyed
in the bone's own rotation mode, continuous with the key before."""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--blend", required=True)
ap.add_argument("--out", default=None, help="default: overwrite --blend")
ap.add_argument("--clips", default="")
ap.add_argument("--tag", default="")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
scene = bpy.context.scene
wanted = [c for c in a.clips.split(",") if c]
n_done = 0
for act in bpy.data.actions:
    p = os.path.join(a.dir, f"contact{a.tag}_{act.name}.npz")
    if (wanted and act.name not in wanted) or not os.path.exists(p):
        continue
    r = np.load(p)
    d, bones = r["d"], [str(b) for b in r["bones"]]
    if not np.abs(d).max() > 1e-6:                           # the solve left this clip as it was
        continue
    arm.animation_data.action = act
    if hasattr(arm.animation_data, "action_slot") and getattr(act, "slots", None):
        arm.animation_data.action_slot = act.slots[0]
    f0 = int(r["f0"])
    keys = {b: [] for b in bones}
    for t in range(d.shape[0]):
        scene.frame_set(f0 + t)
        for k, b in enumerate(bones):
            v = Vector(d[t, k].tolist())
            ang = v.length
            R = Matrix.Rotation(math.radians(ang), 4, v.normalized()) if ang > 1e-6 else Matrix.Identity(4)
            keys[b].append((f0 + t, (arm.pose.bones[b].matrix_basis @ R).to_quaternion()))
    for b, ks in keys.items():
        pb = arm.pose.bones[b]
        prev = None
        for fr, q in ks:
            if pb.rotation_mode == "QUATERNION":
                if prev is not None and prev.dot(q) < 0:
                    q = -q
                prev = q
                pb.rotation_quaternion = q
                pb.keyframe_insert("rotation_quaternion", frame=fr)
            else:
                e = q.to_euler(pb.rotation_mode, prev) if prev is not None else q.to_euler(pb.rotation_mode)
                prev = e
                pb.rotation_euler = e
                pb.keyframe_insert("rotation_euler", frame=fr)
    n_done += 1
bpy.ops.wm.save_as_mainfile(filepath=a.out or a.blend)
print(f"[contact] turns written into {n_done} clips -> {a.out or a.blend}", flush=True)
