"""A rig's joints, frame by frame, in world space - for measuring a clip against the video it was made
from without a renderer or a pose model in between (tools/motion_stages.py).

    blender -b -noaudio --python dump_joints.py -- --blend final.blend --clip punch_combo \
        --frames 124 --fps 24 --out joints.npz
    blender -b -noaudio --python dump_joints.py -- --fbx punch_combo.fbx --frames 124 --fps 24 --out joints.npz

Writes `points` (frames, 13, 3): the 13 points of the pose model's layout (head; shoulders, elbows,
wrists, hips, knees, ankles, left before right) at the bones' heads, with frame i the clip at time
i / fps - the timing blender/render_match.py renders at. Also `rest` (13, 3), the same points in the
rig's rest pose, and `rest_dirs`: each mapped bone's direction in the rest pose, joint to joint,
which a rotation-copy retarget carries into every frame; `rest_mats` (13, 4, 4), each point's bone
in the rest pose in world space, and `bones`, their names.
"""
import argparse
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", default=None)
ap.add_argument("--fbx", default=None)
ap.add_argument("--clip", default=None)
ap.add_argument("--frames", type=int, default=0, help="0: the rest pose only")
ap.add_argument("--fps", type=float, default=24.0)
ap.add_argument("--out", required=True)
ap.add_argument("--calib", default=None, help="joint_calib.json: also write the face points (nose, eyes, ears)")
a = ap.parse_args(argv)

CF = ["head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
      "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]
MX = ["Head", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand",
      "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]
# bones by the joint they start at and the joint they point to
DIRS = {"spine": (("pelvis", "neck"), ("Hips", "Neck")),
        "neck": (("neck", "head"), ("Neck", "Head")),
        "l_collar": (("left_collar", "left_shoulder"), ("LeftShoulder", "LeftArm")),
        "r_collar": (("right_collar", "right_shoulder"), ("RightShoulder", "RightArm")),
        "l_upper_arm": (("left_shoulder", "left_elbow"), ("LeftArm", "LeftForeArm")),
        "r_upper_arm": (("right_shoulder", "right_elbow"), ("RightArm", "RightForeArm")),
        "l_forearm": (("left_elbow", "left_wrist"), ("LeftForeArm", "LeftHand")),
        "r_forearm": (("right_elbow", "right_wrist"), ("RightForeArm", "RightHand")),
        "l_hand": (("left_wrist", "left_middle1"), ("LeftHand", "LeftHandMiddle1")),
        "r_hand": (("right_wrist", "right_middle1"), ("RightHand", "RightHandMiddle1")),
        "l_hip_line": (("right_hip", "left_hip"), ("RightUpLeg", "LeftUpLeg")),
        "l_thigh": (("left_hip", "left_knee"), ("LeftUpLeg", "LeftLeg")),
        "r_thigh": (("right_hip", "right_knee"), ("RightUpLeg", "RightLeg")),
        "l_shin": (("left_knee", "left_ankle"), ("LeftLeg", "LeftFoot")),
        "r_shin": (("right_knee", "right_ankle"), ("RightLeg", "RightFoot")),
        "l_foot": (("left_ankle", "left_foot"), ("LeftFoot", "LeftToeBase")),
        "r_foot": (("right_ankle", "right_foot"), ("RightFoot", "RightToeBase"))}

if a.blend:
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    names, col = CF, 0
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=a.fbx, ignore_leaf_bones=True, automatic_bone_orientation=False)
    names, col = MX, 1
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")


def find(key):
    for b in rig.pose.bones:
        if b.name == key or b.name.endswith(":" + key):
            return b
    return None


W = rig.matrix_world
pbs = [find(n) for n in names]
if any(p is None for p in pbs):
    raise SystemExit(f"[dump] missing bones: {[n for n, p in zip(names, pbs) if p is None]}")
rest = np.array([tuple(W @ p.bone.head_local) for p in pbs])
# each point's bone in the rest pose, world space: where a point read off a render is carried from
rest_mats = np.array([np.array(W @ p.bone.matrix_local) for p in pbs])
rest_dirs = {}
for key, pair in DIRS.items():
    b0, b1 = find(pair[col][0]), find(pair[col][1])
    if b0 is None or b1 is None:
        continue
    d = np.array(tuple((W @ b1.bone.head_local) - (W @ b0.bone.head_local)))
    rest_dirs[key] = d / max(np.linalg.norm(d), 1e-9)

pts, face = [], []
face_local = None
if a.calib:
    import json
    face_local = [f["local"] for f in json.load(open(a.calib))["face"]]
    head_pb = find("head")
if a.frames:
    act = bpy.data.actions.get(a.clip) if a.clip else rig.animation_data.action
    if act is None:
        raise SystemExit(f"[dump] no clip {a.clip!r}")
    if rig.animation_data is None:
        rig.animation_data_create()
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    rig.data.pose_position = "POSE"
    f0, f1 = act.frame_range
    scene_fps = sc.render.fps / sc.render.fps_base
    for i in range(a.frames):
        f = min(f0 + i * scene_fps / a.fps, f1)
        sc.frame_set(int(math.floor(f)), subframe=float(f - math.floor(f)))
        pts.append([tuple(W @ p.head) for p in pbs])
        if face_local is not None:
            M = W @ head_pb.matrix
            face.append([tuple(M @ Vector(l)) for l in face_local])
np.savez_compressed(a.out, points=np.array(pts, np.float64).reshape(-1, 13, 3), rest=rest, rest_mats=rest_mats,
                    bones=np.array([p.name for p in pbs]), face=np.array(face, np.float64).reshape(-1, 5, 3),
                    rest_dir_names=np.array(list(rest_dirs)), rest_dirs=np.array(list(rest_dirs.values())))
print(f"[dump] {len(pts)} frames of {a.clip or a.fbx or 'the rest pose'} -> {a.out}", flush=True)
