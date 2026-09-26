"""A new clip from a video: a library clip replayed at the video's timing, each bone turned so the
body's joints land where pipeline/video_motion.py --fit put them.

    blender -b -noaudio --python fit_clip.py -- --fbx <library clip.fbx> --fit fit.npz --out move.fbx
        [--report move_fit.json]

fit.npz holds, per video frame: the clip's time (library frames, at the library's rate), the
fitted positions of the 13 library points in the clip's world, and whether to turn the bones at
all (a clip that already matches the video is only retimed). The output is a Mixamo skeleton with
one key per video frame, which blender/retarget.py treats as any other Mixamo download.

Bones are turned parents first, each by the smallest rotation that points it where the fit says,
so a bone's twist - which 13 points cannot see - stays the capture's: the hips move to the fitted
hip midpoint, the spine points from the hips to the shoulders, the collarbones to the shoulders,
upper arm to elbow, forearm to wrist, thigh to knee, shin to ankle. The neck and head, hands, feet
and fingers keep the capture's pose relative to them (a pose model's head point is the nose and
ears, not a joint).

After export the FBX is imported again and its joints measured against the fit (--report): the
check that axes, units and the solve all survived the round trip.
"""
import argparse
import json
import math
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--fbx", required=True)
ap.add_argument("--fit", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--report", default=None)
a = ap.parse_args(argv)

POINTS = ["Head", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand",
          "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]
# (bone to turn, its reference point, the point it should aim at): indices into the 13 points,
# -1 = between the hips, -2 = between the shoulders
SPINE = ("Spine", "Spine1", "Spine2")
CHAIN = [("LeftShoulder", None, 1), ("RightShoulder", None, 4),
         ("LeftArm", 1, 2), ("LeftForeArm", 2, 3), ("RightArm", 4, 5), ("RightForeArm", 5, 6),
         ("LeftUpLeg", 7, 8), ("LeftLeg", 8, 9), ("RightUpLeg", 10, 11), ("RightLeg", 11, 12)]


def bone(rig, key):
    for b in rig.pose.bones:
        if b.name == key or b.name.endswith(":" + key):
            return b
    return None


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=True, automatic_bone_orientation=False)
    return next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")


fit = np.load(a.fit)
times, target, turn = fit["times"], fit["points"].astype(np.float64), bool(fit["turn"])
lib_fps, out_fps = float(fit["lib_fps"]), float(fit["fps"])
lib_len = int(fit["lib_frames"])

src = load(a.fbx)
sc = bpy.context.scene
src_fps = sc.render.fps / sc.render.fps_base
f0 = int(src.animation_data.action.frame_range[0])
step = src_fps / lib_fps

# the new clip's skeleton: a copy of the capture's, driven by hand below
dst = src.copy()
dst.data = src.data.copy()
dst.animation_data_clear()
sc.collection.objects.link(dst)
W = src.matrix_world.copy()
Wr = W.to_3x3().normalized()
Wr_inv = Wr.inverted()
names = [pb.name for pb in dst.pose.bones]
for pb in dst.pose.bones:
    pb.rotation_mode = "QUATERNION"
pt_bones = [bone(dst, n) for n in POINTS]


def points(rig_bones):
    return [W @ b.head for b in rig_bones]


def mid(P, i):
    if i == -1:
        return (P[7] + P[10]) / 2
    if i == -2:
        return (P[1] + P[4]) / 2
    return P[i]


def turn_bone(pb, rot_world):
    """Turn a pose bone about its head by a rotation given in world space."""
    Ra = (Wr_inv @ rot_world.to_matrix() @ Wr).to_4x4()
    h = pb.head.copy()
    pb.matrix = Matrix.Translation(h) @ Ra @ Matrix.Translation(-h) @ pb.matrix
    bpy.context.view_layer.update()


keys = {n: [] for n in names}
for i, t in enumerate(times):
    k = float(t) % lib_len
    f = f0 + k * step
    sc.frame_set(int(math.floor(f)), subframe=float(f - math.floor(f)))
    for pb in dst.pose.bones:
        s_pb = src.pose.bones[pb.name]
        pb.location, pb.rotation_quaternion, pb.scale = s_pb.location, s_pb.matrix_basis.to_quaternion(), s_pb.scale
    bpy.context.view_layer.update()
    if turn:
        T = [Vector(p) for p in target[i]]
        # the hips: onto the fitted midpoint, then turned so the hip line lies along the fitted one
        hips = bone(dst, "Hips")
        P = points(pt_bones)
        shift = mid(T, -1) - mid(P, -1)
        m = hips.matrix.copy()
        m.translation = W.inverted() @ (W @ m.translation + shift)
        hips.matrix = m
        bpy.context.view_layer.update()
        P = points(pt_bones)
        turn_bone(hips, (P[7] - P[10]).rotation_difference(T[7] - T[10]))
        # the spine: its three bones share the bend and the twist. Each in turn is rotated about its
        # own head by the rotation that best carries both shoulders onto the fitted ones, the head
        # at half weight (Kabsch, three points), three passes. Aiming one bone along the hips-to-shoulders line left a
        # kick's 30-deg lean with each shoulder 5.7 cm off; aiming it from its own head, or
        # bringing only the shoulders' midpoint home, put Aoi's upright spell 2-4 cm off.
        for _ in range(3):
            for name in reversed(SPINE):
                pb = bone(dst, name)
                if pb is None:
                    continue
                P = points(pt_bones)
                pivot = np.array(W @ pb.head)
                A = np.array([P[1][:], P[4][:], P[0][:]]) - pivot
                B = np.array([T[1][:], T[4][:], T[0][:]]) - pivot
                wts = np.array([1.0, 1.0, 0.5])[:, None]         # the head: kept near, not forced
                U, _, Vt = np.linalg.svd((A * wts).T @ B)
                d = np.sign(np.linalg.det(Vt.T @ U.T))
                Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
                turn_bone(pb, Matrix(Rm.tolist()).to_quaternion())
        for name, ref, aim in CHAIN:
            pb = bone(dst, name)
            if pb is None:
                continue
            P = points(pt_bones)
            start = W @ pb.head if ref is None else mid(P, ref)
            now, want = mid(P, aim) - start, mid(T, aim) - (start if ref is None else mid(T, ref))
            if now.length < 1e-6 or want.length < 1e-6:
                continue
            turn_bone(pb, now.rotation_difference(want))
    for pb in dst.pose.bones:
        mb = pb.matrix_basis
        keys[pb.name].append((tuple(mb.to_translation()), tuple(mb.to_quaternion())))

# the new action: one key per video frame (keyframe_insert: Blender 5's layered actions)
seq = {}
for n in names:
    q = np.array([k[1] for k in keys[n]], np.float64)
    for j in range(1, len(q)):                                     # no flips between frames
        if np.dot(q[j], q[j - 1]) < 0:
            q[j] = -q[j]
    seq[n] = (np.array([k[0] for k in keys[n]], np.float64), q)
for i in range(len(times)):
    for n in names:
        pb = dst.pose.bones[n]
        pb.location = seq[n][0][i]
        pb.rotation_quaternion = seq[n][1][i]
        pb.keyframe_insert("location", frame=i + 1)
        pb.keyframe_insert("rotation_quaternion", frame=i + 1)
sc.frame_start, sc.frame_end = 1, len(times)
sc.render.fps, sc.render.fps_base = int(round(out_fps)), round(out_fps) / out_fps
# The rest pose. Blender writes each bone's FBX node with the pose showing at export, and an
# armature exported without a skinned mesh has no bind pose, so an importer takes that pose as
# the rest: the first export put Mara's boxing guard there, and retarget.py - which carries each
# bone's change from rest - turned every guard into a T-pose. Export with the capture's rest
# (T-pose) showing: a key at frame 0, outside the exported range.
for pb in dst.pose.bones:
    pb.location, pb.rotation_quaternion, pb.scale = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    pb.keyframe_insert("location", frame=0)
    pb.keyframe_insert("rotation_quaternion", frame=0)
sc.frame_set(0)
rest_src = {b.name: (src.matrix_world @ b.head_local).copy() for b in src.data.bones}

bpy.data.objects.remove(src, do_unlink=True)
for o in list(sc.objects):
    o.select_set(o == dst)
bpy.context.view_layer.objects.active = dst
dst.name = "Armature"
bpy.ops.export_scene.fbx(filepath=a.out, use_selection=True, object_types={"ARMATURE"}, add_leaf_bones=True,
                         bake_anim=True, bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                         bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0)

# the round trip: the exported clip's joints against the fit
chk = load(a.out)
cb = [bone(chk, n) for n in POINTS]
Wc = chk.matrix_world
cf0 = int(chk.animation_data.action.frame_range[0])
err, err_src = [], []
for i in range(len(times)):
    bpy.context.scene.frame_set(cf0 + i)
    P = np.array([(Wc @ b.head)[:] for b in cb])
    err.append(np.linalg.norm(P - target[i], axis=1))
err = np.array(err)
# the round trip must also bring back the capture's rest pose (see the export above)
rest_gap = max(((Wc @ b.head_local) - rest_src[b.name]).length for b in chk.data.bones if b.name in rest_src)
if rest_gap > 0.01:
    raise SystemExit(f"[fit] the exported rest pose is not the capture's: a bone head is {rest_gap * 100:.1f} cm off")
# The fit is on the video's own body (pipeline/video_motion.py), which a Mixamo skeleton cannot take:
# what carries over to the character is each bone's direction, so that is what the round trip checks
# (degrees between the exported bone and the fitted one); joint distances are reported beside it.
BONES_RT = [(1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9), (10, 11), (11, 12), (10, 7), (4, 1)]
ang = []
for i in range(len(times)):
    bpy.context.scene.frame_set(cf0 + i)
    P = np.array([(Wc @ b.head)[:] for b in cb])
    for p_, q_ in BONES_RT:
        u_, v_ = P[q_] - P[p_], target[i][q_] - target[i][p_]
        c_ = float(np.dot(u_, v_) / max(np.linalg.norm(u_) * np.linalg.norm(v_), 1e-9))
        ang.append(np.degrees(np.arccos(np.clip(c_, -1, 1))))
ang = np.array(ang)
# the head is left to the capture (it turns with the chest; its point is not a joint to a pose
# model), so the round trip is scored on the twelve fitted joints and the head reported apart
fitted = err[:, 1:]
rep = {"frames": len(times), "fps": out_fps, "turned": turn, "rest_gap_cm": round(rest_gap * 100, 3),
       "bone_angle_deg_mean": float(ang.mean()), "bone_angle_deg_p95": float(np.percentile(ang, 95)),
       "joint_error_cm_mean": float(fitted.mean() * 100), "joint_error_cm_p95": float(np.percentile(fitted, 95) * 100),
       "head_cm_mean": float(err[:, 0].mean() * 100),
       "per_point_cm": {n: round(float(err[:, j].mean() * 100), 2) for j, n in enumerate(POINTS)}}
print(f"[fit] {a.out}: {len(times)} frames at {out_fps:g} fps; bones {rep['bone_angle_deg_mean']:.1f} deg from the "
      f"fit's on average (95% within {rep['bone_angle_deg_p95']:.1f}); joints {rep['joint_error_cm_mean']:.1f} cm (the "
      f"capture's bone lengths, not the video's)", flush=True)
if a.report:
    json.dump(rep, open(a.report, "w"), indent=1)
