"""Retarget Mixamo captures onto a CharForge rig, as clips a character controller can drive.

Rotations. The naive version - a Copy Rotation constraint in world space per bone - copies the
source bone's *absolute* orientation, and with it the source rig's rest pose and bone roll, so
every limb picks up a constant twist. What is wanted is the source's change from its own rest
pose, re-expressed in the target's rest frame:

    tgt_pose_world = src_pose_world @ src_rest_world^-1 @ tgt_rest_world

with every matrix in WORLD space. Mixamo FBX is Y-up and the importer puts the axis correction
(and the 0.01 unit scale) on the armature object, so deltas computed from armature-space matrices
mix a Y-up source with a Z-up target: the first retargeted run flew horizontally. Bones are
posed parent-first, keeping the target's own bone heads, so the character keeps its proportions.

Heading. A controller moves the model the way it faces, so a travelling clip is turned to face
its own direction of travel (a stationary one by its mean hip line). Whatever angle is left
between the two becomes a sideways foot slide of speed x sin(angle).

Root motion. Clips ship in place, with the ground speed in the report: the runtime moves the
character and plays the clip at the matching rate. Only the pelvis's MEAN velocity is removed -
its surge and sway within the stride stay, or the planted feet inherit them as slide. The root is
set through `pose_bone.matrix` in armature space; `pose_bone.location` is in the bone's own rest
frame, and writing a world offset to it once sent a wave's vertical motion out of the frame.

Feet. ground_and_plant() puts the soles on the floor key by key and pins each foot where it is
down, reading contact from the capture - see its docstring for the measurements that forced it.

Run: blender -b -noaudio --python retarget.py -- --rig rig_t.blend --clips clips.json \
         --out animated.glb [--blend-out animated.blend] [--json report.json]
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--rig", required=True)
ap.add_argument("--clips", required=True, help="JSON: {clip_name: fbx_path}")
ap.add_argument("--out", required=True)
ap.add_argument("--blend-out", default=None)
ap.add_argument("--fps", type=int, default=60,
                help="bake rate. 60: at 30 a run's ground contact is 4-6 keys, and interpolating fast leg\n"
                     "rotations between them slid the pinned foot 13%% in the browser against 3%% at the keys")
ap.add_argument("--max-frames", type=int, default=600)
ap.add_argument("--json", default=None)
ap.add_argument("--calib", default=None,
                help="the character's joint_calib.json (tools/calibrate_joints.py): its face points, to turn the\n"
                     "head of a clip made from a video to the video's face")
ap.add_argument("--keep-root-motion", action="store_true",
                help="bake horizontal root travel into the clip (default: in-place + metadata)")
a = ap.parse_args(argv)

# mixamorig -> CharForge / SMPL-H. Only bones the rig actually has.
MAP = {
    "Hips": "pelvis",
    "Spine": "spine1", "Spine1": "spine2", "Spine2": "spine3",
    "Neck": "neck", "Head": "head",
    "LeftShoulder": "left_collar", "RightShoulder": "right_collar",
    "LeftArm": "left_shoulder", "RightArm": "right_shoulder",
    "LeftForeArm": "left_elbow", "RightForeArm": "right_elbow",
    "LeftHand": "left_wrist", "RightHand": "right_wrist",
    "LeftUpLeg": "left_hip", "RightUpLeg": "right_hip",
    "LeftLeg": "left_knee", "RightLeg": "right_knee",
    "LeftFoot": "left_ankle", "RightFoot": "right_ankle",
}
# fingers: three per chain, where the rig has them (a rig built by rig_build.py does)
for _S, _s in (("Left", "left"), ("Right", "right")):
    for _F in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for _i in (1, 2, 3):
            MAP[f"{_S}Hand{_F}{_i}"] = f"{_s}_{_F.lower()}{_i}"
PREFIXES = ("mixamorig:", "mixamorig1:", "mixamorig2:", "")


def src_name(rig, key):
    for p in PREFIXES:
        if p + key in rig.pose.bones:
            return p + key
    return None


def action_fcurves(act):
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            for slot in act.slots:
                cb = strip.channelbag(slot)
                if cb:
                    out.extend(cb.fcurves)
    return out


# ---- load the target rig -----------------------------------------------------------------
if a.rig.endswith(".blend"):
    bpy.ops.wm.open_mainfile(filepath=a.rig)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=a.rig)
scene = bpy.context.scene
scene.render.fps = a.fps
tgt = next((o for o in scene.objects if o.type == "ARMATURE"), None)
if tgt is None:
    raise SystemExit("[retarget] no armature in the target rig")
tgt_rest = {b.name: b.matrix_local.copy() for b in tgt.data.bones}

# Clear any actions already in the file. The target .blend carries the earlier procedural clips
# under exactly the names used here, so Blender silently renames the new ones to walk.001,
# run.001 and so on - and every downstream tool, including the review renderer, then plays the
# *old* clips while the log happily reports that five new ones were baked. That is how a broken
# procedural `run` got reviewed as if it were the retargeted one.
_stale = [act for act in bpy.data.actions]
for act in _stale:
    act.use_fake_user = False
    bpy.data.actions.remove(act)
if tgt.animation_data:
    tgt.animation_data.action = None
print(f"[retarget] cleared {len(_stale)} pre-existing action(s)", flush=True)
def leg_length(arm, pelvis, ankle):
    """Pelvis-to-ankle distance in WORLD space.

    Two traps here, both of which produced a character that played every clip on the spot.

    First, hip height above the ground is not a usable reference on this rig: its armature
    origin sits near the chest, so the pelvis world Z is ~0.005. A bone-chain length is
    independent of where the armature is placed.

    Second - and this is the one that survived the first fix - `head_local` is in armature
    space, which for an imported FBX is still in the file's own units. Blender's FBX importer
    leaves Mixamo's centimetres in the armature data and puts the 0.01 conversion on the object
    matrix instead. So head_local reports a leg of 84.8 while the hips genuinely stand 0.84 m
    off the floor, and dividing one rig's metres by the other's centimetres scaled the root
    motion down by another 100x. Measuring both rigs in world space makes the ratio a pure
    proportion, which is all it should ever have been.
    """
    b = arm.data.bones
    if pelvis not in b or ankle not in b:
        return None
    M = arm.matrix_world
    return ((M @ b[pelvis].head_local) - (M @ b[ankle].head_local)).length


tgt_leg = leg_length(tgt, "pelvis", "left_ankle") or 1.0
print(f"[retarget] target rig {tgt.name}, {len(tgt.pose.bones)} bones, "
      f"leg length {tgt_leg:.3f}", flush=True)

LEGS = (("left_hip", "left_knee", "left_ankle", "left_foot", 0, 1),
        ("right_hip", "right_knee", "right_ankle", "right_foot", 2, 3))

# A clip made from a video carries the points pipeline/video_motion.py --fit solved for: the pose
# model's 13 points (head; shoulders, elbows, wrists, hips, knees, ankles, left before right) with
# depth, on the video's own body. Copying rotations from the fitted Mixamo skeleton carries each
# bone's rest-pose difference into every frame - the two skeletons' thighs differ by 4-12 deg at
# rest, their collarbones by 16-27 - so the character's limbs are aimed along the fitted bones
# directly, frame by frame, after the copy (fit_to_points).
J13 = ["head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
       "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]
MIRROR13 = [0, 4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9]
AIMS = [("left_shoulder", 1, 2), ("left_elbow", 2, 3), ("right_shoulder", 4, 5), ("right_elbow", 5, 6),
        ("left_hip", 7, 8), ("left_knee", 8, 9), ("right_hip", 10, 11), ("right_knee", 11, 12)]
SPINE_T = ("spine1", "spine2", "spine3")
FACE_MIRROR = [0, 2, 1, 4, 3]


def kabsch(A, B):
    """The rotation that best carries point set A onto B, both about their centres."""
    A, B = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Matrix((Vt.T @ np.diag([1.0, 1.0, d]) @ U.T).tolist()).to_quaternion()


def fit_to_points(tgt, F, TW, rig_torso, fit_torso, stats, face=None, face_local=None):
    """Pose the character so its skeleton follows the fitted points F (13 Vectors, world space, already
    mirrored and turned as the clip is): the pelvis turned onto the fitted hip line; the spine's three
    bones turned about their heads so both shoulders come onto the fitted ones (Kabsch, three passes;
    the neck held at the fit's angle to them, half weight); each upper arm, forearm, thigh and shin
    turned about its head to point along its fitted bone. Collarbones, hands, feet and fingers keep
    the copied pose - the 13 points do not see them. A bone's twist stays the copy's. With the fit's
    face points (face) and the character's own (face_local, in its head bone's frame) the neck and
    head then turn, half each, until the face lies as the video's does."""
    pbs = tgt.pose.bones
    TWr = TW.to_3x3().normalized()
    TWr_inv = TWr.inverted()

    def turn(pb, q):
        Ra = (TWr_inv @ q.to_matrix() @ TWr).to_4x4()
        h = pb.head.copy()
        pb.matrix = Matrix.Translation(h) @ Ra @ Matrix.Translation(-h) @ pb.matrix
        bpy.context.view_layer.update()
        stats.setdefault(pb.name, []).append(math.degrees(q.angle))

    def pts():
        return [TW @ pbs[n].head for n in J13]

    pelvis = pbs.get("pelvis")
    if pelvis is not None:
        P = pts()
        turn(pelvis, (P[7] - P[10]).rotation_difference(F[7] - F[10]))
    # the fitted body laid on the rig's: hips on hips, torso length for torso length
    P = pts()
    k = rig_torso / max(fit_torso, 1e-9)
    hr, hf = (P[7] + P[10]) / 2, (F[7] + F[10]) / 2
    G = [hr + (f - hf) * k for f in F]
    sm_f = (F[1] + F[4]) / 2
    neck = (F[0] - sm_f).normalized()
    for _ in range(3):
        for name in reversed(SPINE_T):
            pb = pbs.get(name)
            if pb is None:
                continue
            P = pts()
            sm_p, sm_g = (P[1] + P[4]) / 2, (G[1] + G[4]) / 2
            head_goal = sm_g + neck * (P[0] - sm_p).length
            pivot = np.array(TW @ pb.head)
            A = np.array([P[1][:], P[4][:], P[0][:]]) - pivot
            B = np.array([G[1][:], G[4][:], head_goal[:]]) - pivot
            wts = np.array([1.0, 1.0, 0.5])[:, None]
            U, _, Vt = np.linalg.svd((A * wts).T @ B)
            d = np.sign(np.linalg.det(Vt.T @ U.T))
            Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
            turn(pb, Matrix(Rm.tolist()).to_quaternion())
    for name, ref, aim in AIMS:
        pb = pbs.get(name)
        if pb is None:
            continue
        P = pts()
        now, want = P[aim] - P[ref], F[aim] - F[ref]
        if now.length > 1e-6 and want.length > 1e-6:
            turn(pb, now.rotation_difference(want))
    head = pbs.get("head")
    if face is not None and face_local is not None and head is not None:
        B = np.array([tuple(v) for v in face])

        def face_now():
            M = TW @ head.matrix
            return np.array([tuple(M @ Vector(l)) for l in face_local])
        q = kabsch(face_now(), B)
        neck = pbs.get("neck")
        if neck is not None:
            turn(neck, Quaternion().slerp(q, 0.5))
            q = kabsch(face_now(), B)
        turn(head, q)


def sole_points(tgt, TW):
    """The sole of each foot as a cloud of points fixed to the foot bone, at most 160 per foot.

    A generated skeleton's foot joints are estimated from renders of a shoe and sit inside it. On
    these rigs the 'ball' joint is at the instep, 9 cm above the floor (Mixamo's is on the floor),
    and pinning that joint pinned a point in mid-air: the solver dragged the pelvis down 6-11 cm to
    reach it. The sole is read off the mesh instead - the rest-pose vertices the foot bones own,
    within 3 cm of the floor - and carried in the foot bone's own frame, so that at any key the
    point of the shoe actually touching the floor is simply the lowest of them."""
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"
              and any(m.type == "ARMATURE" and m.object == tgt for m in o.modifiers)]
    out = {}
    for hip_b, knee_b, ank_b, ball_b, _, _ in LEGS:
        pts = []
        for o in meshes:
            gi = {g.index for g in o.vertex_groups if g.name in (ank_b, ball_b)}
            if not gi:
                continue
            mw = o.matrix_world
            for vtx in o.data.vertices:
                if sum(g.weight for g in vtx.groups if g.group in gi) > 0.5:
                    q = mw @ vtx.co
                    if q.z < 0.03:
                        pts.append(q)
        if len(pts) < 20:
            # no sole found: stand in with points under the skeleton's own heel and ball
            ank = TW @ tgt.data.bones[ank_b].head_local
            bl = TW @ tgt.data.bones[ball_b].head_local
            pts = [Vector((ank.x, ank.y, 0.0)), Vector((bl.x, bl.y, 0.0))]
            how = "no sole vertices found - two points under the skeleton's own joints"
        else:
            ys = [q.y for q in pts]
            how = f"sole from {len(pts)} vertices, {(max(ys) - min(ys))*100:.1f} cm long"
            pts = pts[::max(1, len(pts) // 160)]
        Minv = (TW @ tgt.data.bones[ank_b].matrix_local).inverted()
        out[ank_b] = np.array([tuple(Minv @ q) + (1.0,) for q in pts])
        print(f"[retarget] {ank_b}: {how}", flush=True)
    return out


def ground_and_plant(tgt, pelvis, n, fps, travelling, src_feet, lap, looping, spd_thr, TW, sole,
                     drift=(0.0, 1.0)):
    """Put the feet on the floor and keep them still while they are down. Returns a stats dict.

    Two defects of rotation-copy retargeting, both measured on every clip before this existed:

      height  no reference taken from the capture survives contact with the floor. Frame 0 as
              "standing" floated the run 4.8 cm, because the run starts on bent knees; the
              capture's rest pose sank every clip 4-5 cm, because that pose does not stand on the
              capture's own floor; and a leg of different proportions reaches the floor at
              different heights through the stride anyway. So the floor is measured where the feet
              are: key by key, the pelvis moves so the lowest point of a planted sole is on the
              floor - smoothed, and interpolated through flight.
      slip    the foot's motion relative to the hips is a sum over thigh and shin, each scaled by
              its own length ratio, so a planted foot drifts through the stance even when its
              average speed is right: 20-30% slip, from captures whose own feet slip under 2%.
              The capture still knows exactly when each foot is down, so contact is read from the
              SOURCE. A foot rolls - heel strike, flat, toe-off - so each contact has its own pivot:
              the point of the sole lowest when the heel lands, and the point lowest when the toe
              leaves. That point is pinned to one place in the world (which, in an in-place clip,
              moves backwards at the ground speed) and a two-bone IK on thigh and shin reaches it,
              keeping the knee in its animated plane and the foot at its animated angle.
    """
    scene = bpy.context.scene
    TW_inv = TW.inverted()
    up = TW_inv.to_3x3() @ Vector((0.0, 0.0, 1.0))
    pbs = tgt.pose.bones
    period = n - 1                                   # on a loop, key n-1 is key 0 again
    RAMP = max(1, round(0.067 * fps))                # keys of blend either side of a contact (67 ms)
    TWn = np.array(TW)

    def at(arr, j):
        return arr[j % period] if looping else arr[min(max(j, 0), n - 1)]

    def contact(k):
        z = [src_feet[i][k].z for i in range(n)]
        zmin = min(z)
        m = []
        for i in range(n):
            if looping and i == 0:
                p0, p1, gap = src_feet[n - 2][k] - lap, src_feet[1][k], 2
            elif looping and i == n - 1:
                p0, p1, gap = src_feet[n - 2][k], src_feet[1][k] + lap, 2
            else:
                i0, i1 = max(i - 1, 0), min(i + 1, n - 1)
                p0, p1, gap = src_feet[i0][k], src_feet[i1][k], i1 - i0
            dv = p1 - p0
            m.append(z[i] < zmin + 0.025 and math.hypot(dv.x, dv.y) * fps / max(gap, 1) < spd_thr)
        # contacts or gaps shorter than ~1/30 s, between runs of the other kind, are noise
        minlen = max(2, round(0.034 * fps))
        runs, i = [], 0
        while i < n:
            j = i
            while j < n and m[j] == m[i]:
                j += 1
            runs.append((i, j, m[i]))
            i = j
        for q in range(1, len(runs) - 1):
            r0, r1, val = runs[q]
            if r1 - r0 < minlen and runs[q - 1][2] == runs[q + 1][2] != val:
                for j in range(r0, r1):
                    m[j] = runs[q - 1][2]
        if looping:
            m[n - 1] = m[0]
        return m

    def intervals(m):
        """Runs of contact as (key, time) lists; a run through the loop point is one run, its early
        keys unwrapped to the next cycle."""
        runs, cur = [], []
        for i in range(n):
            if m[i]:
                cur.append(i)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        out = [[(i, float(i)) for i in r] for r in runs]
        if looping and len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n - 1:
            out = [out[-1] + [(i, float(i + period)) for i in runs[0]]] + out[1:-1]
        return out

    def offset(i, j):
        o = i - j
        if looping:
            o = (o + period // 2) % period - period // 2
        return o

    def points():
        """Per key, per leg: (sole points in world space as an array, ankle position)."""
        rows = []
        for i in range(n):
            scene.frame_set(i + 1)
            row = []
            for leg in LEGS:
                ank = leg[2]
                M = TWn @ np.array(pbs[ank].matrix)
                row.append(((sole[ank] @ M.T)[:, :3], TW @ pbs[ank].head))
            rows.append(row)
        return rows

    masks = [(contact(sa), contact(sb)) for (_, _, _, _, sa, sb) in LEGS]
    down = [[ma[i] or mb[i] for i in range(n)] for ma, mb in masks]
    n_contact = sum(1 for li in range(2) for i in range(n) if down[li][i])

    # ---- height ----------------------------------------------------------------------------
    rows = points()
    if n_contact:
        need = [None] * n
        for i in range(n):
            # the HIGHER of the planted feet comes down to the floor; the lower one is then
            # lifted back onto it by its own leg below - a leg can always bend, not always reach
            zs = [float(rows[i][li][0][:, 2].min()) for li in range(2) if down[li][i]]
            need[i] = -max(zs) if zs else None
        known = [i for i in range(n) if need[i] is not None]
        for i in range(n):                           # through flight: interpolate between contacts
            if need[i] is None:
                if looping:
                    prv = max((j for j in known if j < i), default=known[-1] - period)
                    nxt = min((j for j in known if j > i), default=known[0] + period)
                else:
                    prv = max((j for j in known if j < i), default=None)
                    nxt = min((j for j in known if j > i), default=None)
                    prv = nxt if prv is None else prv
                    nxt = prv if nxt is None else nxt
                a_, b_ = at(need, prv), at(need, nxt)
                need[i] = a_ if nxt == prv else a_ + (b_ - a_) * (i - prv) / (nxt - prv)
        sig = max(1.0, 0.033 * fps)                  # ~33 ms of smoothing, whatever the bake rate
        rad = int(3 * sig)
        ks = [math.exp(-0.5 * (o / sig) ** 2) for o in range(-rad, rad + 1)]
        need = [sum(k * at(need, i + o) for k, o in zip(ks, range(-rad, rad + 1))) / sum(ks) for i in range(n)]
        if looping:
            need[n - 1] = need[0]
    else:                                            # never down: lowest sole point to the floor
        need = [-min(float(r[li][0][:, 2].min()) for r in rows for li in range(2))] * n
    if max(abs(x) for x in need) > 0.25:
        print(f"[retarget]   WARNING the feet would need {max(need, key=abs)*100:+.1f} cm to reach "
              f"the floor - height left as captured (a kneel or a fall?)", flush=True)
        need = [0.0] * n
    for i in range(n):
        scene.frame_set(i + 1)
        M = pelvis.matrix.copy()
        M.translation = M.translation + up * need[i]
        pelvis.matrix = M
        bpy.context.view_layer.update()
        pelvis.keyframe_insert("location", frame=i + 1)
    rows = points()

    # ---- pivots: where each contact rolls about ----------------------------------------------
    # heel contacts pivot on the sole point lowest when the heel lands, ball contacts on the one
    # lowest when the toe leaves; the ball's pivot takes over wherever both are down
    tracks = []                                      # (leg, interval, pivot index)
    pivot = [[None] * n for _ in range(2)]
    for li, (ma, mb) in enumerate(masks):
        for which, m in (("heel", ma), ("ball", mb)):
            for iv in intervals(m):
                key = iv[0][0] if which == "heel" else iv[-1][0]
                idx = int(np.argmin(rows[key][li][0][:, 2]))
                tracks.append((li, which, iv, idx))
                for i, _ in iv:
                    if which == "ball" or pivot[li][i] is None:
                        pivot[li][i] = idx

    def pos(rows, li, i, idx):
        return Vector(rows[i][li][0][idx])

    def vel(rows, li, i, idx):
        if looping and i in (0, n - 1):
            return (pos(rows, li, 1, idx) - pos(rows, li, n - 2, idx)) * (fps / 2)
        i0, i1 = max(i - 1, 0), min(i + 1, n - 1)
        return (pos(rows, li, i1, idx) - pos(rows, li, i0, idx)) * (fps / max(i1 - i0, 1))

    def planted(rows):
        return [(vel(rows, li, i, pivot[li][i]), pos(rows, li, i, pivot[li][i]))
                for li in range(2) for i in range(n - 1 if looping else n) if pivot[li][i] is not None]

    # In an in-place clip a planted foot drifts opposite to the way the character travels: backwards
    # for a walk, sideways for a strafe, forwards for a backpedal. `drift` is that direction in the
    # ground plane; speed and slip are measured along and across it.
    dx_, dy_ = drift
    ps = planted(rows)
    v = sum(p[0].x * dx_ + p[0].y * dy_ for p in ps) / len(ps) if (travelling and ps) else 0.0
    ref = v if v >= 1.0 else 1.0

    def slip_of(ps):
        return sum(math.hypot(p[0].x - v * dx_, p[0].y - v * dy_) for p in ps) / len(ps) / ref if ps else None
    slip_before = slip_of(ps)

    # ---- plant ------------------------------------------------------------------------------
    def shift(t):
        return Vector((v * dx_ * t / fps, v * dy_ * t / fps, 0.0))

    lock = {(li, w): [(0.0, None, None)] * n for li in range(2) for w in ("heel", "ball")}
    for li, which, iv, idx in tracks:
        P = sum((pos(rows, li, i, idx) - shift(t) for i, t in iv), Vector()) / len(iv)
        tt = dict(iv)
        out = lock[(li, which)]
        for i in range(n):
            if i in tt:
                out[i] = (1.0, P + shift(tt[i]), idx)
                continue
            j = min(tt, key=lambda j: abs(offset(i, j)))
            o = offset(i, j)
            if 0 < abs(o) <= RAMP:
                w = 0.5 * (1 + math.cos(math.pi * abs(o) / (RAMP + 1)))
                if w > out[i][0]:
                    out[i] = (w, P + shift(tt[j] + o), idx)
    if looping:
        for k in lock:
            lock[k][n - 1] = lock[k][0]

    # keys from each key to that leg's nearest contact, around the loop on a cycle
    dist = [[min((abs(offset(i, j)) for j in range(n) if pivot[li][j] is not None), default=99)
             for i in range(n)] for li in range(2)]
    goals = {}
    for li in range(2):
        for i in range(n):
            (wa, pa, ia), (wb, pb_, ib) = lock[(li, "heel")][i], lock[(li, "ball")][i]
            A = rows[i][li][1]
            goal = A.copy()
            if pa is not None and wa > 0:
                goal = A.lerp(pa - (pos(rows, li, i, ia) - A), wa)
            if pb_ is not None and wb > 0:
                goal = goal.lerp(pb_ - (pos(rows, li, i, ib) - A), wb)
            # height. While the foot is down its lowest sole point sits ON the floor - not the
            # pivot, which holds a rolling foot still sideways but would drive the rest of the
            # sole through the floor as it turns. While it is up it may never go below the floor,
            # and keeps a clearance that eases in from contact (15 cm/s, up to 15 mm): without
            # it the retargeted swing foot brushed the floor two keys before landing, lifted, and
            # landed again, and dipped 3 cm through it just after toe-off - both invisible to a
            # contact-only solve, both plain on the mesh.
            low = float(rows[i][li][0][:, 2].min())
            if pivot[li][i] is not None:
                goal.z = A.z - low
            else:
                goal.z = A.z + max(0.0, min(0.015, 0.15 * dist[li][i] / fps) - low)
            if (goal - A).length > 1e-5:
                goals.setdefault(i, {})[li] = goal

    locked = 0
    for i in sorted(goals):
        scene.frame_set(i + 1)
        for li, goal in goals[i].items():
            hip_b, knee_b, ank_b = (pbs[k] for k in LEGS[li][:3])
            fk_euler = {b.name: b.rotation_euler.copy() for b in (hip_b, knee_b, ank_b)}
            foot_fk = ank_b.matrix.copy()
            H, K, Aa = hip_b.head.copy(), knee_b.head.copy(), ank_b.head.copy()
            T = TW_inv @ goal
            L1, L2 = (K - H).length, (Aa - K).length
            u = T - H
            dist = min(max(u.length, abs(L1 - L2) + 1e-4), (L1 + L2) * 0.9995)
            u.normalize()
            wv = (K - H) - (K - H).dot(u) * u        # the knee stays in its animated plane
            if wv.length < 1e-6:
                continue
            wv.normalize()
            ca = max(-1.0, min(1.0, (L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist)))
            K2 = H + L1 * (ca * u + math.sqrt(max(0.0, 1 - ca * ca)) * wv)
            A2 = H + dist * u
            q1 = (K - H).normalized().rotation_difference((K2 - H).normalized())
            hip_b.matrix = (Matrix.Translation(H) @ q1.to_matrix().to_4x4()
                            @ Matrix.Translation(-H) @ hip_b.matrix)
            bpy.context.view_layer.update()
            Kc, Ac = knee_b.head.copy(), ank_b.head.copy()
            q2 = (Ac - Kc).normalized().rotation_difference((A2 - Kc).normalized())
            knee_b.matrix = (Matrix.Translation(Kc) @ q2.to_matrix().to_4x4()
                             @ Matrix.Translation(-Kc) @ knee_b.matrix)
            bpy.context.view_layer.update()
            foot_fk.translation = ank_b.head.copy()   # the foot keeps its animated angle
            ank_b.matrix = foot_fk
            bpy.context.view_layer.update()
            for b in (hip_b, knee_b, ank_b):
                # the same rotation, in the Euler branch of the key it replaces, so the curve does
                # not spin through 360 degrees between this key and its neighbours
                e = b.rotation_euler.copy()
                e.make_compatible(fk_euler[b.name])
                b.rotation_euler = e
                b.keyframe_insert("rotation_euler", frame=i + 1)
            locked += 1

    rows = points()
    if os.environ.get("CF_DEBUG_FEET"):
        for i in range(n):
            print("[feet-debug] key %2d need %+.3f | " % (i, need[i]) + " | ".join(
                f"{'LR'[li]} down {int(down[li][i])} piv {pivot[li][i]} low {float(rows[i][li][0][:, 2].min())*100:+5.1f}cm"
                f" goal {'y' if (i in goals and li in goals[i]) else '-'}" for li in range(2)), flush=True)
    ps = planted(rows)
    zs = [p[1].z for p in ps]
    lowest = min(float(r[li][0][:, 2].min()) for r in rows for li in range(2))
    # airborne: both soles clear of the floor - what a controller launches and lands on
    sole_z = [min(float(rows[i][li][0][:, 2].min()) for li in range(2)) for i in range(n)]
    flights, start = [], None
    for i in range(n):
        if sole_z[i] > 0.02 and start is None:
            start = i
        elif sole_z[i] <= 0.02 and start is not None:
            flights.append([start, i])
            start = None
    if start is not None:
        flights.append([start, n])
    return {"speed": v, "slip_before": slip_before, "slip": slip_of(ps), "locked": locked,
            "looping": looping, "contact_keys": n_contact,
            "planted_height_cm": (round(100 * sum(abs(z) for z in zs) / len(zs), 2) if zs else None),
            "planted_height_range_cm": ([round(100 * min(zs), 2), round(100 * max(zs), 2)] if zs else None),
            "sole_lowest_cm": round(100 * lowest, 2),
            "flights": flights, "apex_sole_m": round(max(sole_z), 3),
            "pelvis_adjust_cm": [round(100 * min(need), 2), round(100 * max(need), 2)]}


SOLE = sole_points(tgt, tgt.matrix_world.copy())
FACE_LOCAL = None
if a.calib and os.path.exists(a.calib):
    FACE_LOCAL = [f["local"] for f in json.load(open(a.calib)).get("face", [])] or None
clips = json.load(open(a.clips))
report = {}

# A clip is a file, or a file with options (animations/default_clips.json):
#   travel_deg  the direction it is meant to travel, degrees from forward, counter-clockwise
#               seen from above: 0 forward, 90 left, -90 right, 180 back. Strafes and backpedals
#               need this - turning them to face their travel, as a forward walk is, would walk
#               them sideways.
#   heading     "auto" (travel direction when travelling, else the mean hip line) or "start"
#               (the hip line at the first frames: a turn in place starts facing forward and
#               ends turned, rather than being centred on its mean)
#   ground      false for a clip that never touches the floor (a fall loop): no planting
for clip_name, spec_ in clips.items():
    opts = spec_ if isinstance(spec_, dict) else {"file": spec_}
    fbx = opts["file"]
    travel_rad = math.radians(float(opts.get("travel_deg", 0.0)))
    # ---- import the source clip into the same scene ---------------------------------------
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.fbx(filepath=fbx, ignore_leaf_bones=True,
                                 automatic_bone_orientation=False)
    except Exception as e:
        print(f"[retarget] {clip_name}: import failed: {e}", flush=True)
        continue
    added = [o for o in bpy.data.objects if o not in before]
    src = next((o for o in added if o.type == "ARMATURE"), None)
    if src is None:
        print(f"[retarget] {clip_name}: no source armature", flush=True)
        continue

    src_act = src.animation_data.action if src.animation_data else None
    f0, f1 = (int(x) for x in src_act.frame_range)
    # The FBX importer sets the scene fps to the clip's own rate. Resample onto the export rate
    # instead of replaying frame-for-frame, or a 60 fps capture plays at half speed.
    src_fps = scene.render.fps or a.fps
    dur = (f1 - f0) / src_fps
    n = min(int(round(dur * a.fps)) + 1, a.max_frames)
    step = (f1 - f0) / max(n - 1, 1)
    scene.render.fps = a.fps

    src_rest = {b.name: b.matrix_local.copy() for b in src.data.bones}
    # A mirrored clip: the source is read reflected across its own centre plane, left bones
    # driving right ones. A world reflection applied to both the pose and the rest cancels in
    # every rotation delta, so what reaches the target is a proper, mirrored motion - a right
    # turn from a left turn, with the same timing and style.
    MIR = bool(opts.get("mirror"))

    def sname_(key):
        if MIR:
            key = key.replace("Left", "\0").replace("Right", "Left").replace("\0", "Right")
        return src_name(src, key)

    hips = sname_("Hips")
    s_pelvis = sname_("Hips")
    s_ankle = sname_("LeftFoot")
    src_leg = leg_length(src, s_pelvis, s_ankle) if s_ankle else None
    scale = (tgt_leg / src_leg) if src_leg else 1.0
    # Mixamo FBX is authored in centimetres and this rig is in metres, so a legitimate scale
    # here is around 0.01. The bound is only here to catch a scale that would collapse the
    # motion entirely; the real check on stride length happens after the bake.
    if not (1e-4 < scale < 1e4):
        raise SystemExit(f"[retarget] {clip_name}: implausible retarget scale {scale:.5f} "
                         f"(target leg {tgt_leg:.3f}, source leg {src_leg}).")

    pairs = []
    for key, tname in MAP.items():
        sname = sname_(key)
        if sname and tname in tgt.pose.bones:
            pairs.append((sname, tname))
    missing = [k for k in MAP if sname_(k) is None]
    print(f"[retarget] {clip_name}: {f1-f0+1} src frames @{src_fps}fps -> {n} @{a.fps}fps, "
          f"{len(pairs)}/{len(MAP)} bones mapped, scale {scale:.3f}" + (f", missing {missing}" if missing else ""), flush=True)

    # parent-first order on the target
    order = []
    seen = set()

    def visit(b):
        if b.name in seen:
            return
        seen.add(b.name)
        if b.parent:
            visit(b.parent)
        order.append(b.name)

    for _, tname in pairs:
        visit(tgt.data.bones[tname])
    tmap = {t: s for s, t in pairs}

    # ---- bake ------------------------------------------------------------------------------
    act = bpy.data.actions.new(clip_name)
    if act.name != clip_name:
        raise SystemExit(f"[retarget] action name collision: asked for {clip_name!r}, "
                         f"Blender gave {act.name!r}. Downstream would play the wrong clip.")
    act.use_fake_user = True
    if tgt.animation_data is None:
        tgt.animation_data_create()
    tgt.animation_data.action = act
    if hasattr(tgt.animation_data, "action_slot") and getattr(act, "slots", None):
        tgt.animation_data.action_slot = act.slots[0]

    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="POSE")
    for pb in tgt.pose.bones:
        pb.rotation_mode = "XYZ"

    SW = src.matrix_world.copy()
    if MIR:
        SW = Matrix.Scale(-1.0, 4, Vector((1.0, 0.0, 0.0))) @ SW

    # ---- heading ------------------------------------------------------------------------------
    # A character controller moves the model along the way it faces, so a travelling clip must
    # travel along the character's forward: any angle between the two becomes a sideways foot
    # slide of speed x sin(angle). Copying world-space rotations copies the capture's heading, and
    # a capture is not always recorded facing its own direction of travel. The first attempt at
    # this removed the mean HIP-LINE heading instead - and turned a walk that travelled dead
    # straight 13 degrees off course, because that capture walks with its pelvis turned. The body
    # cues (hips, shoulders, feet) disagree with each other by up to 25 degrees on the same clip;
    # the travel direction is the one thing the controller and the planted feet both depend on.
    # So: a travelling clip is turned to face its travel; a stationary one by its mean hip line.
    s_lup, s_rup = sname_("LeftUpLeg"), sname_("RightUpLeg")

    def heading(lp, rp):
        side = lp - rp
        return math.atan2(side.y, side.x)          # facing is perpendicular; offsets cancel below
    rest_h = heading((SW @ src_rest[s_lup]).to_translation(), (SW @ src_rest[s_rup]).to_translation())
    rest_fwd = rest_h - math.pi / 2                 # side x up: the rest pose's forward, in the plane
    ss, cs = 0.0, 0.0
    hips_path = []
    s_feet = [sname_(k) for k in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")]
    src_feet = []
    for i in range(n):
        t = f0 + i * step
        scene.frame_set(int(math.floor(t)), subframe=float(t - math.floor(t)))
        h = heading(SW @ src.pose.bones[s_lup].head, SW @ src.pose.bones[s_rup].head) - rest_h
        ss += math.sin(h); cs += math.cos(h)
        hips_path.append((SW @ src.pose.bones[hips].head).copy())
        if all(s_feet):
            src_feet.append([(SW @ src.pose.bones[k].head).copy() for k in s_feet])
    hip_yaw = math.atan2(ss, cs)
    d = hips_path[-1] - hips_path[0]
    d.z = 0.0
    src_hip_h = (SW @ src_rest[hips]).to_translation().z
    travelling = d.length > 0.25 * src_hip_h and d.length / max((n - 1) / a.fps, 1e-6) > 0.3 * src_hip_h
    head_mode = opts.get("heading", "auto")
    if head_mode == "start":
        k0 = max(2, n // 20)
        s0 = c0 = 0.0
        for i in range(k0):
            t = f0 + i * step
            scene.frame_set(int(math.floor(t)), subframe=float(t - math.floor(t)))
            h = heading(SW @ src.pose.bones[s_lup].head, SW @ src.pose.bones[s_rup].head) - rest_h
            s0 += math.sin(h); c0 += math.cos(h)
        yaw = math.atan2(s0, c0)
        how = "hip line at the first frames (a turn: starts facing forward)"
        travelling = False
    elif travelling:
        want = rest_fwd + travel_rad                  # where its travel should point, after the turn
        yaw = math.atan2(math.sin(math.atan2(d.y, d.x) - want), math.cos(math.atan2(d.y, d.x) - want))
        how = (f"travel direction (hip line says {math.degrees(hip_yaw):+.1f})" if not travel_rad else
               f"travel direction set to {math.degrees(travel_rad):+.0f} deg from forward "
               f"(body then faces {math.degrees(math.atan2(math.sin(hip_yaw - yaw), math.cos(hip_yaw - yaw))):+.1f})")
    else:
        yaw = hip_yaw
        how = "mean hip line (stationary clip)"
    UNYAW = Matrix.Rotation(-yaw, 4, "Z")
    # how far the body turns from first frame to last (a turn in place: about +-90)
    scene.frame_set(f0)
    h_first = heading(SW @ src.pose.bones[s_lup].head, SW @ src.pose.bones[s_rup].head)
    scene.frame_set(f1)
    h_last = heading(SW @ src.pose.bones[s_lup].head, SW @ src.pose.bones[s_rup].head)
    turn_total = math.atan2(math.sin(h_last - h_first), math.cos(h_last - h_first))
    print(f"[retarget]   heading {math.degrees(yaw):+.1f} deg off forward by {how} - removed", flush=True)

    # ---- root motion ----------------------------------------------------------------------------
    # A character controller moves the model at a constant velocity, so an in-place clip must keep
    # everything the pelvis does EXCEPT that velocity. The first in-place version pinned the pelvis
    # outright, which also threw away its surge and sway within each stride, and the planted feet
    # inherited both as slide: 10-30% of ground speed on clips whose source feet slip under 2%.
    # Removing only the straight line from the first frame's position to the last keeps the sway,
    # and on a clip that is one whole cycle the residual is zero at both ends, so it still loops.
    TW = tgt.matrix_world.copy()
    TW_inv = TW.inverted()
    R3 = UNYAW.to_3x3()
    rest_hips_z = (SW @ src_rest[hips]).to_translation().z
    offs = [R3 @ ((p - hips_path[0]) * scale) for p in hips_path]   # character frame, target scale
    last = offs[-1].copy()
    last.z = 0.0
    travel = last.length
    dur = max((n - 1) / a.fps, 1e-6)                 # n keys span n-1 intervals
    per_sec = travel / dur / tgt_leg

    def root_offset(i):
        o = offs[i].copy()
        if not a.keep_root_motion:
            o -= last * (i / max(n - 1, 1))          # the mean velocity goes; the sway stays
        o.z = (hips_path[i].z - rest_hips_z) * scale # height: re-referenced to the floor below
        return o

    # a clip made from a video: its fitted points, mirrored and turned as the clip is (fit_to_points)
    FIT = FACE = None
    fit_stats = {}
    if opts.get("fitted") and opts.get("fit_points") and os.path.exists(opts["fit_points"]):
        FIT = np.load(opts["fit_points"])["points"].astype(np.float64)
        if MIR:
            FIT = FIT[:, MIRROR13] * np.array([-1.0, 1.0, 1.0])
        FIT = FIT @ np.array(R3).T
        _sm = (FIT[:, 1] + FIT[:, 4]) / 2
        _hm = (FIT[:, 7] + FIT[:, 10]) / 2
        fit_torso = float(np.median(np.linalg.norm(_sm - _hm, axis=1)))
        _rb = tgt.data.bones
        _r = {k_: TW @ _rb[k_].head_local for k_ in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")}
        rig_torso = ((_r["left_shoulder"] + _r["right_shoulder"]) / 2 - (_r["left_hip"] + _r["right_hip"]) / 2).length
        print(f"[retarget]   aiming the limbs along the video's fitted points ({len(FIT)} frames)", flush=True)
        _fz = np.load(opts["fit_points"])
        if "face" in _fz and FACE_LOCAL is not None:
            FACE = _fz["face"].astype(np.float64)
            if MIR:
                FACE = FACE[:, FACE_MIRROR] * np.array([-1.0, 1.0, 1.0])
            FACE = FACE @ np.array(R3).T
            print("[retarget]   and the head to the video's face", flush=True)

    pelvis = tgt.pose.bones.get("pelvis")
    for i in range(n):
        t = f0 + i * step
        scene.frame_set(int(math.floor(t)), subframe=float(t - math.floor(t)))
        bpy.context.view_layer.update()

        for tname in order:
            if tname not in tmap:
                continue
            spb = src.pose.bones[tmap[tname]]
            tpb = tgt.pose.bones[tname]
            src_pose_w = SW @ spb.matrix
            src_rest_w = SW @ src_rest[spb.name]
            tgt_rest_w = TW @ tgt_rest[tname]
            M_world = UNYAW @ src_pose_w @ src_rest_w.inverted() @ tgt_rest_w
            M = TW_inv @ M_world
            # normalise: the source's unit scale must not ride along into the target
            M = Matrix.LocRotScale(M.to_translation(), M.to_quaternion(), Vector((1, 1, 1)))
            # keep the target's own bone head; only the root carries translation
            M.translation = tpb.matrix.to_translation()
            tpb.matrix = M
            bpy.context.view_layer.update()

        # root translation, in armature space, via the matrix - not via pose_bone.location
        if pelvis is not None:
            M = pelvis.matrix.copy()
            M.translation = tgt_rest["pelvis"].to_translation() + TW_inv.to_3x3() @ root_offset(i)
            pelvis.matrix = M
            bpy.context.view_layer.update()

        if FIT is not None:
            j = (t - f0) / max(f1 - f0, 1e-9) * (len(FIT) - 1)
            j0 = min(int(math.floor(j)), len(FIT) - 1)
            j1 = min(j0 + 1, len(FIT) - 1)
            w_ = j - j0
            fit_to_points(tgt, [Vector(tuple(v)) for v in (1 - w_) * FIT[j0] + w_ * FIT[j1]], TW,
                          rig_torso, fit_torso, fit_stats,
                          face=None if FACE is None else [Vector(tuple(v)) for v in (1 - w_) * FACE[j0] + w_ * FACE[j1]],
                          face_local=FACE_LOCAL)

        for tname in order:
            if tname in tmap:
                tgt.pose.bones[tname].keyframe_insert("rotation_euler", frame=i + 1)
        if pelvis is not None:
            pelvis.keyframe_insert("location", frame=i + 1)

    # every clip keyframes every bone, so switching clips cannot inherit a stale pose
    for pb in tgt.pose.bones:
        if pb.name not in tmap:
            pb.rotation_euler = (0, 0, 0)
            pb.keyframe_insert("rotation_euler", frame=1)
            pb.keyframe_insert("rotation_euler", frame=n)

    # ---- feet: on the floor, and still while they are down (see ground_and_plant) -------------
    rel0 = [src_feet[0][k] - hips_path[0] for k in range(4)] if src_feet else []
    rel1 = [src_feet[-1][k] - hips_path[-1] for k in range(4)] if src_feet else []
    looping = bool(rel0) and max((x - y).length for x, y in zip(rel0, rel1)) < 0.01 * src_hip_h
    st = {}
    if opts.get("ground", True) is False:
        print("[retarget]   airborne clip: captured height kept, no floor planting", flush=True)
    elif os.environ.get("CF_NO_PLANT"):                  # measurement only: the fit's legs untouched
        print("[retarget]   CF_NO_PLANT: feet left as fitted", flush=True)
    elif pelvis is not None and len(src_feet) == n:
        cap_src = d.length / dur if travelling else 0.0
        st = ground_and_plant(tgt, pelvis, n, float(a.fps), travelling, src_feet,
                              hips_path[-1] - hips_path[0], looping, max(0.25, 0.12 * cap_src),
                              TW, SOLE, drift=(-math.sin(travel_rad), math.cos(travel_rad)))
    stride = st.get("speed") if travelling else None
    if travelling and stride and not (0.6 * travel / dur < stride < 1.4 * travel / dur):
        print(f"[retarget]   WARNING the feet say {stride:.2f} m/s against {travel/dur:.2f} "
              f"captured", flush=True)
    unit = "%" if (stride or 0) >= 1.0 else " m/s"
    fmt = (lambda x: f"{x*100:.1f}%") if (stride or 0) >= 1.0 else (lambda x: f"{x:.3f} m/s")
    if st:
        print(f"[retarget]   feet: pelvis moved {st['pelvis_adjust_cm'][0]:+.1f}..{st['pelvis_adjust_cm'][1]:+.1f} cm "
              f"to put the planted sole on the floor (now {st['planted_height_cm']} cm off it on average); "
              f"{st['locked']} leg-keys planted, slip "
              + (f"{fmt(st['slip_before'])} -> {fmt(st['slip'])}" if st.get('slip') is not None else "n/a"),
              flush=True)
    if FIT is not None:
        turned = {k_: round(float(np.mean(v_)), 2) for k_, v_ in fit_stats.items()}
        print("[retarget]   aimed along the fit, mean turn per bone (deg): "
              + ", ".join(f"{k_} {v_:.1f}" for k_, v_ in turned.items()), flush=True)
    report[clip_name] = {"frames": n, "fps": a.fps, "bones_mapped": len(pairs), "scale": round(scale, 5),
                         "curves": len(action_fcurves(act)), "source": os.path.basename(fbx),
                         "root_travel_m": round(travel, 3),
                         "capture_speed_mps": round(travel / dur, 3),
                         "stride_speed_mps": round(stride, 3) if stride else None,
                         "looping": st.get("looping"),
                         "foot_slip": round(st["slip"], 4) if st.get("slip") is not None else None,
                         "foot_slip_before_planting": round(st["slip_before"], 4) if st.get("slip_before") is not None else None,
                         "planted_height_cm": st.get("planted_height_cm"),
                         "planted_height_range_cm": st.get("planted_height_range_cm"),
                         "pelvis_adjust_cm": st.get("pelvis_adjust_cm"),
                         "sole_lowest_cm": st.get("sole_lowest_cm"),
                         # airborne spans, in seconds from the first key; a key at frame f is at
                         # (f - 1) / fps once the exporter slides the clip to start at zero
                         "flights_s": [[round(f0_ / a.fps, 3), round(f1_ / a.fps, 3)]
                                       for f0_, f1_ in st.get("flights", [])],
                         "apex_sole_m": st.get("apex_sole_m"),
                         "feet_planted_keys": st.get("locked"),
                         "heading_removed_deg": round(math.degrees(yaw), 2),
                         "heading_from": ("start" if head_mode == "start" else
                                          "travel" if travelling else "hip line"),
                         "travel_deg": round(math.degrees(travel_rad), 1),
                         "grounded": opts.get("ground", True) is not False,
                         "turn_deg": round(math.degrees(turn_total), 1),
                         "body_vs_travel_deg": round(math.degrees(math.atan2(math.sin(hip_yaw - yaw), math.cos(hip_yaw - yaw))), 1) if travelling else None,
                         "leg_lengths_per_sec": round(per_sec, 3)}
    print(f"[retarget]   root travel {travel:.3f} m ({per_sec:.2f} leg-lengths/s)"
          + (f"; ground speed {stride:.2f} m/s (capture {travel/dur:.2f})" if stride else ""), flush=True)
    print(f"[retarget]   -> action '{clip_name}': {report[clip_name]['curves']} curves",
          flush=True)

    bpy.ops.object.mode_set(mode="OBJECT")
    # Remove the imported objects *and* the actions that came in with them. Deleting only the
    # objects leaves each source clip's action orphaned in the file, and the glTF exporter ships
    # every action it finds - five junk "Armature|mixamo.com|Layer0" animations rode along in
    # the first build, bloating the download and polluting the clip list the runtime sees.
    src_actions = set()
    for o in added:
        if o.animation_data and o.animation_data.action:
            src_actions.add(o.animation_data.action)
        if o.type == "MESH" and o.data.shape_keys and o.data.shape_keys.animation_data:
            if o.data.shape_keys.animation_data.action:
                src_actions.add(o.data.shape_keys.animation_data.action)
        bpy.data.objects.remove(o, do_unlink=True)
    for sa in src_actions:
        if sa.name != clip_name:
            sa.use_fake_user = False
            try:
                bpy.data.actions.remove(sa)
            except Exception:
                pass

# ---- export ---------------------------------------------------------------------------------
if a.blend_out:
    bpy.ops.wm.save_as_mainfile(filepath=a.blend_out)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", export_animations=True,
                          export_animation_mode="ACTIONS", export_skins=True)
if a.json:
    json.dump(report, open(a.json, "w"), indent=2)
print(f"[retarget] {len(report)} clips -> {a.out}", flush=True)
