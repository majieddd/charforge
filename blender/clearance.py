"""Hold every clip inside the character's own joint limits (joint_limits.py), so thick limbs stop sinking into each
other - and do it without putting kinks into the motion.

    blender -b -noaudio --python clearance.py -- --blend final.blend --limits limits.json \
        [--out final.blend] [--json clearance.json] [--soft 10] [--ease 2.5] [--clips a,b]

Per key, parent before child:
  shoulder  the upper arm's direction relative to the chest (so a lean does not count) is read as a
            swing (forward/back) and a depth below horizontal; past the limit, the arm goes to the
            nearest allowed direction - out to the side if that is the shorter way (Bo's belly blocks an
            arm swung forward and down; raising it cost up to 44 degrees, moving it out far less). The
            nearest direction is searched on a 2.5 degree grid of swings. Faded out between 60 and 75
            degrees of swing - the limits were measured with the arm beside the body, not reaching forward
  elbow     the angle between upper arm and forearm; past the limit it is opened about its own bend axis
  knee      the same, only while that foot is off the ground: its weight rises from 0 to 1 as the ankle
            climbs from 8 to 16 cm above the clip's lowest, so the planted feet from retarget.py stay where
            they were put; and the opening is held off the floor (see below)
A soft limit: angles more than --soft below the limit are untouched and the last --soft degrees are approached
smoothly (tanh), so the correction has no corner in the pose. Fades use the C2 smootherstep, not smoothstep,
so their second derivative does not jump at the ends of the ramp.

Then each correction is eased in time. Left alone, a correction is a hard
per-key clamp: where the nearest allowed direction changes, or where a limit engages, the joint's jerk spikes
(mara's idle left shoulder: 7x the capture's jerk). Each shoulder turn and elbow opening is smoothed over the keys
with a zero-phase Gaussian of --ease keys (default 2.5 keys = 42 ms at 60 fps; 0 turns it off). A knee's opening is
smoothed over --knee-ease keys (default 1): smoothing it longer spreads it into the planted keys either side and
slides the foot (cadet's sprint slip 28 -> 45%). The smoothed correction is multiplied by the knee's lift weight
again, so it is exactly zero while the foot is planted.

A knee that would put the shoe through the floor is capped at the largest opening that keeps it on. That bound is
found per key on the captured pose (a bisection), and it can release abruptly (a foot lifting off), so the feasible
opening is eroded over 3 sigma of keys before it is smoothed: the eased opening stays under the bound everywhere, and
ramps in after the release instead of stepping to it (cadet's crouch walk: an 11 degree step at lift-off before).
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--limits", required=True)
ap.add_argument("--out", default=None, help="default: overwrite --blend")
ap.add_argument("--json", default=None)
ap.add_argument("--soft", type=float, default=10.0)
ap.add_argument("--ease", type=float, default=2.5,
                help="Gaussian sigma, in keys, on every correction over time (0 = off)")
ap.add_argument("--knee-ease", type=float, default=1.0,
                help="knees only: sigma in keys. A knee's correction is held to 1 key so that it does not spread into "
                     "the stance keys either side (foot slip: sprint 28.3 -> 35-45% at 2.5 keys)")
ap.add_argument("--gate", type=float, default=0.0,
                help="degrees past the limit before an elbow or knee is corrected (0: the correction is continuous)")
ap.add_argument("--floor-iters", type=int, default=6, help="bisection steps for the knee's floor bound")
ap.add_argument("--clips", default="")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
pb = rig.pose.bones
L = json.load(open(a.limits))["joints"]
for o in sc.objects:                                   # posing only
    for m in getattr(o, "modifiers", []):
        if m.type == "ARMATURE":
            m.show_viewport = False


def ramp(t):
    """C2 smootherstep, 0 below 0 and 1 above 1"""
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def soft(x, lim):
    s = lim - a.soft
    return x if x <= s else s + a.soft * math.tanh((x - s) / a.soft)


def use_action(act):
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]


def turn(b, R_arm):
    """Turn pose bone b by R_arm (armature space, about its head); rotation keys only."""
    bpy.context.view_layer.update()
    Cm = b.matrix.to_3x3() @ b.matrix_basis.to_3x3().inverted()
    R = Cm.inverted() @ R_arm @ Cm @ b.matrix_basis.to_3x3()
    b.rotation_euler = R.to_euler(b.rotation_mode if b.rotation_mode != "QUATERNION" else "XYZ", b.rotation_euler)


def direction(b):
    return (b.tail - b.head).normalized()


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skin_points  # noqa: E402


def boot_cloud(side):
    """The shoe below the ankle as points posed by their own skin weights (as retarget.py plants it)."""
    ank_b, ball_b = f"{side}_ankle", f"{side}_foot"
    if ank_b not in rig.data.bones:
        return None
    meshes = [o for o in sc.objects if o.type == "MESH" and any(m.type == "ARMATURE" and m.object == rig for m in o.modifiers)]
    cut = max(0.03, float((rig.matrix_world @ rig.data.bones[ank_b].head_local).z))
    c = skin_points.build(meshes, rig, lambda q, w: q.z < cut and w.get(ank_b, 0) + w.get(ball_b, 0) > 0.5)
    return c if c and c["n"] >= 20 else None


def boot_low(cloud):
    bpy.context.view_layer.update()
    return float(skin_points.pose(cloud, rig)[:, 2].min())


def gauss(x, sigma):
    """zero-phase Gaussian over the key axis (first axis), held at the ends. numpy: Blender's Python has no scipy"""
    x = np.asarray(x, float)
    if sigma <= 0:
        return x
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.zeros_like(x)
    for i, w in enumerate(k):
        out += w * pad[i:i + x.shape[0]]
    return out


def erode(x, r):
    """running minimum over 2r+1 keys, held at the ends"""
    if r <= 0:
        return np.asarray(x, float)
    pad = np.pad(np.asarray(x, float), (r, r), mode="edge")
    return np.array([pad[i:i + 2 * r + 1].min() for i in range(len(x))])


def floor_bound(cloud, b, axis, theta, z0):
    """The largest opening of theta (same sign) that keeps the shoe on the floor, or None if the full one does.
    The pose is restored exactly afterwards."""
    saved = b.rotation_euler.copy()
    turn(b, Matrix.Rotation(theta, 3, axis))
    ok_full = boot_low(cloud) >= min(0.0, z0) - 0.002
    b.rotation_euler = saved
    bpy.context.view_layer.update()
    if ok_full:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(a.floor_iters):
        mid = (lo + hi) / 2
        b.rotation_euler = saved
        turn(b, Matrix.Rotation(theta * mid, 3, axis))
        ok = boot_low(cloud) >= min(0.0, z0) - 0.002
        lo, hi = (mid, hi) if ok else (lo, mid)
    b.rotation_euler = saved
    bpy.context.view_layer.update()
    return theta * lo


chest = pb.get("spine3") or pb.get("spine2")
chest_rest = chest.bone.matrix_local.to_3x3()
sides = [s for s in ("left", "right") if f"{s}_shoulder" in L]
lateral_of = {s: (1.0 if L[f"{s}_shoulder"]["rest_dir"][0] > 0 else -1.0) for s in sides}
clips = [c for c in (a.clips.split(",") if a.clips else [x.name for x in bpy.data.actions]) if bpy.data.actions.get(c)]
report = {"soft_deg": a.soft, "ease_keys": a.ease, "clips": {}}
BOOT = {s: boot_cloud(s) for s in sides}


def arm_dir(side, sw, b):
    return Vector((lateral_of[side] * math.cos(math.radians(b)) * math.cos(math.radians(sw)),
                   -math.cos(math.radians(b)) * math.sin(math.radians(sw)),
                   -math.sin(math.radians(b))))


for clip in clips:
    act = bpy.data.actions[clip]
    use_action(act)
    f0, f1 = (int(x) for x in act.frame_range)
    frames = list(range(f0, f1 + 1))
    K = len(frames)
    # each ankle's lowest point in the clip, for "is this foot off the ground"
    low = {s: 1e9 for s in sides}
    for fr in frames:
        sc.frame_set(fr)
        for s in sides:
            if f"{s}_ankle" in pb:
                low[s] = min(low[s], pb[f"{s}_ankle"].head.z)
    stats = {}

    # pass 1: every correction from the clip as captured, nothing applied yet
    shoulder_rv = {s_: np.zeros((K, 3)) for s_ in sides}
    for s_ in sides:
        cone = L[f"{s_}_shoulder"]
        for k, fr in enumerate(frames):
            sc.frame_set(fr)
            bpy.context.view_layer.update()
            Rc = chest.matrix.to_3x3() @ chest_rest.inverted()               # the chest's turn from rest
            d = (Rc.inverted() @ direction(pb[f"{s_}_shoulder"])).normalized()
            below = math.degrees(math.asin(max(-1.0, min(1.0, -d.z))))
            swing = math.degrees(math.atan2(-d.y, lateral_of[s_] * d.x))
            fade = ramp((75.0 - abs(swing)) / 15.0)                          # full within 60 deg of swing, none past 75
            if fade <= 0:
                continue
            lim = float(np.interp(swing, cone["swing_deg"], cone["below_limit_deg"]))
            if below <= lim - a.soft:
                continue
            best = None
            for sw in np.arange(-75.0, 75.01, 2.5):
                l_ = float(np.interp(sw, cone["swing_deg"], cone["below_limit_deg"]))
                t = arm_dir(s_, sw, min(below, soft(below, l_)))
                cost = d.angle(t)
                if best is None or cost < best[0]:
                    best = (cost, t)
            if best[0] > 1e-3:
                q = d.rotation_difference(best[1])
                shoulder_rv[s_][k] = np.array(q.axis) * q.angle * fade
        shoulder_rv[s_] = gauss(shoulder_rv[s_], a.ease)

    # elbow and knee: the amount past the limit, from the captured pose; the knee's floor bound where it binds
    joint_theta, joint_cap, joint_w = {}, {}, {}
    for s_ in sides:
        for jn in (f"{s_}_elbow", f"{s_}_knee"):
            if jn not in L or jn not in pb:
                continue
            th = np.zeros(K)
            cap = np.full(K, np.nan)
            wv = np.zeros(K)
            cloud = BOOT.get(s_) if jn.endswith("knee") else None
            for k, fr in enumerate(frames):
                sc.frame_set(fr)
                bpy.context.view_layer.update()
                w = 1.0
                if jn.endswith("knee") and f"{s_}_ankle" in pb:
                    # foot down: planted by retarget, so no correction (see the docstring)
                    w = ramp((pb[f"{s_}_ankle"].head.z - low[s_] - 0.08) / 0.08)
                    if w <= 0:
                        continue
                wv[k] = w
                b, par = pb[jn], pb[jn].parent
                u, f = direction(par), direction(b)
                ang = math.degrees(u.angle(f))
                na = soft(ang, L[jn]["limit_deg"])
                if na < ang - a.gate:
                    th[k] = math.radians((na - ang) * w)
                    if cloud and abs(th[k]) > 1e-4:
                        bnd = floor_bound(cloud, b, u.cross(f).normalized(), th[k], boot_low(cloud))
                        cap[k] = np.nan if bnd is None else bnd
            joint_theta[jn] = th
            joint_cap[jn] = cap
            joint_w[jn] = wv

    for jn in list(joint_theta):
        th, cap, wv = joint_theta[jn], joint_cap[jn], joint_w[jn]
        sig = a.ease if (not jn.endswith("knee") or a.knee_ease is None) else a.knee_ease
        # the smoothed correction is multiplied by the same weight again: a Gaussian spreads a lift into the stance
        # keys either side, where the foot is planted and must not move (foot slip 13.1 -> 15.0% without it, cadet)
        if jn.endswith("knee") and not np.all(np.isnan(cap)):
            sg = np.sign(th)
            U = np.where(np.isnan(cap), np.abs(th), np.minimum(np.abs(th), np.abs(np.nan_to_num(cap))))
            r = int(math.ceil(3 * sig)) if sig > 0 else 0
            G = gauss(erode(U, r), sig) if r > 0 else U.copy()
            joint_theta[jn] = wv * sg * np.minimum(G, U)
            joint_cap[jn] = np.where(np.isnan(cap), np.inf, np.abs(np.nan_to_num(cap)))
        else:
            joint_theta[jn] = wv * gauss(th, sig)
            joint_cap[jn] = np.full(K, np.inf)

    # pass 2: apply, parent before child
    for k, fr in enumerate(frames):
        sc.frame_set(fr)
        changed = set()
        bpy.context.view_layer.update()
        Rc = chest.matrix.to_3x3() @ chest_rest.inverted()
        for s_ in sides:
            rv = Vector(shoulder_rv[s_][k].tolist())
            if rv.length > math.radians(0.05):
                sh = pb[f"{s_}_shoulder"]
                turn(sh, Rc @ Matrix.Rotation(rv.length, 3, rv.normalized()) @ Rc.inverted())
                changed.add(sh.name)
                st = stats.setdefault(sh.name, [0, 0.0]); st[0] += 1; st[1] = max(st[1], math.degrees(rv.length))
            # ---- elbow and knee: folded into the limb above -----------------------------------------
            for jn in (f"{s_}_elbow", f"{s_}_knee"):
                if jn not in joint_theta or jn not in pb:
                    continue
                theta = float(joint_theta[jn][k])
                if abs(theta) <= 1e-4:
                    continue
                b, par = pb[jn], pb[jn].parent
                u, f = direction(par), direction(b)
                axis = u.cross(f).normalized()
                cap_k = float(joint_cap[jn][k])
                turn(b, Matrix.Rotation(theta, 3, axis))
                if np.isfinite(cap_k) and abs(theta) >= abs(cap_k) - 1e-6:
                    fc = stats.setdefault(f"{jn} (floor)", [0, 0.0]); fc[0] += 1
                changed.add(jn)
                st = stats.setdefault(jn, [0, 0.0]); st[0] += 1; st[1] = max(st[1], math.degrees(abs(theta)))
        for bn in changed:
            pb[bn].keyframe_insert("rotation_euler", frame=fr)
    n = f1 - f0 + 1
    report["clips"][clip] = {bn: {"frames_pct": round(100 * c / n, 1), "max_deg": round(m, 1)} for bn, (c, m) in stats.items()}
    if stats:
        print(f"[clearance] {clip:16s} " + ", ".join(f"{bn} {100 * c / n:.0f}% of keys (up to {m:.0f} deg)"
                                                     for bn, (c, m) in sorted(stats.items())), flush=True)
rig.animation_data.action = None
for o in sc.objects:
    for m in getattr(o, "modifiers", []):
        if m.type == "ARMATURE":
            m.show_viewport = True
bpy.ops.wm.save_as_mainfile(filepath=a.out or a.blend)
if a.json:
    json.dump(report, open(a.json, "w"), indent=1)
print(f"[clearance] {len(clips)} clips -> {a.out or a.blend}", flush=True)
