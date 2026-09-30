"""Hold every clip inside the character's own joint limits (joint_limits.py), so thick limbs stop
sinking into each other.

    blender -b -noaudio --python clearance.py -- --blend final.blend --limits limits.json \
        [--out final.blend] [--json clearance.json] [--soft 10] [--clips a,b]

Per key, parent before child:
  shoulder  the upper arm's direction relative to the chest (so a lean does not count) is read as a
            swing (forward/back) and a depth below horizontal; past the limit, the arm goes to the
            nearest allowed direction - out to the side if that is the shorter way (Bo's belly blocks an
            arm swung forward and down; raising it cost up to 44 degrees, moving it out far less). The
            per-key turns are smoothed over +-2 keys before they are applied, so the nearest direction
            cannot hop between two near-equal choices. Faded out between 60 and 75 degrees of swing - the
            limits were measured with the arm beside the body, not reaching forward
  elbow     the angle between upper arm and forearm; past the limit it is opened about its own bend axis
  knee      the same, only while that foot is off the ground: faded in as its ankle rises from 8 to 16 cm
            above the clip's lowest, so the planted feet from retarget.py stay where they were put
A soft limit: angles more than --soft below the limit are untouched and the last --soft degrees are
approached smoothly (tanh), so the correction has no corner in time. Changed keys are rewritten as
Euler angles continuous with the key before.
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


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


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


chest = pb.get("spine3") or pb.get("spine2")
chest_rest = chest.bone.matrix_local.to_3x3()
sides = [s for s in ("left", "right") if f"{s}_shoulder" in L]
lateral_of = {s: (1.0 if L[f"{s}_shoulder"]["rest_dir"][0] > 0 else -1.0) for s in sides}
clips = [c for c in (a.clips.split(",") if a.clips else [x.name for x in bpy.data.actions]) if bpy.data.actions.get(c)]
report = {"soft_deg": a.soft, "clips": {}}

for clip in clips:
    act = bpy.data.actions[clip]
    use_action(act)
    f0, f1 = (int(x) for x in act.frame_range)
    # each ankle's lowest point in the clip, for "is this foot off the ground"
    low = {s: 1e9 for s in sides}
    for fr in range(f0, f1 + 1):
        sc.frame_set(fr)
        for s in sides:
            if f"{s}_ankle" in pb:
                low[s] = min(low[s], pb[f"{s}_ankle"].head.z)
    stats = {}

    def arm_dir(sw, b):
        return Vector((lateral_of[s_] * math.cos(math.radians(b)) * math.cos(math.radians(sw)),
                       -math.cos(math.radians(b)) * math.sin(math.radians(sw)),
                       -math.sin(math.radians(b))))

    # pass 1: each key's shoulder turn (a rotation vector in the chest's frame)
    frames = list(range(f0, f1 + 1))
    shoulder_rv = {s_: np.zeros((len(frames), 3)) for s_ in sides}
    for k, fr in enumerate(frames):
        sc.frame_set(fr)
        bpy.context.view_layer.update()
        Rc = chest.matrix.to_3x3() @ chest_rest.inverted()               # the chest's turn from rest
        for s_ in sides:
            cone = L[f"{s_}_shoulder"]
            d = (Rc.inverted() @ direction(pb[f"{s_}_shoulder"])).normalized()
            below = math.degrees(math.asin(max(-1.0, min(1.0, -d.z))))
            swing = math.degrees(math.atan2(-d.y, lateral_of[s_] * d.x))
            fade = smooth((75.0 - abs(swing)) / 15.0)                    # full within 60 deg of swing, none past 75
            if fade <= 0:
                continue
            lim = float(np.interp(swing, cone["swing_deg"], cone["below_limit_deg"]))
            if below <= lim - a.soft:
                continue
            best = None
            for sw in np.arange(-75.0, 75.01, 2.5):
                l_ = float(np.interp(sw, cone["swing_deg"], cone["below_limit_deg"]))
                t = arm_dir(sw, min(below, soft(below, l_)))
                cost = d.angle(t)
                if best is None or cost < best[0]:
                    best = (cost, t)
            if best[0] > 1e-3:
                q = d.rotation_difference(best[1])
                shoulder_rv[s_][k] = np.array(q.axis) * q.angle * fade
    # smooth the turns over +-2 keys
    kern = np.array([1, 4, 6, 4, 1], float); kern /= kern.sum()
    for s_ in sides:
        rv = shoulder_rv[s_]
        pad = np.pad(rv, ((2, 2), (0, 0)), mode="edge")
        shoulder_rv[s_] = np.stack([np.convolve(pad[:, i], kern, mode="valid") for i in range(3)], 1)

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
                if jn not in L or jn not in pb:
                    continue
                w = 1.0
                if jn.endswith("knee") and f"{s_}_ankle" in pb:
                    # foot down: planted by retarget, so no correction; faded in between 8 and 16 cm of lift,
                    # as a switch at one height started the correction with a jump (Wren's sprint)
                    w = smooth((pb[f"{s_}_ankle"].head.z - low[s_] - 0.08) / 0.08)
                    if w <= 0:
                        continue
                bpy.context.view_layer.update()
                b, par = pb[jn], pb[jn].parent
                u, f = direction(par), direction(b)
                ang = math.degrees(u.angle(f))
                na = soft(ang, L[jn]["limit_deg"])
                if na < ang - 0.05:
                    axis = u.cross(f).normalized()
                    turn(b, Matrix.Rotation(math.radians((na - ang) * w), 3, axis))
                    changed.add(jn)
                    st = stats.setdefault(jn, [0, 0.0]); st[0] += 1; st[1] = max(st[1], (ang - na) * w)
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
