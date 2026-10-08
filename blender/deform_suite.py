"""Extreme-pose battery: how each character's skin holds at the poses a clip may never reach (experiment E153).

    blender -b -noaudio --python deform_suite.py -- --blend final.blend --out stress.blend

An auto-rigger's weights look right in a neutral stance and in the clips it was tried on, and break where the
clips do not go: arms overhead, forearms across the chest, a deep squat, a bend with a twist, a high kick, the head
turned hard. Studio guides and reference-asset-compiler's "five-pose deformation suite" make the same point: skinning
is checked at the extremes, not at rest. This writes eight actions - stress_<name>, 13 frames each: rest at frame 1,
the pose at frame 7 held to 13 - into a copy of the rig, for aberration_audit.py to measure like any clip
(tools/deform_suite.py runs both and tabulates).

Each pose is a list of turns about a bone's head in armature space (X across the body, Y front-to-back with the front at
-Y, Z up). The sign of a turn is whichever moves its child joint along the goal direction, so the battery needs
no knowledge of a rig's bone axes. A turn that cannot move its child toward its goal (a joint already past it) is
skipped, and the result is logged."""
import argparse
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
pb = rig.pose.bones
SIDES = [s for s in ("left", "right") if f"{s}_elbow" in pb]
SGN = {s: (1.0 if pb[f"{s}_shoulder"].head.x > pb["pelvis"].head.x else -1.0) for s in SIDES}   # +1: on the +X side
X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
FRONT, UP = -Y, Z


def use_action(act):
    rig.animation_data_create()
    rig.animation_data.action = act
    if act is not None and hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]


def rest():
    for b in pb:
        b.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def turn(bone, axis, deg, child, goal):
    """Turn `bone` about its head, armature-space `axis`, by deg in whichever direction moves `child`'s head along
    `goal`; returns how far that head moved along the goal (cm), or None if neither direction helps."""
    b, c = pb[bone], pb[child]
    bpy.context.view_layer.update()
    h0 = c.head.copy()
    keep = b.matrix_basis.copy()
    if callable(goal):                    # a target point, known only once the parents are posed: swing the child to it
        P = goal()
        d = P - c.head
        if axis is None:
            axis = (c.head - b.head).cross(d)
            if axis.length < 1e-6:
                return None
        Cm = b.matrix.to_3x3() @ b.matrix_basis.to_3x3().inverted()
        best = None
        for sgn in (1.0, -1.0):
            b.matrix_basis = keep
            bpy.context.view_layer.update()
            R = Matrix.Rotation(math.radians(sgn * deg), 3, axis.normalized())
            b.matrix_basis = (Cm.inverted() @ R @ Cm @ b.matrix_basis.to_3x3()).to_4x4()
            bpy.context.view_layer.update()
            gain = (P - h0).length - (P - c.head).length            # how much closer to the point
            if best is None or gain > best[0]:
                best = (gain, b.matrix_basis.copy())
        if best[0] <= 1e-4:
            b.matrix_basis = keep
            bpy.context.view_layer.update()
            return None
        b.matrix_basis = best[1]
        bpy.context.view_layer.update()
        return best[0] * 100
    if axis is None:                      # a hinge: about the axis that swings the child toward the goal
        axis = (c.head - b.head).cross(goal)
        if axis.length < 1e-6:
            return None
    if goal is None:                      # a twist about the bone's own line: no direction to choose, no child to move
        Cm = b.matrix.to_3x3() @ b.matrix_basis.to_3x3().inverted()
        R = Matrix.Rotation(math.radians(deg), 3, axis.normalized())
        b.matrix_basis = (Cm.inverted() @ R @ Cm @ b.matrix_basis.to_3x3()).to_4x4()
        bpy.context.view_layer.update()
        return 0.0
    best = None
    for sgn in (1.0, -1.0):
        b.matrix_basis = keep
        bpy.context.view_layer.update()
        Cm = b.matrix.to_3x3() @ b.matrix_basis.to_3x3().inverted()
        R = Matrix.Rotation(math.radians(sgn * deg), 3, axis.normalized())
        b.matrix_basis = (Cm.inverted() @ R @ Cm @ b.matrix_basis.to_3x3()).to_4x4()
        bpy.context.view_layer.update()
        gain = (c.head - h0).dot(goal.normalized())
        if best is None or gain > best[0]:
            best = (gain, b.matrix_basis.copy())
    if best[0] <= 1e-4:
        b.matrix_basis = keep
        bpy.context.view_layer.update()
        return None
    b.matrix_basis = best[1]
    bpy.context.view_layer.update()
    return best[0] * 100


def poses():
    """name -> list of (bone, axis, degrees, child, goal), applied in this order (parents first)."""
    P = {}
    P["arms_overhead"] = [(f"{s}_shoulder", Y, 110, f"{s}_elbow", UP) for s in SIDES]
    P["arms_forward"] = [(f"{s}_shoulder", X, 100, f"{s}_elbow", FRONT) for s in SIDES]
    across = []
    for s in SIDES:
        across += [(f"{s}_shoulder", X, 50, f"{s}_elbow", FRONT),
                   (f"{s}_shoulder", Z, 55, f"{s}_elbow", X * -SGN[s]),
                   (f"{s}_elbow", None, 110, f"{s}_wrist", (lambda s=s: Vector((0.0, pb["spine3"].head.y - 0.12, pb["spine3"].head.z - 0.05))))]
    P["arms_across"] = across
    P["arms_behind"] = [(f"{s}_shoulder", X, 60, f"{s}_elbow", -FRONT) for s in SIDES] + \
                       [(f"{s}_elbow", X, 70, f"{s}_wrist", -FRONT) for s in SIDES]
    P["squat"] = [("spine1", X, 20, "spine3", FRONT)] + \
                 [(f"{s}_hip", X, 105, f"{s}_knee", FRONT) for s in SIDES] + \
                 [(f"{s}_knee", X, 120, f"{s}_ankle", -FRONT) for s in SIDES]
    P["bend_twist"] = [(b, X, 22, "head", FRONT) for b in ("spine1", "spine2", "spine3")] + \
                      [(b, Z, 22, "head", X * SGN[SIDES[0]]) for b in ("spine1", "spine2", "spine3")]
    s0, s1 = SIDES[0], SIDES[-1]
    P["kick"] = [(f"{s0}_hip", X, 85, f"{s0}_knee", FRONT), (f"{s1}_hip", Y, 50, f"{s1}_knee", X * SGN[s1])]
    # the feet: a boot's ankle is where soft weights bleed (Cadet's chunky boots popped in a run after E154's weights)
    P["tiptoe"] = [(f"{s}_ankle", X, 50, f"{s}_foot", -UP) for s in SIDES] + [(f"{s}_knee", X, 25, f"{s}_ankle", FRONT) for s in SIDES]
    P["heel_strike"] = [(f"{s}_hip", X, 35, f"{s}_knee", FRONT) for s in SIDES[:1]] + [(f"{s}_ankle", X, 28, f"{s}_foot", UP) for s in SIDES[:1]]
    P["head_turn"] = [("neck", Z, 35, "head", None), ("head", Z, 35, "head_top", None), ("head", Y, 20, "head_top", X * -1)]
    return {k: [t for t in v if t[0] in pb and t[3] in pb] for k, v in P.items()}


made = {}
for name, steps in poses().items():
    use_action(None)                      # the last pose's keys would be re-evaluated over this one's turns
    rest()
    got = [turn(*t) for t in steps]
    print(f"[stress]   {name} gains (cm): " + ", ".join(f"{t[0]}:{'-' if g is None else round(g, 1)}" for t, g in zip(steps, got)), flush=True)
    skipped = [t[0] for t, g in zip(steps, got) if g is None]
    act = bpy.data.actions.new(f"stress_{name}")
    act.use_fake_user = True              # unused actions are not saved with the file
    use_action(act)
    snap = {b.name: b.matrix_basis.copy() for b in pb}
    for frame, state in ((1, None), (7, snap), (13, snap)):
        for b in pb:
            b.matrix_basis = Matrix.Identity(4) if state is None else state[b.name]
            for prop in (("rotation_quaternion",) if b.rotation_mode == "QUATERNION" else ("rotation_euler",)):
                b.keyframe_insert(prop, frame=frame)
    sc.frame_start, sc.frame_end = 1, 13
    use_action(None)
    made[name] = {"turns": len(steps), "skipped": skipped}
    print(f"[stress] {name}: {len(steps) - len(skipped)} of {len(steps)} turns" + (f" (skipped {skipped})" if skipped else ""), flush=True)
rest()
use_action(None)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
print(f"[stress] {len(made)} poses -> {a.out}", flush=True)
