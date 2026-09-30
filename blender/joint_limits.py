"""How far each joint of this character can turn before one limb sinks into another, from its own mesh.

    blender -b -noaudio --python joint_limits.py -- --blend final.blend --out limits.json [--tol 1.0]

A captured motion was performed by someone of a given build. A thicker character doing it folds a
forearm into a puffy sleeve (Aoi's spell cast, 15 cm), a heel into a thick thigh, arms hanging into a wide
torso (Bo's idle, 4-7 cm; measured by aberration_audit.py). Game rigs carry joint limits for this; here
they are measured per character instead of guessed:

  elbow, knee   the hinge axis is read from the character's own clips (the axis of each frame's bend,
                in the parent bone's frame, averaged over bent frames); from the rest pose the joint
                is bent about it step by step and the forearm/hand into the upper arm (shin/foot into
                the thigh) measured with mesh_contact.py - the limit is the last angle under --tol
  shoulder      the arm, elbow straight, is lowered from the T-pose at five swings (60 and 30 degrees
                forward, straight to the side, 30 and 60 back): the limit at each swing is how far
                below horizontal it can go before arm, forearm or hand sink into the torso or thighs
                by more than --tol (not the head, whose region includes hair that should swing aside)

clearance.py holds the clips inside these limits. The depth curves are kept for the paper. The tolerance is the
audit's own threshold for deep penetration (2 cm): a limit forbids what the audit would count, not contact.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_contact import Contact  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tol", type=float, default=2.0,
                help="penetration (cm) a limit allows: the audit's 'deep' - flesh squashes where a rigid mesh can only overlap, "
                     "and at 1 cm Mara's boxing guard was opened by 25 degrees")
ap.add_argument("--step", type=float, default=5.0, help="sweep step, degrees")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
meshes = [o for o in sc.objects if o.type == "MESH" and o.visible_get()
          and any(m.type == "ARMATURE" for m in o.modifiers)]
for o in meshes:                                   # measured as the viewers show it
    for m in o.modifiers:
        if m.type == "ARMATURE":
            m.use_deform_preserve_volume = True
C = Contact(rig, meshes)
TOL = a.tol / 100
pb = rig.pose.bones
SIDES = [s for s in ("left", "right") if f"{s}_elbow" in pb]
print(f"[limits] {os.path.basename(os.path.dirname(os.path.abspath(a.blend)))}: sides {SIDES}, tolerance {a.tol} cm", flush=True)


def set_modifiers(on):
    for o in meshes:
        for m in o.modifiers:
            if m.type == "ARMATURE":
                m.show_viewport = on


def use_action(act):
    rig.animation_data.action = act
    if act is not None and hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]


# ---- hinge axes from the clips ------------------------------------------------------------------
def hinge_axes():
    """For each elbow and knee: the unit axis (in the parent bone's pose frame) the joint bends about,
    averaged over every frame bent more than 20 degrees, weighted by how far it bends; and the most
    any clip bends it."""
    set_modifiers(False)                             # posing only: no mesh evaluation per frame
    acc = {f"{s}_{j}": [Vector(), 0.0, 0.0] for s in SIDES for j in ("elbow", "knee")}
    for act in [x for x in bpy.data.actions if x.users or x.use_fake_user]:
        use_action(act)
        f0, f1 = (int(x) for x in act.frame_range)
        for fr in range(f0, f1 + 1, 2):
            sc.frame_set(fr)
            for jn, s_ in acc.items():
                b, par = pb[jn], pb[jn].parent
                u = (par.tail - par.head).normalized()          # armature space
                f = (b.tail - b.head).normalized()
                ang = math.degrees(u.angle(f))
                s_[2] = max(s_[2], ang)
                if ang > 20:
                    axis = u.cross(f).normalized()
                    s_[0] += (par.matrix.to_3x3().inverted() @ axis) * ang   # in the parent's frame
                    s_[1] += ang
    use_action(None)
    set_modifiers(True)
    return {jn: ((s_[0] / s_[1]).normalized() if s_[1] else None, s_[2]) for jn, s_ in acc.items()}


def rest():
    for b in pb:
        b.matrix_basis = Matrix.Identity(4)


def turn(bone, R_arm):
    """Turn a pose bone by R_arm (armature space, about its head)."""
    b = pb[bone]
    bpy.context.view_layer.update()
    Cm = b.matrix.to_3x3() @ b.matrix_basis.to_3x3().inverted()
    b.matrix_basis = (Cm.inverted() @ R_arm @ Cm @ b.matrix_basis.to_3x3()).to_4x4()


def depth_now(pairs):
    bpy.context.view_layer.update()
    V, _, _ = C.posed(bpy.context.evaluated_depsgraph_get())
    return C.depth_between(V, pairs)


def sweep(label, angles, pose_at, pairs):
    """Depth (cm) at each angle; the limit is the last angle before the depth first exceeds --tol."""
    curve, limit = [], None
    for ang in angles:
        rest()
        pose_at(ang)
        d = depth_now(pairs) * 100
        curve.append([round(float(ang), 1), round(d, 2)])
        if d > a.tol and limit is None:
            limit = curve[-2][0] if len(curve) > 1 else float(ang)
    if limit is None:
        limit = float(angles[-1])
    print(f"[limits] {label:28s} limit {limit:5.1f} deg  depth " +
          " ".join(f"{x:.0f}:{d:.1f}" for x, d in curve), flush=True)
    return limit, curve


out = {"blend": os.path.basename(a.blend), "tol_cm": a.tol, "step_deg": a.step, "joints": {}}
axes = hinge_axes()
steps = np.arange(60, 175 + 1e-6, a.step)
for s in SIDES:
    S = "L" if s == "left" else "R"
    # the forearm into the upper arm (a hand may touch its own shoulder); the shin or heel into the thigh
    for joint, parent_region, child_regions in (("elbow", f"upperarm.{S}", (f"forearm.{S}",)),
                                               ("knee", f"thigh.{S}", (f"shin.{S}", f"foot.{S}"))):
        jn = f"{s}_{joint}"
        axis_p, clip_max = axes[jn]
        if axis_p is None:
            continue
        pairs = {(c, parent_region) for c in child_regions}

        def bend(ang, jn=jn, axis_p=axis_p):
            bpy.context.view_layer.update()
            axis = (pb[jn].parent.matrix.to_3x3() @ axis_p).normalized()
            turn(jn, Matrix.Rotation(math.radians(ang), 3, axis))
        limit, curve = sweep(f"{jn} bend", steps, bend, pairs)
        out["joints"][jn] = {"kind": "hinge", "axis_parent": [round(x, 5) for x in axis_p], "limit_deg": limit,
                             "clips_max_deg": round(clip_max, 1), "curve_cm": curve}

    # the arm lowered from the T-pose at five swings, elbow straight
    jn = f"{s}_shoulder"
    # not the head: its region includes the hair, which should swing aside (spring chains), not stop the
    # arm - Aoi's twin tails "limited" her right arm's back swing to 30 degrees
    body = ("torso", f"thigh.{S}")
    pairs = {(x, y) for x in (f"upperarm.{S}", f"forearm.{S}", f"hand.{S}") for y in body}
    swings, limits, curves = (-60, -30, 0, 30, 60), [], []
    rest(); bpy.context.view_layer.update()
    d0 = (pb[jn].tail - pb[jn].head).normalized()          # the T-pose arm, armature space
    lateral = 1.0 if d0.x > 0 else -1.0
    for sw in swings:
        def lower(below, sw=sw):
            # target direction: `below` degrees under horizontal, swung `sw` degrees forward (the character faces -Y)
            t = Vector((lateral * math.cos(math.radians(below)) * math.cos(math.radians(sw)),
                        -math.cos(math.radians(below)) * math.sin(math.radians(sw)),
                        -math.sin(math.radians(below)))).normalized()
            turn(jn, d0.rotation_difference(t).to_matrix())
        limit, curve = sweep(f"{jn} lower, swing {sw:+d}", np.arange(30, 110 + 1e-6, a.step), lower, pairs)
        limits.append(limit); curves.append(curve)
    out["joints"][jn] = {"kind": "cone", "swing_deg": list(swings), "below_limit_deg": limits, "curves_cm": curves,
                         "rest_dir": [round(x, 5) for x in d0]}
rest()
json.dump(out, open(a.out, "w"), indent=1)
print(f"[limits] -> {a.out}", flush=True)
