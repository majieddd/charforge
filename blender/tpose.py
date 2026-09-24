"""Rebake the rig's rest pose to a true T-pose, so captured motion retargets correctly.

This is the fix for the single worst artefact in the pipeline: retargeted Mixamo clips swung the
character's arms down through its own torso and out behind its back.

The cause is a rest-pose mismatch, and it is measurable. Retargeting applies the source's
rotation *relative to the source's own rest* onto the target's rest:

    tgt_pose_world = src_pose_world @ src_rest_world⁻¹ @ tgt_rest_world

Mixamo authors against a T-pose - its upper arm rest direction is (1, 0, 0), exactly 90 degrees
from straight down. The rig skeleton.py infers from a generated character is a narrow A-pose:
(0.48, 0.11, -0.87), only 29.4 degrees from down. A walk clip that lowers the arm ~70 degrees
from horizontal therefore lands at 29 + 70 ≈ 99 degrees past vertical on this rig, which is
inside the ribcage. The legs escaped because their rests agree to within 6 degrees, which is
why the failure read as "the arms are broken" rather than "retargeting is broken".

Rather than special-case the arms in the retargeter - which would need a per-bone fudge for
every rig - the rig is brought into the space the motion was authored in. Three steps, and the
order matters:

  1. pose the arm chain until each bone points along ±X (the T)
  2. bake that pose into every skinned mesh, by applying a *copy* of its armature modifier.
     Skipping this is what breaks people's characters: apply the rest pose without baking the
     meshes and the geometry stays in the old pose while the skeleton moves under it.
  3. make the pose the new rest with pose.armature_apply

Vertex weights are untouched throughout, so the skinning solved in the A-pose still holds; only
the frame it is expressed in changes.

Run: blender -b -noaudio --python tpose.py -- --blend cloth_ready.blend --out tposed.blend
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
if rig is None:
    raise SystemExit("[tpose] no armature")
meshes = [o for o in scene.objects if o.type == "MESH"
          and any(m.type == "ARMATURE" for m in o.modifiers)]

# Target world directions for the T. Only the arm chain moves; legs and spine already agree
# with Mixamo's rest to within a few degrees and are left alone.
TARGETS = {
    "left_shoulder": Vector((1, 0, 0)), "left_elbow": Vector((1, 0, 0)),
    "left_wrist": Vector((1, 0, 0)),
    "right_shoulder": Vector((-1, 0, 0)), "right_elbow": Vector((-1, 0, 0)),
    "right_wrist": Vector((-1, 0, 0)),
}
ORDER = ["left_shoulder", "left_elbow", "left_wrist",
         "right_shoulder", "right_elbow", "right_wrist"]


def world_dir(pb):
    M = rig.matrix_world
    return ((M @ pb.tail) - (M @ pb.head)).normalized()


def report(tag):
    out = {}
    for n in ORDER:
        if n in rig.pose.bones:
            d = world_dir(rig.pose.bones[n])
            out[n] = {"dir": [round(v, 3) for v in d],
                      "deg_from_down": round(math.degrees(d.angle(Vector((0, 0, -1)))), 1)}
    print(f"[tpose] {tag}: " + ", ".join(
        f"{k} {v['deg_from_down']}deg" for k, v in out.items()), flush=True)
    return out


bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
for pb in rig.pose.bones:
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = (1, 0, 0, 0)
    pb.location = (0, 0, 0)
bpy.context.view_layer.update()
before = report("rest before")

# ---- 1. pose into the T, parent-first so each correction builds on the last ------------------
for name in ORDER:
    pb = rig.pose.bones.get(name)
    if pb is None:
        continue
    bpy.context.view_layer.update()
    cur = world_dir(pb)
    tgt = TARGETS[name]
    if cur.angle(tgt) < math.radians(0.5):
        continue
    R = cur.rotation_difference(tgt).to_matrix().to_4x4()
    M = pb.matrix.copy()
    head = M.to_translation()
    # rotate about the bone's own head, in world space
    W = rig.matrix_world
    Winv = W.inverted()
    Mw = W @ M
    hw = Mw.to_translation()
    Mw = (_T := __import__("mathutils").Matrix.Translation(hw)) @ R @ \
         __import__("mathutils").Matrix.Translation(-hw) @ Mw
    pb.matrix = Winv @ Mw
    bpy.context.view_layer.update()

# ---- 1b. square the hands -------------------------------------------------------------------------
# The retargeter copies whole world rotations, so the hand's roll about the forearm must match the
# animation library's rest exactly: palm down, thumb forward. Raising the arm by the shortest
# rotation leaves whatever roll the generated pose had; with modelled fingers there is a knuckle
# line to measure it by (little finger to index should point forward, -Y), so it is rotated out.
import mathutils as _mu
for side, sgn in (("left", 1.0), ("right", -1.0)):
    pb = rig.pose.bones.get(f"{side}_wrist")
    i1, p1 = rig.pose.bones.get(f"{side}_index1"), rig.pose.bones.get(f"{side}_pinky1")
    if pb is None or i1 is None or p1 is None:
        continue
    bpy.context.view_layer.update()
    W = rig.matrix_world
    knuckles = (W @ i1.head) - (W @ p1.head)
    kx = Vector((0.0, knuckles.y, knuckles.z))
    if kx.length < 1e-6:
        continue
    ang = kx.angle(Vector((0, -1, 0)))
    if ang < math.radians(0.5):
        continue
    axis = Vector((sgn, 0, 0))
    # sign: which way about the forearm brings the knuckle line to -Y
    Rp = _mu.Matrix.Rotation(ang, 4, axis)
    Rm = _mu.Matrix.Rotation(-ang, 4, axis)
    R = Rp if (Rp.to_3x3() @ kx).angle(Vector((0, -1, 0))) < (Rm.to_3x3() @ kx).angle(Vector((0, -1, 0))) else Rm
    Mw = W @ pb.matrix
    hw = Mw.to_translation()
    Mw = _mu.Matrix.Translation(hw) @ R @ _mu.Matrix.Translation(-hw) @ Mw
    pb.matrix = W.inverted() @ Mw
    bpy.context.view_layer.update()
    print(f"[tpose] {side} hand squared: rolled {math.degrees(ang):.1f} deg about the forearm "
          "(palm down, thumb forward)", flush=True)

after_pose = report("posed to T")

# ---- 2. bake the pose into every skinned mesh -------------------------------------------------
bpy.ops.object.mode_set(mode="OBJECT")
for ob in meshes:
    if ob.data.shape_keys is not None:
        raise SystemExit(f"[tpose] {ob.name} carries shape keys; applying a modifier would "
                         "discard them. Re-pose before baking cloth, not after.")
    bpy.context.view_layer.objects.active = ob
    arm_mod = next(m for m in ob.modifiers if m.type == "ARMATURE")
    n_before = len(ob.data.vertices)
    bpy.ops.object.modifier_copy(modifier=arm_mod.name)
    copy_mod = next(m for m in ob.modifiers
                    if m.type == "ARMATURE" and m.name != arm_mod.name)
    bpy.ops.object.modifier_apply(modifier=copy_mod.name)
    if len(ob.data.vertices) != n_before:
        raise SystemExit(f"[tpose] {ob.name} vertex count changed during bake")
    print(f"[tpose] baked T-pose into {ob.name} ({n_before:,} verts, "
          f"{len(ob.vertex_groups)} groups intact)", flush=True)

# ---- 3. make it the rest pose -----------------------------------------------------------------
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
bpy.ops.pose.armature_apply()
bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.view_layer.update()

bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
for pb in rig.pose.bones:
    pb.rotation_quaternion = (1, 0, 0, 0)
bpy.context.view_layer.update()
after = report("rest after")
bpy.ops.object.mode_set(mode="OBJECT")

worst = max(abs(v["deg_from_down"] - 90.0) for v in after.values()) if after else 999
if worst > 6.0:
    raise SystemExit(f"[tpose] rest pose is still {worst:.1f}deg off the T - refusing to write "
                     "a file that would retarget as badly as the one it replaces.")

# any pre-existing clips were authored against the OLD rest and are now wrong
stale = [act for act in bpy.data.actions]
for act in stale:
    act.use_fake_user = False
    bpy.data.actions.remove(act)
if rig.animation_data:
    rig.animation_data.action = None
print(f"[tpose] dropped {len(stale)} action(s) authored against the old rest pose", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump({"before": before, "after": after, "max_deg_off_T": round(worst, 2)},
              open(a.json, "w"), indent=2)
print(f"[tpose] -> {a.out}", flush=True)
