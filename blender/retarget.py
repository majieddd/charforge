"""Retarget a Mixamo clip onto the CharForge rig.

The procedural gait in gait.py is sampled from published curves and reads far better than the
eyeballed keyframes it replaced, but it is still a synthesis. Mixamo is captured motion, and the
rig was deliberately given SMPL-H bone names precisely so captured clips could be dropped onto
it. This does the dropping.

The naive version of this - a Copy Rotation constraint in world space per bone - is wrong in a
way that is easy to miss, because it looks nearly right. It copies the source bone's *absolute*
orientation, which silently also copies the source rig's rest pose and bone roll. Mixamo rigs
are T-pose with their own roll conventions; this rig is A-pose with roll that came out of
skeleton.py. Copy the absolute orientation and every limb picks up a constant twist.

What is actually wanted is the source's *change from its own rest pose*, re-expressed in the
target's rest frame:

    tgt_pose_world = src_pose_world @ src_rest_world⁻¹ @ tgt_rest_world

and every one of those matrices has to be in WORLD space, not armature space. Mixamo FBX is
Y-up; Blender is Z-up, and the importer puts that -90 deg X correction on the armature *object*
matrix while leaving the bone rest matrices in the file's own frame. Computing the delta from
`pose_bone.matrix` and `bone.matrix_local` - both armature-space - therefore mixes a Y-up source
with a Z-up target, and the result is a character bent 90 degrees forward: the retargeted run
came out horizontal, flying like Superman. Multiplying each matrix by its own armature's
`matrix_world` first puts both rigs in the same frame, and the 0.01 unit scale cancels out of
the delta on its own.

with the target bone's own head position preserved, so the target keeps its proportions rather
than being stretched onto the source skeleton.

The root is handled separately, and deliberately not the obvious way. Two points:

  * `pose_bone.location` is expressed in the *bone's own* rest frame, not world space. Assigning
    a world offset to it sends vertical motion down whatever axis the pelvis bone happens to
    point, which is how a wave ended up lifting the character out of frame. The root offset is
    therefore applied through `pose_bone.matrix`, in armature space, like every other channel.
  * horizontal root travel is measured but **not baked in**. These clips feed a playground where
    code drives the character's position; a clip that also walks forward on its own fights it and
    slides. What ships instead is an in-place clip plus the measured ground speed as metadata, so
    the runtime can match playback rate to actual velocity. Vertical motion is kept, because a
    jump has to leave the floor.

Bones are processed parent-first, because setting `pose_bone.matrix` on a child depends on the
parent already being posed.

Run: blender -b -noaudio --python retarget.py -- --rig rigged.blend --clips clips.json \
         --out animated_mixamo.glb
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--rig", required=True)
ap.add_argument("--clips", required=True, help="JSON: {clip_name: fbx_path}")
ap.add_argument("--out", required=True)
ap.add_argument("--blend-out", default=None)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--max-frames", type=int, default=120)
ap.add_argument("--json", default=None)
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

clips = json.load(open(a.clips))
report = {}

for clip_name, fbx in clips.items():
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
    hips = src_name(src, "Hips")
    s_pelvis = src_name(src, "Hips")
    s_ankle = src_name(src, "LeftFoot")
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
        sname = src_name(src, key)
        if sname and tname in tgt.pose.bones:
            pairs.append((sname, tname))
    missing = [k for k in MAP if src_name(src, k) is None]
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
    TW = tgt.matrix_world.copy()
    TW_inv = TW.inverted()
    root_ref = None
    min_off = max_off = None
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
            M_world = src_pose_w @ src_rest_w.inverted() @ tgt_rest_w
            M = TW_inv @ M_world
            # normalise: the source's unit scale must not ride along into the target
            M = Matrix.LocRotScale(M.to_translation(), M.to_quaternion(), Vector((1, 1, 1)))
            # keep the target's own bone head; only the root carries translation
            M.translation = tpb.matrix.to_translation()
            tpb.matrix = M
            bpy.context.view_layer.update()

        # root translation, in armature space, via the matrix - not via pose_bone.location
        pelvis = tgt.pose.bones.get("pelvis")
        shp = src.pose.bones[hips]
        world = (src.matrix_world @ shp.head)
        if root_ref is None:
            root_ref = world.copy()
        off = (world - root_ref) * scale
        flat = math.hypot(off.x, off.y)
        min_off = flat if min_off is None else min(min_off, flat)
        max_off = flat if max_off is None else max(max_off, flat)
        if pelvis is not None:
            applied = Vector((off.x, off.y, off.z)) if a.keep_root_motion \
                else Vector((0.0, 0.0, off.z))
            M = pelvis.matrix.copy()
            M.translation = tgt_rest["pelvis"].to_translation() + applied
            pelvis.matrix = M
            bpy.context.view_layer.update()

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

    # Stride check: how far the root actually travelled, in leg-lengths. A walk should cover
    # something on the order of one leg length per second; a clip that retargets to ~0 has been
    # scaled into nothing even though every bone "mapped" fine.
    travel = (max_off - min_off) if (max_off is not None and min_off is not None) else 0.0
    per_sec = travel / max(n / a.fps, 1e-6) / tgt_leg
    report[clip_name] = {"frames": n, "bones_mapped": len(pairs), "scale": round(scale, 5),
                         "curves": len(action_fcurves(act)), "source": os.path.basename(fbx),
                         "root_travel_m": round(travel, 3),
                         "leg_lengths_per_sec": round(per_sec, 3)}
    print(f"[retarget]   root travel {travel:.3f} m ({per_sec:.2f} leg-lengths/s)", flush=True)
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
