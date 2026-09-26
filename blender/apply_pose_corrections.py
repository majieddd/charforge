"""Write pipeline/refine_pose.py's turns into a clip: every key of every corrected bone turned by its
frame's correction, in the bone's own frame.

    blender -b -noaudio --python apply_pose_corrections.py -- --blend final.blend --clip punch_combo \
        --corr corrections.npz --out final_refined.blend [--as punch_combo_refined]

refine_pose.py poses the skinned mesh as M'_bone = M'_parent . L_bone . R(delta), L_bone being the
clip's own parent-to-bone transform; in Blender's terms that is the bone's matrix_basis times
R(delta), so the correction goes on each key of that bone and its children follow as they do in
the fit. The corrections are per video frame; a key between two video frames takes the turn
interpolated between them. With --as the clip is copied first and the copy corrected.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--clip", required=True)
ap.add_argument("--corr", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--as", dest="as_", default=None)
ap.add_argument("--masks", default=None,
                help="the video's figure masks (npz 'masks'): each key's lowest shoe goes to the height the video's "
                     "figure stands above its own floor line")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
act = bpy.data.actions.get(a.clip)
if act is None:
    raise SystemExit(f"[apply] no clip {a.clip!r}")
if a.as_:
    act = act.copy()
    act.name = a.as_
    act.use_fake_user = True
rig.animation_data.action = act
if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
    rig.animation_data.action_slot = act.slots[0]
C = np.load(a.corr)
delta, bones, vfps = C["delta"], [str(b) for b in C["bones"]], float(C["fps"])
T = len(delta)
f0, f1 = (int(round(x)) for x in act.frame_range)
scene_fps = sc.render.fps / sc.render.fps_base


def rot(v):
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return Matrix.Identity(4)
    return Matrix.Rotation(th, 4, Vector((v / th).tolist()))


pbs = [rig.pose.bones.get(b) for b in bones]

# The feet: the retarget put the planted soles on the floor (blender/retarget.py, ground_and_plant), and
# turning the hips, knees and ankles toward the video lifts them off it - Aoi's spell cast stood 2-6 cm
# in the air after the refine, seen from the side. Each key's lowest sole point is measured before the
# turn and after, and the pelvis moves by the difference: the lowest foot goes back to where the
# retarget had it (on the floor while planted, as high as it was in a jump).
SOLE = []
MW = rig.matrix_world
for o in sc.objects:
    if o.type != "MESH" or not any(m.type == "ARMATURE" for m in o.modifiers):
        continue
    groups = {g.index: g.name for g in o.vertex_groups if g.name.endswith(("_ankle", "_foot")) and g.name in rig.data.bones}
    if not groups:
        continue
    cand = []
    for v in o.data.vertices:
        best = max(((groups[g.group], g.weight) for g in v.groups if g.group in groups), key=lambda x: x[1], default=None)
        if best and best[1] > 0.5:
            cand.append((best[0], MW.inverted() @ (o.matrix_world @ v.co)))
    if cand:
        zmin = min(c[1].z for c in cand)
        for bn, co in cand:
            if co.z < zmin + 0.03 * (1 if abs(MW.to_scale().z - 1) < 1e-6 else 1 / MW.to_scale().z):
                SOLE.append((bn, rig.data.bones[bn].matrix_local.inverted() @ co))
SOLE = SOLE[::max(1, len(SOLE) // 400)]


floor0 = 0.0                                   # the ground: the pipeline stands every character on z = 0


def lowest_sole():
    if not SOLE:
        return None
    return min((MW @ (rig.pose.bones[bn].matrix @ p)).z for bn, p in SOLE)


pelvis = rig.pose.bones.get("pelvis")
# the video's floor: the figure's lowest pixel, frame by frame; the floor line is where it is lowest over the
# clip (a robust max), and a frame whose figure ends higher is off the ground by that much (a hop), in metres
# at the refine's scale (pixels per metre of the video)
target = None
if a.masks and os.path.exists(a.masks) and "scale" in C.files:
    mk = np.load(a.masks)["masks"]
    bottom = np.array([np.nonzero(m.any(1))[0].max() if m.any() else np.nan for m in mk], float)
    # a figure running off the bottom of the frame says nothing about the floor (Aoi's feet leave it in the
    # middle of her spell cast: 767 of 768 rows) - those frames keep the retarget's planting
    ok = np.isfinite(bottom) & (bottom < mk.shape[1] - 2)
    if ok.sum() > 10:
        floor_px = float(np.median(bottom[ok]))                         # standing, most of any move
        target = np.clip((floor_px - bottom) / float(C["scale"]), 0.0, None)
        target[~ok] = np.nan
lift = {}
prev = {}
n_keys = 0
for f in range(f0, f1 + 1):
    sc.frame_set(f)
    h0 = lowest_sole()
    j = (f - f0) / scene_fps * vfps                                  # the video frame this key falls on
    j0 = min(int(math.floor(j)), T - 1)
    j1 = min(j0 + 1, T - 1)
    w = j - j0
    d = (1 - w) * delta[j0] + w * delta[j1]
    for k, pb in enumerate(pbs):
        if pb is None:
            continue
        pb.matrix_basis = pb.matrix_basis @ rot(d[k])
    for k, pb in enumerate(pbs):
        if pb is None:
            continue
        if pb.rotation_mode == "QUATERNION":
            pb.keyframe_insert("rotation_quaternion", frame=f)
        else:
            e = pb.rotation_euler.copy()
            if pb.name in prev:
                e.make_compatible(prev[pb.name])
            pb.rotation_euler = e
            prev[pb.name] = e.copy()
            pb.keyframe_insert("rotation_euler", frame=f)
        n_keys += 1
    if h0 is not None:
        bpy.context.view_layer.update()
        h1 = lowest_sole()
        goal = h0 - floor0
        if target is not None:
            tv = (1 - w) * target[j0] + w * target[j1]
            if np.isfinite(tv):
                goal = tv
        lift[f] = (h1 - floor0) - goal
if lift and pelvis is not None:
    fs = sorted(lift)
    dz = np.array([lift[f] for f in fs])
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)                 # a light smoothing, 2 keys
    dz_s = np.convolve(np.pad(dz, 6, mode="edge"), k / k.sum(), mode="valid")
    to_local = (MW.to_3x3() @ pelvis.bone.matrix_local.to_3x3()).inverted()
    for f, z in zip(fs, dz_s):
        sc.frame_set(f)
        pelvis.location = pelvis.location + to_local @ Vector((0.0, 0.0, -float(z)))
        pelvis.keyframe_insert("location", frame=f)
    how = "to the height the video's figure stands above its floor line" if target is not None else "back to where the retarget had it"
    print(f"[apply] feet: the lowest sole was {dz.min() * 100:+.1f}..{dz.max() * 100:+.1f} cm off "
          f"(mean {np.abs(dz).mean() * 100:.1f}) - the pelvis moved it {how}", flush=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
print(f"[apply] {n_keys} keys turned on {sum(p is not None for p in pbs)} bones of {act.name} -> {a.out}", flush=True)
