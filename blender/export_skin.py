"""A character's skinned mesh and a clip's pose at every video frame, as arrays - for
pipeline/refine_pose.py, which re-poses the mesh outside Blender to match a video pixel by pixel.

    blender -b -noaudio --python export_skin.py -- --blend final.blend --clip punch_combo \
        --frames 124 --fps 24 --out skin.npz [--points 24000] [--calib joint_calib.json]

Writes, all in world space (Z up, metres):
  verts       (N, 3) points on the mesh in its rest pose - every vertex, or --points spread evenly
              over the surface by area, which is plenty for a silhouette
  bone_idx, bone_w   (N, 4) each point's four strongest bones and weights (normalised)
  bones, parents     deforming bones and their ancestors, parents first; parent index (-1 at the root)
  rest        (B, 4, 4) each bone's rest matrix
  pose        (T, B, 4, 4) each bone's matrix at video frame i (the clip at time i / fps - the timing
              blender/render_match.py renders at)
  joints      (13,) bone index of each of the pose model's 13 points (a joint sits at its bone's head)
  face_local  (5, 3) the calibrated nose, eyes and ears in the head bone's frame (with --calib)
  joint_local (13, 3) where the pose model sees each of the 13 points, in its bone's frame (--calib)
  hand_bone, hand_local  (2, 21), (2, 21, 3) DWPose's 21 points of each hand (tools/dwpose.py; left
              then right) as a bone and a point in its frame: the wrist and each finger's three joints
              at their bones' heads, each fingertip where the mesh ends past its last bone's head
  palm_normal (2, 3) out of each palm at rest (the side the fingers curl to)
The armature deforms the mesh by linear blend skinning (no preserve-volume), so these reproduce
Blender's deformation; the face's shape keys are left at rest.
"""
import argparse
import json
import math
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--clip", required=True)
ap.add_argument("--frames", type=int, required=True)
ap.add_argument("--fps", type=float, default=24.0)
ap.add_argument("--points", type=int, default=0, help="sample this many vertices (0: all)")
ap.add_argument("--calib", default=None)
ap.add_argument("--hand-calib", default=None, help="tools/calibrate_hands.py: where DWPose sees each hand point")
ap.add_argument("--out", required=True)
a = ap.parse_args(argv)

J13 = ["head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
       "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
meshes = [o for o in sc.objects if o.type == "MESH" and o.visible_get()
          and any(m.type == "ARMATURE" and m.object == rig for m in o.modifiers)]
W = rig.matrix_world

# bones: every deforming bone and its ancestors, parents first
need = set()
for b in rig.data.bones:
    if b.use_deform or b.name in J13:
        x = b
        while x is not None:
            need.add(x.name)
            x = x.parent
order = [b.name for b in rig.data.bones if b.name in need]      # Blender lists parents first
index = {n: i for i, n in enumerate(order)}
parents = np.array([index[rig.data.bones[n].parent.name] if rig.data.bones[n].parent else -1 for n in order])
rest = np.array([np.array(W @ rig.data.bones[n].matrix_local) for n in order])

# points and weights (rest pose, shape keys at rest): with --points, spread evenly over the surface
# by area - a generated mesh puts its vertices in the hair and shoes and few on a vest or sleeve, so
# vertices alone leave holes in a silhouette - each point's weights blended from its triangle's corners
B = len(order)
Vs, Ws = [], []
for o in meshes:
    me = o.data
    M = np.array(o.matrix_world)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    gname = {g.index: g.name for g in o.vertex_groups}
    Wd = np.zeros((len(me.vertices), B), np.float32)
    for vi, v in enumerate(me.vertices):
        for g in v.groups:
            n_ = gname.get(g.group)
            if n_ in index and g.weight > 0:
                Wd[vi, index[n_]] = g.weight
    me.calc_loop_triangles()
    tri = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    Vs.append((co, tri.reshape(-1, 3)))
    Ws.append(Wd)
if a.points:
    rng = np.random.default_rng(0)
    area = [0.5 * np.linalg.norm(np.cross(c[t[:, 1]] - c[t[:, 0]], c[t[:, 2]] - c[t[:, 0]]), axis=1) for c, t in Vs]
    tot = np.concatenate(area)
    pick = rng.choice(len(tot), a.points, p=tot / tot.sum())
    offs = np.cumsum([0] + [len(x) for x in area])
    V, Wp = [], []
    for k, ((c, t), Wd) in enumerate(zip(Vs, Ws)):
        sel = pick[(pick >= offs[k]) & (pick < offs[k + 1])] - offs[k]
        r1, r2 = rng.random(len(sel)), rng.random(len(sel))
        s1 = np.sqrt(r1)
        bary = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
        tt = t[sel]
        V.append(np.einsum("nk,nkd->nd", bary, c[tt]))
        Wp.append(np.einsum("nk,nkb->nb", bary, Wd[tt]))
    V, Wp = np.concatenate(V), np.concatenate(Wp)
else:
    V = np.concatenate([c for c, _ in Vs])
    Wp = np.concatenate(Ws)
keep = Wp.sum(1) > 0
V, Wp = V[keep], Wp[keep]
BI = np.argsort(-Wp, axis=1)[:, :4].astype(np.int32)
BW = np.take_along_axis(Wp, BI, 1)
BW = (BW / BW.sum(1, keepdims=True)).astype(np.float32)

# the clip at the video's timing
act = bpy.data.actions.get(a.clip)
if act is None:
    raise SystemExit(f"[skin] no clip {a.clip!r} in {a.blend}")
rig.animation_data.action = act
if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
    rig.animation_data.action_slot = act.slots[0]
f0, f1 = act.frame_range
scene_fps = sc.render.fps / sc.render.fps_base
pose = np.zeros((a.frames, len(order), 4, 4))
for i in range(a.frames):
    f = min(f0 + i * scene_fps / a.fps, f1)
    sc.frame_set(int(math.floor(f)), subframe=float(f - math.floor(f)))
    for j, n in enumerate(order):
        pose[i, j] = np.array(W @ rig.pose.bones[n].matrix)

extra = {}
# the hands as DWPose reads them: wrist; thumb, index, middle, ring and little finger - three joints and a tip
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
if all(f"{s}_{f}3" in index for s in ("left", "right") for f in FINGERS):
    Vall = np.concatenate([c for c, _ in Vs])
    Wall = np.concatenate(Ws)
    top, topw = Wall.argmax(1), Wall.max(1)
    hb, hl, pn = np.zeros((2, 21), np.int32), np.zeros((2, 21, 3), np.float32), np.zeros((2, 3), np.float32)
    for h, s in enumerate(("left", "right")):
        hb[h, 0] = index[f"{s}_wrist"]
        for k, f in enumerate(FINGERS):
            for i in range(3):
                hb[h, 1 + 4 * k + i] = index[f"{s}_{f}{i + 1}"]
            b3 = index[f"{s}_{f}3"]
            hb[h, 4 + 4 * k] = b3
            # the tip: the end of the vertices the last bone carries, along the bone
            sel = (top == b3) & (topw >= 0.5)
            loc = (np.c_[Vall[sel], np.ones(sel.sum())] @ np.linalg.inv(rest[b3]).T)[:, :3]
            if len(loc) >= 5:
                far = loc[:, 1] >= loc[:, 1].max() - 0.002
                hl[h, 4 + 4 * k] = [loc[far, 0].mean(), loc[:, 1].max(), loc[far, 2].mean()]
            else:                                                       # no mesh on it: the bone's own tail
                hl[h, 4 + 4 * k] = [0.0, rig.data.bones[order[b3]].length, 0.0]
        # out of the palm: across the knuckles from the index to the little finger, turned to the thumb's side
        w0 = rest[index[f"{s}_wrist"]][:3, 3]
        n = np.cross(rest[index[f"{s}_index1"]][:3, 3] - w0, rest[index[f"{s}_pinky1"]][:3, 3] - w0)
        n /= np.linalg.norm(n)
        tip = (rest[hb[h, 4]] @ np.r_[hl[h, 4], 1])[:3]
        pn[h] = n if (tip - w0) @ n > 0 else -n
    if a.hand_calib:                                                    # where DWPose sees them on this character
        hl = hl + np.array(json.load(open(a.hand_calib))["offset_local"], np.float32)
    extra.update(hand_bone=hb, hand_local=hl, palm_normal=pn)
    print("[skin] fingertips past the last bone's head: " + ", ".join(
        f"{f} {hl[0, 4 + 4 * k, 1] * 100:.1f} cm" for k, f in enumerate(FINGERS)) + " (left hand)", flush=True)
if a.calib:
    cal = json.load(open(a.calib))
    extra["face_local"] = np.array([f["local"] for f in cal["face"]])
    # where the pose model sees each joint, in its bone's frame (tools/calibrate_joints.py)
    extra["joint_local"] = np.array([pt["local"] for pt in cal["points"]])
np.savez_compressed(a.out, verts=V.astype(np.float32), bone_idx=BI, bone_w=BW, bones=np.array(order),
                    parents=parents, rest=rest.astype(np.float32), pose=pose.astype(np.float32),
                    lengths=np.array([rig.data.bones[n].length for n in order], np.float32),
                    joints=np.array([index[n] for n in J13]), **extra)
print(f"[skin] {len(V)} points, {len(order)} bones, {a.frames} frames of {a.clip} -> {a.out}", flush=True)
