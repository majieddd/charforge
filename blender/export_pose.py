"""A rig's rest mesh and every clip's bone matrices, for the contact solve (pipeline/contact_solve.py, E151).
    blender -b -noaudio --python export_pose.py -- --blend final.blend --out dir [--clips a,b]
Writes dir/rest.npz (world rest vertices, normals, skin weights over the rig's bones, rest bone matrices, names,
parents) and dir/clip_<name>.npz (per frame, each bone's world matrix; the action's frame range)."""
import argparse
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--clips", default="")
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=a.blend)

arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
names = [b.name for b in arm.data.bones]
idx = {n: i for i, n in enumerate(names)}
parents = np.array([idx[b.parent.name] if b.parent else -1 for b in arm.data.bones])
rest = np.array([np.array(arm.matrix_world @ b.matrix_local) for b in arm.data.bones])

V, N, W = [], [], []
for o in bpy.data.objects:
    if o.type != "MESH" or not any(m.type == "ARMATURE" for m in o.modifiers):
        continue
    mw = np.array(o.matrix_world)
    v = np.array([list(x.co) for x in o.data.vertices])
    n = np.array([list(x.normal) for x in o.data.vertices])
    V.append(v @ mw[:3, :3].T + mw[:3, 3])
    N.append(n @ np.linalg.inv(mw[:3, :3]))
    w = np.zeros((len(v), len(names)), np.float32)
    gmap = {g.index: idx.get(g.name, -1) for g in o.vertex_groups}
    for vx in o.data.vertices:
        for g in vx.groups:
            j = gmap.get(g.group, -1)
            if j >= 0:
                w[vx.index, j] += g.weight
    W.append(w)
V, N, W = np.concatenate(V), np.concatenate(N), np.concatenate(W)
N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
np.savez(os.path.join(a.out, "rest.npz"), verts=V, normals=N, weights=W, rest=rest, names=np.array(names),
         parents=parents)

wanted = [c for c in a.clips.split(",") if c]
scene = bpy.context.scene
for act in bpy.data.actions:
    if wanted and act.name not in wanted:
        continue
    arm.animation_data.action = act
    if hasattr(arm.animation_data, "action_slot") and getattr(act, "slots", None):
        arm.animation_data.action_slot = act.slots[0]
    f0, f1 = (int(x) for x in act.frame_range)
    M = np.zeros((f1 - f0 + 1, len(names), 4, 4))
    for t, fr in enumerate(range(f0, f1 + 1)):
        scene.frame_set(fr)
        for i, pb in enumerate(arm.pose.bones):
            M[t, i] = np.array(arm.matrix_world @ pb.matrix)
    np.savez(os.path.join(a.out, f"clip_{act.name}.npz"), mats=M, f0=f0, f1=f1)
print(f"[export_pose] {len(V)} verts, {len(names)} bones -> {a.out}", flush=True)
