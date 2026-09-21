"""Find geometry that is hidden at rest but pushes through the surface under animation.

The garment is one mesh carrying both the jacket and the t-shirt under it, and 44% of the faces
that collapse are never externally visible - they are the inner layer. That is harmless while
they stay inside. It stops being harmless the moment a layer crosses: a patch of t-shirt
surfacing through the jacket reads as a grey flicker on the chest for a few frames, which is
exactly the class of artefact that gets described as "weird" and is invisible in any rest-pose
check.

A face is externally visible when a ray from its centre, along its own normal, escapes the
character without hitting anything. This classifies every face at rest, then re-tests the ones
that were hidden on every sampled animation frame. A face hidden at rest and visible later has
crossed a layer.

Run: blender -b -noaudio --python pokethrough.py -- --blend final.blend --clips walk,run
"""
import argparse
import json
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--clips", default="walk,run,idle")
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--json", default=None)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next(o for o in scene.objects if o.type == "ARMATURE")
meshes = [o for o in scene.objects if o.type == "MESH" and o.find_armature() == rig]


def posed(dg):
    """Combined BVH of the whole posed character, plus per-mesh face centres and normals."""
    verts, faces, per = [], [], {}
    for ob in meshes:
        ev = ob.evaluated_get(dg)
        m = ev.to_mesh()
        off = len(verts)
        V = np.empty(len(m.vertices) * 3)
        m.vertices.foreach_get("co", V)
        V = V.reshape(-1, 3)
        mw = np.array(ob.matrix_world)
        V = V @ mw[:3, :3].T + mw[:3, 3]
        verts.extend(Vector(p) for p in V)
        faces.extend([[i + off for i in p.vertices] for p in m.polygons])
        C = np.empty(len(m.polygons) * 3)
        m.polygons.foreach_get("center", C)
        N = np.empty(len(m.polygons) * 3)
        m.polygons.foreach_get("normal", N)
        C = C.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        N = N.reshape(-1, 3) @ mw[:3, :3].T
        per[ob.name] = (C, N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12))
        ev.to_mesh_clear()
    return BVHTree.FromPolygons(verts, faces, all_triangles=False, epsilon=0.0), per, verts


def visible(bvh, C, N, idx, eps, reach):
    out = np.zeros(len(idx), bool)
    for k, fi in enumerate(idx):
        o = Vector(C[fi] + N[fi] * eps)
        out[k] = bvh.ray_cast(o, Vector(N[fi]), reach)[0] is None
    return out


# ---- rest ------------------------------------------------------------------------------------
if rig.animation_data:
    rig.animation_data.action = None
for pb in rig.pose.bones:
    pb.matrix_basis.identity()
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
bvh, per, verts = posed(dg)
P = np.array([v[:] for v in verts])
EXT = float(np.linalg.norm(P.max(0) - P.min(0)))
eps, reach = 1e-4 * EXT, 0.30 * EXT

hidden = {}
for ob in meshes:
    C, N = per[ob.name]
    vis = visible(bvh, C, N, np.arange(len(C)), eps, reach)
    hidden[ob.name] = np.where(~vis)[0]
    print(f"[poke] {ob.name}: {len(C):,} faces, {int((~vis).sum()):,} hidden at rest "
          f"({100*(~vis).mean():.1f}%)", flush=True)

# ---- under animation ---------------------------------------------------------------------------
surfaced = {k: np.zeros(len(v), bool) for k, v in hidden.items()}
frames_checked = 0
for clip in [c.strip() for c in a.clips.split(",") if c.strip()]:
    act = bpy.data.actions.get(clip)
    if act is None:
        continue
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    f0, f1 = (int(x) for x in act.frame_range)
    for fr in range(f0, f1 + 1, a.step):
        scene.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        bvh2, per2, _ = posed(dg)
        frames_checked += 1
        for ob in meshes:
            idx = hidden[ob.name]
            if len(idx) == 0:
                continue
            C, N = per2[ob.name]
            surfaced[ob.name] |= visible(bvh2, C, N, idx, eps, reach)

out = {"frames_checked": frames_checked, "meshes": {}}
print(f"\n[poke] checked {frames_checked} frames across {a.clips}", flush=True)
for ob in meshes:
    idx, s = hidden[ob.name], surfaced[ob.name]
    n_total = len(per[ob.name][0])
    n = int(s.sum())
    print(f"[poke] {ob.name}: {n:,} of {len(idx):,} rest-hidden faces surface at some point "
          f"= {100*n/max(n_total,1):.2f}% of the mesh", flush=True)
    out["meshes"][ob.name] = {"faces": n_total, "hidden_at_rest": int(len(idx)),
                              "surfaced": n, "pct_of_mesh": round(100 * n / max(n_total, 1), 3)}
if a.json:
    json.dump(out, open(a.json, "w"), indent=2)
