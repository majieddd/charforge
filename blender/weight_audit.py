"""Why the garment collapses: measure the weights themselves, not their effect.

deform_audit.py showed clothing faces collapsing to 4-18% of rest area with face normals
swinging 100-160 degrees past their own bone's rotation. Both are signatures of the same
thing - a vertex blended between bones that are far apart in the skeleton. Linear blend
skinning interpolates *matrices*, so a vertex weighted half to spine1 and half to left_elbow
lands somewhere between two unrelated transforms, and the surface around it folds.

So the question is not "are the weights smooth" (they are - they were Laplacian-smoothed 18
times) but "are they anatomically coherent". Three measurements:

  influence spread   for each vertex, the maximum distance *through the skeleton tree* between
                     any two bones influencing it. 0-1 is normal (a vertex belongs to a bone
                     and its parent). 4+ means the vertex is being pulled by two unrelated
                     limbs and will collapse.
  outliers           a vertex whose weight vector disagrees sharply with its surface neighbours
                     is a local defect - it gets dragged away from the surface it belongs to,
                     which reads as a spike or a balloon. Measured as L1 distance to the
                     neighbour mean.
  cover              for each clothing vertex, the distance to the nearest body vertex. This
                     decides whether the garment *can* inherit the body's weights: if there is
                     no skin geometry under the jacket, there is nothing to inherit from and a
                     surface transfer is not available.

Also re-runs the penetration test at REST, as a baseline. A metric that already reports 39%
at rest is measuring its own bias, not the animation.

Run: blender -b -noaudio --python weight_audit.py -- --blend p4_final.blend
"""
import argparse
import sys
from collections import defaultdict, deque

import bpy
import numpy as np
from mathutils import kdtree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next(o for o in scene.objects if o.type == "ARMATURE")
meshes = [o for o in scene.objects if o.type == "MESH"]

# ---- skeleton graph distance -------------------------------------------------------------
bones = [b.name for b in rig.data.bones]
bi = {n: i for i, n in enumerate(bones)}
adj = defaultdict(set)
for b in rig.data.bones:
    if b.parent:
        adj[bi[b.name]].add(bi[b.parent.name])
        adj[bi[b.parent.name]].add(bi[b.name])
NB = len(bones)
D = np.full((NB, NB), 99, np.int32)
for s in range(NB):
    D[s, s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if D[s, v] == 99:
                D[s, v] = D[s, u] + 1
                q.append(v)

print(f"[waudit] skeleton: {NB} bones, max graph distance {int(D[D < 99].max())}", flush=True)

body = next((o for o in meshes if "skin" in o.name.lower() or "body" in o.name.lower()), None)

for ob in meshes:
    me = ob.data
    n = len(me.vertices)
    names = [vg.name for vg in ob.vertex_groups]
    gidx_to_bone = [bi.get(nm, -1) for nm in names]

    W = [dict() for _ in range(n)]
    for v in me.vertices:
        for g in v.groups:
            if g.weight > 1e-4:
                W[v.index][g.group] = g.weight

    spread = np.zeros(n, np.int32)
    ninf = np.zeros(n, np.int32)
    for i, w in enumerate(W):
        bs = [gidx_to_bone[k] for k in w if gidx_to_bone[k] >= 0]
        ninf[i] = len(bs)
        if len(bs) > 1:
            sub = D[np.ix_(bs, bs)]
            spread[i] = int(sub[sub < 99].max())

    # only count spread that carries real weight: a 2% tail influence is harmless
    heavy_spread = np.zeros(n, np.int32)
    for i, w in enumerate(W):
        bs = [gidx_to_bone[k] for k, ww in w.items() if ww >= 0.15 and gidx_to_bone[k] >= 0]
        if len(bs) > 1:
            sub = D[np.ix_(bs, bs)]
            heavy_spread[i] = int(sub[sub < 99].max())

    print(f"\n[waudit] {ob.name}: {n:,} vertices, mean {ninf.mean():.2f} influences", flush=True)
    for label, arr in (("any influence", spread), ("influence >=15%", heavy_spread)):
        hist = np.bincount(arr, minlength=7)[:7]
        pct = 100 * hist / n
        print(f"[waudit]   bone-graph spread ({label}): "
              + "  ".join(f"{d}:{pct[d]:5.1f}%" for d in range(7)), flush=True)
        bad = int((arr >= 3).sum())
        print(f"[waudit]     spread>=3 (unrelated limbs blended): {bad:,} ({100*bad/n:.2f}%)",
              flush=True)

    # ---- neighbour disagreement ----------------------------------------------------------
    V = np.empty(n * 3, np.float64)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    ext = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    tol = 1e-5 * ext
    bucket = defaultdict(list)
    keys = np.round(V / tol).astype(np.int64)
    nbr = [set() for _ in range(n)]
    for e in me.edges:
        i, j = e.vertices
        nbr[i].add(j)
        nbr[j].add(i)
    for i in range(n):
        bucket[tuple(keys[i])].append(i)
    for mem in bucket.values():
        if len(mem) < 2:
            continue
        u = set()
        for m in mem:
            u |= nbr[m]
        for m in mem:
            nbr[m] = u

    dis = np.zeros(n)
    for i in range(n):
        nb = nbr[i]
        if not nb:
            continue
        acc = defaultdict(float)
        for j in nb:
            for k, w in W[j].items():
                acc[k] += w
        inv = 1.0 / len(nb)
        keys_all = set(acc) | set(W[i])
        dis[i] = sum(abs(W[i].get(k, 0.0) - acc.get(k, 0.0) * inv) for k in keys_all)
    print(f"[waudit]   neighbour disagreement: mean {dis.mean():.3f}  p99 {np.percentile(dis,99):.3f}"
          f"  >0.5: {int((dis>0.5).sum()):,} ({100*(dis>0.5).mean():.2f}%)", flush=True)

    # worst bones by disagreement, to match against the deform audit
    dom = np.array([max(W[i].items(), key=lambda kv: kv[1])[0] if W[i] else -1 for i in range(n)])
    rows = []
    for g in np.unique(dom):
        if g < 0:
            continue
        m = dom == g
        rows.append((names[g], int(m.sum()), float(dis[m].mean()), float(np.percentile(dis[m], 99)),
                     float(heavy_spread[m].mean())))
    rows.sort(key=lambda r: -r[3])
    print(f"[waudit]   worst bones by p99 neighbour disagreement:", flush=True)
    for nm, cnt, mn, p99, hs in rows[:5]:
        print(f"[waudit]     {nm:<18s} {cnt:6,}v  mean {mn:.3f}  p99 {p99:.3f}  "
              f"mean heavy-spread {hs:.2f}", flush=True)

# ---- is there body geometry under the garment? ----------------------------------------------
cloth = next((o for o in meshes if "cloth" in o.name.lower()), None)
if body is not None and cloth is not None:
    BV = np.array([(body.matrix_world @ v.co)[:] for v in body.data.vertices])
    CV = np.array([(cloth.matrix_world @ v.co)[:] for v in cloth.data.vertices])
    tree = kdtree.KDTree(len(BV))
    for i, p in enumerate(BV):
        tree.insert(p, i)
    tree.balance()
    vn = np.empty(len(body.data.vertices) * 3, np.float64)
    body.data.vertices.foreach_get("normal", vn)
    vn = vn.reshape(-1, 3)

    idx = np.linspace(0, len(CV) - 1, min(6000, len(CV))).astype(int)
    d_near, signed = [], []
    for i in idx:
        p = CV[i]
        _, j, dd = tree.find(p)
        d_near.append(dd)
        signed.append(float(np.dot(p - BV[j], vn[j])))
    d_near = np.array(d_near)
    signed = np.array(signed)
    print(f"\n[waudit] garment -> nearest body vertex (REST pose):", flush=True)
    print(f"[waudit]   distance  median {np.median(d_near)*100:.2f}cm  "
          f"p90 {np.percentile(d_near,90)*100:.2f}cm  max {d_near.max()*100:.2f}cm", flush=True)
    print(f"[waudit]   'inside' by the animation metric: {100*(signed<-1e-4).mean():.2f}% "
          f"at REST  (baseline - anything near this at rest means the metric is biased)",
          flush=True)
    print(f"[waudit]   body bbox z {BV[:,2].min():.3f}..{BV[:,2].max():.3f}   "
          f"cloth bbox z {CV[:,2].min():.3f}..{CV[:,2].max():.3f}", flush=True)
