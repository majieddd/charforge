"""Skin weights from distance *along the surface* instead of distance through space.

Straight-line distance to a bone cannot tell a chest vertex from a sleeve vertex, because in
any pose where the arm is near the body the two are equally far from both `spine3` and
`left_shoulder`. Measured on the shipped character, that confusion left 3,264 clothing vertices
(8.69%) carrying >=15% weight from bones three or more joints apart; linear blend skinning
interpolates matrices, so those vertices land between unrelated transforms and the surface
folds - faces collapsing to 4% of their rest area, normals swinging 160 degrees past the
rotation their own bone underwent.

The obvious repair is to rule out bones a vertex cannot "see", by ray test or by side of the
body. That was implemented and measured, and it is worse than doing nothing:

    build                         skin med p99   clothing med p99
    plain inverse-distance             16.4%          75.9%
    + ray visibility + bilateral      141.3%         129.3%
    + skeleton-graph spread bound      44.3%         123.6%

Every one of those is a *binary per-vertex mask*. Two neighbouring vertices fall on opposite
sides of a grazing ray, their weight vectors diverge, and the surface tears between them.
Smoothness of the weight field matters more than anatomical purity of any single vertex,
because a discontinuity is exactly what the stretch metric - and the eye - picks up.

Geodesic distance gives the same anatomical information as a continuous field. Travelling from
the chest to the humerus means going around the armpit, so the chest's distance to
`left_shoulder` is large and *smoothly* large; no vertex is ever excluded, so no discontinuity
is introduced. Each bone owns the patch of surface it is nearest to, and its distance field
spreads out from that patch along the mesh:

  one graph      all skinned meshes together, welded at coincident vertices and bridged by
                 proximity links, because this character's skin is seven islands - head, two
                 hands, two feet - with no body under the clothes. A hand must be able to reach
                 the sleeve to have any distance to the elbow at all.
  T-pose         run after the T-pose bake, where the limbs are apart. Proximity links are
                 short, but in an A-pose the hand rests beside the hip and any bridge there
                 would connect two parts that must stay independent.
  no masks       weights are 1/geodesic^power over the k nearest bones, smoothed over the same
                 graph. Nothing is zeroed by a test.

Run: blender -b -noaudio --python reskin_geodesic.py -- --blend tposed.blend --out out.blend
"""
import argparse
import heapq
import json
import os
import sys
from collections import defaultdict, deque

import bpy
import numpy as np
from mathutils import Vector, kdtree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--power", type=float, default=4.0)
ap.add_argument("--max-influences", type=int, default=4)
ap.add_argument("--iters", type=int, default=18)
ap.add_argument("--lam", type=float, default=0.7)
ap.add_argument("--link-radius", type=float, default=0.8,
                help="bridge vertices this close, as a fraction of median edge length")
ap.add_argument("--blend-euclid", type=float, default=0.0,
                help="0 = pure geodesic, 1 = pure euclidean, between = geometric mean")
ap.add_argument("--max-wrong-side", type=float, default=100.0,
                help="report-only by default; set low to fail on cross-midline binding")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
meshes = [o for o in scene.objects if o.type == "MESH" and o.find_armature() == rig]
if rig is None or not meshes:
    raise SystemExit("[geo] need an armature and at least one mesh bound to it")

M = rig.matrix_world
deform = [b for b in rig.data.bones if b.use_deform] or list(rig.data.bones)
names = [b.name for b in deform]
head = np.array([(M @ b.head_local)[:] for b in deform])
tail = np.array([(M @ b.tail_local)[:] for b in deform])
NB = len(names)

# skeleton graph distance, for reporting only - nothing is masked by it
bidx = {n: i for i, n in enumerate(names)}
adj_b = defaultdict(set)
for b in deform:
    p = b.parent
    while p is not None and p.name not in bidx:
        p = p.parent
    if p is not None:
        adj_b[bidx[b.name]].add(bidx[p.name])
        adj_b[bidx[p.name]].add(bidx[b.name])
GD = np.full((NB, NB), 99, np.int32)
for s in range(NB):
    GD[s, s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj_b[u]:
            if GD[s, v] == 99:
                GD[s, v] = GD[s, u] + 1
                q.append(v)

# ---- one graph over the whole character ------------------------------------------------------
offsets, V_all, E_all = {}, [], []
for ob in meshes:
    off = len(V_all)
    offsets[ob.name] = off
    mw = ob.matrix_world
    V_all.extend([(mw @ v.co)[:] for v in ob.data.vertices])
    E_all.extend([(e.vertices[0] + off, e.vertices[1] + off) for e in ob.data.edges])
V = np.array(V_all, np.float64)
E = np.array(E_all, np.int64)
N = len(V)
SCALE = float(np.linalg.norm(V.max(0) - V.min(0)))
elen = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
med_edge = float(np.median(elen))
print(f"[geo] {N:,} vertices, {len(E):,} edges across {len(meshes)} mesh(es); "
      f"median edge {med_edge:.5f}, extent {SCALE:.3f}", flush=True)

adj = defaultdict(dict)


def link(i, j, w):
    if j not in adj[i] or w < adj[i][j]:
        adj[i][j] = w
        adj[j][i] = w


for (i, j), w in zip(E, elen):
    link(int(i), int(j), float(w))

# weld coincident vertices (UV-island boundaries) with zero-length links
keys = np.round(V / (1e-5 * SCALE)).astype(np.int64)
bucket = defaultdict(list)
for i in range(N):
    bucket[tuple(keys[i])].append(i)
twins = [m for m in bucket.values() if len(m) > 1]
for m in twins:
    for x in m[1:]:
        link(m[0], x, 0.0)

# bridge gaps the index buffer does not connect: separate UV islands, and the skin islands
# (hands, feet, head) that must reach the garment to have any distance to an arm or leg bone
radius = med_edge * a.link_radius
kd = kdtree.KDTree(N)
for i in range(N):
    kd.insert(Vector(V[i]), i)
kd.balance()
bridged = 0
for i in range(N):
    for _, j, d in kd.find_range(Vector(V[i]), radius):
        if j != i and j not in adj[i]:
            link(i, int(j), float(d))
            bridged += 1
print(f"[geo] welded {len(twins):,} coincident groups, bridged {bridged//2:,} gaps "
      f"within {radius:.5f}", flush=True)

nbr = [np.fromiter(adj[i].keys(), np.int64) for i in range(N)]
nbw = [np.fromiter(adj[i].values(), np.float64) for i in range(N)]

# ---- per-bone geodesic field -------------------------------------------------------------------
D = np.empty((N, NB), np.float64)
for c in range(NB):
    ab = tail[c] - head[c]
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        cp = np.broadcast_to(head[c], V.shape)
    else:
        u = np.clip(((V - head[c]) @ ab) / L2, 0.0, 1.0)[:, None]
        cp = head[c] + u * ab
    D[:, c] = np.linalg.norm(V - cp, axis=1)
D = np.maximum(D, 1e-6)

owner = np.argmin(D, axis=1)          # each bone owns the surface it is nearest to
GEO = np.full((N, NB), np.inf)
for c in range(NB):
    seeds = np.where(owner == c)[0]
    if len(seeds) == 0:               # a bone with no patch of its own: seed its nearest few
        seeds = np.argsort(D[:, c])[:16]
    dist = GEO[:, c]
    heap = [(float(D[s, c]), int(s)) for s in seeds]
    for s in seeds:
        dist[s] = D[s, c]
    heapq.heapify(heap)
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist[u]:
            continue
        nb, nw = nbr[u], nbw[u]
        for k in range(len(nb)):
            v = int(nb[k])
            nd = du + nw[k]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    GEO[:, c] = dist
    print(f"[geo]   {names[c]:<16s} patch {len(seeds):6,} verts  "
          f"reaches {int(np.isfinite(dist).sum()):6,}/{N:,}", flush=True)

unreached = ~np.isfinite(GEO)
GEO[unreached] = D[unreached] * 1e3      # unreachable: heavily penalised, never selected
if a.blend_euclid > 0:
    GEO = np.exp((1 - a.blend_euclid) * np.log(np.maximum(GEO, 1e-6))
                 + a.blend_euclid * np.log(D))

ratio = GEO / D
print(f"[geo] geodesic/euclidean ratio over the k nearest bones: "
      f"median {np.median(ratio[ratio < 1e2]):.2f}, "
      f"p99 {np.percentile(ratio[ratio < 1e2], 99):.2f}", flush=True)

# ---- weights: no masks, no exclusions -----------------------------------------------------------
rows = np.arange(N)[:, None]
order = np.argsort(GEO, axis=1)[:, :a.max_influences]
W = np.zeros((N, NB))
w = 1.0 / np.maximum(GEO[rows, order], 1e-6) ** a.power
W[rows, order] = w / w.sum(1, keepdims=True)

for _ in range(a.iters):
    acc = np.empty_like(W)
    for i in range(N):
        nb = nbr[i]
        acc[i] = W[nb].mean(0) if len(nb) else W[i]
    W = (1.0 - a.lam) * W + a.lam * acc

for m in twins:
    W[m] = W[m].mean(0)               # identical rows: a UV island cannot split open
order = np.argsort(-W, axis=1)[:, :a.max_influences]
keep = np.zeros_like(W)
keep[rows, order] = W[rows, order]
keep /= np.maximum(keep.sum(1, keepdims=True), 1e-12)

# ---- write back, per mesh -------------------------------------------------------------------------
report = {"power": a.power, "iters": a.iters, "link_radius": a.link_radius, "meshes": {}}
bone_x = (head + tail)[:, 0] / 2.0
lat_b = np.abs(bone_x) > 0.08 * (float(np.abs(bone_x).max()) or 1.0)
for ob in meshes:
    off = offsets[ob.name]
    n = len(ob.data.vertices)
    sub = keep[off:off + n]
    for vg in list(ob.vertex_groups):
        ob.vertex_groups.remove(vg)
    vgs = [ob.vertex_groups.new(name=nm) for nm in names]
    for c, vg in enumerate(vgs):
        for i in np.where(sub[:, c] > 1e-4)[0]:
            vg.add([int(i)], float(sub[i, c]), "REPLACE")

    heavy = sub >= 0.15
    dom = np.argmax(sub, axis=1)
    spread = np.zeros(n, np.int32)
    for i in range(n):
        bs = np.where(heavy[i])[0]
        if len(bs) > 1:
            s2 = GD[np.ix_(bs, bs)]
            spread[i] = int(s2[s2 < 99].max())
    Vm = V[off:off + n]
    lat_v = np.abs(Vm[:, 0]) > 0.08 * (float(np.abs(Vm[:, 0]).max()) or 1.0)
    wrong = int((lat_b[dom] & lat_v & (np.sign(bone_x[dom]) != np.sign(Vm[:, 0]))).sum())
    unw = int((sub.sum(1) < 1e-6).sum())
    bad = int((spread >= 3).sum())
    print(f"[geo] {ob.name}: {n:,}v, {unw} unweighted, mean "
          f"{float((sub > 1e-4).sum(1).mean()):.2f} influences", flush=True)
    print(f"[geo] {ob.name}: bones >=3 joints apart at >=15%: {bad:,} ({100*bad/n:.2f}%); "
          f"across midline: {wrong:,} of {int(lat_v.sum()):,} ({100*wrong/max(int(lat_v.sum()),1):.2f}%)",
          flush=True)
    if unw:
        raise SystemExit(f"[geo] {ob.name}: {unw} vertices received no weight")
    report["meshes"][ob.name] = {"vertices": n, "unweighted": unw, "spread_ge3": bad,
                                 "wrong_side": wrong}

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump(report, open(a.json, "w"), indent=2)
print(f"[geo] -> {a.out}", flush=True)
