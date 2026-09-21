"""Re-solve skin weights so a vertex is only bound to bones it can actually see.

The weights this replaces were built from Euclidean distance to bone segments, solved on the
A-pose rest (shoulders 29.4 degrees from down) and only afterwards baked to a T-pose. In that
A-pose the upper-arm bone runs alongside the ribcage, so a vertex on the chest and a vertex on
the inside of the sleeve are the same distance from both `spine3` and `left_shoulder`. The
solver could not tell them apart, and 18 rounds of Laplacian smoothing then spread the
confusion further along the fabric - a jacket is one continuous surface from cuff to cuff, so
arm weight bleeds through the shoulder into the chest and back out the other sleeve.

Measured on the shipped character: 3,264 clothing vertices (8.69%) carry >=15% weight from two
bones three or more joints apart in the skeleton. Linear blend skinning interpolates *matrices*,
so such a vertex lands between two unrelated transforms. That is the collapse the audit found -
faces shrinking to 4% of their rest area, and face normals swinging up to 160 degrees past the
rotation their own bone underwent.

Three changes, in order of how much they matter:

  visibility    a bone is a candidate for a vertex only if the straight line from the vertex to
                the nearest point on that bone segment is not blocked by the character's own
                surface. From a point on the chest, the upper-arm bone is behind the arm; from
                the inside of the sleeve, the spine is behind the ribcage. This is the idea
                Blender's bone heat is built on, minus the Laplacian solve that fails outright
                on an open shell like a jacket (it did fail here, twice).
                Hits nearer than a quarter of the distance to the closest bone are ignored, so
                the far side of a thin garment shell does not occlude everything. That floor
                scales with limb radius rather than being a fixed number tuned to one mesh.
  graph bound   influences are restricted to bones within `--hops` joints of the vertex's
                dominant bone, re-applied after *every* smoothing iteration. Smoothing can then
                soften a boundary without ever reconnecting the chest to the elbow.
  weld-consistent  coincident vertices (this mesh is 38-53% coincident, one pair per UV island
                boundary) are averaged and written identically, so an island cannot split open.

Run: blender -b -noaudio --python reskin_visible.py -- --blend in.blend --out out.blend
"""
import argparse
import json
import os
import sys
from collections import defaultdict, deque

import bpy
import numpy as np
from mathutils import Vector
from mathutils import kdtree
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--hops", type=int, default=2,
                help="max skeleton-graph distance from a vertex's dominant bone, during smoothing")
ap.add_argument("--max-spread", type=int, default=2,
                help="max skeleton-graph distance between ANY TWO bones in the final weight set")
ap.add_argument("--power", type=float, default=3.0)
ap.add_argument("--max-influences", type=int, default=4)
ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--lam", type=float, default=0.6)
ap.add_argument("--vote-rounds", type=int, default=3)
ap.add_argument("--occl-floor", type=float, default=0.25,
                help="ignore occluders nearer than this fraction of the closest-bone distance")
ap.add_argument("--cand-ratio", type=float, default=4.0,
                help="only ray-test bones within this multiple of the closest bone's distance")
ap.add_argument("--facing-tol", type=float, default=0.0,
                help="a bone counts as behind the surface when dot(to_bone, normal) < tol*dist")
ap.add_argument("--link-radius", type=float, default=0.6,
                help="link vertices this close (as a fraction of median edge length) for smoothing")
ap.add_argument("--lateral-frac", type=float, default=0.08,
                help="a bone/vertex counts as off-midline past this fraction of the body's half-width")
ap.add_argument("--plain", action="store_true",
                help="control: plain inverse-distance weights, every added constraint disabled")
ap.add_argument("--max-wrong-side", type=float, default=1.0,
                help="fail if more than this %% of lateral vertices bind across the midline")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
meshes = [o for o in scene.objects if o.type == "MESH" and o.find_armature() == rig]
if rig is None or not meshes:
    raise SystemExit("[reskin] need an armature and at least one mesh bound to it")
print(f"[reskin] {len(meshes)} skinned mesh(es): "
      + ", ".join(f"{o.name} ({len(o.data.vertices):,}v)" for o in meshes), flush=True)

# ---- skeleton ------------------------------------------------------------------------------
M = rig.matrix_world
deform = [b for b in rig.data.bones if b.use_deform] or list(rig.data.bones)
names = [b.name for b in deform]
bidx = {n: i for i, n in enumerate(names)}
head = np.array([(M @ b.head_local)[:] for b in deform])
tail = np.array([(M @ b.tail_local)[:] for b in deform])
NB = len(names)

adj_b = defaultdict(set)
for b in deform:
    p = b.parent
    while p is not None and p.name not in bidx:
        p = p.parent                      # skip non-deform bones, keep the chain connected
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
print(f"[reskin] skeleton: {NB} deform bones, graph diameter {int(GD[GD < 99].max())}", flush=True)

# ---- one BVH over the whole character: occluders come from any part -------------------------
bvh_verts, bvh_faces = [], []
for ob in meshes:
    off = len(bvh_verts)
    mw = ob.matrix_world
    bvh_verts.extend([mw @ v.co for v in ob.data.vertices])
    bvh_faces.extend([[i + off for i in p.vertices] for p in ob.data.polygons])
bvh = BVHTree.FromPolygons(bvh_verts, bvh_faces, all_triangles=False, epsilon=0.0)
allV = np.array([v[:] for v in bvh_verts])
SCALE = float(np.linalg.norm(allV.max(0) - allV.min(0)))
print(f"[reskin] occlusion BVH: {len(bvh_verts):,} verts, {len(bvh_faces):,} faces, "
      f"extent {SCALE:.3f}", flush=True)

report = {"hops": a.hops, "power": a.power, "iters": a.iters, "meshes": {}}

for ob in meshes:
    me = ob.data
    n = len(me.vertices)
    V = np.array([(ob.matrix_world @ v.co)[:] for v in me.vertices])

    # ---- distance to every bone segment ------------------------------------------------------
    D = np.empty((n, NB), np.float64)
    CP = np.empty((n, NB, 3), np.float64)
    for c in range(NB):
        ab = tail[c] - head[c]
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            CP[:, c] = head[c]
        else:
            u = np.clip(((V - head[c]) @ ab) / L2, 0.0, 1.0)[:, None]
            CP[:, c] = head[c] + u * ab
        D[:, c] = np.linalg.norm(V - CP[:, c], axis=1)
    D = np.maximum(D, 1e-6)
    dmin = D.min(1)

    # ---- visibility: can this vertex see that bone? -------------------------------------------
    # Two conditions, and the second is not optional. An unobstructed line of sight alone put
    # 23% of one leg's vertices on the *other* leg's bones: from a point on the inner thigh the
    # ray to the opposite femur crosses the gap between the legs and hits nothing at all, so the
    # wrong bone looked perfectly visible. A bone that deforms a surface point lies behind that
    # surface, so the direction to it must oppose the outward normal.
    NRM = np.empty(n * 3, np.float64)
    me.vertices.foreach_get("normal", NRM)
    NRM = (np.array(ob.matrix_world.to_3x3().normalized()) @ NRM.reshape(-1, 3).T).T

    to_bone = CP - V[:, None, :]
    facing = (to_bone * NRM[:, None, :]).sum(2) < a.facing_tol * D
    if a.plain:
        facing = np.ones_like(facing)

    # Bilateral prior. Ray visibility cannot settle laterality on its own: the gap between the
    # legs is narrower than a limb radius, and stray skin islands near the feet have no
    # neighbouring surface to block a ray at all, so the opposite ankle stays "visible" from the
    # outer edge of a foot. That a biped's left surface is driven by its left bones is not a
    # heuristic though - it is a fact about the skeleton, and it is cheap to state directly.
    # Sides are read off the rig, so this holds for any bilaterally symmetric character.
    bone_x = (head + tail)[:, 0] / 2.0
    lat_b = np.abs(bone_x) > a.lateral_frac * (float(np.abs(bone_x).max()) or 1.0)
    lat_v = np.abs(V[:, 0]) > a.lateral_frac * (float(np.abs(V[:, 0]).max()) or 1.0)
    opposite = (np.sign(V[:, 0])[:, None] * np.sign(bone_x)[None, :] < 0)
    same_side = ~(opposite & lat_b[None, :] & lat_v[:, None])

    if a.plain:
        same_side = np.ones_like(same_side)
    cand = (D <= (dmin * a.cand_ratio)[:, None]) & facing & same_side
    print(f"[reskin] {ob.name}: bilateral prior removed "
          f"{int((~same_side & (D <= (dmin * a.cand_ratio)[:, None])).sum()):,} "
          f"opposite-side bone candidates", flush=True)
    visible = np.zeros((n, NB), bool)
    eps = 1e-4 * SCALE
    tested = 0
    nfall = 0
    if a.plain:
        # k nearest bone segments by distance, exactly as the replaced solver chose them
        visible[np.arange(n)[:, None], np.argsort(D, axis=1)[:, :a.max_influences]] = True
    for i in (range(n) if not a.plain else ()):
        origin = Vector(V[i])
        for c in np.where(cand[i])[0]:
            d = float(D[i, c])
            direction = Vector((CP[i, c] - V[i]) / d)
            tested += 1
            # Walk the ray and look for a FRONT-facing hit. Which way the hit surface faces is
            # what separates "this bone is inside my own limb" from "this bone is across a gap":
            # entering another volume hits its outer surface front-on (normal opposes the ray),
            # while passing out through the far side of my own thin shell hits it from behind.
            # A plain distance floor cannot tell those apart - tuned wide enough to forgive a
            # garment shell it also forgives the gap between the legs, which is how 23% of one
            # leg ended up bound to the other.
            t, blocked = eps, False
            for _ in range(6):
                hit = bvh.ray_cast(origin + direction * t, direction, d - t)
                if hit[0] is None or hit[3] is None:
                    break
                t += hit[3] + eps
                if t >= d - eps:
                    break
                if hit[1].dot(direction) < 0.0:      # front-facing: a genuine occluder
                    blocked = True
                    break
            if not blocked:
                visible[i, c] = True
        if not visible[i].any():
            # Nothing passed. This happens where a vertex sits almost *on* its own bone - a thin
            # ankle or wrist - because the candidate radius is a multiple of the closest-bone
            # distance, which has collapsed to near zero, and the normal test is ill-conditioned
            # when the bone is that close. Degrade in order of how much is being given up, and
            # never past the bilateral prior: an earlier version fell straight back to "nearest
            # bone that faces inward", which handed 345 shin vertices to the opposite knee at
            # weight 0.97 and tore the legs apart.
            # Every pool is bounded by distance. Without that bound the fallback once handed a
            # vertex on top of the foot to `right_wrist` at weight 0.64, because the ankle joint
            # sits *above* the top of the foot and so failed the normal test while some far bone
            # happened to pass it. Whatever else is given up, a vertex is deformed by a bone near
            # it; the last pool always contains the nearest bone, so this never falls through.
            near = D[i] <= dmin[i] * a.cand_ratio
            nfall += 1
            for pool in (np.where(facing[i] & same_side[i] & near)[0],
                         np.where(same_side[i] & near)[0],
                         np.where(near)[0]):
                if len(pool):
                    visible[i, int(pool[np.argmin(D[i, pool])])] = True
                    break
    seen = visible.sum(1)
    print(f"[reskin] {ob.name}: {nfall:,} vertices ({100*nfall/n:.2f}%) saw no bone and used "
          f"the bounded fallback", flush=True)
    print(f"[reskin] {ob.name}: normal test removed "
          f"{int((~facing & (D <= (dmin * a.cand_ratio)[:, None])).sum()):,} "
          f"bone candidates that sat in front of the surface", flush=True)
    print(f"[reskin] {ob.name}: {tested:,} rays, visible bones per vertex "
          f"mean {seen.mean():.2f} (was {cand.sum(1).mean():.2f} by distance alone)", flush=True)

    # ---- weights from visible bones only -----------------------------------------------------
    Dv = np.where(visible, D, np.inf)
    Wm = np.zeros((n, NB))
    inv = np.where(np.isfinite(Dv), 1.0 / np.maximum(Dv, 1e-6) ** a.power, 0.0)
    Wm = inv / np.maximum(inv.sum(1, keepdims=True), 1e-12)

    # ---- welded adjacency --------------------------------------------------------------------
    keys = np.round(V / (1e-5 * SCALE)).astype(np.int64)
    adj = [set() for _ in range(n)]
    for e in me.edges:
        i, j = e.vertices
        adj[i].add(j)
        adj[j].add(i)
    bucket = defaultdict(list)
    for i in range(n):
        bucket[tuple(keys[i])].append(i)
    twins = [m for m in bucket.values() if len(m) > 1]
    for m in twins:
        u = set()
        for x in m:
            u |= adj[x]
        for x in m:
            adj[x] = u

    # Exact coincidence is not enough. Two vertices 2.3mm apart - a fifth of one edge - sat in
    # different UV islands with no edge between them and no weld, so smoothing could never
    # equalise them: one ended up on `left_hand` and the other on `left_hip`, and that 2.3mm
    # edge stretched to 286mm under animation. Whether two points on a surface should deform
    # alike is a question about distance, not about the index buffer, so the smoothing graph
    # gets a proximity link wherever two vertices are closer than a fraction of the mesh's own
    # median edge length.
    rest_len = np.linalg.norm(V[[e.vertices[0] for e in me.edges]]
                              - V[[e.vertices[1] for e in me.edges]], axis=1)
    radius = float(np.median(rest_len)) * a.link_radius
    kd = kdtree.KDTree(n)
    for i in range(n):
        kd.insert(Vector(V[i]), i)
    kd.balance()
    linked = 0
    for i in range(n):
        for _, j, _ in kd.find_range(Vector(V[i]), radius):
            if j != i and j not in adj[i]:
                adj[i].add(j)
                adj[j].add(i)
                linked += 1
    print(f"[reskin] {ob.name}: median edge {np.median(rest_len):.5f}, {linked//2:,} proximity "
          f"links added within {radius:.5f} that the index buffer did not connect", flush=True)
    nbr = [np.fromiter(adj[i], dtype=np.int64) for i in range(n)]

    # ---- dominant bone, majority-voted over the surface ---------------------------------------
    dom = np.argmax(Wm, axis=1)
    for _ in range(a.vote_rounds):
        nxt = dom.copy()
        for i in range(n):
            nb = nbr[i]
            if len(nb) == 0:
                continue
            cnt = np.bincount(dom[nb], minlength=NB)
            best = int(cnt.argmax())
            if cnt[best] * 2 > len(nb):
                nxt[i] = best
        dom = nxt

    allowed = GD[dom] <= a.hops             # (n, NB) bool

    # ---- smooth, re-masking onto the allowed set each iteration --------------------------------
    Wm = np.where(allowed, Wm, 0.0)
    Wm /= np.maximum(Wm.sum(1, keepdims=True), 1e-12)
    for _ in range(a.iters):
        acc = np.empty_like(Wm)
        for i in range(n):
            nb = nbr[i]
            acc[i] = Wm[nb].mean(0) if len(nb) else Wm[i]
        Wm = (1.0 - a.lam) * Wm + a.lam * acc
        Wm = np.where(allowed, Wm, 0.0)
        Wm /= np.maximum(Wm.sum(1, keepdims=True), 1e-12)

    # ---- top-k under a pairwise skeleton-diameter bound -----------------------------------------
    # `allowed` bounds each influence's distance to the *dominant* bone, which still lets two
    # influences sit 2*hops apart through it - with hops=2 that is a spread of 4, exactly the
    # defect this script exists to remove. So the final set is built greedily instead: take the
    # strongest bone, then add each next-strongest only while every pair in the set stays within
    # `--max-spread` joints of each other.
    # Coincident vertices are unified FIRST, before the selection. Averaging two already-pruned
    # rows takes the *union* of their influence sets, which silently breaks both bounds this
    # section exists to enforce - it produced vertices carrying nine influences spanning
    # left_hand to left_ankle, and those were the 12,000% edges in the audit. Averaging first
    # leaves every twin with an identical row, so the greedy pass below makes the identical
    # choice for each of them and the island still cannot split open.
    for m in twins:
        Wm[m] = Wm[m].mean(0)

    keep = np.zeros_like(Wm)
    ranked = np.argsort(-Wm, axis=1)
    for i in range(n):
        chosen = []
        for c in ranked[i]:
            if Wm[i, c] <= 1e-6 or len(chosen) >= a.max_influences:
                break
            if all(GD[c, o] <= a.max_spread for o in chosen):
                chosen.append(int(c))
        for c in chosen:
            keep[i, c] = Wm[i, c]
    keep /= np.maximum(keep.sum(1, keepdims=True), 1e-12)

    # ---- write back ----------------------------------------------------------------------------
    for vg in list(ob.vertex_groups):
        ob.vertex_groups.remove(vg)
    vgs = [ob.vertex_groups.new(name=nm) for nm in names]
    for c, vg in enumerate(vgs):
        idx = np.where(keep[:, c] > 1e-4)[0]
        for i in idx:
            vg.add([int(i)], float(keep[i, c]), "REPLACE")

    # ---- report the thing this script exists to fix ---------------------------------------------
    heavy = keep >= 0.15
    spread = np.zeros(n, np.int32)
    for i in range(n):
        bs = np.where(heavy[i])[0]
        if len(bs) > 1:
            sub = GD[np.ix_(bs, bs)]
            spread[i] = int(sub[sub < 99].max())
    bad = int((spread >= 3).sum())
    unw = int((keep.sum(1) < 1e-6).sum())

    # Verify the array that was actually written, not the one that was intended. Both of these
    # bounds were silently violated by an ordering mistake that no intermediate print revealed.
    n_inf = (keep > 1e-4).sum(1)
    if int(n_inf.max()) > a.max_influences:
        raise SystemExit(f"[reskin] {ob.name}: {int((n_inf > a.max_influences).sum())} vertices "
                         f"exceed {a.max_influences} influences (max {int(n_inf.max())})")
    if bad and a.max_spread < 3:
        raise SystemExit(f"[reskin] {ob.name}: {bad} vertices blend bones >=3 joints apart "
                         f"despite a max-spread of {a.max_spread}")

    # the specific confusion this script targets: torso geometry bound to an arm, or vice versa
    TORSO = [bidx[x] for x in ("spine1", "spine2", "spine3", "pelvis") if x in bidx]
    ARM = [bidx[x] for x in names if any(k in x for k in ("shoulder", "elbow", "wrist", "hand"))]
    cross = 0
    if TORSO and ARM:
        dom_t = np.isin(dom, TORSO)
        cross = int((dom_t & heavy[:, ARM].any(1)).sum())
        dom_a = np.isin(dom, ARM)
        cross += int((dom_a & heavy[:, TORSO].any(1)).sum())
    print(f"[reskin] {ob.name}: torso<->arm cross-binding at >=15% weight: {cross:,} "
          f"({100*cross/n:.2f}%)", flush=True)

    # ---- laterality gate ------------------------------------------------------------------
    # A vertex well to one side of the midline must not be driven by a bone on the other side.
    # Which axis is lateral and which sign is which limb are both read off the skeleton, so
    # this holds for any rig, not just this one.
    off = lat_v
    wrong = 0
    if lat_b.any() and off.any():
        dom_lat = lat_b[dom]
        opposite_sel = dom_lat & off & (np.sign(bone_x[dom]) != np.sign(V[:, 0]))
        wrong = int(opposite_sel.sum())
    if wrong and os.environ.get("RESKIN_DEBUG"):
        w = np.where(opposite_sel)[0][:12]
        print("[reskin] DEBUG offenders (vx, dom bone, bone_x, cand_ok, voted_from):", flush=True)
        for i in w:
            print(f"[reskin]   x={V[i,0]:+.3f} z={V[i,2]:+.3f} dom={names[dom[i]]} "
                  f"bone_x={bone_x[dom[i]]:+.3f} cand_allowed={bool(cand[i, dom[i]])} "
                  f"visible={bool(visible[i, dom[i]])} same_side={bool(same_side[i, dom[i]])} "
                  f"w={keep[i, dom[i]]:.3f}", flush=True)
    frac = 100.0 * wrong / max(int(off.sum()), 1)
    print(f"[reskin] {ob.name}: vertices bound across the midline: {wrong:,} of "
          f"{int(off.sum()):,} lateral ({frac:.2f}%)", flush=True)
    if frac > a.max_wrong_side:
        raise SystemExit(
            f"[reskin] {ob.name}: {frac:.2f}% of lateral vertices are driven by a bone on the "
            f"opposite side of the body (limit {a.max_wrong_side}%). Left/right confusion "
            f"explodes the mesh under animation - refusing to write these weights.")
    print(f"[reskin] {ob.name}: {len(twins):,} coincident groups unified, "
          f"{unw} unweighted, mean {int((keep>1e-4).sum(1).mean()*100)/100:.2f} influences",
          flush=True)
    print(f"[reskin] {ob.name}: vertices blending bones >=3 joints apart at >=15% weight: "
          f"{bad:,} ({100*bad/n:.2f}%)", flush=True)
    if unw:
        raise SystemExit(f"[reskin] {ob.name}: {unw} vertices received no weight")
    report["meshes"][ob.name] = {
        "vertices": n, "rays": int(tested), "visible_mean": round(float(seen.mean()), 3),
        "coincident_groups": len(twins), "unweighted": unw,
        "spread_ge3": bad, "spread_ge3_pct": round(100 * bad / n, 3), "torso_arm_cross": cross, "wrong_side": wrong, "wrong_side_pct": round(frac, 3),
    }

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump(report, open(a.json, "w"), indent=2)
print(f"[reskin] -> {a.out}", flush=True)
