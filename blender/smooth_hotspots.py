"""Soften the weight field only where it is steep, and by a continuously varying amount.

`collapse_why.py` measures what collapsing faces have in common. Steepness of the weight
gradient across the face carries a lift of 2.48x and accounts for a quarter of all collapses:
the face's corners follow different bones, so it shears and loses area.

Smoothing the weights is the obvious response, and it has failed twice on this project - once
globally (smoothing everything regressed the median, because the field was already smooth) and
once through masks (ray visibility, a bilateral prior, a skeleton-graph bound), each of which
took the skin's median per-bone p99 stretch from 16.4% to between 44% and 141%. The lesson from
both is the same: **a binary decision per vertex is the thing that tears.** Two neighbours land
on opposite sides of a threshold, their weights diverge, and the surface opens between them.

So the selection here is not a selection. Every vertex gets a smoothing strength `alpha` that
varies continuously with how steep its neighbourhood is - zero over the flat majority of the
mesh, rising smoothly to full only in the steepest tail. A vertex just below the hot region and
one just inside it receive almost the same treatment, so no new discontinuity is created
anywhere. That is the whole design.

    blender -b -noaudio --python smooth_hotspots.py -- --blend in.blend --out out.blend
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--meshes", default="clothing",
                help="comma-separated name fragments; the skin is left alone by default")
ap.add_argument("--lo-pct", type=float, default=75.0,
                help="gradient percentile where smoothing starts to ramp in (alpha=0)")
ap.add_argument("--hi-pct", type=float, default=97.0,
                help="gradient percentile where smoothing reaches full strength (alpha=1)")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--max-influences", type=int, default=4)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next(o for o in scene.objects if o.type == "ARMATURE")
keys = [k.strip().lower() for k in a.meshes.split(",") if k.strip()]
meshes = [o for o in scene.objects
          if o.type == "MESH" and o.find_armature() == rig
          and any(k in o.name.lower() for k in keys)]
if not meshes:
    raise SystemExit(f"[hot] no mesh matched {keys}")

report = {"lo_pct": a.lo_pct, "hi_pct": a.hi_pct, "iters": a.iters, "lam": a.lam, "meshes": {}}

for ob in meshes:
    me = ob.data
    n = len(me.vertices)
    names = [vg.name for vg in ob.vertex_groups]
    NB = len(names)

    W = np.zeros((n, NB))
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    W0 = W.copy()

    # ---- adjacency, welded and proximity-linked ------------------------------------------
    # glTF splits a vertex at every UV seam, and this mesh is 38% coincident. Smoothing over the
    # raw index graph would run inside thousands of disconnected patches.
    V = np.empty(n * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    ext = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    adj = [set() for _ in range(n)]
    for e in me.edges:
        i, j = e.vertices
        adj[i].add(j)
        adj[j].add(i)
    bucket = defaultdict(list)
    kq = np.round(V / (1e-5 * ext)).astype(np.int64)
    for i in range(n):
        bucket[tuple(kq[i])].append(i)
    twins = [m for m in bucket.values() if len(m) > 1]
    for m in twins:
        u = set()
        for x in m:
            u |= adj[x]
        for x in m:
            adj[x] = u
    nbr = [np.fromiter(adj[i], np.int64) for i in range(n)]

    # ---- how steep is the field around each vertex? ---------------------------------------
    grad = np.zeros(n)
    for i in range(n):
        nb = nbr[i]
        if len(nb) == 0:
            continue
        grad[i] = np.abs(W[nb] - W[i]).sum(1).max()

    lo, hi = np.percentile(grad, a.lo_pct), np.percentile(grad, a.hi_pct)
    if hi - lo < 1e-9:
        raise SystemExit("[hot] gradient has no spread; nothing to target")
    t = np.clip((grad - lo) / (hi - lo), 0.0, 1.0)
    alpha = t * t * (3.0 - 2.0 * t)          # smoothstep: continuous, and flat at both ends
    # coincident vertices must receive identical treatment or the island splits
    for m in twins:
        alpha[m] = alpha[m].max()

    print(f"[hot] {ob.name}: gradient p50 {np.percentile(grad,50):.3f} "
          f"p{a.lo_pct:.0f} {lo:.3f} p{a.hi_pct:.0f} {hi:.3f} max {grad.max():.3f}", flush=True)
    print(f"[hot] {ob.name}: alpha>0 on {int((alpha>0).sum()):,} vertices ({100*(alpha>0).mean():.1f}%), "
          f"alpha>0.5 on {int((alpha>0.5).sum()):,} ({100*(alpha>0.5).mean():.1f}%)", flush=True)

    # ---- smooth, weighted by alpha ---------------------------------------------------------
    A = alpha[:, None]
    for _ in range(a.iters):
        acc = np.empty_like(W)
        for i in range(n):
            nb = nbr[i]
            acc[i] = W[nb].mean(0) if len(nb) else W[i]
        W = W + A * a.lam * (acc - W)
        W = np.maximum(W, 0.0)
        W /= np.maximum(W.sum(1, keepdims=True), 1e-12)

    for m in twins:
        W[m] = W[m].mean(0)
    rows = np.arange(n)[:, None]
    order = np.argsort(-W, axis=1)[:, :a.max_influences]
    keep = np.zeros_like(W)
    keep[rows, order] = W[rows, order]
    keep /= np.maximum(keep.sum(1, keepdims=True), 1e-12)

    moved = np.abs(keep - W0).sum(1)
    print(f"[hot] {ob.name}: weights moved by L1 mean {moved.mean():.4f}, "
          f"p99 {np.percentile(moved,99):.4f}, max {moved.max():.4f}", flush=True)
    unw = int((keep.sum(1) < 1e-6).sum())
    if unw:
        raise SystemExit(f"[hot] {ob.name}: {unw} vertices lost all weight")

    # the whole point: the steep tail should be flatter, the flat majority untouched
    g2 = np.zeros(n)
    for i in range(n):
        nb = nbr[i]
        if len(nb):
            g2[i] = np.abs(keep[nb] - keep[i]).sum(1).max()
    print(f"[hot] {ob.name}: gradient p99 {np.percentile(grad,99):.3f} -> "
          f"{np.percentile(g2,99):.3f}   p50 {np.percentile(grad,50):.3f} -> "
          f"{np.percentile(g2,50):.3f} (should barely move)", flush=True)

    for vg in list(ob.vertex_groups):
        ob.vertex_groups.remove(vg)
    vgs = [ob.vertex_groups.new(name=nm) for nm in names]
    for c, vg in enumerate(vgs):
        for i in np.where(keep[:, c] > 1e-4)[0]:
            vg.add([int(i)], float(keep[i, c]), "REPLACE")

    report["meshes"][ob.name] = {
        "vertices": n, "alpha_gt0": int((alpha > 0).sum()),
        "grad_p99_before": round(float(np.percentile(grad, 99)), 4),
        "grad_p99_after": round(float(np.percentile(g2, 99)), 4),
        "grad_p50_before": round(float(np.percentile(grad, 50)), 4),
        "grad_p50_after": round(float(np.percentile(g2, 50)), 4),
        "l1_moved_mean": round(float(moved.mean()), 5),
    }

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump(report, open(a.json, "w"), indent=2)
print(f"[hot] -> {a.out}", flush=True)
