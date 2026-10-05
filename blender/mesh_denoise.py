"""Feature-preserving smoothing of a generated surface: bilateral normal filtering (E135).

    blender -b -noaudio --python mesh_denoise.py -- --mesh mesh.glb --out mesh_smooth.glb \
        [--sigma-r 0.35] [--normal-iters 8] [--vertex-iters 20]

A generated surface carries two kinds of roughness a professional model does not: facets left by decimating
TRELLIS's mesh (flat triangles a centimetre or two across, a few degrees apart), and voxel noise. Both are a
few degrees of turn between neighbouring faces; a designed crease - a plate's edge, a collar - is tens of
degrees. Bilateral normal filtering (Zheng et al., 2011) averages each face's normal with its neighbours',
weighted by their area, their distance (sigma_s, the mean spacing of neighbouring faces) and how alike the two
normals are (sigma_r, on the normals' difference), so the first kind is smoothed away and the second kept; the
vertices are then moved to agree with the filtered normals (the update of Sun et al., 2007).

Positions only change: the mesh keeps its vertices, UVs and texture, so it can stand in for the original in the
pipeline. glTF splits vertices along UV seams; they are welded for the filtering and moved together.
"""
import argparse
import sys
import time

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sigma-r", type=float, default=0.35, help="normal difference |n_i - n_j| kept apart (0.35 ~ 20 deg)")
ap.add_argument("--normal-iters", type=int, default=8)
ap.add_argument("--vertex-iters", type=int, default=20)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"
        and not any(c.name == "glTF_not_exported" for c in o.users_collection)]
t0 = time.time()
for o in objs:
    me = o.data
    nv = len(me.vertices)
    V = np.empty(nv * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    me.calc_loop_triangles()
    F = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", F)
    F = F.reshape(-1, 3)
    # weld seam duplicates
    key = np.round(V * 1e6).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    W = V[first].copy()
    T = inv[F]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    m, n = len(T), len(W)

    # faces sharing a vertex: all pairs among each vertex's faces
    fv = T.ravel()
    ff = np.repeat(np.arange(m), 3)
    order = np.argsort(fv, kind="stable")
    fv, ff = fv[order], ff[order]
    starts = np.flatnonzero(np.r_[True, fv[1:] != fv[:-1]])
    counts = np.diff(np.r_[starts, len(fv)])
    rows, cols = [], []
    for k in range(1, counts.max()):                      # pair each face with the k-th face after it, per vertex
        has = counts > k
        idx = starts[has]
        for j in range(k):
            a_, b_ = ff[idx + j], ff[idx + k]
            rows.append(a_); cols.append(b_)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    pair = np.unique(np.stack([np.minimum(rows, cols), np.maximum(rows, cols)], 1), axis=0)
    pair = pair[pair[:, 0] != pair[:, 1]]
    r_, c_ = np.r_[pair[:, 0], pair[:, 1]], np.r_[pair[:, 1], pair[:, 0]]

    def faces(P):
        e1, e2 = P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]
        cr = np.cross(e1, e2)
        ar = np.linalg.norm(cr, axis=1)
        return P[T].mean(1), cr / np.maximum(ar, 1e-20)[:, None], ar / 2

    C, N, A = faces(W)
    sigma_s = float(np.mean(np.linalg.norm(C[r_] - C[c_], axis=1)))
    N0 = N.copy()
    for _ in range(a.normal_iters):
        w = (A[c_] * np.exp(-((C[r_] - C[c_]) ** 2).sum(1) / (2 * sigma_s ** 2))
             * np.exp(-((N[r_] - N[c_]) ** 2).sum(1) / (2 * a.sigma_r ** 2)))
        Nn = np.stack([np.bincount(r_, w * N[c_, i], minlength=m) for i in range(3)], 1) + A[:, None] * N
        N = Nn / np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-20)
    deg = np.bincount(T.ravel(), minlength=n).astype(float)
    P = W.copy()
    for _ in range(a.vertex_iters):
        C, _, _ = faces(P)
        d = np.zeros((n, 3))
        for col in range(3):
            v = T[:, col]
            proj = (N * (C - P[v])).sum(1)[:, None] * N
            for i in range(3):
                d[:, i] += np.bincount(v, proj[:, i], minlength=n)
        P += d / np.maximum(deg, 1)[:, None]
    moved = np.linalg.norm(P - W, axis=1)
    turn = np.degrees(np.arccos(np.clip((N * N0).sum(1), -1, 1)))
    print(f"[denoise] {o.name}: {m:,} faces, sigma_s {sigma_s * 1000:.2f} mm-units, normals turned median "
          f"{np.median(turn):.2f} p90 {np.quantile(turn, 0.9):.2f} deg; vertices moved median {np.median(moved):.5f} "
          f"p99 {np.quantile(moved, 0.99):.5f} (units of the file)", flush=True)
    V_new = P[inv]                                          # every seam duplicate with its welded vertex
    me.vertices.foreach_set("co", V_new.ravel())
    me.update()
bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=False)
print(f"[denoise] done in {time.time() - t0:.1f} s -> {a.out}", flush=True)
