"""Where a shipped mesh folds back on itself within a few millimetres: the grooves that render black.

    OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 ../.venv/bin/python tools/marks_grooves.py \
        --mesh work/rowan/retopo.glb [--labels work/rowan/labels.json] [--reach 0.004] [--hops 3] \
        [--json out.json]

A vertex is flagged when a vertex that is NOT within --hops edges of it (so not on its own sheet)
lies within --reach metres of it: two surfaces meet across a gap of a few millimetres, such as the
crease where a sleeve meets the torso at the armpit, or a shoulder blade under a fold. Those grooves
are occluded from the world light and render as the black squiggles on the 0.19 jackets (lane marks,
E-defects). The texture cannot darken or lighten them: they are shading from geometry.

Reports the flagged share of the mesh, the share per part label (when --labels is given), and the
clusters of flagged vertices (centre in the Blender frame, Z up, and extent), largest first. Distances
do not depend on the axis convention, so the merge and the test run on the glTF positions; centres are
printed in the Blender frame (x, -z, y) that uv_maps.py and the lit renders use.

Reads only. Writes nothing unless --json is given.
"""
import argparse
import json
import sys
from collections import deque

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--labels", default=None, help="labels.json: per-vertex part classes (fine 'classes' key)")
ap.add_argument("--reach", type=float, default=0.004, help="metres: a second sheet this close marks a groove")
ap.add_argument("--hops", type=int, default=3, help="edges within which vertices share a sheet")
ap.add_argument("--json", default=None)
ap.add_argument("--clusters", type=int, default=8, help="how many clusters to print")
a = ap.parse_args()


def load_merged(path):
    s = trimesh.load(path)
    g = max(s.geometry.values(), key=lambda m: len(m.faces))
    V = np.asarray(g.vertices, float)
    F = np.asarray(g.faces)
    # glTF export splits vertices at UV and normal seams: merge them by position so the sheets are real
    q = np.round(V / 1e-6).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    Fm = inv[F]
    keep = (Fm[:, 0] != Fm[:, 1]) & (Fm[:, 1] != Fm[:, 2]) & (Fm[:, 0] != Fm[:, 2])
    Fm = Fm[keep]
    Vm = np.zeros((inv.max() + 1, 3))
    np.add.at(Vm, inv, V)
    Vm /= np.bincount(inv)[:, None]
    return Vm, Fm, inv


def main():
    V, F, inv = load_merged(a.mesh)
    n = len(V)
    adj = [set() for _ in range(n)]
    for x, y, z in F:
        adj[x].update((y, z)); adj[y].update((x, z)); adj[z].update((x, y))
    tree = cKDTree(V)
    flagged = np.zeros(n, bool)
    partner = np.full(n, -1)
    for v in range(n):
        nb = tree.query_ball_point(V[v], a.reach)
        if len(nb) <= 1:
            continue
        # vertices within `hops` edges of v share its sheet
        seen = {v}
        frontier = [v]
        for _ in range(a.hops):
            nxt = []
            for x in frontier:
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        nxt.append(y)
            frontier = nxt
        for u in nb:
            if u not in seen:
                flagged[v] = True
                partner[v] = u
                break
    share = float(flagged.mean())
    print(f"[grooves] {a.mesh}: {n} vertices, {int(flagged.sum())} flagged ({share:.2%}) within {a.reach*1000:.1f} mm "
          f"across a {a.hops}-edge sheet", flush=True)
    per_label = {}
    if a.labels:
        J = json.load(open(a.labels))
        raw = np.asarray(J["classes"])
        names = J["class_names"]
        if len(raw) == len(inv):
            # labels are per glTF vertex; each merged vertex takes the label of the vertices it came from
            cls = np.zeros(n, int)
            cls[inv] = raw
            for c in np.unique(cls):
                m = cls == c
                per_label[names.get(str(int(c)), str(int(c)))] = [int(flagged[m].sum()), int(m.sum())]
            print("[grooves] per part (flagged / vertices):",
                  {k: f"{v[0]}/{v[1]}" for k, v in per_label.items() if v[0]}, flush=True)
        else:
            print(f"[grooves] labels cover {len(raw)} vertices, mesh has {len(inv)}: per-part counts skipped", flush=True)
    # clusters of flagged vertices, in the Blender frame (x, -z, y)
    P = V[flagged]
    clusters = []
    if len(P):
        vox = np.floor(P / 0.012).astype(np.int64)
        vox -= vox.min(0)
        grid = np.zeros(vox.max(0) + 1, bool)
        grid[vox[:, 0], vox[:, 1], vox[:, 2]] = True
        lbl, m = ndimage.label(grid, structure=np.ones((3, 3, 3)))
        cl = lbl[vox[:, 0], vox[:, 1], vox[:, 2]]
        sizes = np.bincount(cl)[1:]
        for k in np.argsort(-sizes)[: a.clusters]:
            Q = P[cl == k + 1]
            B = np.stack([Q[:, 0], -Q[:, 2], Q[:, 1]], 1)
            c = B.mean(0)
            ext = B.max(0) - B.min(0)
            clusters.append({"vertices": int(len(Q)), "centre_blender": [round(float(t), 4) for t in c],
                             "extent_m": [round(float(t), 4) for t in ext]})
            print(f"[grooves] cluster {len(clusters)}: {len(Q)} vertices, centre X {c[0]:+.3f} Y {c[1]:+.3f} "
                  f"Z {c[2]:+.3f}, extent {ext[0]*100:.1f}x{ext[1]*100:.1f}x{ext[2]*100:.1f} cm", flush=True)
    if a.json:
        json.dump({"mesh": a.mesh, "vertices": int(n), "flagged": int(flagged.sum()), "share": share,
                   "reach_m": a.reach, "hops": a.hops, "per_part": per_label, "clusters": clusters},
                  open(a.json, "w"), indent=1)
        print(f"[grooves] -> {a.json}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
