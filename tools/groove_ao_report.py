"""Occlusion at the groove vertices: how dark the flagged crevices are, from tools/groove_ao.py's output.

    ../.venv/bin/python tools/groove_ao_report.py --mesh proxy.glb --ao ao.npz [--reach 0.004] [--hops 3] [--json out.json]

The flags are tools/marks_grooves.py's rule (a vertex with a vertex of another sheet within --reach across more
than --hops edges), recomputed here on the same mesh so the AO positions and the flags agree. Reports the mean
AO at flagged vertices, at vertices within --reach of a flagged one, and elsewhere, and the share of flagged
vertices below AO 0.5. Reads and writes nothing else.
"""
import argparse
import json
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--ao", required=True)
ap.add_argument("--reach", type=float, default=0.004)
ap.add_argument("--hops", type=int, default=3)
ap.add_argument("--json", default=None)
ap.add_argument("--hand-x", type=float, default=0.2, help="flagged vertices beyond this |X| (m) and below --hand-z are the hands and wrists")
ap.add_argument("--hand-z", type=float, default=0.15, help="height (m, Blender frame) below which |X| > --hand-x counts as the hands")
a = ap.parse_args()


def flagged_blender(path):
    s = trimesh.load(path)
    g = max(s.geometry.values(), key=lambda m: len(m.faces)) if hasattr(s, "geometry") else s
    V = np.asarray(g.vertices, float)
    F = np.asarray(g.faces)
    q = np.round(V / 1e-6).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    Fm = inv[F]
    keep = (Fm[:, 0] != Fm[:, 1]) & (Fm[:, 1] != Fm[:, 2]) & (Fm[:, 0] != Fm[:, 2])
    Fm = Fm[keep]
    Vm = np.zeros((inv.max() + 1, 3))
    np.add.at(Vm, inv, V)
    Vm /= np.bincount(inv)[:, None]
    n = len(Vm)
    adj = [set() for _ in range(n)]
    for x, y, z in Fm:
        adj[x].update((y, z)); adj[y].update((x, z)); adj[z].update((x, y))
    tree = cKDTree(Vm)
    flag = np.zeros(n, bool)
    for v in range(n):
        nb = tree.query_ball_point(Vm[v], a.reach)
        if len(nb) <= 1:
            continue
        seen = {v}
        fr = [v]
        for _ in range(a.hops):
            nx = []
            for x in fr:
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        nx.append(y)
            fr = nx
        if any(u not in seen for u in nb):
            flag[v] = True
    P = Vm[flag]
    return np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1), flag


P, flag = flagged_blender(a.mesh)
D = np.load(a.ao)
pos, ao = D["pos"], D["ao"]
tree = cKDTree(pos)
# AO of the flagged vertices: the nearest AO sample to each flagged position
d_f, i_f = tree.query(P)
ao_flag = ao[i_f]
# AO of every sample within reach of a flagged vertex
near = tree.query_ball_point(P, a.reach)
idx = np.unique(np.concatenate([np.asarray(x, int) for x in near if len(x)])) if any(len(x) for x in near) else np.array([], int)
in_reach = np.zeros(len(pos), bool)
in_reach[idx] = True
rest = ~in_reach
body = ~((np.abs(P[:, 0]) > a.hand_x) & (P[:, 2] < a.hand_z)) if len(P) else np.zeros(0, bool)
out = {"mesh": a.mesh, "flagged": int(len(P)), "vertices": int(len(pos)),
       "flagged_body": int(body.sum()), "dark_body": int(((ao_flag < 0.5) & body).sum()) if len(P) else 0,
       "ao_flagged_mean": round(float(ao_flag.mean()), 3) if len(P) else None,
       "ao_near_flagged_mean": round(float(ao[in_reach].mean()), 3) if in_reach.any() else None,
       "ao_elsewhere_mean": round(float(ao[rest].mean()), 3),
       "share_flagged_ao_below_0_5": round(float((ao_flag < 0.5).mean()), 3) if len(P) else None}
print(f"[groove_ao_report] {a.mesh}: {out['flagged']} flagged; AO at flagged {out['ao_flagged_mean']}, "
      f"within {a.reach*1000:.0f} mm of flagged {out['ao_near_flagged_mean']}, elsewhere {out['ao_elsewhere_mean']}; "
      f"flagged below 0.5: {out['share_flagged_ao_below_0_5']}", flush=True)
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
