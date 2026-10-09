"""Where a character's grooves come from: the free_arms carve, or narrow outside air in the solidify output.

    ../.venv/bin/python tools/groove_attribution.py --char work/rowan [--reach 0.004] [--near 6]

Reads <char>/sdf.npz, <char>/retopo.glb (the shipped low-poly), <char>/solid.npz (solidify.py's output),
<char>/solid_cut.npz (after cut_hands) and <char>/solid_free.npz (after free_arms.py). Reports, for the
flagged retopo vertices (tools/marks_grooves.py's rule):
  carve   - share within --near voxels (1.5 mm each) of the voxels free_arms.py took out of the solid
  narrow  - share within --near voxels of a narrow outside slot in solid.npz (air connected to the box edge
            that does not survive an opening by a ball of radius 2 voxels; slots narrower than ~7.5 mm)
  narrow_after - the same for solid_free.npz (after the arms were cut)
Reads only.
"""
import argparse

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--char", required=True, help="a work folder, e.g. work/rowan")
ap.add_argument("--reach", type=float, default=0.004)
ap.add_argument("--hops", type=int, default=3)
ap.add_argument("--near", type=int, default=6, help="voxels (1.5 mm) of tolerance round a carved or narrow voxel")
ap.add_argument("--open-radius", type=int, default=2)
a = ap.parse_args()


def flagged(path):
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
    return np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1), n


def narrow_outside(S):
    air = S >= 0
    lab, _ = ndimage.label(air)
    edge = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                     lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    edge = edge[edge > 0]
    ext = np.isin(lab, edge)
    R = a.open_radius
    g = np.mgrid[-R:R + 1, -R:R + 1, -R:R + 1]
    ball = (g ** 2).sum(0) <= R * R + 0.5
    pad = R + 1
    op = ndimage.binary_opening(np.pad(ext, pad, constant_values=True), structure=ball)[pad:-pad, pad:-pad, pad:-pad]
    return ext & ~op


d = np.load(f"{a.char}/sdf.npz")
v = float(d["voxel"])
o = d["origin_ijk"].astype(np.int64)
P, n = flagged(f"{a.char}/retopo.glb")
cut = np.load(f"{a.char}/solid_cut.npz")["sdf"] < 0
free = np.load(f"{a.char}/solid_free.npz")["sdf"]
carve = cut & ~(free < 0)
near_carve = ndimage.binary_dilation(carve, iterations=a.near)
ijk = np.round(P / v).astype(np.int64) - o
ok = np.all((ijk >= 0) & (ijk < np.array(carve.shape)), axis=1)


def share(mask):
    hit = np.zeros(len(P), bool)
    m = np.zeros(len(P), bool)
    m[ok] = mask[ijk[ok][:, 0], ijk[ok][:, 1], ijk[ok][:, 2]]
    hit[:] = m
    return int(hit.sum())


near_narrow_pre = ndimage.binary_dilation(narrow_outside(np.load(f"{a.char}/solid.npz")["sdf"]), iterations=a.near)
near_narrow_post = ndimage.binary_dilation(narrow_outside(free), iterations=a.near)
print(f"{a.char}: {len(P)} flagged of {n} retopo vertices")
print(f"  within {a.near * v * 1000:.0f} mm of the free_arms carve:            {share(near_carve):>5} ({share(near_carve) / max(1, len(P)):.0%})")
print(f"  within {a.near * v * 1000:.0f} mm of a narrow outside slot in solid.npz: {share(near_narrow_pre):>5} ({share(near_narrow_pre) / max(1, len(P)):.0%})")
print(f"  within {a.near * v * 1000:.0f} mm of a narrow outside slot after arms:   {share(near_narrow_post):>5} ({share(near_narrow_post) / max(1, len(P)):.0%})")
