"""Skin weights from distance measured THROUGH the body: geodesic voxel binding.

Every weighting scheme this project tried before measured straight-line distance from a vertex
to a bone, and straight lines cross air. In the A-pose a generated character is built in, the
side of a jacket hangs a few centimetres from the upper arm, so the jacket's side was bound to
the arm; raising the arms to a T dragged the torso up with them into "bat wings" from armpit to
waist. Hands hanging beside the thighs pulled the trousers the same way. Bone heat fixes this
with a visibility test, but Blender's solver fails silently on these meshes (100% unweighted on
a clean, closed, manifold character) and cannot be inspected.

Dionne & de Lasa's geodesic voxel binding (the method behind Maya's production binder) measures
distance inside the solid instead. The solid is solidify.py's volume; a path from a vertex to a
bone may only pass through solid voxels, so from the side of the torso to the upper arm it has
to go up and over through the armpit, and the spine wins. Weights fall off with that distance,
the four strongest are kept, and a short smoothing pass over the surface removes voxel steps.

Output: weights.npz with per-vertex bone indices and weights (in the vertex order trimesh loads
the mesh, which is the order Blender imports it in - labels.json relies on the same), plus the
bone names.

Run: python geodesic_weights.py --mesh retopo.glb --solid solid.npz --sdf sdf.npz \
         --joints joints.json --out weights.npz [--extra-bones hands.json]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import trimesh
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True, help="the skinned mesh (glTF, Y up), in the SAME frame as the solid")
ap.add_argument("--solid", required=True)
ap.add_argument("--sdf", required=True, help="sdf_io.py output - its points give the joints' frame")
ap.add_argument("--joints", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--coarsen", type=int, default=2, help="voxel downsampling for the graph")
ap.add_argument("--power", type=float, default=4.0)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--smooth", type=int, default=6, help="surface smoothing iterations on the weights")
ap.add_argument("--radius", type=float, default=0.0,
                help="measure from each body part's surface, not its bone: subtract this share of the part's own radius (E159)")
ap.add_argument("--inward", type=float, default=0.0,
                help="find each vertex's voxel from a point this many graph voxels inside the surface, along the vertex normal (E159)")
ap.add_argument("--radius-groups", default="",
                help="with --radius: apply the radius only to vertices of these label groups (comma list of body, clothing, "
                     "hair, accessory, from --labels); the rest keep their bone distances (E161)")
ap.add_argument("--labels", default="", help="labels.json (its per-vertex group) for --radius-groups")
ap.add_argument("--radius-skip-arm", type=float, default=0.0,
                help="with --radius: vertices whose default weights give at least this share to the arm bones (shoulder, collar, elbow, wrist) keep their bone distances, so a sleeve stays with its arm (E161)")
ap.add_argument("--radius-bones", default="",
                help="with --radius: subtract the radius only for these bones (comma list, e.g. pelvis,spine1,spine2,spine3); "
                     "the others keep their bone distances (E161)")
ap.add_argument("--radius-skip-leg", type=float, default=0.0,
                help="with --radius: vertices whose default weights give at least this share to the leg bones (hip, knee, ankle) "
                     "keep their bone distances, as --radius-skip-arm does for the arm (E161)")
a = ap.parse_args()
t0 = time.time()

S = np.load(a.solid)
sdf, v, o, H = S["sdf"], float(S["voxel"]), S["origin_ijk"], float(S["height"])
P = np.load(a.sdf)["points"]
lo, hi = P.min(0), P.max(0)
kk = 2.0 / float(hi[2] - lo[2])
cc = (lo + hi) / 2
Jd = json.load(open(a.joints))
J = {n: np.array(p) / kk + cc for n, p in Jd["joints"].items()}
PAR = Jd["parents"]

# ---- bones as segments (the same rule the rig uses: tail at the first child) ---------------------
children = {}
for n, p in PAR.items():
    if p:
        children.setdefault(p, []).append(n)
ENDS = {"head_top", "left_hand", "right_hand", "left_foot", "right_foot"}
bones = []
for n in J:
    if n in ENDS:
        continue
    kids = children.get(n, [])
    if n == "spine3":
        kids = ["neck"]
    tail = J[kids[0]] if kids else J[n] + np.array([0, 0, 0.03 * H])
    bones.append((n, J[n], tail))
names = [b[0] for b in bones]

# ---- the solid at graph resolution ----------------------------------------------------------------
c = a.coarsen
solid_f = sdf < 0
sh = tuple(s // c for s in solid_f.shape)
blk = solid_f[:sh[0] * c, :sh[1] * c, :sh[2] * c].reshape(sh[0], c, sh[1], c, sh[2], c)
# "at least a quarter of the fine voxels" keeps thin parts without bridging gaps a coarse
# "any" would close (a hand 5 mm from a thigh must stay apart)
solid = blk.sum(axis=(1, 3, 5)) >= max(1, (c ** 3) // 4)
vc = v * c
oc = o.astype(np.float64) + (c - 1) / 2.0            # world position of coarse voxel (0,0,0) / v
idx = -np.ones(sh, np.int64)
cells = np.argwhere(solid)
idx[tuple(cells.T)] = np.arange(len(cells))
centres = (cells * c + oc) * v
print(f"[weights] graph: {len(cells):,} solid voxels of {vc / H * 1000:.1f}/1000 height", flush=True)

rows, cols, wts = [], [], []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if (dx, dy, dz) <= (0, 0, 0):
                continue
            nb = cells + np.array([dx, dy, dz])
            okb = np.all((nb >= 0) & (nb < np.array(sh)), axis=1)
            j = -np.ones(len(cells), np.int64)
            j[okb] = idx[tuple(nb[okb].T)]
            m = j >= 0
            rows.append(np.nonzero(m)[0])
            cols.append(j[m])
            wts.append(np.full(m.sum(), vc * np.sqrt(dx * dx + dy * dy + dz * dz)))
G = sparse.csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(cells), len(cells)))
G = G + G.T
tree = cKDTree(centres)

# ---- one multi-source shortest-path run per bone -------------------------------------------------
D = np.full((len(cells), len(bones)), np.inf)
for bi, (n, h, t) in enumerate(bones):
    seg = np.linspace(0, 1, max(2, int(np.linalg.norm(t - h) / (0.5 * vc)) + 1))[:, None]
    pts = h + seg * (t - h)
    d_, near = tree.query(pts)
    src = np.unique(near[d_ < 3 * vc])
    if len(src) == 0:                                # bone entirely outside the solid
        d_, near = tree.query(pts)
        src = np.unique(near[np.argsort(d_)[:3]])
    D[:, bi] = dijkstra(G, directed=False, indices=src, min_only=True)
print(f"[weights] {len(bones)} bones, geodesic distances in {time.time() - t0:.1f}s", flush=True)
if a.radius > 0:
    # Distance from a bone's line favours a thin limb over a thick torso: the side of a loose jacket hangs nearer the upper
    # arm's bone than the spine's, though it is the torso's surface it belongs to, and it rose into fins with the arm (Vex,
    # Pip's vest). Measured from each part's own surface instead - the bone's distance less its radius, the depth of the
    # solid along the bone's middle - the torso's reach grows by its girth and a limb's by its own (E159).
    depth = ndimage.distance_transform_edt(solid) * vc          # the stored SDF is truncated near the surface
    rad = np.zeros(len(bones))
    for bi, (n, h, t) in enumerate(bones):
        pts = h + np.linspace(0.2, 0.8, 9)[:, None] * (t - h)
        ijk = np.clip(np.round((pts / v - oc) / c).astype(int), 0, np.array(solid.shape) - 1)
        rad[bi] = float(np.median(depth[tuple(ijk.T)]))
    # the trunk is one girth: its bones share the largest of their radii, or a vertebra a little off the body's middle
    # line takes the trunk from its neighbours and a bend shears between them (Vex's bend-and-twist 3.2 -> 5.7%)
    trunk = [i for i, b in enumerate(bones) if b[0] in ("pelvis", "spine1", "spine2", "spine3")]
    if trunk:
        rad[trunk] = rad[trunk].max()
    if a.radius_bones:
        # only the named bones' radii are subtracted (E161): a thick segment's radius lets it take the weight of a thin joint
        # next to it along the same chain (the wrist's radius is 0 cm, the forearm's 2.6: the forearm took the hand's weight)
        keep = {s.strip() for s in a.radius_bones.split(",") if s.strip()}
        rad = rad * np.array([b[0] in keep for b in bones], float)
    if not (a.radius_groups or a.radius_skip_arm > 0 or a.radius_skip_leg > 0):   # else applied per vertex below
        D = D - a.radius * rad[None, :]
    print("[weights] part radii (cm): " + ", ".join(f"{bones[i][0]} {rad[i] * 100:.1f}" for i in range(len(bones))), flush=True)

# ---- vertices: nearest solid voxel, plus the step to it ---------------------------------------
mesh = trimesh.load(a.mesh, force="mesh", process=False)
Vgl = np.asarray(mesh.vertices)
V = np.stack([Vgl[:, 0], -Vgl[:, 2], Vgl[:, 1]], axis=1)
dv, nv = tree.query(V)
if a.inward > 0:
    # A loose jacket hangs a few millimetres from the arm in the A-pose, the carved gap between them (free_arms.py) is thinner
    # than a voxel, and the side panel's nearest voxel was often the arm's: Vex's panel below the armpit carried half its
    # weight on the upper arm and rose into fins with it. A surface belongs to the solid behind it, so the voxel is found from
    # a point stepped inward along the normal; where that lands no nearer the solid (a strand thinner than the step), the
    # vertex keeps its own nearest voxel.
    Ngl = np.asarray(mesh.vertex_normals)
    Nn = np.stack([Ngl[:, 0], -Ngl[:, 2], Ngl[:, 1]], axis=1)
    di, ni = tree.query(V - Nn * (a.inward * vc))
    use = (di <= dv + 0.5 * vc) & (ni != nv)
    print(f"[weights] inward step {a.inward:g} voxels: {int(use.sum()):,} vertices take a voxel behind their surface", flush=True)
    nv = np.where(use, ni, nv)
    dv = np.linalg.norm(V - centres[nv], axis=1)
# The step out to a vertex is capped: a pocket flap or a strap the voxels do not hold (Pip's cargo
# pockets stood up to 3 cm off the solid) took that whole step onto every bone's distance, which
# evens them out - the flap's own thigh fell to a third of its weight, the pelvis and the other leg
# took the rest, and a squat pulled it into a plank. Off the solid a vertex takes its nearest voxel's
# distances, as a pocket follows the thigh it is sewn to.
DV = D[nv] + np.minimum(dv, 1.5 * vc)[:, None]
if a.radius > 0 and (a.radius_groups or a.radius_skip_arm > 0 or a.radius_skip_leg > 0):
    # the radius per vertex, not for the whole field (E161): only the label groups named (e.g. the clothing alone), and/or
    # not for the vertices the default weights already bind to an arm (a sleeve keeps its arm)
    on = np.ones(len(V), bool)
    if a.radius_groups:
        if not a.labels:
            raise SystemExit("--radius-groups needs --labels (labels.json)")
        lab = json.load(open(a.labels))
        grp = np.asarray(lab["labels"])
        if len(grp) != len(V):
            raise SystemExit(f"labels.json has {len(grp):,} vertices, the mesh {len(V):,}")
        on &= np.isin(grp, [lab["group_ids"][g] for g in a.radius_groups.split(",")])
    if a.radius_skip_arm > 0:
        W0 = np.maximum(DV, 0.25 * vc) ** (-a.power)
        W0[~np.isfinite(W0)] = 0
        arm = np.array([any(t in n for t in ("shoulder", "collar", "elbow", "wrist")) for n in names])
        share = W0[:, arm].sum(1) / np.maximum(W0.sum(1), 1e-30)
        bound = share >= a.radius_skip_arm
        print(f"[weights] {int((bound & on).sum()):,} vertices bound to an arm by the default keep their bone distances", flush=True)
        on &= ~bound
    if a.radius_skip_leg > 0:
        # the same for a leg (E161): a thigh's or shin's skin, bound by the default to its hip, knee or ankle, keeps its bone
        # distances - the thick thigh's radius otherwise takes the weight of the thinner knee and ankle beside it
        W0 = np.maximum(DV, 0.25 * vc) ** (-a.power)
        W0[~np.isfinite(W0)] = 0
        leg = np.array([n.endswith(("_hip", "_knee", "_ankle")) for n in names])
        share_l = W0[:, leg].sum(1) / np.maximum(W0.sum(1), 1e-30)
        bound_l = share_l >= a.radius_skip_leg
        print(f"[weights] {int((bound_l & on).sum()):,} vertices bound to a leg by the default keep their bone distances", flush=True)
        on &= ~bound_l
    DV[on] -= a.radius * rad[None, :]
    print(f"[weights] radius {a.radius:g} on {int(on.sum()):,} of {len(V):,} vertices", flush=True)
unreached = ~np.isfinite(DV).any(axis=1)
if unreached.any():                                   # a vertex whose voxel island has no bone
    eu = np.array([[np.linalg.norm(np.cross(t - h, h - x)) / max(np.linalg.norm(t - h), 1e-9)
                    for (_, h, t) in bones] for x in V[unreached]])
    DV[unreached] = eu
DV = np.maximum(DV, 0.25 * vc)
W = DV ** (-a.power)
W[~np.isfinite(W)] = 0

# ---- keep the strongest k, smooth over the surface, keep k again ------------------------------------
def topk(Wm, k):
    if Wm.shape[1] > k:
        cut = np.partition(Wm, -k, axis=1)[:, -k][:, None]
        Wm = np.where(Wm >= cut, Wm, 0.0)
    s = Wm.sum(1, keepdims=True)
    return Wm / np.maximum(s, 1e-12)


W = topk(W, a.k)
# Smooth on the welded surface: the file splits vertices along UV seams, and smoothing that
# stops at a seam gives the two copies of a seam vertex different weights - a crack when it bends.
nvtx = len(V)
_, weld = np.unique(np.round(V / (1e-6 * H)).astype(np.int64), axis=0, return_inverse=True)
weld = weld.ravel()
nw = int(weld.max()) + 1
edges = weld[mesh.edges_unique]
edges = edges[edges[:, 0] != edges[:, 1]]
A = sparse.coo_matrix((np.ones(len(edges) * 2), (np.r_[edges[:, 0], edges[:, 1]],
                                                  np.r_[edges[:, 1], edges[:, 0]])), shape=(nw, nw)).tocsr()
deg = np.asarray(A.sum(1)).ravel()
Dinv = sparse.diags(1.0 / np.maximum(deg, 1))
cnt = np.bincount(weld, minlength=nw).astype(np.float64)
Ww = np.zeros((nw, W.shape[1]))
np.add.at(Ww, weld, W)
Ww /= cnt[:, None]
for _ in range(a.smooth):
    Ww = 0.5 * Ww + 0.5 * (Dinv @ (A @ Ww))
W = topk(Ww[weld], a.k)
order = np.argsort(-W, axis=1)[:, :a.k]
ww = np.take_along_axis(W, order, axis=1)
np.savez_compressed(a.out, bones=np.array(names), index=order.astype(np.int16), weight=ww.astype(np.float32))
dom = np.bincount(order[:, 0], minlength=len(names))
print(f"[weights] {nvtx:,} vertices; dominant bone counts: "
      + ", ".join(f"{names[i]} {dom[i]}" for i in np.argsort(-dom)[:8]) + f"; {time.time() - t0:.1f}s",
      flush=True)
