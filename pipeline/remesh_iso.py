"""Isotropic remesh of a clean solid to a triangle budget, with a sizing field (E168).

The dense solid (solid_hands.glb, ~1.4M triangles) is the reference: every vertex of the output lies on it. The
cage to remesh is the default decimated retopo.glb (already on the solid), remeshed Botsch-Kobbelt style: long edges
split, short edges collapsed, valences evened by flips, one tangential relaxation, and every vertex projected back
onto the solid. Collapse decimation leaves long thin triangles whose tilt against their own smooth normals is what
a normal map then has to carry (E135); a remesh gives near-equilateral triangles whose size follows the surface:

    target size h(x) = s * g(x) * m_joint(x) * m_hand(x) * m_head(x)
    g(x)     = sqrt(8 e / kappa(x)), clipped to [0.4%, 3%] of the height. The chordal error of a triangle of size h
               on a surface of curvature kappa is h^2 kappa / 8, so equal e everywhere puts small triangles where
               the surface turns (faces, fingers, folds) and large ones on flat cloth.
    m_joint  0.6 within 6% of the height of an elbow, knee, shoulder or hip, back to 1 at 12%: deformation rings
    m_hand   0.5 within 10% of the height of a wrist-to-hand segment, back to 1 at 20%
    m_head   solved so the head (retopo.py's head region) takes --head-share of the triangles (default 28%)
    s        solved so the count estimate, sum of area / (sqrt3/4 h^2), is the budget; one calibration rerun
             corrects the estimate's bias.

Curvature: cotangent-Laplacian mean curvature |H| and angle-deficit Gaussian K on the dense solid; the largest
principal curvature |H| + sqrt(H^2 - K), averaged over 20 neighbours to take out the solid's marching-cubes noise.

Run: ../.venv/bin/python pipeline/remesh_iso.py --solid work/<c>/solid_hands.glb --start work/<c>/retopo.glb \
         --mesh work/<c>/mesh.glb --joints work/<c>/joints_refined.json --tris 60000 --head-share 0.28 \
         --out work/<c>/e168/iso/low_iso.glb
Writes the GLB (glTF frame, for retopo.py --low) and low_iso_remesh.json beside it.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remesh_io as R  # noqa: E402

COS_FLIP = 0.3        # a flip may turn a new triangle no further than this from the old normal (cosine)
COS_COLLAPSE = 0.5    # a collapse may not turn a star triangle more than 60 degrees
E_REL = 3e-4          # chordal tolerance, fraction of the height
H_MIN_REL = 0.004     # smallest target edge, fraction of the height
H_MAX_REL = 0.03      # largest target edge, fraction of the height
JOINT_NAMES = ["left_elbow", "right_elbow", "left_knee", "right_knee", "left_shoulder", "right_shoulder",
               "left_hip", "right_hip"]
HAND_SEGMENTS = [("left_wrist", "left_hand"), ("right_wrist", "right_hand")]


def curvature(V, F):
    """Largest principal curvature magnitude per vertex: cotangent Laplacian (mean) and angle deficit (Gaussian)."""
    n = len(V)
    _, area = R.face_normals_areas(V, F)
    ang_sum = np.zeros(n)
    rows, cols, w = [], [], []
    for k in range(3):
        i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        a = V[j] - V[i]
        b = V[l] - V[i]
        cosv = np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
        th = np.arccos(np.clip(cosv, -1, 1))
        ang_sum += np.bincount(i, th, minlength=n)
        cot = np.cos(th) / np.maximum(np.sin(th), 1e-12)     # the opposite edge (j, l) gets this weight
        rows += [j, l]
        cols += [l, j]
        w += [cot, cot]
    W = sparse.coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    A_i = np.maximum(np.bincount(F.ravel(), np.repeat(area / 3, 3), minlength=n), 1e-300)
    rowsum = np.asarray(W.sum(axis=1)).ravel()
    lap = (W @ V - rowsum[:, None] * V) / (2 * A_i)[:, None]          # Delta x = 2 H n
    Hm = np.linalg.norm(lap, axis=1) / 2
    K = (2 * np.pi - ang_sum) / A_i
    return Hm + np.sqrt(np.maximum(Hm * Hm - K, 0.0))


def smooth_scalar(V, val, k=20):
    _, idx = cKDTree(V).query(V, k=k)
    return val[idx].mean(axis=1)


def region_fields(Vb, J, H):
    """Per dense vertex (Blender frame): head membership and the joint and hand multipliers."""
    neck, head = J["neck"], J["head"]
    in_head = (Vb[:, 2] > neck[2]) & (np.linalg.norm(Vb[:, :2] - head[:2], axis=1) < 0.12 * H)
    d_joint = np.full(len(Vb), np.inf)
    for n in JOINT_NAMES:
        if n in J:
            d_joint = np.minimum(d_joint, np.linalg.norm(Vb - J[n], axis=1))
    d_hand = np.full(len(Vb), np.inf)
    for w_name, h_name in HAND_SEGMENTS:
        if w_name in J and h_name in J:
            p0, p1 = J[w_name], J[h_name]
            ab = p1 - p0
            t = np.clip(((Vb - p0) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)[:, None]
            d_hand = np.minimum(d_hand, np.linalg.norm(Vb - (p0 + t * ab), axis=1))
    m_joint = 0.6 + 0.4 * np.clip((d_joint - 0.06 * H) / (0.06 * H), 0, 1)
    m_hand = 0.5 + 0.5 * np.clip((d_hand - 0.10 * H) / (0.10 * H), 0, 1)
    return in_head, m_joint, m_hand


def solve_sizes(Fd, Vd, base_d, in_head_f, tris, head_share):
    """Head multiplier and global scale in closed form, from the dense faces (count estimate per unit scale)."""
    _, area = R.face_normals_areas(Vd, Fd)
    gbar = base_d[Fd].mean(axis=1)
    q = area / ((np.sqrt(3) / 4) * gbar ** 2)
    Qh = q[in_head_f].sum()
    Qb = q[~in_head_f].sum()
    if head_share is None or Qh <= 0:
        m_h, X = 1.0, Qh
    else:
        S = head_share
        m_h = np.sqrt(Qh * (1 - S) / (S * Qb))
        X = Qh / m_h ** 2
    s = np.sqrt((X + Qb) / tris)
    return s, m_h


def compact(V, F, h, frozen):
    used = np.zeros(len(V), bool)
    used[F.ravel()] = True
    newidx = np.cumsum(used) - 1
    return V[used], newidx[F], h[used], frozen[used]


class Projector:
    """Vertices back onto the dense solid, with the target size there (barycentric blend of the dense sizes)."""

    def __init__(self, Vd, Fd, G_d, index):
        self.Fd, self.G_d, self.index = Fd, G_d, index

    def __call__(self, V, h, fallback=None, gate=True):
        Q, face, bary, _ = self.index.query(V)
        g = np.sum(bary * self.G_d[self.Fd[face]], axis=1)
        if not gate:
            return Q, g, 0
        ok = np.linalg.norm(Q - V, axis=1) <= 0.5 * h
        base = V if fallback is None else fallback
        Vn = np.where(ok[:, None], Q, base)
        return Vn, np.where(ok, g, h), int((~ok).sum())


def split_long(V, F, h, frozen):
    """Split every edge longer than 4/3 of its target: midpoints, each face cut by its split edges."""
    n = len(V)
    m = len(F)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    Eu, inv = np.unique(E, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    L = np.linalg.norm(V[Eu[:, 0]] - V[Eu[:, 1]], axis=1)
    te = 0.5 * (h[Eu[:, 0]] + h[Eu[:, 1]])
    ok = ~(frozen[Eu[:, 0]] | frozen[Eu[:, 1]])
    ids = np.nonzero((L > (4.0 / 3.0) * te) & ok)[0]
    if len(ids) == 0:
        return V, F, h, frozen, 0
    mid = np.full(len(Eu), -1, np.int64)
    mid[ids] = n + np.arange(len(ids))
    Vm = 0.5 * (V[Eu[ids, 0]] + V[Eu[ids, 1]])
    hm = 0.5 * (h[Eu[ids, 0]] + h[Eu[ids, 1]])
    M = mid[inv].reshape(3, m).T                       # per face: the midpoint of edges 01, 12, 20 (or -1)
    Fl, Ml = F.tolist(), M.tolist()
    out = []
    for f in range(m):
        a, b, c = Fl[f]
        mab, mbc, mca = Ml[f]
        s = (mab >= 0) + (mbc >= 0) + (mca >= 0)
        if s == 0:
            out.append((a, b, c))
        elif s == 1:
            if mab >= 0:
                out += [(a, mab, c), (mab, b, c)]
            elif mbc >= 0:
                out += [(a, b, mbc), (a, mbc, c)]
            else:
                out += [(a, b, mca), (mca, b, c)]
        elif s == 2:
            if mab >= 0 and mbc >= 0:
                out += [(a, mab, c), (mab, b, mbc), (mab, mbc, c)]
            elif mbc >= 0 and mca >= 0:
                out += [(b, mbc, a), (mbc, c, mca), (mbc, mca, a)]
            else:
                out += [(c, mca, b), (mca, a, mab), (mca, mab, b)]
        else:
            out += [(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]
    Vn = np.vstack([V, Vm])
    hn = np.concatenate([h, hm])
    fr = np.concatenate([frozen, np.zeros(len(Vm), bool)])
    return Vn, np.array(out, dtype=np.int64), hn, fr, len(ids)


def collapse_short(V, F, h, frozen):
    """Collapse every edge shorter than 4/5 of its target that passes the link condition and keeps normals."""
    n = len(V)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    Eu = np.unique(E, axis=0)
    L = np.linalg.norm(V[Eu[:, 0]] - V[Eu[:, 1]], axis=1)
    te = 0.5 * (h[Eu[:, 0]] + h[Eu[:, 1]])
    cand = np.nonzero((L < 0.8 * te) & ~(frozen[Eu[:, 0]] | frozen[Eu[:, 1]]))[0]
    if len(cand) == 0:
        return V, F, h, frozen, 0
    cand = cand[np.argsort(L[cand] / te[cand])]
    Fl = F.tolist()
    vf = [[] for _ in range(n)]
    nb = [set() for _ in range(n)]
    for f, (a, b, c) in enumerate(Fl):
        for x in (a, b, c):
            vf[x].append(f)
        nb[a].update((b, c))
        nb[b].update((a, c))
        nb[c].update((a, b))
    touched = np.zeros(n, bool)
    remap = np.arange(n)
    dead = set()
    accepted = []
    for e in cand.tolist():
        u, v = int(Eu[e, 0]), int(Eu[e, 1])
        if touched[u] or touched[v]:
            continue
        shared = [f for f in vf[u] if v in Fl[f]]
        if len(shared) != 2:
            continue
        third = {x for f in shared for x in Fl[f] if x != u and x != v}
        if len(third) != 2 or (nb[u] & nb[v]) != third:          # link condition
            continue
        p = 0.5 * (V[u] + V[v])
        # Botsch-Kobbelt: a collapse may not create an edge longer than 4/3 of the target, or the split of that edge
        # later undoes it (without this check the edges split and collapse back every iteration and never settle)
        te_m = 0.5 * (h[u] + h[v])
        if any(np.linalg.norm(V[x] - p) > (4.0 / 3.0) * te_m for x in (nb[u] | nb[v]) if x != u and x != v):
            continue
        ok = True
        for f in (set(vf[u]) | set(vf[v])) - set(shared):
            tri = Fl[f]
            old = V[tri]
            new = np.array([p if x in (u, v) else V[x] for x in tri])
            n_old = np.cross(old[1] - old[0], old[2] - old[0])
            n_new = np.cross(new[1] - new[0], new[2] - new[0])
            nn, no = np.linalg.norm(n_new), np.linalg.norm(n_old)
            if nn < 1e-15 or no < 1e-15 or np.dot(n_old, n_new) / (nn * no) < COS_COLLAPSE:
                ok = False
                break
        if not ok:
            continue
        accepted.append((u, v, p))
        touched[u] = touched[v] = True
        for x in nb[u] | nb[v]:
            touched[x] = True
        remap[v] = u
        dead.update(shared)
    if not accepted:
        return V, F, h, frozen, 0
    V = V.copy()
    h = h.copy()
    for u, v, p in accepted:
        V[u] = p
        h[u] = 0.5 * (h[u] + h[v])
    keep = np.ones(len(F), bool)
    keep[list(dead)] = False
    Fn = remap[F[keep]]
    Vc, Fc, hc, fc = compact(V, Fn, h, frozen)
    return Vc, Fc, hc, fc, len(accepted)


def flip_edges(V, F, frozen, valence=6):
    """Flip interior edges whose flip brings the valences nearer 6 and keeps the new triangles facing their old ones."""
    n = len(V)
    Fl = F.tolist()
    emap = {}
    nb = [set() for _ in range(n)]
    for f, (a, b, c) in enumerate(Fl):
        for x, y in ((a, b), (b, c), (c, a)):
            emap.setdefault((x, y) if x < y else (y, x), []).append(f)
            nb[x].add(y)
            nb[y].add(x)
    touched = np.zeros(n, bool)
    flips = 0
    for key in list(emap.keys()):
        fs = emap.get(key)
        if not fs or len(fs) != 2:
            continue
        a0, b0 = key
        f1, f2 = fs
        t1, t2 = Fl[f1], Fl[f2]
        # orient: f1 runs a -> b; f2 must run b -> a (consistent winding), else leave the edge alone
        i = t1.index(a0)
        a, b = (a0, b0) if t1[(i + 1) % 3] == b0 else (b0, a0)
        j = t2.index(b)
        if t2[(j + 1) % 3] != a:
            continue
        if touched[a] or touched[b] or frozen[a] or frozen[b]:
            continue
        c = next(x for x in t1 if x != a and x != b)
        d = next(x for x in t2 if x != a and x != b)
        if c == d or frozen[c] or frozen[d] or touched[c] or touched[d] or d in nb[c]:
            continue
        if len(nb[a]) <= 3 or len(nb[b]) <= 3:
            continue
        t = valence
        da, db, dc, dd = len(nb[a]), len(nb[b]), len(nb[c]), len(nb[d])
        before = abs(da - t) + abs(db - t) + abs(dc - t) + abs(dd - t)
        after = abs(da - 1 - t) + abs(db - 1 - t) + abs(dc + 1 - t) + abs(dd + 1 - t)
        if after >= before:
            continue
        o1 = np.cross(V[t1[1]] - V[t1[0]], V[t1[2]] - V[t1[0]])
        o2 = np.cross(V[t2[1]] - V[t2[0]], V[t2[2]] - V[t2[0]])
        n1 = np.cross(V[a] - V[c], V[d] - V[c])                  # (c, a, d)
        n2 = np.cross(V[b] - V[d], V[c] - V[d])                  # (d, b, c)
        ok = True
        for nn_, oo_ in ((n1, o1), (n2, o2)):
            ln = np.linalg.norm(nn_) * np.linalg.norm(oo_)
            if ln < 1e-30 or np.dot(nn_, oo_) / ln < COS_FLIP:
                ok = False
        if not ok:
            continue
        Fl[f1] = [c, a, d]
        Fl[f2] = [d, b, c]
        del emap[key]
        emap[(min(c, d), max(c, d))] = [f1, f2]
        ka = (min(a, d), max(a, d))
        emap[ka] = [f1 if x == f2 else x for x in emap[ka]]
        kb = (min(b, c), max(b, c))
        emap[kb] = [f2 if x == f1 else x for x in emap[kb]]
        nb[a].discard(b)
        nb[b].discard(a)
        nb[c].add(d)
        nb[d].add(c)
        touched[a] = touched[b] = touched[c] = touched[d] = True
        flips += 1
    return np.array(Fl, dtype=np.int64), flips


def relax_tangential(V, F, frozen, lam):
    """Move each vertex towards its neighbours' centroid, in its tangent plane only."""
    n = len(V)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.concatenate([E, E[:, ::-1]])
    A = sparse.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)
    d = (A @ V) / deg[:, None] - V
    N = R.vertex_normals(V, F)
    d = d - np.sum(d * N, 1)[:, None] * N
    return V + np.where(frozen[:, None], 0.0, lam * d)


def tilt_summary(V, F):
    n_f, _ = R.face_normals_areas(V, F)
    N = R.vertex_normals(V, F)
    corners = np.degrees(np.arccos(np.clip(np.sum(N[F] * n_f[:, None, :], axis=2), -1, 1))).ravel()
    return float(np.median(corners)), float(np.quantile(corners, 0.99))


def remesh(args, project, V0, F0, h0, frozen0):
    V, F, h, fr = V0.copy(), F0.copy(), h0.copy(), frozen0.copy()
    V, h, _ = project(V, h, gate=False)
    log = []
    for it in range(args.iters):
        t0 = time.time()
        V, F, h, fr, ns = split_long(V, F, h, fr)
        V, F, h, fr, nc = collapse_short(V, F, h, fr)
        F, nf = flip_edges(V, F, fr)
        Vr = relax_tangential(V, F, fr, args.relax)
        V, h, nrej = project(Vr, h, fallback=V)
        log.append({"it": it, "splits": int(ns), "collapses": int(nc), "flips": int(nf), "rejected": int(nrej),
                    "triangles": int(len(F))})
        print(f"[remesh] it {it:2d}: split {ns:6,d} collapse {nc:6,d} flip {nf:6,d} rejected {nrej:4d} "
              f"-> {len(F):,} triangles ({time.time() - t0:.1f}s)", flush=True)
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    V, F, h, fr = compact(V, F[ok], h, fr)
    return V, F, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solid", required=True, help="the clean solid (solidify.py + hands.py): the reference surface")
    ap.add_argument("--start", required=True, help="the cage to remesh: the default decimated retopo.glb")
    ap.add_argument("--mesh", required=True, help="the generator's mesh.glb, whose box gives the joints' frame")
    ap.add_argument("--joints", required=True)
    ap.add_argument("--tris", type=int, default=60000)
    ap.add_argument("--head-share", type=float, default=0.28)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--relax", type=float, default=0.5)
    ap.add_argument("--e-rel", type=float, default=E_REL)
    ap.add_argument("--h-min", type=float, default=H_MIN_REL)
    ap.add_argument("--h-max", type=float, default=H_MAX_REL)
    ap.add_argument("--calib-rounds", type=int, default=3,
                    help="reruns that correct the scale by the count reached, until within 1%% of the budget")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t_start = time.time()

    D = R.read_mesh(args.solid)
    Vd, Fd = D["V"], D["F"]
    print(f"[remesh] solid {len(Vd):,} vertices, {len(Fd):,} triangles", flush=True)
    lo, hi = R.bbox_blender(R.read_mesh(args.mesh)["V"])
    Jw, H, centre = R.joints_world(R.load_joints(args.joints), lo, hi)
    Vb = R.to_blender(Vd)

    kmax = smooth_scalar(Vd, curvature(Vd, Fd))
    g0 = np.clip(np.sqrt(8 * args.e_rel * H / np.maximum(kmax, 1e-9)), args.h_min * H, args.h_max * H)
    in_head_v, m_joint, m_hand = region_fields(Vb, Jw, H)
    in_head_f = in_head_v[Fd].mean(axis=1) > 0.5
    base = g0 * m_joint * m_hand
    print(f"[remesh] height {H:.4f}; base size p5/p50/p95 {np.round(np.percentile(g0, [5, 50, 95]) / H * 100, 2)}"
          f" % of height; head {in_head_f.mean():.1%} of the solid", flush=True)

    S = R.read_mesh(args.start)
    V0, F0 = R.weld(S["V"], S["F"])
    b0, nm0 = R.boundary_and_nonmanifold(F0)
    print(f"[remesh] start cage: {len(V0):,} vertices, {len(F0):,} triangles, boundary {b0}, non-manifold {nm0}",
          flush=True)
    E = np.sort(np.concatenate([F0[:, [0, 1]], F0[:, [1, 2]], F0[:, [2, 0]]]), axis=1)
    Eu, cnt = np.unique(E, axis=0, return_counts=True)
    frozen0 = np.zeros(len(V0), bool)
    frozen0[Eu[cnt != 2].ravel()] = True                    # boundary and non-manifold: left where they are
    index = R.TriangleIndex(Vd, Fd, k=24)

    def run(est_target):
        s, m_h = solve_sizes(Fd, Vd, base, in_head_f, est_target, args.head_share)
        G = base * s * np.where(in_head_v, m_h, 1.0)
        proj = Projector(Vd, Fd, G, index)
        h0 = np.full(len(V0), args.h_min * H)
        V, F, log = remesh(args, proj, V0, F0, h0, frozen0)
        return V, F, log, s, m_h, G

    # The closed-form count is an estimate: the edges settle at a different size from the target, so the scale is
    # corrected by the count the converged mesh reaches (N ~ 1/s^2), until it is within 1% of the budget.
    corr, rounds = 1.0, []
    for rnd in range(max(1, args.calib_rounds)):
        est = args.tris / corr
        V, F, log, s, m_h, G = run(est)
        rounds.append({"round": rnd, "triangles": int(len(F)), "scale": round(float(s), 5),
                       "head_multiplier": round(float(m_h), 4)})
        print(f"[remesh] round {rnd}: {len(F):,} triangles against the {args.tris:,} budget "
              f"(s {s:.4f}, head multiplier {m_h:.3f})", flush=True)
        if abs(len(F) / args.tris - 1) < 0.01:
            break
        corr = len(F) / est
    b1, nm1 = R.boundary_and_nonmanifold(F)

    # where the triangles sit against their targets, and where the lean is
    qo, fo, bo, _ = index.query(V)
    g_out = np.sum(bo * G[Fd[fo]], axis=1)
    mj_out = np.sum(bo * m_joint[Fd[fo]], axis=1)
    mh_out = np.sum(bo * m_hand[Fd[fo]], axis=1)
    hd_out = np.sum(bo * in_head_v[Fd[fo]].astype(float), axis=1) > 0.5
    Lv = np.linalg.norm(V[F[:, [0, 1, 2]]] - V[F[:, [1, 2, 0]]], axis=2)
    te = 0.5 * (g_out[F[:, [0, 1, 2]]] + g_out[F[:, [1, 2, 0]]])
    rat = (Lv / te).ravel()
    cen_b = R.to_blender(V)[F].mean(axis=1)
    in_head_out = (cen_b[:, 2] > Jw["neck"][2]) & (np.linalg.norm(cen_b[:, :2] - Jw["head"][:2], axis=1) < 0.12 * H)
    n_f, _ = R.face_normals_areas(V, F)
    VN = R.vertex_normals(V, F)
    corners = np.degrees(np.arccos(np.clip(np.sum(VN[F] * n_f[:, None, :], axis=2), -1, 1)))
    region = np.where(in_head_out, "head", np.where(mh_out[F].mean(1) < 0.999, "hands",
                      np.where(mj_out[F].mean(1) < 0.999, "joints", "body")))
    reg = {}
    for name in ("head", "hands", "joints", "body"):
        m = region == name
        if m.any():
            reg[name] = {"faces": int(m.sum()), "edge_over_target_p50": round(float(np.median(
                Lv[m] / te[m])), 3), "tilt_p50": round(float(np.median(corners[m])), 2),
                "tilt_p99": round(float(np.quantile(corners[m], 0.99)), 2)}
    e3 = Lv.ravel()
    stats = {"triangles": int(len(F)), "vertices": int(len(V)), "budget": args.tris,
             "head_share": round(float(in_head_out.mean()), 4), "boundary_edges": b1, "non_manifold_edges": nm1,
             "edge_mm_at_1750_p50": round(float(np.median(e3)) / H * 1750, 2),
             "edge_mm_at_1750_p5_p95": [round(float(x) / H * 1750, 2) for x in np.percentile(e3, [5, 95])],
             "edge_over_target_p5_p50_p95": [round(float(x), 3) for x in np.percentile(rat, [5, 50, 95])],
             "tilt_median_deg": round(float(np.median(corners)), 2),
             "tilt_p99_deg": round(float(np.quantile(corners, 0.99)), 2),
             "by_region": reg, "rounds": rounds, "e_rel": args.e_rel, "h_min_rel": args.h_min,
             "h_max_rel": args.h_max, "start": args.start, "seconds": round(time.time() - t_start, 1),
             "iterations": log}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    R.write_glb(args.out, V, F)
    json.dump(stats, open(os.path.splitext(args.out)[0] + "_remesh.json", "w"), indent=2)
    print("[remesh] " + json.dumps({k: v for k, v in stats.items() if k not in ("iterations", "rounds")}), flush=True)


if __name__ == "__main__":
    main()
