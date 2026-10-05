"""How far a model's shape and texture are from a professionally made model's (E132).

    python tools/model_quality.py --npz q.npz [--work work/<name>] --out quality.json

--npz comes from blender/model_quality.py (the surface in metres, scaled to one height, smooth normals,
topology counts). Every measure is taken the same way on our characters and on professional baselines, so
each number has a target beside it. Shape (every model):

  noise_deg     how far each surface point's normal turns from the mean normal of the surface within
                1 cm of it (same side only, so a finger's far side does not count): fine bumpiness
  lump_mm       how far the surface within 4 cm of a point departs from the smooth quadric patch that
                fits it best, the worst quarter of points left out of the fit (so a designed crease
                does not count against a smooth patch): bumps a few centimetres across
  symmetry_mm   distance from the mirrored surface to the surface (left-right, about the best mirror plane)
  skinny_tri    share of triangles with an angle under 10 degrees; quads, the share of faces from quads
  tilt_deg      how far each triangle leans against the smooth normals at its corners - a low-poly that lies along
                its surface needs only small corrections from a normal map (professional meshes 2.4-8.4 median;
                a decimated voxel solid 15-21)
  open, non-manifold edges, pieces, cuts (faces of one object passing through each other) per 10k triangles

Texture (--work: the character's albedo.png, baked_ao.png, baked_orm.png and texproj/uv_position.npy,
uv_normal.npy, which say where on the body each texel lies). Garments are found as colour clusters (the
parser's classes miss armour), and each is compared with itself:

  light_r2      how much of a garment's lightness its surface's facing and ambient occlusion explain -
                lighting drawn into the albedo (an unlit albedo explains little)
  front_back_de CIEDE2000 between the same garment's colour where it faces front and where it faces back
  back_sharp    the back's fine detail (Laplacian energy) against the front's, same garment
  seam_de       colour step across a UV seam, against the step between neighbouring texels inside an island
  dark_pct, bright_pct   non-metal albedo darker than sRGB 30 or brighter than 240 (the PBR guide's limits)
  texel_mm      the size of a texel on the body, and the spread of that size (p95/p5)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None
RNG = np.random.default_rng(0)


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# ---------------------------------------------------------------------------------------------- shape
def sample_surface(V, F, VN, n):
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1) / 2
    idx = RNG.choice(len(F), n, p=a / a.sum())
    s, t = np.sqrt(RNG.random(n)), RNG.random(n)
    w = np.stack([1 - s, s * (1 - t), s * t], 1)
    P = np.einsum("nk,nkj->nj", w, V[F[idx]])
    N = np.einsum("nk,nkj->nj", w, VN[F[idx]])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return P, N, a.sum()


def noise_deg(P, N, tree, radius=0.01, k=48):
    d, j = tree.query(P, k=k, distance_upper_bound=radius)
    ok = np.isfinite(d) & (j < len(P))
    jj = np.where(ok, j, 0)
    nb = N[jj] * ok[..., None]
    same = (nb * N[:, None]).sum(-1) > 0                 # the near side of a thin part only
    m = (nb * same[..., None]).sum(1)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip((m * N).sum(1), -1, 1)))


def lump_mm(P, N, tree, centres, radius=0.04, k=400):
    out = []
    for c in centres:
        d, j = tree.query(P[c], k=k, distance_upper_bound=radius)
        j = j[np.isfinite(d) & (j < len(P))]
        j = j[(N[j] @ N[c]) > 0.3]
        if len(j) < 30:
            continue
        n = N[c]
        u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        q = P[j] - P[c]
        x, y, h = q @ u, q @ v, q @ n
        A = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], 1)
        keep = np.ones(len(h), bool)
        for _ in range(2):                                  # trimmed: a crease is not a lump
            coef, *_ = np.linalg.lstsq(A[keep], h[keep], rcond=None)
            res = np.abs(A @ coef - h)
            keep = res <= np.quantile(res, 0.75)
        out.append(np.sqrt(np.mean(res[keep] ** 2)) * 1000)
    return np.array(out)


def shape(npz):
    D = np.load(npz)
    V, F, VN = D["V"].astype(np.float64), D["F"], D["VN"].astype(np.float64)
    P, N, area = sample_surface(V, F, VN, 200_000)
    tree = cKDTree(P)
    nz = noise_deg(P, N, tree)
    lm = lump_mm(P, N, tree, RNG.choice(len(P), 6000, replace=False))
    # left-right symmetry about the best mirror plane, not the file's origin: the mirrored surface is laid on the
    # surface by a rigid ICP first, so a model a centimetre off centre or turned a degree is not read as lopsided
    M = P * [-1, 1, 1]
    S_ = M[RNG.choice(len(P), 20_000, replace=False)]
    Rm, tm = np.eye(3), np.zeros(3)
    for _ in range(20):
        Q_ = S_ @ Rm.T + tm
        d_, j_ = tree.query(Q_)
        k_ = d_ < np.quantile(d_, 0.8)
        A_, B_ = Q_[k_], P[j_[k_]]
        ca, cb = A_.mean(0), B_.mean(0)
        U, _, Vt = np.linalg.svd((A_ - ca).T @ (B_ - cb))
        Ri = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        Rm, tm = Ri @ Rm, Ri @ tm + (cb - Ri @ ca)
    sym = tree.query(M[RNG.choice(len(P), 50_000, replace=False)] @ Rm.T + tm)[0] * 1000
    e = np.stack([V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 1]], V[F[:, 0]] - V[F[:, 2]]], 1)
    L = np.linalg.norm(e, axis=2)
    cosang = np.stack([-(e[:, 2] * e[:, 0]).sum(1), -(e[:, 0] * e[:, 1]).sum(1), -(e[:, 1] * e[:, 2]).sum(1)], 1) \
        / np.maximum(L[:, [2, 0, 1]] * L, 1e-15)
    min_ang = np.degrees(np.arccos(np.clip(cosang, -1, 1))).min(1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    tilt = np.degrees(np.arccos(np.clip((VN[F] * fn[:, None]).sum(-1), -1, 1))).ravel()
    n = len(V)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    g = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    pieces = connected_components(g, directed=False)[0] - int((np.bincount(F.ravel(), minlength=n) == 0).sum())
    per10k = 1e4 / len(F)
    return {
        "triangles": int(len(F)), "area_m2": r(area), "edge_mm": r(np.median(L) * 1000, 2),
        "quads": r((D["sides"] == 4).mean()),
        "noise_deg": {"p50": r(np.median(nz), 2), "p90": r(np.quantile(nz, 0.9), 2)},
        "lump_mm": {"p50": r(np.median(lm), 3), "p75": r(np.quantile(lm, 0.75), 3)},
        "symmetry_mm": {"p50": r(np.median(sym), 2), "p90": r(np.quantile(sym, 0.9), 2)},
        "skinny_tri": r((min_ang < 10).mean(), 4),
        "tilt_deg": {"p50": r(np.median(tilt), 2), "p99": r(np.quantile(tilt, 0.99), 2)},
        "open_edges_per10k": r(int(D["boundary"]) * per10k, 2), "non_manifold_per10k": r(int(D["non_manifold"]) * per10k, 2),
        "cuts_per10k": r(int(D["self_cuts"]) * per10k, 2), "pieces": int(pieces),
    }


# -------------------------------------------------------------------------------------------- texture
def srgb_to_lab(c):
    """c: (..., 3) sRGB in 0..1 -> CIELAB (D65)."""
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / [0.95047, 1.0, 1.08883]
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ciede2000(l1, l2):
    L1, a1, b1 = l1[..., 0], l1[..., 1], l1[..., 2]
    L2, a2, b2 = l2[..., 0], l2[..., 1], l2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(np.abs(dh) > 180, dh - 360 * np.sign(dh), dh)
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hs, np.where(np.abs(h1p - h2p) > 180, (hs + 360) / 2 % 360, hs / 2))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC, SH = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    RT = (-2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
          * np.sin(np.radians(60 * np.exp(-(((hbp - 275) / 25) ** 2)))))
    return np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))


def pool(x, f):
    h, w = x.shape[:2]
    return x.reshape(h // f, f, w // f, f, *x.shape[2:]).mean((1, 3))


def kmeans(X, k, iters=25):
    C = X[RNG.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([X[lab == i].mean(0) if (lab == i).any() else C[i] for i in range(k)])
    return C


def texture(work, scale, res=1024, albedo="albedo.png"):
    w = Path(work)
    alb = np.asarray(Image.open(w / albedo).convert("RGB"), np.float32) / 255
    f = alb.shape[0] // res
    pos = np.load(w / "texproj" / "uv_position.npy").astype(np.float32)
    nrm = np.load(w / "texproj" / "uv_normal.npy").astype(np.float32)
    valid = np.isfinite(pos).all(-1) & (np.abs(pos).sum(-1) > 0)
    valid = pool(valid.astype(np.float32), f) > 0.999            # texels wholly on the surface
    pos = pool(np.nan_to_num(pos), f) * scale
    nrm = pool(np.nan_to_num(nrm), f)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
    alb = pool(alb, f)
    ao = pool(np.asarray(Image.open(w / "baked_ao.png").convert("L"), np.float32) / 255, f) \
        if (w / "baked_ao.png").exists() else np.ones(valid.shape)
    orm = w / "baked_orm.png"
    metal = pool(np.asarray(Image.open(orm).convert("RGB"), np.float32)[..., 2] / 255, f) if orm.exists() else np.zeros(valid.shape)
    lab = srgb_to_lab(alb)

    # garments as colour clusters: chroma first, lightness lightly (baked light moves lightness most)
    X = np.concatenate([lab[..., 1:], 0.35 * lab[..., :1]], -1)
    sub = np.flatnonzero(valid.ravel())
    C = kmeans(X.reshape(-1, 3)[RNG.choice(sub, min(60_000, len(sub)), replace=False)], 8)
    cl = np.argmin(((X.reshape(-1, 3)[:, None] - C[None]) ** 2).sum(-1), 1).reshape(valid.shape)
    cl[~valid] = -1
    front, back = valid & (nrm[..., 1] < -0.5), valid & (nrm[..., 1] > 0.5)

    # lighting drawn in: lightness explained by facing and occlusion, per garment
    r2, wts, de, de_w, sharp, sh_w = [], [], [], [], [], []
    Lstar = lab[..., 0]
    lap = np.zeros_like(Lstar)
    lap[1:-1, 1:-1] = 4 * Lstar[1:-1, 1:-1] - Lstar[:-2, 1:-1] - Lstar[2:, 1:-1] - Lstar[1:-1, :-2] - Lstar[1:-1, 2:]
    same = np.zeros(valid.shape, bool)
    same[1:-1, 1:-1] = ((cl[1:-1, 1:-1] >= 0) & (cl[:-2, 1:-1] == cl[1:-1, 1:-1]) & (cl[2:, 1:-1] == cl[1:-1, 1:-1])
                        & (cl[1:-1, :-2] == cl[1:-1, 1:-1]) & (cl[1:-1, 2:] == cl[1:-1, 1:-1]))
    for i in range(len(C)):
        m = cl == i
        if m.sum() < 0.02 * valid.sum():
            continue
        y = Lstar[m]
        A = np.column_stack([np.ones(m.sum()), nrm[m], ao[m]])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2.append(1 - np.var(y - A @ coef) / max(np.var(y), 1e-9)); wts.append(m.sum())
        fm, bm = m & front, m & back
        if fm.sum() > 0.005 * valid.sum() and bm.sum() > 0.005 * valid.sum():
            de.append(float(ciede2000(np.median(lab[fm], 0), np.median(lab[bm], 0)))); de_w.append(min(fm.sum(), bm.sum()))
            sf, sb = np.abs(lap[fm & same]).mean(), np.abs(lap[bm & same]).mean()
            if sf > 0:
                sharp.append(sb / sf); sh_w.append(min(fm.sum(), bm.sum()))

    # seams: texels at an island's edge next, on the body, to another island's edge texel
    inner = valid.copy()
    inner[1:-1, 1:-1] &= valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    edge = valid & ~inner
    ey, ex = np.nonzero(edge)
    step = np.linalg.norm(pos[:, 1:] - pos[:, :-1], axis=-1)[valid[:, 1:] & valid[:, :-1]]
    texel = float(np.median(step))
    t = cKDTree(pos[ey, ex])
    d, j = t.query(pos[ey, ex], k=8, distance_upper_bound=1.5 * texel)
    seam = []
    for a_ in range(1, 8):
        ok = np.isfinite(d[:, a_]) & (j[:, a_] < len(ey))
        jj = np.where(ok, j[:, a_], 0)
        far = ok & (np.hypot(ey - ey[jj], ex - ex[jj]) > 6)        # near on the body, far on the atlas
        if far.any():
            s_ = np.flatnonzero(far)
            seam.append(ciede2000(lab[ey[s_], ex[s_]], lab[ey[jj[s_]], ex[jj[s_]]]))
    seam = np.concatenate(seam) if seam else np.array([])
    iy, ix = np.nonzero(inner[:, :-1] & inner[:, 1:])
    pick = RNG.choice(len(iy), min(50_000, len(iy)), replace=False)
    interior = ciede2000(lab[iy[pick], ix[pick]], lab[iy[pick], ix[pick] + 1])

    nonmetal = valid & (metal < 0.5)
    lum = (0.2126 * alb[..., 0] + 0.7152 * alb[..., 1] + 0.0722 * alb[..., 2]) * 255
    full_res_mm = texel * 1000 / f                                    # a full-resolution texel on the body, in mm
    wa = lambda x, w_: r(np.average(x, weights=w_), 3) if x else None
    return {
        "light_r2": wa(r2, wts), "front_back_de": wa(de, de_w),
        "back_sharp": r(np.exp(np.average(np.log(sharp), weights=sh_w)), 3) if sharp else None,
        "seam_de": {"p50": r(np.median(seam), 2) if len(seam) else None, "interior_p50": r(np.median(interior), 2),
                    "pairs": int(len(seam))},
        "dark_pct": r(100 * (lum[nonmetal] < 30).mean(), 2), "bright_pct": r(100 * (lum[nonmetal] > 240).mean(), 2),
        "texel_mm": {"p50": r(full_res_mm, 3), "spread_p95_p5": r(np.quantile(step, 0.95) / max(np.quantile(step, 0.05), 1e-12), 2)},
        "coverage": r(valid.mean(), 3), "clusters": int(len(wts)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--work", default=None, help="the character's work dir, for the texture measures")
    ap.add_argument("--albedo", default="albedo.png", help="which albedo in --work (baked_base_color.png: TRELLIS's own)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"shape": shape(a.npz)}
    if a.work:
        out["texture"] = texture(a.work, float(np.load(a.npz)["scale"]), albedo=a.albedo)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
