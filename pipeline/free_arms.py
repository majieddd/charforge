"""Part the arms from whatever they were generated against.

A generator draws the character in an A-pose, arms at its sides, and wherever a sleeve lay against
the body - a puffy vest, an oversized jacket, a hand resting on a cargo pocket - the two came out as
one solid. Nothing downstream can part them: when the arm rises, the glued strip has to stretch
with it. Pip's vest rose into wings with his arms overhead, and his pocket into a plank when his
hand swung away from it. Re-weighting cannot fix a glued surface - it only chooses which side
tears (REVIEW, Tried and dropped); the surface itself has to open.

How. Walking down each arm below the armpit, a cross-section at a time: the arm's centre is the
deepest point of the solid near the traced centre line (the trace can run off-centre), and from it
rays go out across the arm. A ray on the free side leaves the solid at the arm's surface; a ray into
the glue runs on into something big - past half again the arm's radius in that direction, carried
round the section from the directions where it is free. There the solid is cut: a thin shell at the
arm's radius, all the way round where anything is glued (Pip's vest wrapped the back of his arm as
well as its inner side; a cut on the inner side alone left the back glued). The sleeve keeps its round
section, the vest or the hip keeps the rest, and above the armpit the arm stays joined, as a shoulder
is. Where nothing touches the arm the rays leave it and nothing is cut; where a sleeve is only baggy
it is not cut either (cutting inside a sleeve hollows it, as carving a braid out of a hood did).

Two first versions failed and are the reason for each rule: a radius from the outer half of an
off-centre trace came out 50% large and cut pockets inside the torso; a watershed seeded on the
centre line and the spine split the sleeve in rings, the body's flood reaching round through the
shoulder.

Run: python free_arms.py --solid solid_cut.npz --sdf sdf.npz --joints joints_refined.json \\
         --out solid_free.npz [--report free_arms.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--solid", required=True, help="the volume after cut_hands.py (solid_cut.npz)")
ap.add_argument("--sdf", required=True, help="sdf.npz (its points give the joints' frame)")
ap.add_argument("--joints", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--report", default=None)
ap.add_argument("--axilla", type=float, default=0.25, help="where the cut starts, as a fraction of the upper arm (where its section parts from the torso on Pip)")
ap.add_argument("--gap", type=float, default=0.010, help="the cut's width, as a fraction of height")
a = ap.parse_args()

S = np.load(a.solid)
sdf = S["sdf"].copy()
v, o, H = float(S["voxel"]), S["origin_ijk"].astype(np.int64), float(S["height"])
P = np.load(a.sdf)["points"]
lo, hi = P.min(0), P.max(0)
kk = 2.0 / float(hi[2] - lo[2])
cc = (lo + hi) / 2
Jd = json.load(open(a.joints))
J = {n: np.array(p) / kk + cc for n, p in Jd["joints"].items()}
SPINE = [J[n] for n in ("pelvis", "spine1", "spine2", "spine3", "neck")]
NA = 36
ANG = np.radians(np.arange(-180, 180, 360 // NA))                 # 0: toward the body


def sample(vol, X):
    return ndimage.map_coordinates(vol, (X / v - o).T, order=1, mode="constant", cval=1.0)


def spine_at(z):
    zs = np.array([p[2] for p in SPINE])
    k = np.clip(np.searchsorted(zs, z) - 1, 0, len(SPINE) - 2)
    f = np.clip((z - zs[k]) / max(zs[k + 1] - zs[k], 1e-9), 0, 1)
    return SPINE[k] * (1 - f) + SPINE[k + 1] * f


report = {"sides": {}}
total = 0
# depth inside the solid, for finding each section's centre
inside_all = sdf < 0
depth = ndimage.distance_transform_edt(inside_all) * v
for side in ("left", "right"):
    Sh, E, W = J[f"{side}_shoulder"], J[f"{side}_elbow"], J[f"{side}_wrist"]
    # the joints' line: refine_joints.py put the joints on the limb's centre, where the trace between
    # them can drift toward the body where the arm is glued to it
    line = np.array([Sh, E, W])
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s_line = np.r_[0, np.cumsum(seg)]
    s = np.arange(0, s_line[-1], v)
    T = np.stack([np.interp(s, s_line, line[:, d]) for d in range(3)], 1)
    X = np.gradient(ndimage.gaussian_filter1d(T, 4, axis=0), axis=0)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Lu = float(np.linalg.norm(E - Sh))
    M = np.stack([spine_at(t[2]) for t in T]) - T
    M -= (M * X).sum(1, keepdims=True) * X
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    N = np.cross(X, M)
    # each section's centre: the deepest point of the solid close to that line, across the arm - close:
    # a wider search found the torso, always deeper than the arm, through the glue
    r_look = 0.012 * H
    uu = np.linspace(-r_look, r_look, 15)
    UU, VV = np.meshgrid(uu, uu)
    C = T.copy()
    for i in range(len(s)):
        cand = T[i] + UU.ravel()[:, None] * M[i] + VV.ravel()[:, None] * N[i]
        dep = sample(depth, cand)
        C[i] = cand[int(np.argmax(dep))]
    C = ndimage.gaussian_filter1d(C, 3, axis=0)
    # rays across each section: where each leaves the solid
    reach = 0.14 * H
    steps = np.arange(v / 2, reach, v / 2)
    Ex = np.zeros((len(s), NA))
    for i in range(len(s)):
        dirs = np.cos(ANG)[:, None] * M[i] + np.sin(ANG)[:, None] * N[i]
        Xr = C[i][None, None, :] + dirs[:, None, :] * steps[None, :, None]
        out = sample(sdf, Xr.reshape(-1, 3)).reshape(NA, len(steps)) > 0
        Ex[i] = steps[np.where(out.any(1), out.argmax(1), len(steps) - 1)]
    outer = np.abs(ANG) >= np.radians(90)                           # away from the body: free
    r_typ = ndimage.median_filter(np.median(Ex[:, outer], 1), size=9, mode="nearest")
    # the arm's own radius all the way round: from the directions where it is free (its surface near
    # the typical radius), carried round the section into those where something is glued to it
    R_in = np.zeros_like(Ex)
    for i in range(len(s)):
        free = Ex[i] <= 1.35 * r_typ[i]
        if free.sum() < 3:
            R_in[i] = r_typ[i]
            continue
        a_f = ANG[free]
        R_in[i] = np.interp(ANG, a_f, Ex[i, free], period=2 * np.pi)
    # glued: the solid runs on past half again the arm's radius in that direction - any direction:
    # a vest wraps round the back of the arm as well as its inner side
    glued = (Ex > 1.5 * R_in) & (np.abs(ANG) <= np.radians(170))[None, :]
    # from the armpit to short of the wrist: cut_hands.py capped the arm at the wrist for the modelled
    # hand, and a cut through that cap left hands.py nothing to join the right hand onto (Pip's: 446
    # vertices joined where there should be 30,000)
    span = (s >= a.axilla * Lu) & (s <= s_line[-1] - 0.3 * (s_line[-1] - s_line[1]))
    glued &= span[:, None]
    glued = ndimage.binary_closing(glued, structure=np.ones((5, 3)), border_value=0) & (np.abs(ANG) <= np.radians(170))[None, :]
    glued = ndimage.binary_opening(glued, structure=np.ones((3, 1))) & span[:, None]
    # past the glue's ends by two directions each way (round the section, which wraps): an arc that
    # stopped exactly where the rays did left slivers a voxel thick at its ends, and a sliver in every
    # section is a bridge - Pip's sleeve still reached his vest's back through one
    wide = glued.copy()
    for k_ in (-2, -1, 1, 2):
        wide |= np.roll(glued, k_, axis=1)
    glued = wide & span[:, None]
    if not glued.any():
        report["sides"][side] = {"carved_voxels": 0}
        print(f"[arms] {side}: free of the body below the armpit; nothing cut", flush=True)
        continue
    # the arm's radius in each glued direction, carried round from its free directions
    r_dir = np.clip(R_in, 0.8 * r_typ[:, None], 1.25 * r_typ[:, None])
    r_dir = ndimage.median_filter(r_dir, size=(5, 1), mode="nearest")
    delta = 1.5 * v
    gap = max(3 * v, a.gap * H)

    def around(rows_, reach_):
        """The voxels of the box round these sections out to reach_: their indices, nearest section,
        distance from its centre line and angle about it."""
        b0 = np.clip(np.floor(C[rows_].min(0) / v - reach_ / v).astype(int) - o, 0, np.array(sdf.shape) - 1)
        b1 = np.clip(np.ceil(C[rows_].max(0) / v + reach_ / v).astype(int) - o + 1, 1, np.array(sdf.shape))
        sl_ = tuple(slice(p, q) for p, q in zip(b0, b1))
        g_ = np.stack(np.meshgrid(*[np.arange(p, q) for p, q in zip(b0, b1)], indexing="ij"), -1).reshape(-1, 3)
        Xw_ = (g_ + o) * v
        _, ni_ = cKDTree(C).query(Xw_)
        d_ = Xw_ - C[ni_]
        d_ -= (d_ * X[ni_]).sum(1, keepdims=True) * X[ni_]
        return sl_, ni_, np.linalg.norm(d_, axis=1), np.arctan2((d_ * N[ni_]).sum(1), (d_ * M[ni_]).sum(1))

    def carve(glued_):
        rows_ = np.nonzero(glued_.any(1))[0]
        sl_, ni_, rho_, phi_ = around(rows_, 1.4 * float(r_dir[rows_].max()) + gap + 4 * v)
        fa_ = (phi_ + np.pi) / (2 * np.pi / NA)                       # fractional angle index
        a0_ = np.floor(fa_).astype(int) % NA
        a1_ = (a0_ + 1) % NA
        wa_ = fa_ - np.floor(fa_)
        on_ = glued_[ni_, a0_] | glued_[ni_, a1_]
        r1_ = (1 - wa_) * r_dir[ni_, a0_] + wa_ * r_dir[ni_, a1_] + delta
        shell_ = np.minimum(rho_ - r1_, r1_ + gap - rho_)             # positive inside the shell
        blk_ = sdf[sl_].reshape(-1)
        new_ = np.where(on_ & (shell_ > -2 * v), np.maximum(blk_, shell_), blk_)
        sdf[sl_] = new_.reshape(sdf[sl_].shape)
        return int(((blk_ < 0) & (new_ >= 0)).sum())

    def still_joined():
        """Stretches of the span, 5% of the upper arm at a time, where the arm's cross-section still reaches
        the body through the solid: the rays go round a section 10 degrees apart, and between two of
        them - or at a section whose centre was misjudged - a web can survive. It did on Vex (at 0.70 of
        her left upper arm, at 0.40 and 0.55 of her right) and on four more of the seven characters, and
        through it the rig's weights reached the jacket's side, which lifted with her arms. A slab is cut
        straight across the joints' line (nearest-section slabs gave torso voxels to the armpit's sections
        and missed the webs)."""
        rows_ = np.nonzero(span)[0]
        r0 = float(np.median(r_typ[rows_]))
        reach_ = 5.0 * r0
        b0 = np.clip(np.floor(T[rows_].min(0) / v - reach_ / v).astype(int) - o, 0, np.array(sdf.shape) - 1)
        b1 = np.clip(np.ceil(T[rows_].max(0) / v + reach_ / v).astype(int) - o + 1, 1, np.array(sdf.shape))
        sl_ = tuple(slice(p, q) for p, q in zip(b0, b1))
        solid_ = sdf[sl_] < 0
        g_ = np.stack(np.meshgrid(*[np.arange(p, q) for p, q in zip(b0, b1)], indexing="ij"), -1).reshape(-1, 3)
        Xw_ = (g_ + o) * v
        best_d, best_s = np.full(len(Xw_), np.inf), np.zeros(len(Xw_))
        for k_ in range(len(line) - 1):
            ab_ = line[k_ + 1] - line[k_]
            u_ = np.clip(((Xw_ - line[k_]) @ ab_) / float(ab_ @ ab_), 0, 1)
            d_ = np.linalg.norm(Xw_ - (line[k_] + u_[:, None] * ab_), axis=1)
            b_ = d_ < best_d
            best_d[b_], best_s[b_] = d_[b_], (s_line[k_] + u_ * seg[k_])[b_]
        sol = solid_.reshape(-1) & (best_d < reach_)
        bad = []
        width = 0.05 * Lu
        s0 = s[rows_].min()
        while s0 < s[rows_].max():
            inslab = sol & (best_s >= s0) & (best_s < s0 + width)
            if inslab.any():
                lab_, _ = ndimage.label(inslab.reshape(solid_.shape))
                lab_ = lab_.reshape(-1)
                core = np.unique(lab_[inslab & (best_d < 0.4 * r0)])
                core = core[core > 0]
                if len(core) and (np.isin(lab_, core) & (best_d > 3.0 * r0)).sum() > 50:
                    bad.append(np.nonzero((s >= s0) & (s < s0 + width))[0])
            s0 += width
        return bad

    # Webs: the generator fills the space between an arm and the body with membranes of solid a few
    # voxels thick - Knight's forearms were tied to his hips by fins that stretched into grey sheets
    # hanging from his arms when he waved. Between the arm and the body only (the body's side of the arm,
    # past its surface, within 3.5 arm radii, down the span), solid thinner than about 9 mm goes: what an
    # opening of the solid by 3 voxels removes. A torso or an arm is far thicker, and fingers, hair and
    # ears are nowhere near this region.
    rows_w = np.nonzero(span)[0]
    if len(rows_w):
        sl_w, ni_w, rho_w, phi_w = around(rows_w, (float(os.environ.get("CF_WEB_R", "3.5")) + 0.1) * float(r_typ[rows_w].max()))
        blk_w = sdf[sl_w]
        # CF_WEB_K / CF_WEB_R: how thick a web (voxels) and how far out (arm radii) - 4 and 4.5 take 1.8x as
        # much of Knight's armour webs, and 60% more of what lies by Pip's arms; the defaults are the tested ones
        k_w = int(os.environ.get("CF_WEB_K", "3"))
        solid_w = blk_w < 0
        core_w = blk_w < -k_w * v                                       # eroded by k voxels
        opened_w = ndimage.distance_transform_edt(~core_w) <= k_w + 0.5  # ...and grown back
        thin_w = (solid_w & ~opened_w).reshape(-1)
        region_w = (span[ni_w] & (rho_w > 1.25 * r_typ[ni_w]) & (rho_w < float(os.environ.get("CF_WEB_R", "3.5")) * r_typ[ni_w])
                    & (np.abs(phi_w) <= np.radians(120)))
        web = thin_w & region_w
        if web.any():
            flat = blk_w.reshape(-1).copy()
            flat[web] = 0.5 * v
            sdf[sl_w] = flat.reshape(blk_w.shape)
            print(f"[arms] {side}: {int(web.sum()):,} voxels of web between the arm and the body taken away",
                  flush=True)
        total += int(web.sum())

    n_cut = carve(glued)
    closed = 0
    for _ in range(3):
        bad = still_joined()
        if not bad:
            break
        # where the arm still reaches the body: carve every direction facing it there (and a little either
        # side), and the glue of the sections round about
        add = np.zeros_like(glued)
        for rows_b in bad:
            if not len(rows_b):
                continue
            lo_i, hi_i = max(0, rows_b.min() - 3), min(len(s), rows_b.max() + 4)
            add[lo_i:hi_i] |= (np.abs(ANG) <= np.radians(120))[None, :]
            add[lo_i:hi_i] |= glued[max(0, lo_i - 8):min(len(s), hi_i + 8)].any(0)[None, :]
            # and further out: where the sleeve bulges past the arm's usual radius the first ring ran inside
            # it, and the sleeve's outer skin still lay on the jacket beyond (Vex's right arm at 0.41)
            r_dir[lo_i:hi_i] = np.minimum(r_dir[lo_i:hi_i] * 1.12, 1.6 * r_typ[lo_i:hi_i, None])
        glued = (glued | add) & span[:, None]
        n_cut += carve(glued)
        closed += len(bad)
    left_joined = still_joined() if closed else []
    if closed:
        print(f"[arms] {side}: {closed} stretch(es) of the arm still joined the body after the first cut - cut again"
              + (f"; {len(left_joined)} still joined" if left_joined else "; now free"), flush=True)
    total += n_cut
    along = s[glued.any(1)] / Lu
    report["sides"][side] = {"glued_from": round(float(along.min()), 2), "glued_to": round(float(along.max()), 2),
                             "arm_radius": round(float(np.median(r_typ) / H), 4), "carved_voxels": n_cut,
                             "recut": closed, "still_joined": len(left_joined)}
    print(f"[arms] {side}: glued to the body from {along.min():.2f} to {along.max():.2f} of the upper arm's length "
          f"down the arm; arm radius {np.median(r_typ) / H * 100:.1f}% of height; opened a {gap / H * 100:.1f}% gap - "
          f"{n_cut:,} voxels", flush=True)

# Pieces the cuts split off the body - the middle of a web cut at both ends floated between Knight's
# forearm and his hip - go: small pieces that were part of the body's one piece before the cuts. Parts
# that stood apart already (a kept hat, say) are left as they were.
lab_in, _ = ndimage.label(S["sdf"] < 0)
sz_in = np.bincount(lab_in.ravel()); sz_in[0] = 0
main_in = int(sz_in.argmax())
lab_out, n_out = ndimage.label(sdf < 0)
sz_out = np.bincount(lab_out.ravel()); sz_out[0] = 0
main_out = int(sz_out.argmax())
dropped = 0
for c in np.nonzero((sz_out > 0) & (sz_out < 0.01 * sz_out.sum()))[0]:
    if c == main_out:
        continue
    where = lab_out == c
    if (lab_in[where] == main_in).all():
        sdf[where] = 0.5 * v
        dropped += int(sz_out[c])
if dropped:
    print(f"[arms] {dropped:,} voxels in pieces the cuts split off the body dropped", flush=True)

np.savez_compressed(a.out, sdf=sdf, origin_ijk=o, voxel=v, background=float(S["background"]), height=H)
if a.report:
    json.dump(report, open(a.report, "w"), indent=1)
print(f"[arms] -> {a.out} ({total:,} voxels opened)", flush=True)
