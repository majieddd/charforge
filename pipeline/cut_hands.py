"""Remove the generated hands from the solid, and say where the new ones go.

A generator's hand is a paddle: the fingers fused, often curled, sometimes welded to the thigh
it hangs beside. It is cut out of the solid volume - not out of the triangles, where a cut
along face edges is ragged and a weld to the thigh would drag the thigh along - and replaced by
blender/hands.py with a modelled hand that has real fingers and a finger skeleton.

Where to cut. The wrist is the narrowest cross-section of the arm near the estimated wrist joint:
below a cuff that is the bare wrist, with no cuff it is the joint itself. The volume beyond that
plane goes if it is nearer the hand's bone than to any leg or spine bone and within reach of
it, so a thigh the hand rested against keeps its surface. The arm is left closed by a flat cap
at the plane; hands.py unions the new hand onto it.

What to build. The hand's length is the old hands' reach past the cut - one length for both, the
median of the two measurements and the adult norm, within human ranges (0.10-0.125 of height);
its wrist cross-section is the arm's at the cut, capped at a wrist's
size when the cut runs through a sleeve. Its orientation follows the rest pose convention of
the animation library: the rotation that lowers a T-posed arm to this arm is applied to a
T-posed hand (palm down, thumb forward), so raising the arm back to a T gives exactly that hand.

Run: python cut_hands.py --solid solid.npz --sdf sdf.npz --joints joints_refined.json \
         --out solid_cut.npz --spec hands_spec.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--solid", required=True)
ap.add_argument("--sdf", required=True)
ap.add_argument("--joints", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--spec", required=True)
ap.add_argument("--parts", default=None, help="parts.json: whether the forearm at the wrist is bare skin")
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


def inside(pts):
    idx = (pts / v - o).T
    return ndimage.map_coordinates(sdf, idx, order=1, mode="constant", cval=1.0) < 0


def seg_dist(X, p0, p1):
    ab = p1 - p0
    t = np.clip(((X - p0) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)[:, None]
    return np.linalg.norm(X - (p0 + t * ab), axis=1)


def rotation_between(u, w):
    u, w = u / np.linalg.norm(u), w / np.linalg.norm(w)
    c = float(u @ w)
    ax = np.cross(u, w)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return np.eye(3)
    ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


spec = {"height": H, "sides": {}}
meas = {}
for side, sgn in (("left", 1.0), ("right", -1.0)):
    E, W, Hd = J[f"{side}_elbow"], J[f"{side}_wrist"], J[f"{side}_hand"]
    # the forearm's direction at the wrist, from refine_joints.py's traced centre line when there
    # is one (the elbow-to-wrist chord bends with the forearm's taper)
    tr = np.array(Jd.get("traces", {}).get(f"{side}_arm", []))
    if len(tr) > 10:
        tr = tr / kk + cc
        iw = int(np.argmin(np.linalg.norm(tr - W, axis=1)))
        back = tr[max(0, iw - max(3, int(0.03 * H / (0.004 * H))))]
        x = (W - back) / np.linalg.norm(W - back)
    else:
        x = (W - E) / np.linalg.norm(W - E)
    # the hand lies beyond the wrist, away from the elbow: a traced line that turns back there (knight2's
    # doubled back up his vambrace) would carve the forearm and keep the hand
    ew = (W - E) / np.linalg.norm(W - E)
    if x @ ew < 0.3:
        print(f"[hands] {side}: the traced forearm turns back at the wrist - the elbow-to-wrist direction used",
              flush=True)
        x = ew
    R = rotation_between(np.array([sgn, 0.0, 0.0]), x)
    z = R @ np.array([0.0, 0.0, 1.0])
    z -= (z @ x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x) * (1.0 if side == "right" else -1.0)      # thumb side; mirrored on the left
    # the wrist cross-section, across the forearm at the refined wrist (its narrowest point)
    rr = np.linspace(0, 0.045 * H, 25)[:, None]
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)[None, :]
    disc_y = (rr * np.cos(th)).ravel()
    disc_z = (rr * np.sin(th)).ravel()
    C = W.copy()
    ins = inside(C + disc_y[:, None] * y + disc_z[:, None] * z)
    lab_, n_ = ndimage.label(ins.reshape(rr.shape[0], th.shape[1]), structure=np.ones((3, 3)))
    ry = float(np.ptp(disc_y[ins])) / 2 if ins.any() else 0.0
    rz = float(np.ptp(disc_z[ins])) / 2 if ins.any() else 0.0
    ts = [0.0]
    k = 0
    fl = 0.0
    # the old hand: solid samples beyond the cut, nearer the hand bone than to the legs
    L0 = 0.108 * H
    ii = np.argwhere(np.abs(sdf) < 3 * v)                     # near-surface voxels only
    Xw = (ii + o) * v
    beyond = (Xw - C) @ x
    cand = (beyond > 0) & (beyond < 1.6 * L0)
    Xc = Xw[cand]
    d_hand = seg_dist(Xc, C, C + x * L0 * 1.2)
    d_other = np.min([seg_dist(Xc, J[f"{s}_hip"], J[f"{s}_knee"]) for s in ("left", "right")]
                     + [seg_dist(Xc, J["pelvis"], J["spine3"])], axis=0)
    mine = (d_hand < 0.6 * L0) & (d_hand < d_other)
    L_meas = float(np.percentile(((Xc[mine] - C) @ x), 99.5)) if mine.any() else L0
    if len(tr) > 10:                                   # the traced hand, when it reached further
        L_meas = max(L_meas, float((tr[-1] - C) @ x))
    meas[side] = dict(C=C, x=x, y=y, z=z, ry=ry, rz=rz, L_meas=L_meas)

# One length for both hands. Each side's measurement fails its own way - a curled or fused hand
# measures short, a hand welded to the thigh long - and clamping each on its own gave aoi a left
# hand of 17.5 cm and a right of 21.9 (measured 7.7 and 25.8): one hand a quarter bigger than the
# other. People's hands match, so the length is the median of the two measurements and the adult
# norm (0.108 of height): two agreeing measurements win, and one wild one cannot. A hand shorter
# than a tenth of the body reads as a child's, so that stays the floor.
L_both = float(np.clip(np.median([meas["left"]["L_meas"], meas["right"]["L_meas"], 0.108 * H]),
                       0.10 * H, 0.125 * H))
# A bare forearm sets the hand's bulk. The modelled hand has a person's proportions, clipped to a
# person's wrist; on Gray - "extremely muscular", 11-13 cm across the wrist - that left a normal hand on
# a forearm twice its width, the forearm ending in a flat step like a cuff. A sleeve measures as wide
# (Pip's cuffs, knight2's gauntlets, Bo's chef's sleeves), so the width alone cannot say; the part
# labels can: the vertices on the forearm just above the cut are "arms" (skin) on a bare one and "top"
# on a sleeve - Gray, Mara and the boy scout 0.92-1.00 skin, every sleeved character 0.13 or less.
bulk, skin_frac = 1.0, None
if a.parts and os.path.exists(a.parts):
    _pl = np.array(json.load(open(a.parts))["class_labels"])
    if len(_pl) == len(P):
        n_all = n_skin = 0
        for side_ in ("left", "right"):
            m_ = meas[side_]
            along_ = (P - m_["C"]) @ m_["x"]
            rad_ = np.linalg.norm((P - m_["C"]) - np.outer(along_, m_["x"]), axis=1)
            band_ = (along_ > -0.8 * 0.108 * H) & (along_ < -0.1 * 0.108 * H) & (rad_ < 0.06 * H) & (_pl > 0)
            n_all += int(band_.sum())
            n_skin += int(np.isin(_pl[band_], (12, 13)).sum())      # "arms", "hands"
        if n_all >= 100:
            skin_frac = n_skin / n_all
    if skin_frac is not None and skin_frac >= 0.6:
        # one bulk for both hands, from the wrists' own cross-sections against a person's (0.15 x 0.10
        # of the hand's length), as far as 1.6x
        L_ = L_both
        ratios = [np.sqrt(meas[s_]["ry"] * meas[s_]["rz"]) / L_ / np.sqrt(0.15 * 0.10) for s_ in ("left", "right")]
        bulk = float(np.clip(np.median(ratios), 1.0, 1.6))
    print(f"[hands] forearm at the wrist: " + ("unlabelled" if skin_frac is None else f"{skin_frac:.0%} bare skin")
          + (f" - hands {bulk:.2f}x as broad and thick, to meet it" if bulk > 1.01 else " - a person's hand"), flush=True)
for side, sgn in (("left", 1.0), ("right", -1.0)):
    m_ = meas[side]
    C, x, y, z, ry, rz, L_meas = (m_[k] for k in ("C", "x", "y", "z", "ry", "rz", "L_meas"))
    L = L_both
    # carve it out of the volume: everything beyond the plane that belongs to this hand
    box_lo = np.floor((np.minimum(C - 1.4 * L, C + 1.4 * L) / v)).astype(int) - o
    box_hi = np.ceil((np.maximum(C - 1.4 * L, C + 1.4 * L) / v)).astype(int) - o
    box_lo = np.clip(box_lo, 0, np.array(sdf.shape) - 1)
    box_hi = np.clip(box_hi, 1, np.array(sdf.shape))
    sl = tuple(slice(l, h) for l, h in zip(box_lo, box_hi))
    g = np.stack(np.meshgrid(*[np.arange(l, h) for l, h in zip(box_lo, box_hi)], indexing="ij"), -1)
    Xb = (g.reshape(-1, 3) + o) * v
    bb = (Xb - C) @ x
    dh = seg_dist(Xb, C, C + x * L * 1.2)
    do = np.min([seg_dist(Xb, J[f"{s}_hip"], J[f"{s}_knee"]) for s in ("left", "right")]
                + [seg_dist(Xb, J["pelvis"], J["spine3"])], axis=0)
    cut = (bb > 0) & (dh < 0.75 * L) & (dh < do)
    blk = sdf[sl].reshape(-1)
    # a signed distance to the carved region: outside wherever the cut says, keeping the
    # remaining surface's values elsewhere; the plane itself becomes the cap
    blk = np.where(cut, np.maximum(blk, np.minimum(bb, 3 * v)), blk)
    sdf[sl] = blk.reshape(sdf[sl].shape)
    spec["sides"][side] = {
        "cut_point": C.tolist(), "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
        "mirror": side == "left", "length": L, "length_measured": L_meas,
        "wrist_radius": [ry / L, rz / L], "bulk": bulk, "bare_skin": skin_frac,
    }
    print(f"[hands] {side}: cut at the wrist; hand length {L / H * 175:.1f} cm (measured {L_meas / H * 175:.1f}); wrist "
          f"{2 * ry / H * 175:.1f} x {2 * rz / H * 175:.1f} cm; carved {int(cut.sum()):,} voxels", flush=True)

np.savez_compressed(a.out, sdf=sdf, origin_ijk=o, voxel=v, background=float(S["background"]), height=H)
json.dump(spec, open(a.spec, "w"), indent=1)
print(f"[hands] -> {a.out}, {a.spec}", flush=True)
