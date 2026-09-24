"""Put every estimated joint inside the body, on the centre line of its limb.

The skeleton comes from a 2D pose model on a front and a side render. The front view fixes
left-right and height well. Depth comes from the side view, where the arms overlap the torso and
each other, and it is the weak axis: on Juno the pose model gave both wrists the same depth, and
both wrists, both hands, both ankles and the toes landed 1-3 cm OUTSIDE the mesh. A bone outside
the surface cannot be skinned by anything that reasons about the inside of the body (bone heat
refuses it: 100% of vertices unweighted), and a joint off its limb's centre bends the limb around
the wrong point.

With the solid's signed distance (pipeline/solidify.py) the limbs can be traced instead. From
the shoulder (or hip), step along the limb and re-centre at each step on the deepest point of
the cross-section - the limb's medial axis - until the limb ends. Each joint then takes its
height and left-right position from the front view, as estimated, and its place on the traced
centre line: the traced point nearest to it in the front plane. Depth comes from the body, not
from the side view. The wrist is refined once more to the narrowest point of the forearm near
it, where a cuff ends and the hand begins; the ankle keeps its estimated height.

The spine, neck and head are centred left-right only - their depth is anatomy (the spine sits
behind the middle of the torso), which the side view already gives.

Output: the joints.json layout plus "traces" (each limb's centre line, in the joints' frame).
Run: python refine_joints.py --joints joints.json --solid solid.npz --sdf sdf.npz --out joints_refined.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--joints", required=True)
ap.add_argument("--solid", required=True, help="solidify.py's output: signed distance, negative inside")
ap.add_argument("--sdf", required=True, help="sdf_io.py's output, for the source mesh's bounding box")
ap.add_argument("--out", required=True)
a = ap.parse_args()

S = np.load(a.solid)
sdf, v, o, H = S["sdf"], float(S["voxel"]), S["origin_ijk"], float(S["height"])
P = np.load(a.sdf)["points"]
lo, hi = P.min(0), P.max(0)
kk = 2.0 / float(hi[2] - lo[2])                      # joints.json frame: 2 units tall, bbox-centred
cc = (lo + hi) / 2
Jd = json.load(open(a.joints))
J0 = {n: np.array(p) / kk + cc for n, p in Jd["joints"].items()}
J = {n: p.copy() for n, p in J0.items()}
BIG = float(np.abs(sdf).max())


def sample(pts):
    idx = (np.atleast_2d(pts) / v - o).T
    return ndimage.map_coordinates(sdf, idx, order=1, mode="constant", cval=BIG)


def basis(axis):
    axis = axis / np.linalg.norm(axis)
    t0 = np.cross(axis, [0, 0, 1.0])
    if np.linalg.norm(t0) < 1e-3:
        t0 = np.cross(axis, [1.0, 0, 0])
    t0 /= np.linalg.norm(t0)
    return axis, t0, np.cross(axis, t0)


def deepest_in_disc(c, axis, R, n_r=10, n_t=20):
    _, t0, t1 = basis(axis)
    rr = np.linspace(0, R, n_r)[:, None]
    th = np.linspace(0, 2 * np.pi, n_t, endpoint=False)[None, :]
    pts = c + (rr * np.cos(th)).reshape(-1, 1) * t0 + (rr * np.sin(th)).reshape(-1, 1) * t1
    val = sample(pts)
    k = int(np.argmin(val + 1e-3 * np.linalg.norm(pts - c, axis=1) / R * v))
    return pts[k], float(val[k])


def section(c, axis, R, n=29):
    """The limb's cross-section through c: the connected piece of the solid in the plane across
    `axis` that contains (or is nearest) c. Returns its centroid and area (0 if empty)."""
    _, t0, t1 = basis(axis)
    u = np.linspace(-R, R, n)
    g0, g1 = np.meshgrid(u, u, indexing="ij")
    pts = c + g0.reshape(-1, 1) * t0 + g1.reshape(-1, 1) * t1
    ins = (sample(pts) < 0).reshape(n, n)
    if not ins.any():
        return c, 0.0
    lab, k = ndimage.label(ins)
    mid = lab[n // 2, n // 2]
    if mid == 0:                                    # the centre missed: take the nearest piece
        ii = np.argwhere(ins)
        j = ii[np.argmin(((ii - n // 2) ** 2).sum(1))]
        mid = lab[j[0], j[1]]
    m = lab == mid
    cen = pts.reshape(n, n, 3)[m].mean(0)
    return cen, float(m.sum()) * (2 * R / (n - 1)) ** 2


def aspect(c, axis, R, n=29):
    """How flat the limb's cross-section is at c (1 = round). A palm is flat, a wrist is not."""
    _, t0, t1 = basis(axis)
    u = np.linspace(-R, R, n)
    g0, g1 = np.meshgrid(u, u, indexing="ij")
    pts = c + g0.reshape(-1, 1) * t0 + g1.reshape(-1, 1) * t1
    ins = (sample(pts) < 0).reshape(n, n)
    lab, _ = ndimage.label(ins)
    mid = lab[n // 2, n // 2]
    if mid == 0 or (lab == mid).sum() < 6:
        return 1.0
    q = np.stack([g0[lab == mid], g1[lab == mid]], 1)
    ev = np.linalg.eigvalsh(np.cov(q.T))
    return float(np.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12)))


def trace(start, direction, max_len, step, stop_growth=None):
    """Follow a limb from start: step along it and re-centre on the centroid of its cross-section
    - the connected piece only, inside a window scaled to the limb's local thickness, so a torso
    or thigh the limb passes beside does not pull it. Stops where the limb ends, or (when
    stop_growth is set) where the cross-section suddenly grows, i.e. the limb joins the torso."""
    pts = [start.copy()]
    d = direction / np.linalg.norm(direction)
    r0 = max(-float(sample(start)[0]), 0.008 * H)
    area = [section(start, d, float(np.clip(2.5 * r0, 0.02 * H, 0.06 * H)))[1]]
    for _ in range(int(max_len / step)):
        pred = pts[-1] + d * step
        r = max(-float(sample(pred)[0]), 0.008 * H)
        p, ar = section(pred, d, float(np.clip(2.5 * r, 0.02 * H, 0.06 * H)))
        if ar <= 0:
            break
        if stop_growth and len(area) >= 6 and ar > stop_growth * np.median(area[:6]):
            break
        dn = p - pts[-1]
        dn /= max(np.linalg.norm(dn), 1e-9)
        d = 0.7 * d + 0.3 * dn
        d /= np.linalg.norm(d)
        pts.append(p)
        area.append(ar)
    return np.array(pts), np.array(area)


def limb(mid_est, toward_root, toward_end, len_root, len_end, step):
    """Trace a limb both ways from a joint in its middle (elbow, knee), where it stands alone."""
    m, _ = deepest_in_disc(mid_est, toward_end, 0.04 * H)
    up, _ = trace(m, toward_root, 1.2 * len_root, step, stop_growth=2.4)
    dn, ar = trace(m, toward_end, 1.5 * len_end, step)
    pts = np.vstack([up[::-1], dn[1:]])
    area = np.concatenate([np.full(len(up), np.nan), ar[1:]])
    return pts, area, len(up) - 1


def nearest_front(trace_pts, target):
    """Index of the traced point nearest the target in the front plane (x, z) - depth ignored."""
    return int(np.argmin(np.hypot(trace_pts[:, 0] - target[0], trace_pts[:, 2] - target[2])))


def into_limb(p, reach):
    """Put a limb joint inside its limb along the axis the front view cannot see: the centre of
    the run of solid along y at the joint's (x, z) nearest the estimate. The pose model's depth
    comes from the side view, where an A-posed arm lies over the torso and the hair (aoi's side
    view put both wrists on her ponytail, 10 cm behind the arms). When no solid lies along y
    there - the front-view point fell just off a thin limb - the nearest (x, z) within reach
    that has some."""
    ys = (np.arange(sdf.shape[1]) + o[1]) * v
    for r in np.linspace(0.0, reach, 7):
        best = None
        for th in (np.linspace(0, 2 * np.pi, 12, endpoint=False) if r > 0 else [0.0]):
            q = np.stack([np.full_like(ys, p[0] + r * np.cos(th)), ys, np.full_like(ys, p[2] + r * np.sin(th))], 1)
            ins = sample(q) < 0
            if not ins.any():
                continue
            lab_, k_ = ndimage.label(ins)
            for i in range(1, k_ + 1):
                c_ = float(ys[lab_ == i].mean())
                if best is None or abs(c_ - p[1]) < abs(best[1] - p[1]):
                    best = (q[0, 0], c_, q[0, 2])
        if best is not None:
            return np.array(best)
    return p


for side in ("left", "right"):
    for jn in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        n_ = f"{side}_{jn}"
        q_ = into_limb(J0[n_], 0.03 * H)
        if np.linalg.norm(q_ - J0[n_]) > 0.012 * H:
            print(f"[joints]   {n_:15s} put inside its limb: {np.linalg.norm(q_ - J0[n_]) / H * 175:.1f} cm "
                  f"(depth {J0[n_][1]:+.3f} -> {q_[1]:+.3f})", flush=True)
        J0[n_] = q_
        J[n_] = q_.copy()

step = 0.004 * H
traces = {}
arms_done = {}
for side in ("left", "right"):
    # arms: traced from the elbow out to the hand and in until the torso
    S_, E_, W_, Hd_ = (J0[f"{side}_{n}"] for n in ("shoulder", "elbow", "wrist", "hand"))
    # out to the fingertips whatever the estimate says: the trace ends where the limb does
    tr, area, ie = limb(E_, S_ - E_, W_ - E_, np.linalg.norm(S_ - E_),
                        max(np.linalg.norm(W_ - E_) + np.linalg.norm(Hd_ - W_), 0.30 * H), step)
    traces[f"{side}_arm"] = tr
    if len(tr) - ie > 10:
        J[f"{side}_elbow"] = tr[ie]
        iw = ie + nearest_front(tr[ie:], W_)
        # A wrist sits a hand's length from the fingertips (~0.108 of height). The pose model,
        # trained on photographs, reads a big anime cuff as the wrist - aoi's came out 20 cm from
        # her fingertips, at the cuff. When the estimate is outside the range a wrist can be, the
        # wrist is the thinnest cross-section inside that range.
        seg_ = np.linalg.norm(np.diff(tr, axis=0), axis=1)
        to_tip = np.concatenate([np.cumsum(seg_[::-1])[::-1], [0.0]])
        if not (0.07 * H <= to_tip[iw] <= 0.16 * H):
            win = np.nonzero((to_tip >= 0.06 * H) & (to_tip <= 0.17 * H) & (np.arange(len(tr)) > ie + 3))[0]
            iw_new, how_ = None, ""
            if len(win):
                # The palm is flat and the wrist is round: walking back from the fingertips, the wrist is
                # where the section stops being palm-flat. (The thinnest section is a finger when the
                # fingers are modelled apart - pip's cartoon hands put the wrist mid-hand.)
                asp = np.array([aspect(tr[i], tr[min(i + 2, len(tr) - 1)] - tr[max(i - 2, 0)], 0.04 * H) for i in win])
                order_ = np.argsort(-to_tip[win])[::-1]            # fingertip end first
                palm_areas, prev_k = [], None
                for k_ in order_:
                    ar_k = area[win[k_]]
                    if palm_areas and asp[k_] < 1.5:
                        iw_new, how_ = int(win[k_]), "where the palm rounds into the wrist"
                        break
                    # ...or where the hand enters a sleeve: pip's cartoon palm is 22 cm2 and his
                    # hoodie cuff 70 cm2 - and as oval as the palm, so it never "rounds"
                    if len(palm_areas) >= 3 and np.isfinite(ar_k) and ar_k > 1.8 * np.median(palm_areas):
                        iw_new, how_ = int(win[prev_k]), "where the hand enters the sleeve"
                        break
                    if asp[k_] > 1.8 and np.isfinite(ar_k):
                        palm_areas.append(ar_k)
                    prev_k = k_
                if iw_new is None and np.isfinite(area[win]).any():
                    iw_new, how_ = int(win[np.nanargmin(area[win])]), "at the thinnest point"
            if iw_new is not None:
                print(f"[joints]   {side} wrist: the estimate was {to_tip[iw] / H * 175:.0f} cm from the fingertips "
                      f"(at 1.75 m) - moved {how_}, {to_tip[iw_new] / H * 175:.0f} cm from them", flush=True)
                iw = iw_new
        # The front view places the wrist well along the arm; but a generated hand is a flat
        # paddle, and if the estimate landed in it the cut would leave half the old hand behind.
        # Walk back toward the elbow while the cross-section is still palm-flat.
        for _ in range(int(0.06 * H / step)):
            if iw - 1 <= ie + 3:
                break
            dvec = tr[min(iw + 2, len(tr) - 1)] - tr[iw - 2]
            if aspect(tr[iw], dvec, 0.04 * H) < 1.9:
                break
            iw -= 1
        arms_done[side] = (tr, to_tip, ie, iw)
        J[f"{side}_wrist"] = tr[iw]
        J[f"{side}_hand"] = tr[-1] if len(tr) - 1 > iw else J[f"{side}_wrist"] + (Hd_ - W_)
        # The elbow started where the pose model put it. Upper arm to forearm is ~1.27 : 1 on a
        # person (0.186 and 0.146 of height); far outside that, the estimate is wrong and the elbow
        # goes where the proportion puts it along the traced arm.
        up_len = np.linalg.norm(tr[ie] - S_)
        fore_len = float(np.sum(np.linalg.norm(np.diff(tr[ie:iw + 1], axis=0), axis=1)))
        if fore_len > 0 and not (0.8 <= up_len / fore_len <= 1.9):
            d_s = np.linalg.norm(tr[:iw + 1] - S_, axis=1)
            arc_w = np.concatenate([np.cumsum(np.linalg.norm(np.diff(tr[:iw + 1], axis=0), axis=1)[::-1])[::-1], [0.0]])
            ie_new = int(np.argmin(np.abs(d_s / np.maximum(d_s + arc_w, 1e-9) - 0.56)))
            print(f"[joints]   {side} elbow: upper arm {up_len / H * 175:.0f} cm to forearm {fore_len / H * 175:.0f} cm "
                  f"- moved to 56% of the way from shoulder to wrist", flush=True)
            J[f"{side}_elbow"] = tr[ie_new]
        # the shoulder keeps its estimated height and side, and takes the traced arm's depth
        J[f"{side}_shoulder"][1] = tr[0][1]
    # legs: traced from the knee down to the sole and up until the crotch
    Hp, K_, A_ = (J0[f"{side}_{n}"] for n in ("hip", "knee", "ankle"))
    tr, area, ik = limb(K_, Hp - K_, A_ - K_, np.linalg.norm(Hp - K_), max(np.linalg.norm(A_ - K_), 0.30 * H), step)
    traces[f"{side}_leg"] = tr
    if len(tr) - ik > 10:
        J[f"{side}_knee"] = tr[ik]
        below = tr[ik:]
        if below[:, 2].min() > A_[2] + 0.03 * H:
            # the trace stopped short of the foot - pip's wide cargo shorts end in a hem the section
            # could not follow into the shin - so the estimated ankle stands (pulled into its limb)
            print(f"[joints]   {side} leg: trace ended {(below[:, 2].min() - A_[2]) / H * 175:.0f} cm above the "
                  f"ankle - the estimate kept", flush=True)
        else:
            ia = ik + int(np.argmin(np.abs(below[:, 2] - A_[2])))
            J[f"{side}_ankle"] = tr[ia]
        J[f"{side}_hip"][1] = tr[0][1]
        # toe: the ankle's offset kept, then pulled inside the shoe at its estimated height
        toe = J[f"{side}_ankle"] + (J0[f"{side}_foot"] - A_)
        p, val = deepest_in_disc(toe, np.array([0, -1.0, 0]), 0.03 * H)
        J[f"{side}_foot"] = p if val < 0 else toe

# The two wrists the same distance from their fingertips: each side's thinnest point wanders by a
# few centimetres, and hands.py builds both hands one length - left alone, one arm came out 3 cm
# shorter than the other.
if len(arms_done) == 2:
    target = float(np.mean([t_[1][t_[3]] for t_ in arms_done.values()]))
    for side, (tr, to_tip, ie, iw) in arms_done.items():
        cand_ = np.nonzero(np.arange(len(tr)) > ie + 3)[0]
        if len(cand_):
            iw2 = int(cand_[np.argmin(np.abs(to_tip[cand_] - target))])
            J[f"{side}_wrist"] = tr[iw2]

# spine, neck, head: centre left-right only. With both legs traced, the midline is halfway
# between them - the one measure of the body's centre that hair cannot move (the deepest point
# across the neck slid 9 cm into aoi's ponytail). Otherwise the deepest point within 3.5 cm.
legs_ok = all(len(traces.get(f"{sd}_leg", [])) > 10 for sd in ("left", "right"))
mid_x = (0.5 * (J["left_knee"][0] + J["right_knee"][0]) + 0.5 * (J["left_ankle"][0] + J["right_ankle"][0])) / 2
for n in ("pelvis", "spine1", "spine2", "spine3", "neck", "head"):
    if legs_ok:
        J[n][0] = mid_x
        continue
    xs = J[n] + np.linspace(-0.02 * H, 0.02 * H, 17)[:, None] * np.array([1.0, 0, 0])
    val = sample(xs)
    k = int(np.argmin(val + 1e-4 * np.abs(xs[:, 0] - J[n][0])))
    if val[k] < 0:
        J[n] = xs[k]
J["head_top"] = J["head"] + (J0["head_top"] - J0["head"])
for side in ("left", "right"):                     # collars sit between the neck and shoulder
    J[f"{side}_collar"][0] = 0.5 * (J["spine3"][0] + J[f"{side}_shoulder"][0])

ENDS = {"left_hand", "right_hand", "head_top", "left_foot", "right_foot"}
before = {n: float(sample(J0[n])[0]) for n in J}
after = {n: float(sample(J[n])[0]) for n in J}
moved = {n: float(np.linalg.norm(J[n] - J0[n])) for n in J}
out = dict(Jd)
out["joints"] = {n: ((J[n] - cc) * kk).tolist() for n in J}
out["traces"] = {k: ((t - cc) * kk).tolist() for k, t in traces.items()}
out["refined"] = {n: f"moved {moved[n] / H * 175:.1f} cm at 1.75 m; depth "
                     f"{before[n] / H * 175:+.1f} -> {after[n] / H * 175:+.1f} cm" for n in J if moved[n] > 1e-9}
json.dump(out, open(a.out, "w"), indent=1)
ob = [n for n in J if before[n] > 0 and n not in ENDS]
oa = [n for n in J if after[n] > 0 and n not in ENDS]
print(f"[joints] outside the body before: {ob or 'none'}; after: {oa or 'none'}", flush=True)
for k, t in traces.items():
    print(f"[joints]   traced {k}: {len(t)} steps, {len(t) * step / H * 175:.0f} cm", flush=True)
for n in sorted(moved, key=lambda n: -moved[n])[:6]:
    print(f"[joints]   {n:15s} moved {moved[n] / H * 175:.1f} cm", flush=True)
