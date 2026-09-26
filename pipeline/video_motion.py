"""Motion from a video: the library clip that moves the way the person in a video does.

    python video_motion.py --video move.mp4 --library work/motion_library.npz --out match.json
    python video_motion.py --selftest 60 --library work/motion_library.npz      # measured accuracy
    python video_motion.py --video move.mp4 --library ... --add-to aoi --as dance   # onto a character

The last link of the text -> image -> video -> motion chain. A pose model reads the person's 2D
joints in every frame; blender/motion_library.py holds every Mixamo clip's joints in 3D; the
answer is the clip, camera angle and timing whose 2D projection best follows the video.

This is retrieval, not reconstruction: it can only return a motion the library has (2,147 clips -
locomotion, gestures, fights, dances, sport), but what it returns is clean, looping motion-capture
that the pipeline already retargets onto every character, where monocular 3D lifting returns
jittery poses with sliding feet. The steps:

  pose      ViTPose on each frame, inside the figure's box (the frame against its background)
  features  13 points (head; shoulders, elbows, wrists, hips, knees, ankles) around the hips,
            in units of the torso's length across the whole video - so a crouch still reads as
            a crouch - and the hips' height, so a jump reads as a jump
  views     each clip projected from 16 camera angles around it, and mirrored (a left-handed
            wave is a right-handed one retargeted with mirror: true); legs compared both ways
            round, because a pose model swaps them where they cross
  timing    dynamic time warping: the clip may play at 0 to 3 frames per video frame and the
            video may cover any stretch of it
  search    all clips at a third of the frame rate, then the best 60 at full rate

--selftest N measures it on N clips from the library itself, each seen from a random angle
(not one of the 16), mirrored or not, with pose noise, a random stretch of 3 s and a random speed
change: how often the right clip comes back first, and how far the angle is off.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--video", default=None, help="a video file, or a directory of frames (PNG, in name order)")
ap.add_argument("--library", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--selftest", type=int, default=0)
ap.add_argument("--body", type=float, default=0.0,
                help="self-test: give each test's body other proportions - every bone (left and right alike) "
                     "scaled by a factor in [1-body, 1+body] - as a stylised character's are")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--add-to", default=None, help="a character's name: add the best match to its clips")
ap.add_argument("--as", dest="clip_name", default="from_video", help="the clip's name on the character")
ap.add_argument("--fit", action="store_true",
                help="turn the matched clip to follow the video (a new clip) instead of using it as it is")
ap.add_argument("--points", default=None, help="the pose model's points already read (npz: pts, conf) instead of --video")
ap.add_argument("--masks", default=None,
                help="the figure's mask per frame (npz 'masks', pipeline/foreground.py --video): boxes for the pose "
                     "model instead of a colour threshold, which takes half of a studio backdrop for the figure")
ap.add_argument("--calib", default=None,
                help="the performer's work/<name>/joint_calib.json (tools/calibrate_joints.py): with it the fit also "
                     "turns the head to the video's face")
ap.add_argument("--capture-lengths", action="store_true",
                help="fit with the capture's body instead of the video's (the old behaviour; for comparison)")
ap.add_argument("--exclude-clip", nargs="*", default=[], help="library clips to leave out of the match (tests)")
ap.add_argument("--exclude-true", action="store_true",
                help="--selftest: take the test clip (and its near-twins) out of the library first, as a "
                     "move the library does not have")
a = ap.parse_args()

LIB = np.load(a.library)
NAMES = LIB["names"]
OFF = LIB["offsets"]
JOINTS = LIB["joints"].astype(np.float32)
FACING = LIB["facing"]
UNIT = LIB["unit"]
NCLIP = len(NAMES)
AZ = np.radians(np.arange(0, 360, 22.5))
MIRROR_PERM = [0, 4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9]        # swap left and right points
ARM_SIGNS = os.environ.get("CF_ARM_SIGNS", "1") != "0"          # 0: the arms' depths as before (comparison)
EXCLUDE = set()                                                  # clips left out of the match


def project(P, facing, az, mirror):
    """(T, 13, 3) world points -> (T, 13, 2) image points for a camera at azimuth az around the
    body (0 = in front of it, looking at its face), orthographic, u right, v up."""
    th = facing + az
    right = np.array([-math.sin(th), math.cos(th), 0.0])            # the body's left, turned by az
    uv = np.stack([P @ right, P[..., 2]], -1)
    if mirror:
        uv = uv[:, MIRROR_PERM] * np.array([-1.0, 1.0])
    return uv


def features(uv, conf=None):
    """Image points -> pose features: around the hips, in torso lengths, plus the hips' height."""
    hip = (uv[:, 7] + uv[:, 10]) / 2
    sh = (uv[:, 1] + uv[:, 4]) / 2
    torso = np.median(np.linalg.norm(sh - hip, axis=1)) + 1e-6
    f = (uv - hip[:, None, :]) / torso
    lift = (hip[:, 1] - np.median(hip[:, 1])) / torso
    return f.astype(np.float32), lift.astype(np.float32)


def dtw(C):
    """Open-ended alignment of every video frame to the clip, 0-3 clip frames per video frame,
    starting and ending anywhere in the clip. C is (views, Tv, Tc); returns per view
    (mean cost, start, end)."""
    V, Tv, Tc = C.shape
    D = C[:, 0].copy()
    start = np.broadcast_to(np.arange(Tc), (V, Tc)).copy()
    for i in range(1, Tv):
        best = D + 0.15                                              # a held frame costs a little
        bstart = start.copy()
        for k in (1, 2, 3):
            if k >= Tc:
                break
            cand = np.full((V, Tc), np.inf, np.float32)
            cand[:, k:] = D[:, :-k] + (0.05 if k == 3 else 0.0)
            better = cand < best
            best = np.where(better, cand, best)
            s_k = np.zeros((V, Tc), np.int64)
            s_k[:, k:] = start[:, :-k]
            bstart = np.where(better, s_k, bstart)
        D = best + C[:, i]
        start = bstart
    j = np.argmin(D, axis=1)
    rows = np.arange(V)
    return D[rows, j] / Tv, start[rows, j], j


VIEWS = [(az, m) for az in AZ for m in (False, True)]


def clip_views(P, facing, step=1, cover=0):
    """A clip's features from every camera view: (views, T, 13, 2) and (views, T). A clip shorter
    than `cover` frames is repeated to cover it: a walk cycle or a wave is a second long, and a
    video of it shows the loop over and over (the right crouch-walk once scored 0.42 - worse than
    fifty wrong clips - because one pass through it could not cover three seconds of video)."""
    if cover and len(P) < cover:
        P = np.concatenate([P] * int(np.ceil(cover / len(P))))
    fs, ls = zip(*[features(project(P[::step], facing, az, m)) for az, m in VIEWS])
    return np.stack(fs), np.stack(ls)


LEGS_SWAPPED = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]


def cost_views(fv, lv, cv, FC, LC):
    """Frame-to-frame distance for every view at once, (views, Tv, Tc): the mean point distance
    (weighted by the pose model's confidence) plus the gap in hip height. Seen from the front, a
    pose model swaps the legs where they cross (on a crouch-walk, every other stride), so the legs
    are compared both ways round and the nearer counts."""
    d = np.linalg.norm(fv[None, :, None] - FC[:, None], axis=-1)     # (V, Tv, Tc, 13)
    ds = np.linalg.norm(fv[None, :, None][..., LEGS_SWAPPED, :] - FC[:, None], axis=-1)
    w = cv[None, :, None, :]
    ws = cv[:, LEGS_SWAPPED][None, :, None, :]
    upper = (d[..., :7] * w[..., :7]).sum(-1)
    legs = np.minimum((d[..., 7:] * w[..., 7:]).sum(-1), (ds[..., 7:] * ws[..., 7:]).sum(-1))
    pose = (upper + legs) / np.maximum(w.sum(-1), 1e-6)
    return pose + 0.5 * np.abs(lv[None, :, None] - LC[:, None, :])


def match(fv, lv, cv, step_coarse=3, keep=60):
    """Rank clips (each at its best view) for one video's features."""
    coarse = []
    fvs, lvs, cvs = fv[::step_coarse], lv[::step_coarse], cv[::step_coarse]
    for c in range(NCLIP):
        P = JOINTS[OFF[c]:OFF[c + 1]]
        if len(P) < 2 * step_coarse or c in EXCLUDE:
            continue
        FC, LC = clip_views(P, FACING[c], step_coarse, cover=int(1.3 * len(fv)))
        cost, _, _ = dtw(cost_views(fvs, lvs, cvs, FC, LC))
        coarse.append((float(cost.min()), c))
    coarse.sort()
    out = []
    for _, c in coarse[:keep]:
        P = JOINTS[OFF[c]:OFF[c + 1]]
        FC, LC = clip_views(P, FACING[c], cover=int(1.3 * len(fv)))
        cost, j0, j1 = dtw(cost_views(fv, lv, cv, FC, LC))
        v = int(np.argmin(cost))
        az, m = VIEWS[v]
        out.append({"cost": float(cost[v]), "clip": int(c), "file": str(NAMES[c]),
                    "azimuth_deg": float(np.degrees(az)), "mirror": bool(m),
                    "clip_frames": [int(j0[v]), int(j1[v])], "speed": float((j1[v] - j0[v] + 1) / len(fv))})
    out.sort(key=lambda r: r["cost"])
    return out


# ---- fitting the matched clip to the video -------------------------------------------------------
# A match is the nearest library clip, and a generated move is rarely exactly one: a jab lands
# higher, a kick turns further. The fit keeps from the clip only what one camera cannot see - how
# far each bone reaches toward or away from it - and takes the rest from the video: every bone is
# turned so its image lies on the pose model's, at the clip's bone length, its depth on the side
# the clip has it (or, where the clip holds it flat to the camera, the side it was on a frame ago).

HIPM, SHM = 13, 14                                   # between the hips, between the shoulders
TREE = [(HIPM, 7), (7, 8), (8, 9), (HIPM, 10), (10, 11), (11, 12), (HIPM, SHM),
        (SHM, 1), (1, 2), (2, 3), (SHM, 4), (4, 5), (5, 6), (SHM, 0)]


def with_mids(X):
    """(..., 13, D) -> (..., 15, D): the two midpoints the body's tree hangs from."""
    hip = (X[..., 7, :] + X[..., 10, :]) / 2
    sh = (X[..., 1, :] + X[..., 4, :]) / 2
    return np.concatenate([X, hip[..., None, :], sh[..., None, :]], -2)


def camera_axes(facing, az):
    """Rows: the camera's right, up and toward-the-camera axes in the clip's world (see project)."""
    th = facing + az
    return np.array([[-math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0], [math.cos(th), math.sin(th), 0.0]])


def smooth(x, sigma, axis=0):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x, sigma, axis=axis, mode="nearest") if sigma > 0 else x


def dtw_path(C):
    """The alignment itself, for one view: C (Tv, Tc) -> the clip frame at each video frame, with
    dtw's steps and costs."""
    Tv, Tc = C.shape
    D = C[0].copy()
    back = np.zeros((Tv, Tc), np.int8)
    for i in range(1, Tv):
        best = D + 0.15
        arg = np.zeros(Tc, np.int8)
        for k in (1, 2, 3):
            if k >= Tc:
                break
            cand = np.full(Tc, np.inf, np.float32)
            cand[k:] = D[:-k] + (0.05 if k == 3 else 0.0)
            better = cand < best
            best = np.where(better, cand, best)
            arg = np.where(better, k, arg)
        D = best + C[i]
        back[i] = arg
    j = int(np.argmin(D))
    path = np.empty(Tv, np.int64)
    for i in range(Tv - 1, -1, -1):
        path[i] = j
        j -= int(back[i, j])
    return path


def clip_at(P, times):
    """A clip's points at fractional frames (looped past its end, as the match covers it)."""
    n = len(P)
    i0 = np.floor(times).astype(int)
    f = (times - i0)[:, None, None]
    return (1 - f) * P[i0 % n] + f * P[(i0 + 1) % n]


def depth_signs(dz, zlib, L, agree=1.0, flip=4.0, extra=None):
    """In front of its parent or behind, per frame, for one bone whose image fixes its depth up to
    sign (dz >= 0): agree with the clip where the clip holds the bone clearly off the image plane,
    change seldom - a flip swaps a forearm from in front of the body to behind it (Viterbi).
    `extra` (T, 2) adds a cost to each sign, per frame."""
    T = len(dz)
    states = np.array([1.0, -1.0])
    clear = np.clip(np.abs(zlib) / L, 0, 1)
    emit = agree * clear[:, None] * (np.sign(zlib)[:, None] != states[None, :])
    if extra is not None:
        emit = emit + extra
    cost, back = emit[0].copy(), np.zeros((T, 2), np.int64)
    for t in range(1, T):
        tot = cost[:, None] + flip * np.abs(states[:, None] * dz[t - 1] - states[None, :] * dz[t]) / L
        back[t] = np.argmin(tot, 0)
        cost = tot.min(0) + emit[t]
    s = np.empty(T, np.int64)
    s[-1] = int(np.argmin(cost))
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return states[s]


PAIRED = {(HIPM, 7): (HIPM, 10), (7, 8): (10, 11), (8, 9): (11, 12),
          (SHM, 1): (SHM, 4), (1, 2): (4, 5), (2, 3): (5, 6)}
PAIRED_INV = {v: k for k, v in PAIRED.items()}


def video_lengths(V, W, Lc, flat_min=0.9):
    """Each bone's length as the video shows it, in torso lengths, left and right averaged - measured
    only where the matched capture says the bone lies nearly across the camera's view (its image at
    least 90% of its length), each image length divided by that fraction, the median of them. The
    video's own longest image of a bone overshoots with the pose model's noise, and a bone that never
    lies flat to the camera - a shoulder line seen from the side - shows nothing of its length: on
    the self-test's side-on, noisy views the fit on those lengths came out 50% worse than on the
    capture's. A bone the capture never shows flat is left to the capture (nan).

    Then each ratio is drawn toward the capture's by how sure it is (shrinkage): a bone within the
    measurement's own error of the capture's length keeps the capture's - on the capture's own body
    the video's estimate only adds noise - and one clearly different takes the video's."""
    est, se = {}, {}
    for b in TREE:
        lib = Lc[:, b[1]] - Lc[:, b[0]]
        flat = np.linalg.norm(lib[:, :2], axis=1) / np.maximum(np.linalg.norm(lib, axis=1), 1e-9)
        ok = (np.minimum(W[:, b[0]], W[:, b[1]]) > 0.5) & (flat > flat_min)
        if ok.sum() < 5:
            est[b], se[b] = np.nan, np.nan
            continue
        l_ = np.linalg.norm(V[ok, b[1]] - V[ok, b[0]], axis=1) / flat[ok]
        est[b] = float(np.median(l_))
        se[b] = 1.253 * 1.4826 * float(np.median(np.abs(l_ - est[b]))) / np.sqrt(ok.sum())
    for l_, r_ in PAIRED.items():
        both = [k for k in (l_, r_) if np.isfinite(est[k])]
        if both:
            m_ = float(np.mean([est[k] for k in both]))
            e_ = float(np.sqrt(np.sum([se[k] ** 2 for k in both]))) / len(both)
        else:
            m_, e_ = np.nan, np.nan
        est[l_] = est[r_] = m_
        se[l_] = se[r_] = e_
    t, st = est[(HIPM, SHM)], se[(HIPM, SHM)]
    if not np.isfinite(t):
        return None
    cap = {b: float(np.median(np.linalg.norm(Lc[:, b[1]] - Lc[:, b[0]], axis=-1))) for b in TREE}
    out = {}
    for b in TREE:
        if not np.isfinite(est[b]):
            out[b] = np.nan
            continue
        rv, rc = est[b] / t, cap[b] / cap[(HIPM, SHM)]
        sr = rv * np.sqrt((se[b] / est[b]) ** 2 + (st / t) ** 2) if b != (HIPM, SHM) else 0.0
        d2 = (rv - rc) ** 2
        lam = d2 / (d2 + sr ** 2 + (0.02 * rc) ** 2) if d2 > 0 else 0.0
        out[b] = rc + lam * (rv - rc)
    return out


def with_proportions(Lc, Lv):
    """The clip's pose on the video's body: each bone keeps the clip's direction, and its stretch from
    frame to frame, at the video's length - the torso keeping the clip's size. A fit built on the
    capture's body bends the video's to fit it: Aoi's legs are 1.03 torso lengths to the capture's
    0.80, and her short torso came out as a 28 deg lean back."""
    cap = {b: float(np.median(np.linalg.norm(Lc[:, b[1]] - Lc[:, b[0]], axis=-1))) for b in TREE}
    k = cap[(HIPM, SHM)]
    out = Lc.copy()
    for b in TREE:                                                    # parents first
        f = Lv[b] * k / max(cap[b], 1e-9) if np.isfinite(Lv[b]) else 1.0
        out[:, b[1]] = out[:, b[0]] + (Lc[:, b[1]] - Lc[:, b[0]]) * f
    return out


PARENT13 = {2: 1, 3: 2, 5: 4, 6: 5, 8: 7, 9: 8, 11: 10, 12: 11}   # elbow <- shoulder, wrist <- elbow, ...


def fill_gaps(pts, conf, lo=0.45, reach=4, filled=0.5):
    """A joint the pose model is unsure of for a few frames, drawn between the frames either side it
    is sure of - relative to its parent joint, so the limb stays on. At the top of Mara's kick the
    ankle was read right at 0.6 and missed on the frames around it; a fit that falls back to the
    capture there blends a leg pointing up with one pointing down and folds it. Gaps longer than
    `reach` frames either side are left alone."""
    P, C = pts.copy(), conf.copy()
    T = len(P)
    for j in range(13):
        par = PARENT13.get(j)
        rel = P[:, j] - (P[:, par] if par is not None else 0.0)
        good = C[:, j] >= lo
        for t in np.nonzero(~good)[0]:
            prv = [u for u in range(max(0, t - reach), t) if good[u]]
            nxt = [u for u in range(t + 1, min(T, t + reach + 1)) if good[u]]
            if not prv or not nxt:
                continue
            t0, t1 = prv[-1], nxt[0]
            f = (t - t0) / (t1 - t0)
            r = (1 - f) * rel[t0] + f * rel[t1]
            P[t, j] = (P[t, par] if par is not None else 0.0) + r
            C[t, j] = max(C[t, j], filled)
    return P, C


def fit_clip(pts, conf, c, az, mirror, times, own_lengths=True):
    """Turn clip c, read at `times` (fractional library frames), to follow the video's points
    (T, 13, 2; pixels, v up; conf per point). Returns the fitted points in the clip's own world
    (T, 13, 3) - unmirrored: a mirrored match stays mirrored in its clip spec - and the numbers."""
    P = JOINTS[OFF[c]:OFF[c + 1]]
    Pt = clip_at(P, times)
    V, w = fill_gaps(pts.astype(np.float64), conf.astype(np.float64))
    if mirror:                                                        # the video in the clip's frame
        V, w = V[:, MIRROR_PERM] * np.array([-1.0, 1.0]), w[:, MIRROR_PERM]
    Vr = with_mids(V)
    V = with_mids(smooth(V, 1.0))                                     # pose-model jitter
    W = np.concatenate([w, np.minimum(w[:, 7], w[:, 10])[:, None], np.minimum(w[:, 1], w[:, 4])[:, None]], 1)

    # the camera angle to the degree: the match knows it to 22.5 deg, and a view 11 deg off
    # reads as every bone disagreeing with the clip - the fit then follows the noisy video
    def image_gap(az_):
        L_ = with_mids(Pt @ camera_axes(FACING[c], az_).T)[..., :2]
        A_ = L_ - L_[:, HIPM:HIPM + 1]
        B_ = V - V[:, HIPM:HIPM + 1]
        k_ = np.sum(A_ * B_) / max(np.sum(B_ * B_), 1e-9)             # best scale, pixels -> metres
        return float(np.sum(np.linalg.norm(A_ - k_ * B_, axis=-1) * W))
    az = min(az + np.radians(np.arange(-12, 12.5, 1.0)), key=image_gap)
    R = camera_axes(FACING[c], az)
    Lc = with_mids(Pt @ R.T)                                          # the clip in camera space
    Lv = video_lengths(V, W, Lc) if own_lengths else None
    if Lv is not None:
        Lc = with_proportions(Lc, Lv)
    L = {b: float(np.median(np.linalg.norm(Lc[:, b[1]] - Lc[:, b[0]], axis=-1))) for b in TREE}
    d2 = {b: V[:, b[1]] - V[:, b[0]] for b in TREE}
    ok = {b: np.minimum(W[:, b[0]], W[:, b[1]]) > 0.3 for b in TREE}

    # pixels -> metres: first from the torso, then refined so that bones neither overshoot their
    # length in the image (too large) nor need much more depth than the clip gives them (too small)
    tv = np.linalg.norm(V[:, SHM] - V[:, HIPM], axis=1)
    tl = np.linalg.norm(Lc[:, SHM, :2] - Lc[:, HIPM, :2], axis=1)
    s0 = float(np.median(tl / np.maximum(tv, 1e-6)))

    def objective(s):
        e = []
        for b in TREE:
            if b[1] == 0 or not ok[b].any():                          # the head point is not a joint
                continue
            l2 = s * np.linalg.norm(d2[b][ok[b]], axis=1)
            zl = np.abs(Lc[ok[b], b[1], 2] - Lc[ok[b], b[0], 2])
            over = np.maximum(l2 - L[b], 0) / L[b]
            dz = np.sqrt(np.maximum(L[b] ** 2 - np.minimum(l2, L[b]) ** 2, 0))
            e.append(np.mean(over ** 2 + 0.5 * ((dz - zl) / L[b]) ** 2))
        return float(np.mean(e))
    grid = s0 * np.linspace(0.8, 1.25, 46)
    s = float(grid[int(np.argmin([objective(g) for g in grid]))])

    T = len(V)
    rel = np.zeros((T, 15, 3))
    flat, kept = [], []
    torso0 = float(np.median(np.linalg.norm(Lc[:, SHM] - Lc[:, HIPM], axis=1)))

    def image_depth(b_):
        """A bone's image (clipped to its length) and the depth that length leaves it, per frame."""
        lib_ = Lc[:, b_[1]] - Lc[:, b_[0]]
        Lt_ = np.maximum(np.linalg.norm(lib_, axis=1), 1e-6)
        v2_ = s * d2[b_]
        l2_ = np.linalg.norm(v2_, axis=1)
        v2_ = np.where((l2_ > Lt_)[:, None], v2_ * (Lt_ / np.maximum(l2_, 1e-9))[:, None], v2_)
        return v2_, np.sqrt(np.maximum(Lt_ ** 2 - np.minimum(l2_, Lt_) ** 2, 0))

    def arm_costs(sh, upper, fore):
        """What an arm's depths cost, per frame: an elbow folded past 150-160 degrees (the motion
        library's elbows reach 150 in 0.05% of frames, 159 at most), and a wrist within the chest's
        width and height behind the torso - 1% of the library's such wrists are; the median is half a
        torso length in front. The image fixes each depth only up to sign, and a hand pressed in front
        of the chest reads the same as one folded back through the arm, or behind the back (Aoi's
        spell cast: the capture's upper arm tipped back, and the elbow came out folded to 176 deg)."""
        P1, P4 = rel[:, SHM] + rel[:, 1], rel[:, SHM] + rel[:, 4]
        mid, hip = (P1 + P4) / 2, (rel[:, 7] + rel[:, 10]) / 2
        tl = np.maximum(np.linalg.norm(mid - hip, axis=1), 1e-6)
        up = (mid - hip) / tl[:, None]
        lat = (P1 - P4) - np.sum((P1 - P4) * up, 1, keepdims=True) * up        # toward the body's left
        wid = np.maximum(np.linalg.norm(lat, axis=1), 1e-6)
        lat = lat / wid[:, None]
        fwd = np.cross(lat, up)
        cos = np.sum(upper * fore, 1) / np.maximum(np.linalg.norm(upper, axis=1) * np.linalg.norm(fore, axis=1), 1e-9)
        bend = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        d = sh + upper + fore - mid
        x, y, z = np.sum(d * lat, 1), np.sum(d * up, 1), np.sum(d * fwd, 1)
        band = (np.abs(x) < 0.6 * wid) & (y < 0) & (y > -0.7 * tl)
        zt = np.sum((hip - mid) * fwd, 1) * np.clip(-y / tl, 0, 1)            # the torso's middle at that height
        return 3.0 * np.clip((bend - 150.0) / 10.0, 0, 1) + 1.5 * (band & (z < zt - 0.05 * tl))

    for b in TREE:
        p, ch = b
        lib = Lc[:, ch] - Lc[:, p]
        # the clip's own length at each frame: the spine, neck and collarbones are not rigid
        # between these points, and a median length put depth where the clip had none
        Lt = np.maximum(np.linalg.norm(lib, axis=1), 1e-6)
        # The image of the bone: the clip's where the clip and the video agree to within the pose
        # model's noise, the video's where they part by clearly more. Depth comes out of the length,
        # sqrt(L^2 - l^2), which noise swings hard when the bone lies near the image plane: taken
        # from the video alone, clips that already matched got 4x worse (self-test, 12 clips).
        raw = s * (Vr[:, ch] - Vr[:, p])
        v2 = s * d2[b]
        # a floor for clean footage: a render's jitter is ~0.4% of the torso, but where a pose model
        # puts a joint differs from where a skeleton has it by more than that
        noise = max(1.4826 * float(np.median(np.abs(raw - v2))) * 0.6, 0.01 * L[b], 0.02 * torso0)
        gap = np.linalg.norm(lib[:, :2] - v2, axis=1)
        w_clip = (1.0 / (1.0 + (gap / (3.0 * noise)) ** 4))[:, None]
        kept.append(float(w_clip.mean()))
        l2 = np.linalg.norm(v2, axis=1)
        over = l2 > Lt
        v2[over] *= (Lt[over] / l2[over])[:, None]
        flat.append(float(np.mean(over[ok[b]])) if ok[b].any() else 0.0)
        dz = np.sqrt(np.maximum(Lt ** 2 - np.minimum(l2, Lt) ** 2, 0))
        extra = None
        if ch in (9, 12):
            # a knee only bends one way: the shin folds back behind the thigh about the hip line
            # (cross(thigh, shin) along the body's left). Of the two depths the image allows, the
            # one that bends the knee backwards more than a few degrees costs. On the self-test it
            # hardly shows (0.178 -> 0.177 torso lengths, 40 moves the library lacks); it is here
            # for the knee a noisier real video would bend backwards, which shows at once.
            thigh = rel[:, p]
            left = Lc[:, 7] - Lc[:, 10]
            left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-9)
            extra = np.zeros((T, 2))
            for j, sg in enumerate((1.0, -1.0)):
                shin = np.concatenate([v2, (sg * dz)[:, None]], 1)
                bend = np.sum(np.cross(thigh, shin) * left, 1) / np.maximum(
                    np.linalg.norm(thigh, axis=1) * np.linalg.norm(shin, axis=1), 1e-9)
                extra[:, j] = 3.0 * (bend < -0.1)
        if ARM_SIGNS and ch in (2, 5):
            # an upper arm's depth with its forearm's in view: each of its two signs costs what the
            # better of the forearm's two would cost with it
            vf, dzf = image_depth((ch, ch + 1))
            sh = rel[:, SHM] + rel[:, p]
            extra = np.zeros((T, 2))
            for j, sg in enumerate((1.0, -1.0)):
                up_ = np.concatenate([v2, (sg * dz)[:, None]], 1)
                extra[:, j] = np.minimum(*(arm_costs(sh, up_, np.concatenate([vf, (sf * dzf)[:, None]], 1))
                                           for sf in (1.0, -1.0)))
        elif ARM_SIGNS and ch in (3, 6):
            sh = rel[:, SHM] + rel[:, p - 1]
            extra = np.stack([arm_costs(sh, rel[:, p], np.concatenate([v2, (sf * dz)[:, None]], 1))
                              for sf in (1.0, -1.0)], 1)
        # the video's bone (depth from its image length), then blended whole with the clip's: a
        # depth recomputed from a blended image moved bones near the image plane by noise alone
        sgn = depth_signs(dz, lib[:, 2], L[b], extra=extra)
        vid = np.concatenate([v2, (sgn * dz)[:, None]], 1)
        if ARM_SIGNS and ch in (2, 3, 5, 6):
            # where the clip's own bone is kept (it agrees with the video's image), on the chosen side too:
            # a capture's elbow tipped back is kept back otherwise, the image being the same either way
            lib = np.concatenate([lib[:, :2], (sgn * np.abs(lib[:, 2]))[:, None]], 1)
        v = w_clip * lib + (1 - w_clip) * vid
        # where the pose model is unsure of either end, the clip's bone; the change to the clip is
        # smoothed in time, not the clip (a jab lasts three frames at 15 fps). The head point is the
        # nose and ears, not a joint: the neck stays the capture's
        wb = np.clip((np.minimum(W[:, p], W[:, ch]) - 0.3) / 0.3, 0, 1)[:, None] * (ch != 0)
        # smoothed by confidence (normalised convolution): an uncertain frame takes the correction of
        # the sure frames beside it, and a sure frame keeps its own - plain smoothing of wb * change
        # diluted a sure frame between unsure ones back toward the capture
        den = smooth(wb, 1.0)
        v = lib + smooth(wb * (v - lib), 1.0) / np.maximum(den, 1e-6) * np.clip(2 * den, 0, 1)
        rel[:, ch] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9) * Lt[:, None]
    X = np.zeros((T, 15, 3))
    X[:, HIPM, :2] = s * (V[:, HIPM] - V[0, HIPM]) + Lc[0, HIPM, :2]  # moves as the video's hips do
    X[:, HIPM, 2] = Lc[:, HIPM, 2]                                     # toward the camera as the clip's
    for p, ch in TREE:
        X[:, ch] = X[:, p] + rel[:, ch]
    world = X[:, :13] @ R
    # how far the clip's and the fit's images are from the pose model's (torso lengths)
    torso = float(np.median(np.linalg.norm(Lc[:, SHM] - Lc[:, HIPM], axis=1)))

    def img_err(Y):
        a_ = Y[:, :13, :2] - Y[:, HIPM:HIPM + 1, :2]
        b_ = s * (V[:, :13] - V[:, HIPM:HIPM + 1])
        m = W[:, :13] > 0.3
        return float(np.linalg.norm(a_ - b_, axis=-1)[m].mean() / torso)
    # the pose model's own jitter, in torso lengths: what a clip that fits leaves unexplained
    jitter = float(1.4826 * np.median(np.linalg.norm(Vr[:, :13] - V[:, :13], axis=-1)) * s / torso)
    return world, {"azimuth_deg": float(np.degrees(az)), "scale_m_per_px": s, "jitter": jitter,
                   "body": "video" if Lv is not None else "capture",
                   "image_error_clip": img_err(Lc), "image_error_fit": img_err(X),
                   "bones_flat_to_camera": float(np.mean(flat)), "clip_kept": float(np.mean(kept)),
                   "torso_m": torso}


def fit_head(face, fconf, calib, world, R, s, mirror):
    """The head's turn, frame by frame, from the pose model's nose, eyes and ears. The capture's head
    only turns with its chest - Aoi's spell looked down and aside while the video's Aoi faced the
    camera throughout. The performer's own five face points (tools/calibrate_joints.py, in 3D on its
    head) are turned about their centre until their image lies on the video's (orthographic, the
    fit's camera and scale); the turn is kept relative to the fitted torso, pulled weakly toward
    looking along it, and smoothed in time. One size for the face over the clip (the median of a
    first pass with the size free), so a head turning away cannot be read as a head shrinking.
    Returns the five points in the clip's world about the fitted head point, (T, 5, 3), and which
    frames had a face to read."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    F, w = face.astype(np.float64) * s, fconf.astype(np.float64)
    if mirror:
        F, w = F[:, FACE_MIRROR] * np.array([-1.0, 1.0]), w[:, FACE_MIRROR]
    X = with_mids(world @ R.T)                                        # camera space: right, up, toward it
    torso_fit = float(np.median(np.linalg.norm(X[:, SHM] - X[:, HIPM], axis=1)))
    model = np.array([f["rest"] for f in calib["face"]])
    m = (model - model.mean(0)) * torso_fit / calib["torso_m"]
    m_local = np.stack([m[:, 0], m[:, 2], -m[:, 1]], 1)               # body frame: its left, up, front
    T = len(F)

    def frame(t):
        right = X[t, 1] - X[t, 4]
        right = right / max(np.linalg.norm(right), 1e-9)
        up = X[t, SHM] - X[t, HIPM]
        up = up - up.dot(right) * right
        up = up / max(np.linalg.norm(up), 1e-9)
        return np.stack([right, up, np.cross(right, up)], 1)

    def solve(t, x0, size=None):
        """A turn (rotation vector, body frame) and, unless given, the face's size (log)."""
        B, wt = frame(t), w[t] * (w[t] > 0.3)
        b = F[t] - (F[t] * wt[:, None]).sum(0) / wt.sum()

        def res(x):
            c = (B @ Rotation.from_rotvec(x[:3]).as_matrix() @ m_local.T).T[:, :2]
            c = c * (np.exp(x[3]) if size is None else size)
            c = c - (c * wt[:, None]).sum(0) / wt.sum()
            return np.concatenate([((c - b) * np.sqrt(wt)[:, None]).ravel(), 0.02 * x[:3]])
        return least_squares(res, x0 if size is None else x0[:3], method="lm").x
    ok = np.array([(w[t] > 0.3).sum() >= 4 for t in range(T)])
    if ok.sum() < 3:
        return None, ok
    x, sizes = np.zeros(4), []
    for t in np.nonzero(ok)[0]:
        x = solve(t, x)
        sizes.append(np.exp(x[3]))
    size = float(np.median(sizes))
    rv, x = np.zeros((T, 3)), np.zeros(3)
    for t in np.nonzero(ok)[0]:
        x = solve(t, x, size)
        rv[t] = x
    idx = np.nonzero(ok)[0]
    for k in range(3):                                                # through unread frames: interpolate
        rv[:, k] = np.interp(np.arange(T), idx, rv[idx, k])
    rv = smooth(rv, 1.0)
    out = np.zeros((T, 5, 3))
    for t in range(T):
        c = (frame(t) @ Rotation.from_rotvec(rv[t]).as_matrix() @ m_local.T).T * size
        out[t] = (c + X[t, 0]) @ R
    return out, ok


def use_fit(st):
    """Fit only where the clip misses the video by more than the pose model's jitter explains:
    on the self-test, clips the library holds left 0.66-0.84 jitters unexplained (a fit only added
    noise to them), clips it lacks mostly 1.5-6."""
    return st["image_error_clip"] > 1.2 * max(st["jitter"], 0.02)


def match_times(fv, lv, cv, r, sigma=2.0):
    """The best match's alignment as smooth, rising clip times (library frames) per video frame."""
    c = r["clip"]
    az, m = math.radians(r["azimuth_deg"]), r["mirror"]
    P = JOINTS[OFF[c]:OFF[c + 1]]
    cover = int(1.3 * len(fv))
    if len(P) < cover:
        P = np.concatenate([P] * int(np.ceil(cover / len(P))))
    FC, LC = features(project(P, FACING[c], az, m))
    path = dtw_path(cost_views(fv, lv, cv, FC[None], LC[None])[0])
    return np.maximum.accumulate(smooth(path.astype(np.float64), sigma))


def pose_video(path, fps):
    """ViTPose keypoints per frame (COCO-17), inside the figure's box."""
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from align import image_mask
    frames = []
    if os.path.isdir(path):                                          # frames rendered at --fps
        for f in sorted(x for x in os.listdir(path) if x.lower().endswith(".png")):
            frames.append(np.asarray(Image.open(os.path.join(path, f)).convert("RGB")))
    else:
        cap = cv2.VideoCapture(path)
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        t, k = 0.0, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            while k / vfps + 1e-9 >= t:              # --fps above the video's repeats frames
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                t += 1.0 / fps
            k += 1
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(dev).eval()
    kps, scs = [], []
    given = np.load(a.masks)["masks"] if a.masks else None
    for i_, fr in enumerate(frames):
        m = given[min(i_, len(given) - 1)] if given is not None else image_mask(fr)
        ys, xs = np.nonzero(m)
        if len(xs) < 50:
            x0, y0, x1, y1 = 0, 0, fr.shape[1], fr.shape[0]
        else:
            pad = 0.1 * max(np.ptp(xs), np.ptp(ys))
            x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
            x1, y1 = min(xs.max() + pad, fr.shape[1]), min(ys.max() + pad, fr.shape[0])
        boxes = [[[float(x0), float(y0), float(x1 - x0), float(y1 - y0)]]]
        inp = proc(Image.fromarray(fr), boxes=boxes, return_tensors="pt").to(dev)
        with torch.no_grad():
            res = proc.post_process_pose_estimation(model(**inp), boxes=boxes)[0][0]
        kps.append(res["keypoints"].cpu().numpy())
        scs.append(res["scores"].cpu().numpy())
    return np.array(kps), np.array(scs), len(frames)


FACE_MIRROR = [0, 2, 1, 4, 3]                                    # nose; eyes and ears swap sides


def coco_face(kp, sc):
    """COCO-17 -> the face: nose, left and right eye, left and right ear, image y flipped to up."""
    return kp[:, :5] * np.array([1.0, -1.0]), np.clip(sc[:, :5], 0, 1)


def coco_to_points(kp, sc):
    """COCO-17 -> the library's 13 points, image y flipped to up; confidence per point."""
    head_idx = [0, 3, 4]
    w = sc[:, head_idx]
    head = (kp[:, head_idx] * w[..., None]).sum(1) / np.maximum(w.sum(1), 1e-6)[:, None]
    order = [5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]
    pts = np.concatenate([head[:, None], kp[:, order]], 1) * np.array([1.0, -1.0])
    conf = np.concatenate([w.mean(1, keepdims=True), sc[:, order]], 1)
    return pts, np.clip(conf, 0, 1)


if a.selftest:
    rng = np.random.default_rng(a.seed)
    ok_exact = ok_top5 = ok_tie = 0
    fit_err = {}
    az_err = []
    long_clips = [c for c in range(NCLIP) if OFF[c + 1] - OFF[c] >= int(3.5 * LIB["fps"])]
    picks = rng.choice(long_clips, size=min(a.selftest, len(long_clips)), replace=False)
    for n_, c in enumerate(picks):
        P = JOINTS[OFF[c]:OFF[c + 1]]
        sp = rng.uniform(0.8, 1.25)
        T = int(3 * LIB["fps"])
        idx = np.clip(np.round(np.arange(T) * sp).astype(int), 0, len(P) - 1)
        s0 = rng.integers(0, max(len(P) - idx[-1], 1))
        Pq = P[np.clip(idx + s0, 0, len(P) - 1)]
        az_true = rng.uniform(0, 2 * math.pi)
        mir = bool(rng.random() < 0.3)
        if a.body > 0:                                               # another body doing the same motion
            fac = {}
            for b_ in TREE:
                fac[b_] = fac.get(PAIRED_INV.get(b_), None) or rng.uniform(1 - a.body, 1 + a.body)
            Xq = with_mids(Pq)
            Yq = Xq.copy()
            for b_ in TREE:
                Yq[:, b_[1]] = Yq[:, b_[0]] + (Xq[:, b_[1]] - Xq[:, b_[0]]) * fac[b_]
            Pq = Yq[:, :13]
        uv = project(Pq, FACING[c], az_true, mir)
        torso = np.median(np.linalg.norm((uv[:, 1] + uv[:, 4]) / 2 - (uv[:, 7] + uv[:, 10]) / 2, axis=1))
        uv = uv + rng.normal(0, 0.04 * torso, uv.shape)
        fv, lv = features(uv)
        res = match(fv, lv, np.ones((len(fv), 13), np.float32))
        top = [r["clip"] for r in res[:5]]
        ok_exact += top[0] == c
        ok_top5 += c in top
        # a miss that is the same motion: Mixamo holds near-duplicates (a clip and its in-place or
        # mirrored twin), so the true clip's cost against the winner's says whether it was a tie
        true_cost = next((r["cost"] for r in res if r["clip"] == c), None)
        tie = top[0] != c and true_cost is not None and true_cost <= 1.1 * res[0]["cost"]
        ok_tie += tie
        if top[0] == c:
            d = abs((math.degrees(az_true) - res[0]["azimuth_deg"] + 180) % 360 - 180)
            az_err.append(d)
        print(f"[motion] test {n_ + 1}/{len(picks)}: {NAMES[c]} at {math.degrees(az_true):.0f} deg"
              f"{' mirrored' if mir else ''} -> {res[0]['file']} at {res[0]['azimuth_deg']:.0f} deg "
              f"(cost {res[0]['cost']:.3f}){'  OK' if top[0] == c else ''}", flush=True)
        if a.fit:
            # the pose the video shows, in its camera's frame, against the clip's and the fit's
            if a.exclude_true:                                       # a move the library lacks
                cut = 1.1 * true_cost if true_cost is not None else -1
                res = [r for r in res if r["clip"] != c and r["cost"] > cut] or res
            r = res[0]
            times = match_times(fv, lv, np.ones((len(fv), 13), np.float32), r)
            world, st = fit_clip(uv, np.ones((len(uv), 13)), r["clip"], math.radians(r["azimuth_deg"]),
                                 r["mirror"], times, own_lengths=not a.capture_lengths)

            def cam(Y, cc, az_, m_):
                Z = Y @ camera_axes(FACING[cc], az_).T
                if m_:
                    Z = Z[:, MIRROR_PERM] * np.array([-1.0, 1.0, 1.0])
                Z = with_mids(Z)
                Z = Z - Z[:, HIPM:HIPM + 1]
                return Z[:, :13] / np.median(np.linalg.norm(Z[:, SHM], axis=1))

            def yaw_to(Y, ref):
                """Turned about the vertical to face as ref does: the way a body faces is free in a
                game (the controller turns it), so it is no error."""
                num = np.sum(Y[..., 2] * ref[..., 0] - Y[..., 0] * ref[..., 2])
                den = np.sum(Y[..., 0] * ref[..., 0] + Y[..., 2] * ref[..., 2])
                t_ = math.atan2(num, den)
                Rz = np.array([[math.cos(t_), 0, math.sin(t_)], [0, 1, 0], [-math.sin(t_), 0, math.cos(t_)]])
                return Y @ Rz.T
            truth = cam(Pq, c, az_true, mir)
            az_r = math.radians(r["azimuth_deg"])
            clip_only = cam(clip_at(JOINTS[OFF[r["clip"]]:OFF[r["clip"] + 1]], times), r["clip"], az_r, r["mirror"])
            fitted = cam(world, r["clip"], az_r, r["mirror"])
            clip_only, fitted = yaw_to(clip_only, truth), yaw_to(fitted, truth)
            gated = fitted if use_fit(st) else clip_only
            for key, Y in (("clip", clip_only), ("fit", fitted), ("gated", gated)):
                fit_err.setdefault(key, []).append(float(np.linalg.norm(Y - truth, axis=-1).mean()))
                fit_err.setdefault(key + "_depth", []).append(float(np.abs(Y[..., 2] - truth[..., 2]).mean()))
            print(f"[motion]   {'(true clip left out) ' if a.exclude_true else ''}pose error, torso lengths: "
                  f"clip {fit_err['clip'][-1]:.3f} -> fit {fit_err['fit'][-1]:.3f} "
                  f"(depth {fit_err['clip_depth'][-1]:.3f} -> {fit_err['fit_depth'][-1]:.3f}) img clip {st['image_error_clip']:.3f} fit {st['image_error_fit']:.3f} jitter {st['jitter']:.3f} kept {st['clip_kept']:.2f}", flush=True)
    n = len(picks)
    summary = {"tests": n, "top1": ok_exact / n, "top1_or_tie": (ok_exact + ok_tie) / n, "top5": ok_top5 / n,
               "angle_error_deg_median": float(np.median(az_err)) if az_err else None}
    if a.fit:
        summary["fit"] = {"true_clip_left_out": bool(a.exclude_true),
                          **{f"pose_error_{k}_mean": float(np.mean(v)) for k, v in fit_err.items()},
                          **{f"pose_error_{k}_median": float(np.median(v)) for k, v in fit_err.items()},
                          "fit_better": float(np.mean(np.array(fit_err["fit"]) < np.array(fit_err["clip"]))),
                          "gate": "fit when the clip's image error > 1.2 x the pose model's jitter"}
        print(f"[motion] fit: pose error (torso lengths, mean over {n}) clip alone "
              f"{summary['fit']['pose_error_clip_mean']:.3f} -> fitted {summary['fit']['pose_error_fit_mean']:.3f}; "
              f"depth alone {summary['fit']['pose_error_clip_depth_mean']:.3f} -> "
              f"{summary['fit']['pose_error_fit_depth_mean']:.3f}; the fit is closer in "
              f"{summary['fit']['fit_better']:.0%} of tests; gated (fit only where the clip misses the video) "
              f"{summary['fit']['pose_error_gated_mean']:.3f}", flush=True)
    print(f"[motion] self-test: right clip first {ok_exact}/{n} ({ok_exact / n:.0%}); first or within 10% of the "
          f"winner's cost {ok_exact + ok_tie}/{n}; in the top 5 {ok_top5}/{n} "
          f"({ok_top5 / n:.0%}); camera angle off by {summary['angle_error_deg_median']} deg (median, when right)",
          flush=True)
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)
    raise SystemExit(0)

if not a.video and not a.points:
    raise SystemExit("give --video, or --selftest N")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
face = face_conf = None
if a.points:                                                         # pose already read (or a test)
    d = np.load(a.points)
    if "kp17" in d:                                                  # the pose model's own 17 points
        pts, conf = coco_to_points(d["kp17"].astype(np.float64), d["sc17"].astype(np.float64))
        face, face_conf = coco_face(d["kp17"].astype(np.float64), d["sc17"].astype(np.float64))
    else:
        pts, conf = d["pts"].astype(np.float64), d["conf"].astype(np.float64)
        if "face" in d:
            face, face_conf = d["face"].astype(np.float64), d["face_conf"].astype(np.float64)
    nfr = len(pts)
else:
    kp, sc, nfr = pose_video(a.video, a.fps)
    pts, conf = coco_to_points(kp, sc)
    face, face_conf = coco_face(kp, sc)
    if a.add_to or a.out:                                            # keep what the pose model read
        side = os.path.splitext(a.out or a.video)[0] + "_pose.npz"
        np.savez_compressed(side, pts=pts, conf=conf, face=face, face_conf=face_conf, fps=a.fps,
                            kp17=kp, sc17=sc)                    # as read (pixels, y down): refine_pose.py
fv, lv = features(pts)
EXCLUDE.update(i for i, n in enumerate(NAMES) if str(n) in set(a.exclude_clip))
res = match(fv, lv, conf.astype(np.float32))
print(f"[motion] {nfr} frames at {a.fps:g} fps; best matches:", flush=True)
for r in res[:a.top]:
    print(f"[motion]   {r['file']}  cost {r['cost']:.3f}  camera {r['azimuth_deg']:.0f} deg"
          f"{'  mirrored' if r['mirror'] else ''}  clip frames {r['clip_frames'][0]}-{r['clip_frames'][1]}"
          f"  speed x{r['speed']:.2f}", flush=True)
source = os.path.abspath(a.video or a.points)
spec = {"file": res[0]["file"], "mirror": res[0]["mirror"], "from_video": source}
if a.fit:
    # a new clip: the match replayed at the video's timing and, where it misses the video by more
    # than the pose model's jitter, turned to follow it (blender/fit_clip.py writes the FBX)
    import glob
    import subprocess
    r = res[0]
    c = r["clip"]
    times = match_times(fv, lv, conf.astype(np.float32), r)
    world, st = fit_clip(pts, conf, c, math.radians(r["azimuth_deg"]), r["mirror"], times,
                         own_lengths=not a.capture_lengths)
    turn = use_fit(st)
    out_dir = (os.path.join(ROOT, "work", a.add_to, "motion_videos") if a.add_to else
               os.path.dirname(os.path.abspath(a.out or source)))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, a.clip_name)
    extra = {}
    if a.calib and face is not None and turn:
        fpts, fok = fit_head(face, face_conf, json.load(open(a.calib)), world,
                             camera_axes(FACING[c], math.radians(st["azimuth_deg"])), st["scale_m_per_px"], r["mirror"])
        if fpts is not None:
            extra = {"face": fpts.astype(np.float32), "face_ok": fok}
            st["face_frames"] = float(fok.mean())
            print(f"[motion] head turned to the video's face in {fok.mean():.0%} of frames", flush=True)
    np.savez_compressed(base + "_fit.npz", times=times, points=world.astype(np.float32), turn=turn,
                        lib_fps=float(LIB["fps"]), fps=a.fps, lib_frames=int(OFF[c + 1] - OFF[c]), **extra)
    snaps = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--jasongzy--Mixamo/snapshots/*/animation")))
    if not snaps:
        raise SystemExit("the Mixamo mirror is not in the Hugging Face cache; the fit needs the matched clip's FBX")
    blender = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
    cmd = [blender, "-b", "-noaudio", "--python", os.path.join(ROOT, "blender", "fit_clip.py"), "--",
           "--fbx", os.path.join(snaps[-1], r["file"]), "--fit", base + "_fit.npz", "--out", base + ".fbx",
           "--report", base + "_fit.json"]
    run = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in run.stdout.splitlines() if ln.startswith("[fit]")]
    if run.returncode != 0 or not lines:
        raise SystemExit(f"fit_clip.py failed:\n{run.stdout[-2500:]}\n{run.stderr[-2500:]}")
    print("[motion] " + lines[-1][6:], flush=True)
    rt = json.load(open(base + "_fit.json"))
    print(f"[motion] {'turned to follow the video' if turn else 'the match already follows the video - retimed only'}: "
          f"image error {st['image_error_clip']:.3f} -> {st['image_error_fit'] if turn else st['image_error_clip']:.3f} "
          f"torso lengths (pose model jitter {st['jitter']:.3f}); camera {st['azimuth_deg']:.0f} deg", flush=True)
    # the camera as seen from the body's average heading - retarget.py turns a stationary clip to
    # face its mean hip line - so a preview can be rendered from where the video was shot
    side = world[:, 7] - world[:, 10]
    face = np.arctan2(-side[:, 0], side[:, 1])
    mean_face = math.atan2(np.sin(face).mean(), np.cos(face).mean())
    yaw = (math.degrees(FACING[c] + math.radians(st["azimuth_deg"]) - mean_face) + 180) % 360 - 180
    st["preview_yaw_deg"] = -yaw if r["mirror"] else yaw
    spec = {"file": base + ".fbx", "mirror": r["mirror"], "from_video": source, "matched": r["file"],
            "match_cost": r["cost"], "fitted": bool(turn),
            "fit": {k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in st.items()},
            # the fitted points themselves: blender/retarget.py aims the character's limbs along them
            "fit_points": base + "_fit.npz",
            "roundtrip_cm": round(rt["joint_error_cm_mean"], 2),
            "roundtrip_deg": round(rt.get("bone_angle_deg_mean", float("nan")), 2)}
if a.out:
    json.dump({"video": source, "frames": nfr, "fps": a.fps, "matches": res[:a.top], "clip_spec": spec},
              open(a.out, "w"), indent=1)
if a.add_to:
    # onto the character: the clip joins its set, and `charforge.py make --name <it> --from animate`
    # retargets it with the rest (planted feet, heading, the lot)
    f = os.path.join(ROOT, "work", a.add_to, "extra_clips.json")
    have = json.load(open(f)) if os.path.exists(f) else {}
    have[a.clip_name] = spec
    json.dump(have, open(f, "w"), indent=1)
    print(f"[motion] added to {a.add_to} as {a.clip_name!r} -> {f}; rerun: charforge.py make --name {a.add_to} "
          f"--from animate", flush=True)
