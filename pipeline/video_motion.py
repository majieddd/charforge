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
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--add-to", default=None, help="a character's name: add the best match to its clips")
ap.add_argument("--as", dest="clip_name", default="from_video", help="the clip's name on the character")
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
        if len(P) < 2 * step_coarse:
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
            if k / vfps + 1e-9 >= t:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                t += 1.0 / fps
            k += 1
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(dev).eval()
    kps, scs = [], []
    for fr in frames:
        m = image_mask(fr)
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
    n = len(picks)
    summary = {"tests": n, "top1": ok_exact / n, "top1_or_tie": (ok_exact + ok_tie) / n, "top5": ok_top5 / n,
               "angle_error_deg_median": float(np.median(az_err)) if az_err else None}
    print(f"[motion] self-test: right clip first {ok_exact}/{n} ({ok_exact / n:.0%}); first or within 10% of the "
          f"winner's cost {ok_exact + ok_tie}/{n}; in the top 5 {ok_top5}/{n} "
          f"({ok_top5 / n:.0%}); camera angle off by {summary['angle_error_deg_median']} deg (median, when right)",
          flush=True)
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)
    raise SystemExit(0)

if not a.video:
    raise SystemExit("give --video, or --selftest N")
kp, sc, nfr = pose_video(a.video, a.fps)
pts, conf = coco_to_points(kp, sc)
fv, lv = features(pts)
res = match(fv, lv, conf.astype(np.float32))
print(f"[motion] {nfr} frames at {a.fps:g} fps; best matches:", flush=True)
for r in res[:a.top]:
    print(f"[motion]   {r['file']}  cost {r['cost']:.3f}  camera {r['azimuth_deg']:.0f} deg"
          f"{'  mirrored' if r['mirror'] else ''}  clip frames {r['clip_frames'][0]}-{r['clip_frames'][1]}"
          f"  speed x{r['speed']:.2f}", flush=True)
if a.out:
    json.dump({"video": a.video, "frames": nfr, "fps": a.fps, "matches": res[:a.top],
               "clip_spec": {"file": res[0]["file"], "mirror": res[0]["mirror"]}}, open(a.out, "w"), indent=1)
if a.add_to:
    # onto the character: the clip joins its set, and `charforge.py make --name <it> --from animate`
    # retargets it with the rest (planted feet, heading, the lot)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    f = os.path.join(root, "work", a.add_to, "extra_clips.json")
    have = json.load(open(f)) if os.path.exists(f) else {}
    have[a.clip_name] = {"file": res[0]["file"], "mirror": res[0]["mirror"], "from_video": os.path.abspath(a.video)}
    json.dump(have, open(f, "w"), indent=1)
    print(f"[motion] added to {a.add_to} as {a.clip_name!r} -> {f}; rerun: charforge.py make --name {a.add_to} "
          f"--from animate", flush=True)
