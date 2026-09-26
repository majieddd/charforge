"""Where DWPose sees a character's hands, against the bones the refine stage turns.

    python tools/calibrate_hands.py --name pip [--clips wave,idle,jump] [--yaws -40,0,40]

pipeline/refine_pose.py holds each hand's 21 points - the wrist, every finger joint, the tips - against
DWPose's reading of the video. On the rig those points are the finger bones' heads (and where the mesh
ends past the last bone); DWPose puts them where a hand looks like it has them, which on a stylised
hand is somewhere else: on Pip's cartoon hands it reads the palm (wrist to middle knuckle) 29-41% longer
than his bones have it, on Aoi's and Mara's 4-13%. Held against DWPose as they are, those points pull
the fingers and the wrist toward a hand of another shape. So, as tools/calibrate_joints.py does for the
body, DWPose reads the character itself: its own clips (a wave, the idle, a jump - hands open, relaxed
and swinging) rendered from three angles (blender/render_match.py), and for every point the offset in
its bone's frame that puts the rig's point where DWPose sees it, solved over all the frames at once - a
linear least-squares fit through the orthographic camera, reweighted against misreads, with a pull to
no offset where the frames say little.

Only the wrist's offset is used. The finger points' move with the pose - DWPose reads a hand a little
differently in every one - and held against a video's poses, which are not the wave's, they made the
hands worse (Aoi 0.24 -> 0.29 palm lengths). The wrist's is where a stylised hand moves DWPose's point
furthest, and it is trusted as far as it holds from clip to clip and stands clear of the ~1.5 cm the
points wander between poses: Pip's 4 cm (his sleeve's cuff) 75-78%, Aoi's and Mara's 1 cm barely.
Writes work/<name>/hand_calib.json (every point's offset, its spread and trust; the wrist's to use) for
blender/export_skin.py --hand-calib.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from motion_fidelity import BLENDER, DW_PY  # noqa: E402

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
NAMES = ["wrist"] + [f"{f}{i}" for f in FINGERS for i in ("1", "2", "3", "_tip")]
SURE = 0.6


def camera(folder):
    """render_match.py's orthographic camera: pixel = c0 + J @ world point."""
    txt = (folder / "camera.txt").read_text()
    yaw = np.radians(float(re.search(r"yaw (-?[\d.]+)", txt).group(1)))
    ortho = float(re.search(r"ortho_scale ([\d.]+)", txt).group(1))
    ctr = np.array([float(x) for x in re.search(r"centre \(([^)]*)\)", txt).group(1).split(",")])
    W, H = (int(x) for x in re.search(r"res (\d+) (\d+)", txt).groups())
    k = H / ortho                                                       # sensor fit vertical: ortho_scale spans H
    side = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    J = k * np.array([side, [0.0, 0.0, -1.0]])
    c0 = np.array([W / 2, H / 2]) - J @ ctr
    return J, c0


def main(a):
    work = ROOT / "work" / a.name
    blend = work / "final.blend"
    out = work / "calib_hands"
    out.mkdir(exist_ok=True)
    rows = {}                                                           # (clip, hand, point) -> [(A (2,3), r (2,), w)]
    before, n_frames = [], 0
    rig = work / "rig_f.blend" if (work / "rig_f.blend").exists() else blend   # the hands change with the rig, not the clips
    fresh = lambda f: f.exists() and f.stat().st_mtime > rig.stat().st_mtime   # noqa: E731
    for clip in a.clips.split(","):
        skin = out / f"{clip}_skin.npz"
        if not fresh(skin):
            subprocess.run([BLENDER, "-b", "-noaudio", "--python", str(ROOT / "blender" / "export_skin.py"), "--",
                            "--blend", str(blend), "--clip", clip, "--frames", str(a.frames), "--fps", str(a.fps),
                            "--points", "2000", "--out", str(skin)], check=True, capture_output=True)
        sk = np.load(skin)
        pose, hb, hl = sk["pose"].astype(np.float64), sk["hand_bone"], sk["hand_local"].astype(np.float64)
        for yaw in (float(y) for y in a.yaws.split(",")):
            ren = out / f"{clip}_{yaw:+.0f}"
            if not fresh(ren / "dw.npz"):
                subprocess.run([BLENDER, "-b", "-noaudio", "--python", str(ROOT / "blender" / "render_match.py"), "--",
                                "--blend", str(blend), "--clip", clip, "--frames", str(a.frames), "--fps", str(a.fps),
                                "--yaw", str(yaw), "--out", str(ren)], check=True, capture_output=True)
                subprocess.run([str(DW_PY), str(ROOT / "tools" / "dwpose.py"), "--frames", str(ren), "--out", str(ren / "dw.npz")],
                               check=True, capture_output=True)
            dw = np.load(ren / "dw.npz")
            K, C = dw["kp133"][:, 91:133].reshape(-1, 2, 21, 2), dw["sc133"][:, 91:133].reshape(-1, 2, 21)
            J, c0 = camera(ren)
            for t in range(min(len(K), len(pose))):
                for h in range(2):
                    if np.median(C[t, h]) < SURE:
                        continue
                    n_frames += 1
                    for j in range(21):
                        if C[t, h, j] < SURE:
                            continue
                        M = pose[t, hb[h, j]]
                        P0 = M[:3, :3] @ hl[h, j] + M[:3, 3]
                        r = K[t, h, j] - (c0 + J @ P0)
                        rows.setdefault((clip, h, j), []).append((J @ M[:3, :3], r, float(C[t, h, j])))
                        before.append(np.linalg.norm(r))
    lam = (a.sigma_px / a.sigma_m) ** 2                                  # a prior: sigma_m of offset costs sigma_px

    def solve(rr):
        A = np.stack([x[0] for x in rr])
        r = np.stack([x[1] for x in rr])
        w = np.array([x[2] for x in rr])
        o = np.zeros(3)
        for _ in range(4):                                             # reweighted: a misread frame counts less
            res = np.linalg.norm(np.einsum("nij,j->ni", A, o) - r, axis=1)
            ww = w * np.minimum(1.0, a.huber_px / np.maximum(res, 1e-6))
            o = np.linalg.solve(np.einsum("n,nki,nkj->ij", ww, A, A) + lam * np.eye(3), np.einsum("n,nki,nk->i", ww, A, r))
        return o, np.linalg.norm(np.einsum("nij,j->ni", A, o) - r, axis=1)

    clips = a.clips.split(",")
    pooled, spread, trust = np.zeros((2, 21, 3)), np.zeros((2, 21)), np.zeros((2, 21))
    after, used = [], np.zeros((2, 21), int)
    for h, j in np.ndindex(2, 21):
        allr = [x for c in clips for x in rows.get((c, h, j), [])]
        if len(allr) < 10:
            continue
        o, res = solve(allr)
        per = [solve(rows[(c, h, j)])[0] for c in clips if len(rows.get((c, h, j), [])) >= 10]
        # how far the offset moves from one clip's poses to another's: DWPose sees a hand a little differently
        # in every pose, and a video's poses are others again - an offset is trusted as far as it holds, and
        # only as far as it stands clear of the ~1.5 cm DWPose's points wander between poses on any hand (a
        # consistent 1.2 cm on Aoi's wrist, from a wave, the idle and a jump, made her spell's hands worse)
        sd = np.sqrt(np.mean([np.sum((p - o) ** 2) for p in per])) if len(per) >= 2 else np.linalg.norm(o)
        pooled[h, j], spread[h, j], used[h, j] = o, sd, len(allr)
        trust[h, j] = o @ o / (o @ o + sd ** 2 + a.floor_m ** 2)
        after += list(res)
    # only the wrist is written: held against the video, the finger points' offsets made the hands worse on
    # every character (REVIEW, Tried and dropped); the wrist is where a stylised hand's shape moves DWPose's
    # point furthest - on Pip's to his sleeve's cuff, 4 cm up his wrist
    keep = np.zeros((2, 21, 1))
    keep[:, 0] = 1
    off = pooled * trust[..., None] * keep
    json.dump({"points": NAMES, "offset_local": off.round(5).tolist(), "pooled_offset_local": pooled.round(5).tolist(),
               "spread_m": spread.round(4).tolist(), "trust": trust.round(3).tolist(), "frames": used.tolist(),
               "clips": clips, "yaws": [float(y) for y in a.yaws.split(",")],
               "residual_px_median": {"before": round(float(np.median(before)), 2) if before else None,
                                      "after": round(float(np.median(after)), 2) if after else None}},
              open(work / "hand_calib.json", "w"), indent=1)
    cm = np.linalg.norm(pooled, axis=-1) * 100
    print(f"[hands] {a.name}: DWPose sure of {n_frames} hand-frames; its points {np.median(before):.1f} px from the rig's "
          f"-> {np.median(after):.1f} px with every point's offset (median {np.median(cm):.1f} cm); the wrists "
          f"{cm[0, 0]:.1f}/{cm[1, 0]:.1f} cm off, {spread[0, 0] * 100:.1f}/{spread[1, 0] * 100:.1f} cm from clip to clip "
          f"-> trusted {trust[0, 0]:.0%}/{trust[1, 0]:.0%}; {work / 'hand_calib.json'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--clips", default="wave,idle,jump", help="the character's own clips to render")
    ap.add_argument("--yaws", default="-40,0,40", help="camera angles, degrees from the front")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--fps", type=float, default=12.0, help="frames are this far apart in each clip")
    ap.add_argument("--sigma-px", type=float, default=3.0)
    ap.add_argument("--sigma-m", type=float, default=0.03)
    ap.add_argument("--huber-px", type=float, default=3.0)
    ap.add_argument("--floor-m", type=float, default=0.015, help="how far DWPose's points wander between poses")
    main(ap.parse_args())
