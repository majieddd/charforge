"""How well a 3D model holds up from every side: its silhouette against a video of the character
turning a full circle (charforge.py move --move turntable).

    python tools/turntable_fidelity.py --name mara --mesh work/mara/mesh.glb --mesh work/mara/pass1.glb \
        [--video work/mara/motion_videos/turntable.mp4] [--hold-out 90,180,270]

The reference image only shows the front, and at rest the front already overlaps it at IoU 0.90-0.93;
what the model is like from the side and the back had no measure. A turntable video is one: each
frame's figure (cut out by pipeline/foreground.py --video) is held against the model's silhouette
from every 5 degrees (blender/render_turntable.py), each laid on the figure by height, feet and
centre, and the best angle kept. The angles follow the turn: a dynamic programme picks, per frame,
the best-fitting angle that moves on from the last by at most 25 degrees (a symmetric silhouette
reads the same from the front and the back, so a free choice flips between them).

Per model: the mean overlap over the turn, by quarter (front, left side, back, right side), and per
frame. With --hold-out, frames within 20 degrees of those angles are left out of the score - frames a
model was built from, to keep the test honest. Writes results/v3/turntable_<name>.json and a sheet.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from motion_fidelity import BLENDER, video_frames  # noqa: E402

RES = (192, 288)                                                     # working resolution (w, h)


def fit_to(sil, target):
    """sil laid on target by height, feet and horizontal centre; both boolean at RES."""
    ys, xs = np.nonzero(sil)
    ty, tx = np.nonzero(target)
    if len(ys) < 20 or len(ty) < 20:
        return np.zeros_like(target)
    s = (ty.max() - ty.min()) / max(ys.max() - ys.min(), 1)
    M = np.float32([[s, 0, (tx.min() + tx.max()) / 2 - s * (xs.min() + xs.max()) / 2], [0, s, ty.max() - s * ys.max()]])
    return cv2.warpAffine(sil.astype(np.uint8), M, (target.shape[1], target.shape[0]), flags=cv2.INTER_NEAREST) > 0


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main(a):
    w = ROOT / "work" / a.name
    video = Path(a.video) if a.video else w / "motion_videos" / "turntable.mp4"
    mfile = video.with_name(video.stem + "_masks.npz")
    if not mfile.exists():
        subprocess.run([str(ROOT / "vendor" / "trellis2mlx" / ".venv" / "bin" / "python"), str(ROOT / "pipeline" / "foreground.py"),
                        "--video", str(video), "--out", str(mfile)], check=True, capture_output=True)
    masks = np.load(mfile)["masks"]
    T = len(masks)
    tgt = [cv2.resize(m.astype(np.uint8), RES, interpolation=cv2.INTER_NEAREST) > 0 for m in masks]
    held = [float(x) for x in a.hold_out.split(",")] if a.hold_out else []
    held_frames = [int(x) for x in a.hold_out_frames.split(",")] if a.hold_out_frames else []
    report, sheets = {}, []
    for mesh in a.mesh:
        mesh = Path(mesh)
        out = w / "turntable" / mesh.stem
        if not (out / "000.png").exists() or (out / "000.png").stat().st_mtime < mesh.stat().st_mtime:
            subprocess.run([BLENDER, "-b", "-noaudio", "--python", str(ROOT / "blender" / "render_turntable.py"), "--",
                            "--mesh", str(mesh), "--out", str(out), "--step", "5"], check=True, capture_output=True)
        views = sorted(out.glob("[0-9][0-9][0-9].png"))
        az = np.array([int(v.stem) for v in views], float)
        sil = [cv2.resize((cv2.imread(str(v), cv2.IMREAD_UNCHANGED)[..., 3] > 127).astype(np.uint8), RES,
                          interpolation=cv2.INTER_NEAREST) > 0 for v in views]
        S = np.array([[iou(fit_to(s, t), t) for s in sil] for t in tgt])    # (frames, angles)
        # the turn as a path through the angles: steps of at most 25 degrees
        n = len(az)
        step = np.abs((az[:, None] - az[None, :] + 180) % 360 - 180) <= 25
        cost, back = -S[0].copy(), np.zeros((T, n), int)
        for t in range(1, T):
            c = np.where(step, cost[:, None], np.inf)
            back[t] = np.argmin(c, 0)
            cost = c[back[t], np.arange(n)] - S[t]
        path = np.zeros(T, int)
        path[-1] = int(np.argmin(cost))
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        ang = az[path]
        score = S[np.arange(T), path]
        keep = np.ones(T, bool)
        for h_ in held:
            keep &= np.abs((ang - h_ + 180) % 360 - 180) > 20
        for f_ in held_frames:                                       # frames a model was built from
            keep[max(0, f_ - 4):f_ + 5] = False
        quarter = {}
        for nm, c in (("front", 0), ("left side", 90), ("back", 180), ("right side", 270)):
            m = keep & (np.abs((ang - c + 180) % 360 - 180) <= 45)
            quarter[nm] = round(float(score[m].mean()), 4) if m.any() else None
        report[mesh.name] = {"iou_mean": round(float(score[keep].mean()), 4), "frames_scored": int(keep.sum()),
                             "by_quarter": quarter, "angle_per_frame": ang.tolist(),
                             "iou_per_frame": [round(float(x), 4) for x in score]}
        print(f"[turntable] {a.name} {mesh.name}: IoU {report[mesh.name]['iou_mean']:.3f} over {int(keep.sum())} frames; "
              + ", ".join(f"{k} {v:.3f}" for k, v in quarter.items() if v is not None)
              + f"; the turn covers {np.ptp(np.unwrap(np.radians(ang))) * 180 / np.pi:.0f} deg", flush=True)
        # sheet: eight frames, the model's outline on the video
        fr = video_frames(video)
        row = []
        for t in np.linspace(0, T - 1, 8).astype(int):
            im = cv2.resize(cv2.cvtColor(fr[t], cv2.COLOR_RGB2BGR), RES)
            fit = fit_to(sil[path[t]], tgt[t])
            cnts, _ = cv2.findContours(fit.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cnts, -1, (255, 140, 40), 1)
            cv2.putText(im, f"{ang[t]:.0f} {score[t]:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
            row.append(im)
        sheets.append(np.hstack(row))
    out = ROOT / "results" / "v3"
    json.dump(report, open(out / f"turntable_{a.name}{'_' + a.tag if a.tag else ''}.json", "w"), indent=1)
    cv2.imwrite(str(out / f"turntable_{a.name}{'_' + a.tag if a.tag else ''}.png"), np.vstack(sheets))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mesh", action="append", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--hold-out", default="", help="angles (deg) whose frames a model was built from, left out")
    ap.add_argument("--hold-out-frames", default="", help="video frames a model was built from (+-4 left out)")
    ap.add_argument("--tag", default="")
    main(ap.parse_args())
