"""Where a clip made from a video loses the video: the same score at each stage of the pipeline.

    python tools/motion_stages.py --name mara --move punch_combo [--performer mara]

Each stage's joints are projected by an orthographic camera, laid over the pose model's reading of
the video exactly as tools/motion_fidelity.py lays a render over it (one scale for the clip from the
torso, the hips per frame), and scored in the video's torso lengths on the twelve body joints (the
pose model's head point is the face, which no skeleton has a joint at):

  fit      the points pipeline/video_motion.py --fit solved for (<move>_fit.npz)
  fbx      the Mixamo clip blender/fit_clip.py wrote, read back (the skeleton the retarget reads)
  rig      the character's own skeleton playing the retargeted clip (final.blend)
  render   the pose model reading the character's render (from motion_fidelity.py's cache)

A skeleton stage is scored from the camera that suits it best (a search over the turn about the
vertical, and a mirror), and from the angle the render was made at. The render's joints are also
held against the rig's own, which isolates where the pose model reads this character's joints
from where its skeleton has them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
NAMES = ["head", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist",
         "l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle"]
MIRROR_PERM = [0, 4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9]
BODY = list(range(1, 13))
MISREAD = 0.3                 # torso lengths between the render's reading and the rig's skeleton


def project(P, yaw_deg, mirror=False):
    """(T, 13, 3) world points, Z up -> (T, 13, 2) image points (y down) for a level orthographic
    camera at `yaw_deg` round from the front (-Y), as blender/render_match.py places it."""
    r = np.radians(yaw_deg)
    uv = np.stack([P @ np.array([np.cos(r), np.sin(r), 0.0]), -P[..., 2]], -1)
    if mirror:
        uv = uv[:, MIRROR_PERM] * np.array([-1.0, 1.0])
    return uv


def align_error(vk, vc, k, kc=None, how="ls"):
    """Per frame and joint, in the video's torso lengths: k laid on vk with one scale for the clip and
    the hips per frame. how="torso" takes the scale from the torsos - motion_fidelity.score's
    alignment, right between two readings by the same pose model; "ls" the scale that best lays all
    twelve joints about the hips, for a skeleton against a pose model: a skeleton's hips sit where
    its builder put them (Aoi's 4 cm above the pose model's), and a torso ratio then stretches
    everything below them."""
    hip = lambda x: (x[:, 7] + x[:, 10]) / 2                          # noqa: E731
    sh = lambda x: (x[:, 1] + x[:, 4]) / 2                            # noqa: E731
    tv = np.nanmedian(np.linalg.norm(sh(vk) - hip(vk), axis=1))
    if how == "torso":
        tk = np.nanmedian(np.linalg.norm(sh(k) - hip(k), axis=1))
        s = tv / max(tk, 1e-9)
    else:
        a_ = (k - hip(k)[:, None])[:, BODY]
        b_ = (vk - hip(vk)[:, None])[:, BODY]
        ok_ = (vc[:, BODY] > 0.3) & np.isfinite(b_).all(-1) & np.isfinite(a_).all(-1)
        s = float(np.sum(a_[ok_] * b_[ok_]) / max(np.sum(a_[ok_] * a_[ok_]), 1e-9))
    al = k * s + (hip(vk) - s * hip(k))[:, None, :]
    err = np.linalg.norm(al - vk, axis=-1) / tv
    ok = vc > 0.3
    if kc is not None:
        ok &= kc > 0.3
    err[~ok] = np.nan
    return err, al


def best_camera(vk, vc, P, around=None):
    """The turn (and mirror) that lays P's projection closest to the video's points."""
    best = None
    yaws = np.arange(-180, 180, 1.0) if around is None else around + np.arange(-30, 30.5, 0.5)
    for m in (False, True):
        for y in yaws:
            e = np.nanmean(align_error(vk, vc, project(P, y, m))[0][:, BODY])
            if best is None or e < best[0]:
                best = (e, float(y), m)
    return best


def dump(args, out):
    # read again when the blend or FBX is newer than the cache: a cache that never went stale kept a
    # rig two rebuilds old, and scored Mara's kick against a skeleton she no longer has
    src = Path(args[args.index("--blend") + 1] if "--blend" in args else args[args.index("--fbx") + 1])
    if not out.exists() or (src.exists() and src.stat().st_mtime > out.stat().st_mtime):
        subprocess.run([BLENDER, "-b", "-noaudio", "--python", str(ROOT / "blender" / "dump_joints.py"), "--",
                        *args, "--out", str(out)], check=True, capture_output=True)
    return np.load(out)


def main(a):
    performer = a.performer or a.name
    vids = ROOT / "work" / performer / "motion_videos"
    mine = ROOT / "work" / a.name / "motion_videos"
    kp = np.load(Path(a.kp) if a.kp else mine / f"{a.move}_fidelity_kp.npz")
    vk, vc, rk, rc, yaw = kp["vid_kp"], kp["vid_conf"], kp["ren_kp"], kp["ren_conf"], float(kp["yaw"])
    n = len(vk)
    cache = mine / f"{a.move}_stages"
    cache.mkdir(exist_ok=True)
    blend = ROOT / "work" / a.name / (a.blend or "final.blend")
    tag = Path(a.blend or "final.blend").stem
    fitdir = Path(a.fit_dir) if a.fit_dir else vids
    stages = {"fit": np.load(fitdir / f"{a.move}_fit.npz")["points"].astype(np.float64),
              "fbx": dump(["--fbx", str(fitdir / f"{a.move}.fbx"), "--frames", str(n)],
                          cache / f"fbx{'_' + fitdir.name if a.fit_dir else ''}.npz")["points"]}
    rig = dump(["--blend", str(blend), "--clip", a.clip or a.move, "--frames", str(n)], cache / f"rig_{tag}.npz")
    stages["rig"] = rig["points"]
    rep = {"character": a.name, "move": a.move, "blend": blend.name, "render_yaw": yaw, "stages": {}}
    for key, P in stages.items():
        e, y, m = best_camera(vk, vc, P)
        err, _ = align_error(vk, vc, project(P, y, m))
        row = {"joints": round(float(np.nanmean(err[:, BODY])), 4), "camera_yaw": y, "mirror": m,
               "per_joint": {NAMES[j]: round(float(np.nanmean(err[:, j])), 3) for j in BODY}}
        if key == "rig":
            er2, _ = align_error(vk, vc, project(P, yaw))
            row["joints_at_render_yaw"] = round(float(np.nanmean(er2[:, BODY])), 4)
        rep["stages"][key] = row
    err, _ = align_error(vk, vc, rk, rc, how="torso")
    err_ls, _ = align_error(vk, vc, rk, rc)
    rep["stages"]["render"] = {"joints": round(float(np.nanmean(err[:, BODY])), 4), "camera_yaw": yaw,
                               "joints_ls": round(float(np.nanmean(err_ls[:, BODY])), 4),
                               "per_joint": {NAMES[j]: round(float(np.nanmean(err[:, j])), 3) for j in BODY}}
    # the pose model on the render against the rig's own joints, from the render's camera
    e_rr, _ = align_error(rk, rc, project(stages["rig"], yaw))
    rep["render_vs_rig"] = {"joints": round(float(np.nanmean(e_rr[:, BODY])), 4),
                            "per_joint": {NAMES[j]: round(float(np.nanmean(e_rr[:, j])), 3) for j in BODY}}
    # frames where the pose model reads the render far from the rig's own skeleton are the reader's
    # failure, not the rig's (the top of Mara's kick: 0.45 against 0.05 elsewhere) - the score without them
    rr = np.nanmean(e_rr[:, BODY], 1)
    misread = np.where(rr > MISREAD)[0]
    keep = np.setdiff1d(np.arange(len(rr)), misread)
    rep["render"] = {"misread_frames": misread.tolist(), "misread_above": MISREAD,
                     "joints_trusted": round(float(np.nanmean(err[keep][:, BODY])), 4) if len(keep) else None}
    json.dump(rep, open(mine / f"{a.move}_stages{'' if tag == 'final' else '_' + tag}.json", "w"), indent=1)
    s = rep["stages"]
    print(f"[stages] {a.name} {a.move} ({blend.name}): fit {s['fit']['joints']:.3f} -> fbx {s['fbx']['joints']:.3f} "
          f"-> rig {s['rig']['joints']:.3f} (at the render's angle {s['rig']['joints_at_render_yaw']:.3f}) "
          f"-> render {s['render']['joints']:.3f} (least-squares scale {s['render']['joints_ls']:.3f}) torso lengths; the pose model's reading of the render "
          f"vs the rig's joints {rep['render_vs_rig']['joints']:.3f}", flush=True)
    if len(misread):
        print(f"[stages]   the pose model misreads the render in {len(misread)} frames ({misread[0]}-{misread[-1]}); "
              f"the render scores {rep['render']['joints_trusted']:.3f} without them", flush=True)
    for key in ("fit", "rig", "render"):
        worst = sorted(s[key]["per_joint"].items(), key=lambda kv: -kv[1])[:4]
        print(f"[stages]   {key:6s} worst: " + ", ".join(f"{k} {v:.2f}" for k, v in worst)
              + (f"  (camera {s[key]['camera_yaw']:.1f}{' mirrored' if s[key].get('mirror') else ''})"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--move", required=True)
    ap.add_argument("--performer", default=None)
    ap.add_argument("--blend", default=None, help="the character's blend to read the clip from (default final.blend)")
    ap.add_argument("--clip", default=None, help="the clip's name in that blend (default: the move)")
    ap.add_argument("--fit-dir", default=None, help="where the fit's npz and FBX are (default next to the video)")
    ap.add_argument("--kp", default=None, help="the fidelity run's pose readings (default next to the video)")
    main(ap.parse_args())
