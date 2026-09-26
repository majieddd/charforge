"""Where the pose model sees a character's joints, as points fixed to its bones - so a clip can be fitted
to the pose model's reading of a video in the character's own terms.

    python tools/calibrate_joints.py --names mara,aoi,pip

A video is read by a pose model (ViTPose), and a skeleton's joints are not where a pose model puts
them: Aoi's rig has its hips 4 cm above where the pose model reads them and her upper arm is 0.78
torso lengths joint to joint against the pose model's 0.59. A fit built on either set of lengths
while aiming at the other bends the body to make up the difference (a short torso read as a 28 deg
lean). So the character is rendered in its rest pose (rig_f.blend, the T-pose clips are retargeted
from) from the front and from each side, the same pose model reads it, and each of its 13 points -
head; shoulders, elbows, wrists, hips, knees, ankles - is placed in 3D: across and up from the
front view, depth from the side that sees it (the arms point at a side camera in a T-pose, so an
elbow or wrist keeps its rig joint's depth). Each point is stored in the frame of the bone whose
head is that joint, and carried by it in any pose.

Writes work/<name>/joint_calib.json: per point its bone and local offset, the rest positions of both
the pose model's points and the rig's joints, and the body's lengths in the pose model's terms (torso
lengths, left and right averaged) for pipeline/video_motion.py --calib.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from motion_fidelity import BLENDER, NAMES, Pose, load_render  # noqa: E402

HIPM, SHM = 13, 14
TREE = [(HIPM, 7), (7, 8), (8, 9), (HIPM, 10), (10, 11), (11, 12), (HIPM, SHM),
        (SHM, 1), (1, 2), (2, 3), (SHM, 4), (4, 5), (5, 6), (SHM, 0)]
PAIRS = {(HIPM, 7): (HIPM, 10), (7, 8): (10, 11), (8, 9): (11, 12), (SHM, 1): (SHM, 4), (1, 2): (4, 5), (2, 3): (5, 6)}
LEFT, RIGHT = (1, 7, 8, 9), (4, 10, 11, 12)                            # read in depth from a side view
FACE = ["nose", "l_eye", "r_eye", "l_ear", "r_ear"]                    # the pose model's first five points
FACE_SIDE = {90: (0, 1, 3), -90: (0, 2, 4)}                            # the face points each side view sees


def render(blend, yaw, out, res):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([BLENDER, "-b", "-noaudio", "--python", str(ROOT / "blender" / "render_match.py"), "--",
                    "--blend", str(blend), "--pose", "rest", "--yaw", str(yaw), "--out", str(out),
                    "--res", str(res[0]), str(res[1])], check=True, capture_output=True)
    cam = (out / "camera.txt").read_text().split()
    return {"yaw": float(cam[1]), "ortho": float(cam[3]), "ctr": np.array(eval(" ".join(cam[5:8]))),
            "res": (int(cam[9]), int(cam[10]))}, load_render(out / "0000.png")


def unproject(uv, cam):
    """Image points (pixels, y down) -> (world coordinate along the image's horizontal axis, height).
    The axis is (cos yaw, sin yaw, 0): +X from the front, +Y from yaw 90, -Y from yaw -90."""
    rx, ry = cam["res"]
    r = np.radians(cam["yaw"])
    along = (uv[:, 0] - rx / 2) / ry * cam["ortho"] + cam["ctr"] @ np.array([np.cos(r), np.sin(r), 0.0])
    z = cam["ctr"][2] - (uv[:, 1] - ry / 2) / ry * cam["ortho"]
    return along, z


def with_mids(X):
    return np.vstack([X, (X[7] + X[10]) / 2, (X[1] + X[4]) / 2])


def main(a):
    pose = Pose()
    for n in a.names.split(","):
        w = ROOT / "work" / n
        # the rest pose clips are retargeted from: rig_f.blend (the animate stage's input; final.blend
        # has the same rest and exists only after it)
        blend = w / "rig_f.blend" if (w / "rig_f.blend").exists() else w / "final.blend"
        dump = w / "calib" / "rest_joints.npz"
        dump.parent.mkdir(exist_ok=True)
        subprocess.run([BLENDER, "-b", "-noaudio", "--python", str(ROOT / "blender" / "dump_joints.py"), "--",
                        "--blend", str(blend), "--out", str(dump)], check=True, capture_output=True)
        d = np.load(dump)
        rig, mats, bones = d["rest"], d["rest_mats"], [str(b) for b in d["bones"]]
        views, faces = {}, {}
        for yaw in (0, 90, -90):
            cam, (rgb, alpha) = render(blend, yaw, w / "calib" / f"yaw{yaw:+d}", a.res)
            kp, conf = pose(rgb, alpha)
            views[yaw] = (cam, kp, conf)
            faces[yaw] = pose.read17(rgb, alpha)
        cam, kp, conf = views[0]
        x, z = unproject(kp, cam)
        V = np.column_stack([x, rig[:, 1], z])                        # depth: the rig joint's, for now
        # depth from the side that sees each point: a camera at yaw +90 sits on the character's left
        # (+X) and its image runs along +Y; at -90, on the right, along -Y
        side = {}
        for yaw, pts, sgn in ((90, LEFT + (0,), 1.0), (-90, RIGHT, -1.0)):
            c2, k2, f2 = views[yaw]
            along, _ = unproject(k2, c2)
            for j in pts:
                if f2[j] > 0.5 and conf[j] > 0.5:
                    side[j] = sgn * along[j]
        for j, y in side.items():
            V[j, 1] = y
        # the points in their bones' rest frames
        local = [np.linalg.inv(mats[j]) @ np.append(V[j], 1.0) for j in range(13)]
        # the face: nose, eyes and ears, across and up from the front, depth from the sides that see
        # them (the nose from both), carried by the head bone - the head's turn in a video is read
        # off these five (pipeline/video_motion.py fits it, blender/retarget.py turns the head to it)
        k17, c17 = faces[0]
        fx, fz = unproject(k17[:5], views[0][0])
        Fd = {i: [] for i in range(5)}
        for yaw, sgn in ((90, 1.0), (-90, -1.0)):
            k2, c2 = faces[yaw]
            along, _ = unproject(k2[:5], views[yaw][0])
            for i in FACE_SIDE[yaw]:
                if c2[i] > 0.3:
                    Fd[i].append(sgn * along[i])
        F3 = np.array([[fx[i], np.mean(Fd[i]) if Fd[i] else rig[0, 1], fz[i]] for i in range(5)])
        face = [{"name": FACE[i], "local": [round(float(v), 5) for v in (np.linalg.inv(mats[0]) @ np.append(F3[i], 1.0))[:3]],
                 "rest": [round(float(v), 5) for v in F3[i]], "confidence": round(float(c17[i]), 3),
                 "depth_from_side": bool(Fd[i])} for i in range(5)]
        Vm = with_mids(V)
        torso = float(np.linalg.norm(Vm[SHM] - Vm[HIPM]))
        L = {}
        for b in TREE:
            l = np.linalg.norm(Vm[b[1]] - Vm[b[0]])
            if b in PAIRS:
                l = (l + np.linalg.norm(Vm[PAIRS[b][1]] - Vm[PAIRS[b][0]])) / 2
            L[b] = float(l / torso)
        for p, q in PAIRS.items():
            L[q] = L[p]
        Rm = with_mids(rig)
        rig_torso = float(np.linalg.norm(Rm[SHM] - Rm[HIPM]))
        rep = {"character": n, "source": f"{blend.name} rest pose, ViTPose base, front + side views",
               "torso_m": torso, "rig_torso_m": rig_torso,
               "points": [{"name": NAMES[j], "bone": bones[j], "local": [round(float(v), 5) for v in local[j][:3]],
                           "rest": [round(float(v), 5) for v in V[j]], "rig": [round(float(v), 5) for v in rig[j]],
                           "confidence": round(float(conf[j]), 3), "depth_from_side": j in side}
                          for j in range(13)],
               "face": face,
               "lengths": {f"{p},{q}": round(v, 4) for (p, q), v in L.items()}}
        json.dump(rep, open(w / "joint_calib.json", "w"), indent=1)
        gap = np.linalg.norm(V - rig, axis=1) / torso
        print(f"[calib] {n:6s} torso {torso:.3f} m (rig joints {rig_torso:.3f}); pose-model points vs rig joints, "
              f"torso lengths: " + " ".join(f"{NAMES[j]} {gap[j]:.2f}" for j in range(13)), flush=True)
        print(f"[calib]        lengths (torso): upper arm {L[(1, 2)]:.2f} forearm {L[(2, 3)]:.2f} thigh {L[(7, 8)]:.2f} "
              f"shin {L[(8, 9)]:.2f} half-shoulders {L[(SHM, 1)]:.2f} half-hips {L[(HIPM, 7)]:.2f} head {L[(SHM, 0)]:.2f}; "
              f"depth from the side for {sorted(NAMES[j] for j in side)}", flush=True)
        nose, ears = F3[0], (F3[3] + F3[4]) / 2
        print(f"[calib]        face: nose {(ears[1] - nose[1]) * 100:.1f} cm in front of the ears, eyes "
              f"{np.linalg.norm(F3[1] - F3[2]) * 100:.1f} cm apart, ears {np.linalg.norm(F3[3] - F3[4]) * 100:.1f} cm; "
              f"confidence {', '.join(f'{FACE[i]} {c17[i]:.2f}' for i in range(5))}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="mara,juno3,rowan,wren,aoi,pip,vex")
    ap.add_argument("--res", type=int, nargs=2, default=(1024, 1536))
    main(ap.parse_args())
