"""How close a finished character is to its own reference image: the 3D model in the reference's pose,
rendered from the front, laid over the picture.

    python tools/image_fidelity.py --names mara,juno3,rowan,wren,aoi,pip,vex

The model is rendered from rig.blend - rigged, textured, in the pose it was generated in (the
reference's A-pose; rig_f.blend is already in the T-pose) - by blender/render_match.py --pose rest. The
render is laid over the reference by the similarity that best overlaps the two silhouettes
(pipeline/align.py), then scored:
  silhouette  IoU of the model's outline with the reference figure's (reference_mask.png)
  joints      the same pose model on both, mean distance in the reference's torso lengths
  colour      mean colour difference (CIE76 dE) where both are the figure
Writes results/v3/image_fidelity.json and a sheet (render outline over each reference).
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
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
from align import image_mask, similarity  # noqa: E402
from motion_fidelity import BLENDER, NAMES, Pose, load_render  # noqa: E402


def lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


def main(a):
    pose = Pose()
    out = ROOT / "results" / "v3"
    report, tiles = {}, []
    for n in a.names.split(","):
        w = ROOT / "work" / n
        ref = cv2.cvtColor(cv2.imread(str(w / "reference.png")), cv2.COLOR_BGR2RGB)
        mfile = w / "reference_mask.png"
        rmask = (cv2.imread(str(mfile), 0) > 127) if mfile.exists() else image_mask(ref)
        rd = w / "fidelity_rest"
        rd.mkdir(exist_ok=True)
        subprocess.run([BLENDER, "-b", "-noaudio", "--python", str(ROOT / "blender" / "render_match.py"), "--",
                        "--blend", str(w / "rig.blend"), "--pose", "rest", "--yaw", "0", "--out", str(rd),
                        "--res", str(ref.shape[1]), str(ref.shape[0])], check=True, capture_output=True)
        rgb, alpha = load_render(sorted(rd.glob("[0-9]*.png"))[0])
        (s, tx, ty), iou = similarity(alpha, rmask)
        M = np.float32([[s, 0, tx], [0, s, ty]])
        h, wd = rmask.shape
        al = cv2.warpAffine(alpha.astype(np.uint8), M, (wd, h), flags=cv2.INTER_NEAREST) > 0
        col = cv2.warpAffine(rgb, M, (wd, h), flags=cv2.INTER_LINEAR)
        iou = float((al & rmask).sum() / max((al | rmask).sum(), 1))
        both = al & rmask
        de = float(np.linalg.norm(lab(col)[both] - lab(ref)[both], axis=1).mean())
        kr, cr = pose(ref, rmask)
        km, cm = pose(rgb, alpha)
        km = km * s + np.array([tx, ty])
        torso = np.linalg.norm((kr[1] + kr[4]) / 2 - (kr[7] + kr[10]) / 2)
        ok = (cr > 0.3) & (cm > 0.3)
        err = np.linalg.norm(km - kr, axis=1) / torso
        report[n] = {"silhouette_iou": round(iou, 4), "colour_dE": round(de, 2),
                     "joint_error_torso": round(float(err[ok].mean()), 4),
                     "per_joint": {NAMES[j]: round(float(err[j]), 3) for j in range(13) if ok[j]}}
        vis = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR).copy()
        cnts, _ = cv2.findContours(al.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (255, 140, 40), 2)
        cv2.putText(vis, f"{n} IoU {iou:.3f} joints {report[n]['joint_error_torso']:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(vis, f"{n} IoU {iou:.3f} joints {report[n]['joint_error_torso']:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(vis, (int(vis.shape[1] * 480 / vis.shape[0]), 480)))
        print(f"[image] {n:6s} silhouette IoU {iou:.3f}  joints {report[n]['joint_error_torso']:.3f} torso lengths  "
              f"colour dE {de:.1f}", flush=True)
    json.dump(report, open(out / "image_fidelity.json", "w"), indent=1)
    cv2.imwrite(str(out / "image_fidelity.png"), np.hstack(tiles))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="mara,juno3,rowan,wren,aoi,pip,vex")
    main(ap.parse_args())
