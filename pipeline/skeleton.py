"""Estimate a humanoid skeleton for a generated character mesh.

Two routes, best first:
  1. pose landmarks on the orthographic front and side renders. The renders' cameras are known
     exactly, so front gives (x, z) and side gives (y, z); combining them recovers 3D joints
     without any depth guesswork.
  2. mesh slice analysis, used when a pose model is unavailable or fails on a stylised
     character: crotch height from the point where a horizontal slice splits into two islands,
     shoulders from the widest slice under the neck, arms from the extreme points of the
     A-pose limbs.

Joint names follow the SMPL-H body layout, so motion from SMPL-based generators
(e.g. HY-Motion) retargets by name with no mapping table.

Output: joints.json  {name: [x, y, z]} in the normalised mesh space (height 2.0, centred).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BODY_JOINTS = [
    "pelvis", "spine1", "spine2", "spine3", "neck", "head", "head_top",
    "left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_hand",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist", "right_hand",
    "left_hip", "left_knee", "left_ankle", "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
]

PARENTS = {
    "pelvis": None, "spine1": "pelvis", "spine2": "spine1", "spine3": "spine2",
    "neck": "spine3", "head": "neck", "head_top": "head",
    "left_collar": "spine3", "left_shoulder": "left_collar", "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow", "left_hand": "left_wrist",
    "right_collar": "spine3", "right_shoulder": "right_collar", "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow", "right_hand": "right_wrist",
    "left_hip": "pelvis", "left_knee": "left_hip", "left_ankle": "left_knee", "left_foot": "left_ankle",
    "right_hip": "pelvis", "right_knee": "right_hip", "right_ankle": "right_knee", "right_foot": "right_ankle",
}


# ---------------------------------------------------------------- route 1: pose landmarks
def _unproject(px, py, res, ortho_scale, axis_h, axis_v_min=None):
    """Pixel -> world for one orthographic view (returns the two world axes the view sees)."""
    x_ndc = (px / res) * 2 - 1
    y_ndc = (1 - py / res) * 2 - 1
    return x_ndc * (ortho_scale / 2), y_ndc * (ortho_scale / 2)


# COCO-17 keypoint order used by ViTPose
COCO = {n: i for i, n in enumerate([
    "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"])}


def from_pose(views_dir: Path, repo="usyd-community/vitpose-base-simple", device="mps",
              min_score=0.3):
    """ViTPose keypoints on the 0deg and 90deg orthographic renders -> 3D joints."""
    try:
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
    except ImportError as e:
        return None, f"vitpose unavailable ({e})"
    from PIL import Image

    meta = json.load(open(views_dir / "meta.json"))
    by_az = {round(v["azimuth"]) % 360: v for v in meta["views"]}
    if 0 not in by_az or 90 not in by_az:
        return None, "need a 0deg and a 90deg view"
    res = meta["res"]

    proc = AutoProcessor.from_pretrained(repo)
    model = VitPoseForPoseEstimation.from_pretrained(repo).to(device).eval()
    out, scores = {}, {}
    for az in (0, 90):
        v = by_az[az]
        im = Image.open(views_dir / f"view_{v['index']:02d}.png").convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(bg, im).convert("RGB")
        # the character fills the frame, so the whole image is the person box
        boxes = [[[0.0, 0.0, float(res), float(res)]]]
        inputs = proc(rgb, boxes=boxes, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**inputs)
        res_kp = proc.post_process_pose_estimation(o, boxes=boxes)[0][0]
        out[az] = res_kp["keypoints"].cpu().numpy()
        scores[az] = res_kp["scores"].cpu().numpy()
    if min(scores[0].mean(), scores[90].mean()) < min_score:
        return None, f"pose confidence too low ({scores[0].mean():.2f}/{scores[90].mean():.2f})"

    os_ = by_az[0]["ortho_scale"]

    # Seen from the side, an A-posed arm lies over the torso and the hair: aoi's side view put
    # both wrists on her ponytail. An arm joint's height is read from the front view alone.
    FRONT_ONLY = {"LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"}

    def world(name):
        i = COCO[name]
        fx, fy = out[0][i]
        sx, sy = out[90][i]
        X, Zf = _unproject(fx, fy, res, os_, None)
        Ys, Zs = _unproject(sx, sy, res, os_, None)
        # The 90deg camera sits at +X looking back at the origin, so its image right is world +Y
        # (the character, facing -Y, looks left in it). This used to be sign-flipped, mirroring
        # every joint's depth front to back: harmless on a slim character, whose limbs the joint
        # refinement found anyway within its 7 cm search, but aoi's legs came out 11 cm behind
        # themselves and no limb could be traced.
        return np.array([X, Ys, Zf if name in FRONT_ONLY else (Zf + Zs) / 2], dtype=float)

    lm = {n: world(n) for n in COCO}
    # COCO has no foot/hand tips: extend the last limb segment
    lm["LEFT_FOOT_INDEX"] = lm["LEFT_ANKLE"] + np.array([0, -0.10, -0.03])
    lm["RIGHT_FOOT_INDEX"] = lm["RIGHT_ANKLE"] + np.array([0, -0.10, -0.03])
    lm["LEFT_INDEX"] = lm["LEFT_WRIST"] + (lm["LEFT_WRIST"] - lm["LEFT_ELBOW"]) * 0.25
    lm["RIGHT_INDEX"] = lm["RIGHT_WRIST"] + (lm["RIGHT_WRIST"] - lm["RIGHT_ELBOW"]) * 0.25

    pelvis = (lm["LEFT_HIP"] + lm["RIGHT_HIP"]) / 2
    neck = (lm["LEFT_SHOULDER"] + lm["RIGHT_SHOULDER"]) / 2
    head = (lm["LEFT_EAR"] + lm["RIGHT_EAR"]) / 2
    j = {
        "pelvis": pelvis,
        "spine1": pelvis + (neck - pelvis) * 0.25,
        "spine2": pelvis + (neck - pelvis) * 0.50,
        "spine3": pelvis + (neck - pelvis) * 0.78,
        "neck": neck, "head": head, "head_top": head + np.array([0, 0, 0.12]),
        "left_collar": neck + (lm["LEFT_SHOULDER"] - neck) * 0.4,
        "right_collar": neck + (lm["RIGHT_SHOULDER"] - neck) * 0.4,
        "left_shoulder": lm["LEFT_SHOULDER"], "right_shoulder": lm["RIGHT_SHOULDER"],
        "left_elbow": lm["LEFT_ELBOW"], "right_elbow": lm["RIGHT_ELBOW"],
        "left_wrist": lm["LEFT_WRIST"], "right_wrist": lm["RIGHT_WRIST"],
        "left_hand": lm["LEFT_INDEX"], "right_hand": lm["RIGHT_INDEX"],
        "left_hip": lm["LEFT_HIP"], "right_hip": lm["RIGHT_HIP"],
        "left_knee": lm["LEFT_KNEE"], "right_knee": lm["RIGHT_KNEE"],
        "left_ankle": lm["LEFT_ANKLE"], "right_ankle": lm["RIGHT_ANKLE"],
        "left_foot": lm["LEFT_FOOT_INDEX"], "right_foot": lm["RIGHT_FOOT_INDEX"],
    }
    return {k: [float(x) for x in v] for k, v in j.items()}, "pose"


# ---------------------------------------------------------------- route 2: mesh analysis
def mesh_in_joint_frame(mesh_path: str):
    """The mesh's vertices in the joints' frame: Z up, bounding-box centred, 2 units tall. glTF
    is Y up; reading its vertices as they are treated depth as height."""
    import trimesh
    m = trimesh.load(mesh_path, force="mesh", process=False)
    V = np.asarray(m.vertices, dtype=float)
    V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)
    lo, hi = V.min(0), V.max(0)
    return (V - (lo + hi) / 2) * (2.0 / float(hi[2] - lo[2]))


def from_mesh(mesh_path: str):
    V = mesh_in_joint_frame(mesh_path)
    zmin, zmax = V[:, 2].min(), V[:, 2].max()
    h = zmax - zmin

    def band(z0, z1):
        return V[(V[:, 2] >= z0) & (V[:, 2] < z1)]

    # crotch: highest slice that still splits into two islands in x
    crotch = zmin + 0.50 * h
    for z in np.linspace(zmin + 0.60 * h, zmin + 0.35 * h, 40):
        b = band(z - 0.01 * h, z + 0.01 * h)
        if len(b) < 20:
            continue
        gap = np.abs(b[:, 0]) < 0.02 * h
        if gap.mean() < 0.02:            # nothing near the centre line -> two legs
            crotch = z
            break

    # shoulders: widest band in the upper torso
    best_w, sh_z = 0, zmin + 0.80 * h
    for z in np.linspace(zmin + 0.70 * h, zmin + 0.88 * h, 30):
        b = band(z - 0.01 * h, z + 0.01 * h)
        if len(b) < 20:
            continue
        w = np.percentile(np.abs(b[:, 0]), 95)
        if w > best_w:
            best_w, sh_z = w, z
    torso_half = best_w

    # neck: narrowest band above the shoulders
    neck_z, best_n = zmin + 0.87 * h, 1e9
    for z in np.linspace(sh_z, zmin + 0.93 * h, 24):
        b = band(z - 0.01 * h, z + 0.01 * h)
        if len(b) < 10:
            continue
        w = np.percentile(np.abs(b[:, 0]), 95)
        if w < best_n:
            best_n, neck_z = w, z

    y_mid = float(np.median(V[:, 1]))
    pelvis = np.array([0.0, y_mid, crotch + 0.06 * h])
    neck = np.array([0.0, y_mid, neck_z])
    head_c = np.array([0.0, y_mid, neck_z + 0.06 * h])

    def arm(sign):
        sh = np.array([sign * torso_half * 0.78, y_mid, sh_z])
        side = V[np.sign(V[:, 0]) == sign] if sign else V
        far = side[np.argmax(sign * side[:, 0])] if len(side) else sh
        d = far - sh
        return sh, sh + d * 0.45, sh + d * 0.82, sh + d * 0.97

    lsh, lel, lwr, lha = arm(1)
    rsh, rel, rwr, rha = arm(-1)
    hip_x = torso_half * 0.33
    ankle_z = zmin + 0.035 * h

    j = {
        "pelvis": pelvis,
        "spine1": pelvis + (neck - pelvis) * 0.25,
        "spine2": pelvis + (neck - pelvis) * 0.50,
        "spine3": pelvis + (neck - pelvis) * 0.78,
        "neck": neck, "head": head_c, "head_top": np.array([0.0, y_mid, zmax]),
        "left_collar": neck + np.array([torso_half * 0.3, 0, 0]),
        "right_collar": neck + np.array([-torso_half * 0.3, 0, 0]),
        "left_shoulder": lsh, "left_elbow": lel, "left_wrist": lwr, "left_hand": lha,
        "right_shoulder": rsh, "right_elbow": rel, "right_wrist": rwr, "right_hand": rha,
        "left_hip": np.array([hip_x, y_mid, crotch + 0.04 * h]),
        "right_hip": np.array([-hip_x, y_mid, crotch + 0.04 * h]),
        "left_knee": np.array([hip_x, y_mid, (crotch + ankle_z) / 2]),
        "right_knee": np.array([-hip_x, y_mid, (crotch + ankle_z) / 2]),
        "left_ankle": np.array([hip_x, y_mid, ankle_z]),
        "right_ankle": np.array([-hip_x, y_mid, ankle_z]),
        "left_foot": np.array([hip_x, y_mid - 0.06 * h, zmin + 0.005 * h]),
        "right_foot": np.array([-hip_x, y_mid - 0.06 * h, zmin + 0.005 * h]),
    }
    return {k: [float(x) for x in v] for k, v in j.items()}, "mesh-analysis"


def validate(joints):
    """Cheap sanity checks. A stylised or featureless character can break a 2D pose model,
    and a bad skeleton is worse than a generic one."""
    problems = []
    j = {k: np.asarray(v) for k, v in joints.items()}
    for side_pair in [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
                      ("left_knee", "right_knee"), ("left_elbow", "right_elbow")]:
        l, r = j[side_pair[0]], j[side_pair[1]]
        if l[0] <= r[0]:
            problems.append(f"{side_pair[0]} is not on the +X side of {side_pair[1]}")
    if j["head_top"][2] <= j["neck"][2]:
        problems.append("head_top below neck")
    if j["pelvis"][2] <= j["left_knee"][2]:
        problems.append("pelvis below knee")
    if j["left_shoulder"][2] <= j["pelvis"][2]:
        problems.append("shoulder below pelvis")
    return problems


def symmetrize(joints):
    """A-pose characters are near-symmetric; mirroring removes most left/right jitter."""
    j = {k: np.asarray(v, dtype=float) for k, v in joints.items()}
    for ln in [k for k in j if k.startswith("left_")]:
        rn = "right_" + ln[5:]
        if rn not in j:
            continue
        l, r = j[ln], j[rn]
        mx = (abs(l[0]) + abs(r[0])) / 2
        my, mz = (l[1] + r[1]) / 2, (l[2] + r[2]) / 2
        j[ln] = np.array([mx, my, mz])
        j[rn] = np.array([-mx, my, mz])
    for c in ["pelvis", "spine1", "spine2", "spine3", "neck", "head", "head_top"]:
        if c in j:
            j[c][0] = 0.0
    return {k: [float(x) for x in v] for k, v in j.items()}


def run(views_dir, mesh=None, out=None, force_mesh=False):
    views_dir = Path(views_dir)
    meta = json.load(open(views_dir / "meta.json"))
    mesh = mesh or meta["mesh"]
    joints, how = (None, "forced") if force_mesh else from_pose(views_dir)
    notes = []
    if joints is not None:
        joints = symmetrize(joints)
        bad = validate(joints)
        if bad:
            notes.append(f"pose result rejected: {'; '.join(bad)}")
            joints = None
    if joints is None:
        reason = how if joints is None and not notes else notes[-1]
        joints, how = from_mesh(mesh)
        joints = symmetrize(joints)
        how = f"{how} (pose route unused: {reason})"
        bad = validate(joints)
        if bad:
            notes.append(f"mesh-analysis warnings: {'; '.join(bad)}")
    # (Joints are pulled inside the limbs by refine_joints.py, against the solid - an earlier snap
    # here compared them with the glTF's raw Y-up vertices and pulled them toward the wrong ones.)
    payload = {"mesh": mesh, "method": how, "notes": notes, "parents": PARENTS, "joints": joints}
    out = out or (views_dir.parent / "joints.json")
    json.dump(payload, open(out, "w"), indent=2)
    print(f"[skeleton] {len(joints)} joints via {how} -> {out}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force-mesh", action="store_true")
    a = ap.parse_args()
    run(a.views, a.mesh, a.out, a.force_mesh)
