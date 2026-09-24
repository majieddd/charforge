"""Measure every Mixamo clip for the moves a game character needs beyond walking forward.

The Mixamo mirror names its 2,453 clips by hash, so a turn, a crouch or a fall has to be found
by what it does. scan_locomotion.py already grades clips that travel; this adds what it cannot
see - how far the body turns, how low the hips go, how long both feet are off the ground, and
where the clip starts and ends - so each of these is a query:

  turn in place    travels < 0.3 m, body turns ~90 degrees between first and last frame
  crouch           hips spend the clip at 55-80% of standing height
  fall loop        both feet off the ground the whole time
  land             starts with both feet off the ground, ends standing
  sprint           travels straight at more than ~5.5 m/s

Heights are ratios of the capture's own rest hip height, directions relative to its own rest
forward, so every clip is measured the same way whatever its scale.

Run: blender -b -noaudio --python scan_moves.py -- --dir <animation/> --out moves.jsonl
         [--frames animation_frames.json --max-frames 400]
"""
import argparse
import glob
import json
import math
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", default=None)
ap.add_argument("--max-frames", type=int, default=400)
a = ap.parse_args(argv)

KEYS = ("Hips", "LeftUpLeg", "RightUpLeg", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase",
        "Head", "LeftHand", "RightHand")


def bone(rig, key):
    for b in rig.pose.bones:
        if b.name == key or b.name.endswith(":" + key):
            return b
    return None


def measure(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=True, automatic_bone_orientation=False)
    except Exception as e:
        return {"error": f"import: {e}"}
    rig = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if rig is None or rig.animation_data is None or rig.animation_data.action is None:
        return {"error": "no animation"}
    act = rig.animation_data.action
    f0, f1 = (int(x) for x in act.frame_range)
    bones = {k: bone(rig, k) for k in KEYS}
    if any(v is None for k, v in bones.items() if k in ("Hips", "LeftUpLeg", "RightUpLeg", "LeftFoot", "RightFoot")):
        return {"error": "missing bones"}
    W = rig.matrix_world
    rest_hip = (W @ rig.data.bones[bones["Hips"].name].head_local).z
    ts = list(range(f0, f1 + 1, max(1, (f1 - f0) // 120 or 1)))
    rows = []
    for f in ts:
        bpy.context.scene.frame_set(f)
        p = {k: np.array((W @ b.head)[:]) for k, b in bones.items() if b is not None}
        rows.append(p)
    H = np.array([r["Hips"] for r in rows])
    side = np.array([r["LeftUpLeg"] - r["RightUpLeg"] for r in rows])
    yaw = np.unwrap(np.arctan2(-side[:, 0], side[:, 1]))        # facing = side x up
    feet = np.array([[r[k][2] for k in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase") if k in r]
                     for r in rows])
    floor = min(0.0, float(feet.min()))
    low = feet.min(1) - floor                                    # lowest foot point per sample
    unit = max(rest_hip, 1e-6)
    air = low > 0.12 * unit
    trav = H[-1, :2] - H[0, :2]
    dur = max((f1 - f0) / max(bpy.context.scene.render.fps, 1), 1e-6)
    heads = np.array([r["Head"][2] for r in rows]) if "Head" in rows[0] else None
    hands = np.array([max(r["LeftHand"][2], r["RightHand"][2]) for r in rows]) if "LeftHand" in rows[0] else None
    out = {
        "seconds": round(dur, 3), "frames": f1 - f0 + 1,
        "travel_m_per_hip": round(float(np.linalg.norm(trav)) / unit, 3),
        "speed_hips_per_s": round(float(np.linalg.norm(trav)) / unit / dur, 3),
        "turn_deg": round(math.degrees(yaw[-1] - yaw[0]), 1),
        "turn_max_deg": round(math.degrees(np.max(np.abs(yaw - yaw[0]))), 1),
        "hip_mean": round(float(H[:, 2].mean() / unit), 3),
        "hip_min": round(float(H[:, 2].min() / unit), 3),
        "hip_max": round(float(H[:, 2].max() / unit), 3),
        "hip_start": round(float(H[0, 2] / unit), 3), "hip_end": round(float(H[-1, 2] / unit), 3),
        "air_frac": round(float(air.mean()), 3),
        "air_start": bool(air[: max(2, len(air) // 10)].all()),
        "ground_end": bool((~air[-max(2, len(air) // 10):]).all()),
        "hands_over_head": round(float(((hands - heads) > 0).mean()), 3) if hands is not None and heads is not None else None,
        "rest_hip_m": round(float(rest_hip), 3),
    }
    return out


files = sorted(glob.glob(os.path.join(a.dir, "*.fbx")))
frames = json.load(open(a.frames)) if a.frames else {}
done = set()
if os.path.exists(a.out):
    done = {json.loads(l).get("file") for l in open(a.out)}
with open(a.out, "a") as fh:
    for i, f in enumerate(files):
        name = os.path.basename(f)
        if name in done:
            continue
        nfr = frames.get(name[:-4])
        if nfr and nfr > a.max_frames:
            continue
        r = measure(f)
        r["file"] = name
        fh.write(json.dumps(r) + "\n")
        fh.flush()
        if i % 100 == 0:
            print(f"[moves] {i}/{len(files)}", flush=True)
print("[moves] done", flush=True)
