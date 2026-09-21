"""Identify Mixamo clips by what they do, because the dataset does not say.

jasongzy/Mixamo ships 2,453 animations named by content hash - 42635a1f2f9d...fbx - with no
index of which is a walk and which is a backflip. The filenames are not hashes of the Mixamo
clip names either (checked). So the only way to find a walk cycle is to look at the motion.

Each clip is imported, and the armature's root and limb trajectories are reduced to a handful of
features that separate the categories we need:

  speed        horizontal root travel per frame, in hip-heights per second - scale-free, so it
               does not matter that Mixamo characters differ in size
  air          the fraction of frames where *both* feet are above their own median height,
               which is what distinguishes a run (flight phase) from a walk (always one foot
               down)
  bob          vertical root range, the signal for a jump
  cadence      dominant frequency of the left-minus-right foot height difference, i.e. steps
  hand_lift    peak wrist height relative to the head, for waves and other arm gestures
  motion       total joint angular travel, to tell an idle from a statue

Writes one JSON line per clip so the scan can be stopped and resumed, and so the classification
can be re-run without re-importing 1,300 FBX files.

Run: blender -b -noaudio --python scan_mixamo.py -- --dir <animation/> --out scan.jsonl --limit 400
"""
import argparse
import json
import math
import os
import sys
import glob

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=400)
ap.add_argument("--max-frames", type=int, default=260)
a = ap.parse_args(argv)

HIP = ("mixamorig:Hips", "mixamorig1:Hips", "Hips")
LFOOT = ("mixamorig:LeftFoot", "mixamorig1:LeftFoot", "LeftFoot")
RFOOT = ("mixamorig:RightFoot", "mixamorig1:RightFoot", "RightFoot")
LHAND = ("mixamorig:LeftHand", "mixamorig1:LeftHand", "LeftHand")
RHAND = ("mixamorig:RightHand", "mixamorig1:RightHand", "RightHand")
HEAD = ("mixamorig:Head", "mixamorig1:Head", "Head")


def pick(rig, names):
    for n in names:
        if n in rig.pose.bones:
            return rig.pose.bones[n]
    return None


def features(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=True,
                                 automatic_bone_orientation=False)
    except Exception as e:
        return {"error": f"import: {type(e).__name__}: {e}"[:160]}
    rig = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if rig is None:
        return {"error": "no armature"}
    act = rig.animation_data.action if rig.animation_data else None
    if act is None:
        return {"error": "no action"}
    f0, f1 = (int(x) for x in act.frame_range)
    n = min(f1 - f0 + 1, a.max_frames)
    if n < 6:
        return {"error": f"too short ({n})"}

    hips, lf, rf = pick(rig, HIP), pick(rig, LFOOT), pick(rig, RFOOT)
    lh, rh, hd = pick(rig, LHAND), pick(rig, RHAND), pick(rig, HEAD)
    if not (hips and lf and rf):
        return {"error": "missing core bones"}

    sc = bpy.context.scene
    P = {k: [] for k in ("hip", "lf", "rf", "lh", "rh", "hd")}
    rots = []
    for i in range(n):
        sc.frame_set(f0 + i)
        M = rig.matrix_world
        P["hip"].append((M @ hips.head).copy())
        P["lf"].append((M @ lf.head).copy())
        P["rf"].append((M @ rf.head).copy())
        if lh: P["lh"].append((M @ lh.head).copy())
        if rh: P["rh"].append((M @ rh.head).copy())
        if hd: P["hd"].append((M @ hd.head).copy())
        rots.append([pb.rotation_quaternion.copy() for pb in rig.pose.bones[:40]])

    hip_h = max(1e-6, sum(p.z for p in P["hip"]) / len(P["hip"]))
    fps = sc.render.fps or 30

    # horizontal speed, in hip-heights per second
    d = 0.0
    for i in range(1, len(P["hip"])):
        p, q = P["hip"][i - 1], P["hip"][i]
        d += math.hypot(q.x - p.x, q.y - p.y)
    speed = d / max(len(P["hip"]) - 1, 1) * fps / hip_h

    zs = [p.z for p in P["hip"]]
    bob = (max(zs) - min(zs)) / hip_h

    lz = [p.z for p in P["lf"]]
    rz = [p.z for p in P["rf"]]
    lmed = sorted(lz)[len(lz) // 2]
    rmed = sorted(rz)[len(rz) // 2]
    air = sum(1 for i in range(len(lz))
              if lz[i] > lmed + 0.02 * hip_h and rz[i] > rmed + 0.02 * hip_h) / len(lz)

    # cadence: zero crossings of (left - right) foot height
    diff = [lz[i] - rz[i] for i in range(len(lz))]
    cross = sum(1 for i in range(1, len(diff)) if (diff[i - 1] <= 0) != (diff[i] <= 0))
    cadence = cross / 2 / (len(diff) / fps)

    hand_lift = 0.0
    if P["lh"] and P["hd"]:
        hand_lift = max((max(P["lh"][i].z, P["rh"][i].z) - P["hd"][i].z) / hip_h
                        for i in range(len(P["hd"])))

    motion = 0.0
    for i in range(1, len(rots)):
        for qa, qb in zip(rots[i - 1], rots[i]):
            motion += (qa - qb).magnitude
    motion /= max(len(rots) - 1, 1)

    return {"frames": n, "fps": fps, "speed": round(speed, 3), "air": round(air, 3),
            "bob": round(bob, 3), "cadence": round(cadence, 2),
            "hand_lift": round(hand_lift, 3), "motion": round(motion, 4)}


files = sorted(glob.glob(os.path.join(a.dir, "*.fbx")))[:a.limit]
done = set()
if os.path.exists(a.out):
    for line in open(a.out):
        try:
            done.add(json.loads(line)["file"])
        except Exception:
            pass
print(f"[scan] {len(files)} files, {len(done)} already scanned", flush=True)

with open(a.out, "a") as fh:
    for i, p in enumerate(files):
        name = os.path.basename(p)
        if name in done:
            continue
        try:
            r = features(p)
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"[:160]}
        r["file"] = name
        fh.write(json.dumps(r) + "\n")
        fh.flush()
        if i % 25 == 0:
            print(f"[scan] {i}/{len(files)} {name} {r.get('error') or ''}", flush=True)
print("[scan] done", flush=True)
