"""Grade Mixamo clips as game locomotion cycles: does the body travel the way it faces?

scan_mixamo.py sorted the dataset into walks, runs and jumps by speed, flight time and bob. That
found clips that MOVE like a run, and the one it picked travels at exactly 45 degrees to its own
rest forward, with the hips turned 28 degrees and the feet 60 - a diagonal run. No rotation of
such a clip can plant its feet under a character controller, which moves the model along the way
it faces: whatever the clip's travel direction disagrees with its body by, the feet slide
sideways by that angle's sine times the speed.

So a locomotion cycle is graded here on what a controller needs from it:

  travel     hips displacement over the clip: direction relative to the rest forward, speed in
             m/s, and straightness (largest sideways deviation of the hips path, per metre)
  agreement  the direction the planted feet point, the hip line, and the shoulder line, each
             against the travel direction - a straight run has all three within a few degrees
  planted    world speed of a foot while it is on the ground, as a fraction of hips speed -
             mocap is near zero; a foot that slides in the source slides on every character
  flight     fraction of frames with neither foot down (a run has flight, a walk has none)
  seam       how far the last frame's pose is from the first - a clip that does not close is
             not a cycle, whatever it looks like

Run: blender -b -noaudio --python scan_locomotion.py -- --dir <animation/> --out loco.jsonl
         [--frames animation_frames.json --max-frames 300]
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
ap.add_argument("--frames", default=None, help="animation_frames.json, to skip long clips unopened")
ap.add_argument("--max-frames", type=int, default=300)
ap.add_argument("--only", nargs="*", default=None, help="file hashes to scan (default: all)")
a = ap.parse_args(argv)

KEYS = ("Hips", "LeftUpLeg", "RightUpLeg", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase",
        "LeftArm", "RightArm")


def bone(rig, key):
    for b in rig.pose.bones:
        if b.name == key or b.name.endswith(":" + key):
            return b
    return None


def ang(v):
    return math.degrees(math.atan2(v[1], v[0]))


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def circ_mean(deg):
    r = np.radians(np.asarray(deg))
    return math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean()))


def forward_of(side):
    """Facing in the ground plane from a left-minus-right vector: side x up."""
    return np.array([side[1], -side[0]])


def grade(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=True,
                                 automatic_bone_orientation=False)
    except Exception as e:
        return {"error": f"import: {e}"}
    sc = bpy.context.scene
    rig = next((o for o in sc.objects if o.type == "ARMATURE"), None)
    if rig is None or rig.animation_data is None or rig.animation_data.action is None:
        return {"error": "no animated armature"}
    pbs = {k: bone(rig, k) for k in KEYS}
    if any(v is None for v in pbs.values()):
        return {"error": "missing bones"}
    act = rig.animation_data.action
    f0, f1 = (int(x) for x in act.frame_range)
    fps = sc.render.fps / sc.render.fps_base
    n = f1 - f0 + 1
    if n < 8:
        return {"error": "too short"}
    W = np.array(rig.matrix_world)

    def wpos(p):
        return (W @ np.append(np.array(p), 1.0))[:3]

    rest = {k: wpos(rig.data.bones[pbs[k].name].head_local) for k in KEYS}
    H = float(rest["Hips"][2] - min(rest["LeftToeBase"][2], rest["RightToeBase"][2]))
    rest_fwd = ang(forward_of(rest["LeftUpLeg"] - rest["RightUpLeg"]))

    P = np.empty((n, len(KEYS), 3))
    Q0 = Q1 = None
    for i, f in enumerate(range(f0, f1 + 1)):
        sc.frame_set(f)
        for j, k in enumerate(KEYS):
            P[i, j] = wpos(pbs[k].head)
        if i == 0:
            Q0 = [pb.matrix_basis.to_quaternion() for pb in rig.pose.bones]
        if i == n - 1:
            Q1 = [pb.matrix_basis.to_quaternion() for pb in rig.pose.bones]
    I = {k: j for j, k in enumerate(KEYS)}
    dur = (n - 1) / fps
    hips = P[:, I["Hips"]]
    d = hips[-1, :2] - hips[0, :2]
    dist = float(np.linalg.norm(d))
    out = {"frames": n, "fps": fps, "seconds": round(dur, 3), "hip_height_m": round(H, 3),
           "rest_forward_deg": round(rest_fwd, 1), "travel_m": round(dist, 3),
           "speed_mps": round(dist / max(dur, 1e-6), 3)}

    # seam: joint rotations and in-place foot placement, last frame against first
    rot = [math.degrees(q0.rotation_difference(q1).angle) for q0, q1 in zip(Q0, Q1)]
    rot = [min(r, 360 - r) for r in rot]
    rel0 = P[0, [I["LeftToeBase"], I["RightToeBase"]]] - hips[0]
    rel1 = P[-1, [I["LeftToeBase"], I["RightToeBase"]]] - hips[-1]
    out["seam_rot_max_deg"] = round(max(rot), 2)
    out["seam_feet_per_H"] = round(float(np.abs(rel1 - rel0).max() / H), 4)

    # contact: a foot is down when its toe or its ankle is within 3% of hip height of its lowest
    down = []
    for s in ("Left", "Right"):
        tz, az = P[:, I[f"{s}ToeBase"], 2], P[:, I[f"{s}Foot"], 2]
        down.append((tz < tz.min() + 0.03 * H) | (az < az.min() + 0.03 * H))
    down = np.array(down)
    out["flight"] = round(float((~down[0] & ~down[1]).mean()), 3)
    out["duty"] = [round(float(x), 3) for x in down.mean(1)]
    if dist < 0.1 * H:
        out["travelling"] = False
        return out
    out["travelling"] = True
    travel = ang(d)
    out["travel_vs_rest_deg"] = round(wrap(travel - rest_fwd), 1)

    # straightness: largest sideways departure of the hips from the start-end line, per metre
    u = d / dist
    rel = hips[:, :2] - hips[0, :2]
    side = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    out["sideways_per_m"] = round(float(side.max() / dist), 4)
    # hip speed steadiness along travel
    v = np.gradient(rel @ u, 1 / fps)
    out["speed_cv"] = round(float(v.std() / max(abs(v.mean()), 1e-6)), 3)

    # planted feet: world speed of the lower of toe/ankle while down, against hips speed
    slip, feet_dir = [], []
    for k, s in enumerate(("Left", "Right")):
        toe, ank = P[:, I[f"{s}ToeBase"]], P[:, I[f"{s}Foot"]]
        vt = np.linalg.norm(np.gradient(toe[:, :2], 1 / fps, axis=0), axis=1)
        va = np.linalg.norm(np.gradient(ank[:, :2], 1 / fps, axis=0), axis=1)
        m = down[k]
        if m.sum() >= 2:
            slip.append(float(np.median(np.minimum(vt, va)[m])))
            dirs = toe[m, :2] - ank[m, :2]
            feet_dir.append(circ_mean(np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))))
    spd = dist / max(dur, 1e-6)
    out["slip_ratio"] = round(max(slip) / spd, 3) if slip else None
    # the rest pose's own toe-out, so a symmetric splay does not count as a turn
    rest_feet = circ_mean([ang(rest[f"{s}ToeBase"][:2] - rest[f"{s}Foot"][:2]) for s in ("Left", "Right")])
    if len(feet_dir) == 2:
        out["feet_vs_travel_deg"] = round(wrap(circ_mean(feet_dir) - rest_feet + rest_fwd - travel), 1)
    hipf = [ang(forward_of(P[i, I["LeftUpLeg"]] - P[i, I["RightUpLeg"]])) for i in range(n)]
    shf = [ang(forward_of(P[i, I["LeftArm"]] - P[i, I["RightArm"]])) for i in range(n)]
    out["hips_vs_travel_deg"] = round(wrap(circ_mean(hipf) - travel), 1)
    out["shoulders_vs_travel_deg"] = round(wrap(circ_mean(shf) - travel), 1)
    return out


files = sorted(glob.glob(os.path.join(a.dir, "*.fbx")))
if a.only:
    want = {h.replace(".fbx", "") for h in a.only}
    files = [f for f in files if os.path.basename(f)[:-4] in want]
elif a.frames:
    nf = json.load(open(a.frames))
    files = [f for f in files if nf.get(os.path.basename(f)[:-4], 0) <= a.max_frames]
done = set()
if os.path.exists(a.out):
    for line in open(a.out):
        try:
            done.add(json.loads(line)["file"])
        except Exception:
            pass
todo = [f for f in files if os.path.basename(f) not in done]
print(f"[loco] {len(files)} clips, {len(done)} already graded, {len(todo)} to go", flush=True)
with open(a.out, "a") as fh:
    for i, f in enumerate(todo):
        try:
            r = grade(f)
        except Exception as e:
            r = {"error": str(e)[:200]}
        r["file"] = os.path.basename(f)
        fh.write(json.dumps(r) + "\n")
        fh.flush()
        if i % 100 == 0:
            print(f"[loco] {i}/{len(todo)}", flush=True)
print("[loco] done", flush=True)
