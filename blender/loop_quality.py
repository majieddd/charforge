"""Measure how well a locomotion capture loops, and where its best loop points are.

A looping clip that is not a whole number of cycles pops at the seam: the retargeted run shipped
with its knee jumping 20 cm at every loop, twice a normal frame step, every half second. Nothing
in the earlier clip selection measured this - it chose clips by speed, cadence and posture, all
of which a clip can satisfy while still being 80% of a stride.

For each clip this finds the period P and start frame s that minimise the pose difference
between frame s and frame s+P, measured on hip-relative joint positions (so ground travel does
not count as a mismatch), and reports the residual as a multiple of an ordinary frame step. A
ratio under ~0.5 is invisible; over ~1.5 is a visible hitch.

Run: blender -b -noaudio --python loop_quality.py -- --files a.fbx b.fbx ... --out loops.json
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-cycle-s", type=float, default=0.35)
a = ap.parse_args(argv)

KEY = ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot",
       "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand", "Head")


def sample(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=True, automatic_bone_orientation=False)
    rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    act = rig.animation_data.action if rig.animation_data else None
    if act is None:
        return None
    f0, f1 = (int(round(x)) for x in act.frame_range)
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    bones = [b for b in (next((pb for pb in rig.pose.bones if pb.name.endswith(k)), None) for k in KEY) if b]
    M = np.array(rig.matrix_world)
    frames = []
    for f in range(f0, f1 + 1):
        bpy.context.scene.frame_set(f)
        P = np.array([(M @ np.append(np.array(pb.head), 1.0))[:3] for pb in bones])
        frames.append(P - P[0])                       # hip-relative: ground travel is not error
    X = np.array(frames)
    return X.reshape(len(X), -1), fps


res = {}
for path in a.files:
    name = os.path.basename(path)
    try:
        out = sample(path)
    except Exception as e:                            # noqa: BLE001
        res[name] = {"error": str(e)[:120]}
        print(f"[loop] {name[:14]}  failed: {e}", flush=True)
        continue
    if out is None:
        continue
    X, fps = out
    N = len(X)
    step = float(np.median(np.linalg.norm(np.diff(X, axis=0), axis=1)))
    minP = max(4, int(a.min_cycle_s * fps))
    best = None
    for P in range(minP, N):
        for s in range(0, N - P):
            # a loop must close at the seam AND be locally smooth there, so compare two frames
            d = np.linalg.norm(X[s] - X[s + P])
            if best is None or d < best[0]:
                best = (d, s, P)
    d, s, P = best
    whole = float(np.linalg.norm(X[0] - X[-1]))
    res[name] = {"frames": N, "fps": fps, "step": round(step, 4),
                 "as_shipped_seam_ratio": round(whole / max(step, 1e-6), 2),
                 "best_start": s, "best_period": P, "best_period_s": round(P / fps, 3),
                 "best_seam_ratio": round(d / max(step, 1e-6), 2)}
    print(f"[loop] {name[:14]}  {N:4d} frames @{fps:.0f}  whole-clip seam {whole/max(step,1e-6):5.2f}x a step   "
          f"best loop: frames {s}..{s+P} ({P/fps:.2f}s) seam {d/max(step,1e-6):5.2f}x", flush=True)

json.dump(res, open(a.out, "w"), indent=2)
