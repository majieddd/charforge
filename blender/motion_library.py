"""Every clip's body pose over time, compact - the library pipeline/video_motion.py matches video against.

    blender -b -noaudio --python motion_library.py -- --dir <Mixamo animation/> --out motion_library.npz \
        [--fps 15] [--limit N]

For each clip: the world positions of 13 points a 2D pose model also reports - the head and, per
side, shoulder, elbow, wrist, hip, knee and ankle - sampled at --fps. The Mixamo mirror names its
clips by hash, so a motion seen in a video can only be found by what it looks like: a pose
model's keypoints are compared with these points projected into the camera (see video_motion.py).

Stored per clip: its rest hip height (the unit its motion is measured in), the direction its body
faces in the first frame (the azimuth a camera angle is measured from), and the frame rate.
"""
import argparse
import glob
import math
import os
import sys
import time

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-seconds", type=float, default=12.0, help="longer clips are cut here")
a = ap.parse_args(argv)

# (pose-model point, Mixamo bone whose head sits there)
POINTS = [("head", "Head"),
          ("l_shoulder", "LeftArm"), ("l_elbow", "LeftForeArm"), ("l_wrist", "LeftHand"),
          ("r_shoulder", "RightArm"), ("r_elbow", "RightForeArm"), ("r_wrist", "RightHand"),
          ("l_hip", "LeftUpLeg"), ("l_knee", "LeftLeg"), ("l_ankle", "LeftFoot"),
          ("r_hip", "RightUpLeg"), ("r_knee", "RightLeg"), ("r_ankle", "RightFoot")]


def bone(rig, key):
    for b in rig.pose.bones:
        if b.name == key or b.name.endswith(":" + key):
            return b
    return None


def sample(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=True, automatic_bone_orientation=False)
    except Exception as e:                                        # noqa: BLE001
        return None, f"import: {e}"
    rig = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if rig is None or rig.animation_data is None or rig.animation_data.action is None:
        return None, "no animation"
    bones = [bone(rig, b) for _, b in POINTS]
    if any(b is None for b in bones):
        return None, "missing bones"
    sc = bpy.context.scene
    src_fps = sc.render.fps / sc.render.fps_base
    f0, f1 = (int(x) for x in rig.animation_data.action.frame_range)
    f1 = min(f1, f0 + int(a.max_seconds * src_fps))
    step = src_fps / a.fps
    W = rig.matrix_world
    frames = np.arange(f0, f1 + 1e-6, step)
    P = np.empty((len(frames), len(POINTS), 3), np.float32)
    for k, f in enumerate(frames):
        sc.frame_set(int(math.floor(f)), subframe=float(f - math.floor(f)))
        for j, b in enumerate(bones):
            P[k, j] = (W @ b.head)[:]
    hips = rig.data.bones[bone(rig, "Hips").name] if bone(rig, "Hips") else None
    unit = float((W @ hips.head_local).z) if hips else float(np.ptp(P[0, :, 2]))
    side = P[0, 7] - P[0, 10]                                     # left hip - right hip
    facing = math.atan2(-side[0], side[1])                        # the body's forward, about +Z
    return {"P": P, "unit": unit, "facing": facing, "fps": a.fps}, None


files = sorted(glob.glob(os.path.join(a.dir, "*.fbx")))
if a.limit:
    files = files[:a.limit]
names, units, facings, offsets, chunks, errors = [], [], [], [0], [], 0
t0 = time.time()
for i, f in enumerate(files):
    r, err = sample(f)
    if r is None:
        errors += 1
        continue
    names.append(os.path.basename(f))
    units.append(r["unit"])
    facings.append(r["facing"])
    chunks.append(r["P"])
    offsets.append(offsets[-1] + len(r["P"]))
    if (i + 1) % 100 == 0:
        print(f"[library] {i + 1}/{len(files)} clips, {time.time() - t0:.0f}s", flush=True)
np.savez_compressed(a.out, names=np.array(names), unit=np.array(units, np.float32),
                    facing=np.array(facings, np.float32), offsets=np.array(offsets, np.int64),
                    joints=np.concatenate(chunks).astype(np.float16), fps=a.fps,
                    points=np.array([p for p, _ in POINTS]))
print(f"[library] {len(names)} clips, {offsets[-1]:,} poses at {a.fps:g} fps ({errors} unreadable) -> {a.out} "
      f"in {time.time() - t0:.0f}s", flush=True)
