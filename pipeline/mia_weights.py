#!/usr/bin/env python3
"""Make-It-Animatable v2's skin weights (pipeline/mia_infer.py, E157) in CharForge's weights format.

    python pipeline/mia_weights.py --mia work/cadet/mia_raw.npz --mesh work/cadet/retopo.glb --out work/cadet/weights_mia.npz

MIA predicts 52 Mixamo joints (no jaw, no head-top, no finger tips); CharForge's weights stage binds 20 body bones and the hands
stage adds the fingers - so each finger joint folds into the wrist, the toes into the foot (ankle), and the strongest four of
the 20 stay, normalised. Vertices are matched by position (the mesh as trimesh loads it)."""
import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree

MAP = {"Hips": "pelvis", "Spine": "spine1", "Spine1": "spine2", "Spine2": "spine3", "Neck": "neck", "Head": "head"}
for s, side in (("Left", "left"), ("Right", "right")):
    MAP.update({f"{s}Shoulder": f"{side}_collar", f"{s}Arm": f"{side}_shoulder", f"{s}ForeArm": f"{side}_elbow",
                f"{s}Hand": f"{side}_wrist", f"{s}UpLeg": f"{side}_hip", f"{s}Leg": f"{side}_knee",
                f"{s}Foot": f"{side}_ankle", f"{s}ToeBase": f"{side}_ankle"})
    for f in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for k in (1, 2, 3):
            MAP[f"{s}Hand{f}{k}"] = f"{side}_wrist"
BONES = ["pelvis", "spine1", "spine2", "spine3", "neck", "head", "left_collar", "right_collar", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
         "left_ankle", "right_ankle"]

ap = argparse.ArgumentParser()
ap.add_argument("--mia", required=True)
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
Z = np.load(a.mia)
bw, names, verts = Z["bw"], [str(n) for n in Z["names"]], Z["verts"]
mesh = trimesh.load(a.mesh, force="mesh", process=False)
V = np.asarray(mesh.vertices, np.float32)
d, j = cKDTree(verts).query(V)
print(f"[mia->weights] {len(V)} mesh vertices, {len(verts)} MIA vertices; nearest-vertex distance max {d.max() * 100:.2f} cm "
      f"(median {np.median(d) * 100:.3f})")
W = np.zeros((len(V), len(BONES)), np.float32)
for k, n in enumerate(names):
    if n in MAP:
        W[:, BONES.index(MAP[n])] += bw[j, k]
    else:
        print("   unmapped joint", n)
top = np.argsort(-W, 1)[:, :4]
w = np.take_along_axis(W, top, 1)
w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
np.savez_compressed(a.out, bones=np.array(BONES), index=top.astype(np.int16), weight=w.astype(np.float32))
dom = np.array(BONES)[W.argmax(1)]
u, c = np.unique(dom, return_counts=True)
print("[mia->weights] dominant bones: " + ", ".join(f"{x} {y}" for x, y in sorted(zip(u, c), key=lambda t: -t[1])[:8]))
