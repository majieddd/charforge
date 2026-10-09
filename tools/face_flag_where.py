"""Where the clip audit's flagged faces rest on a face: inside or outside the face rig's density boxes.

    /Applications/Blender.app/Contents/MacOS/Blender -b -noaudio --python tools/face_flag_where.py -- \
        --blend work/kaito/final.blend --face work/kaito/face.json --out qa/face_flags.json [--npz qa/face_flags.npz]

For every clip, every --step frames, the crushed / stretched / sheared tests of blender/aberration_audit.py
(its definitions, reused through blender/mesh_contact.py) are applied to each rest face and counted. Faces
are then grouped by where they rest, in this order (the first that applies):
  eye           inside an eye box blender/face_rig.py subdivides (the eye's height read from the texture, uncapped)
  mouth_lips    inside the mouth box at the lips' width (0.6 of the eye spacing, as the cut uses it)
  mouth_extra_above   inside the mouth box at the redness' width (face_rig.py's HEAD box) but not the lips', above the chin
  mouth_extra_below   the same, below the chin (the neck the jaw's weight fades into)
  neck          below the chin and within 1.5 eye spacings of it (outside the boxes)
  other         everything else
The boxes are those of blender/face_rig.py (centre, half-widths, the depth band of half an eye spacing). The
output gives, per group, the faces, the flagged face-frames of each kind and the rate per 1000 face-frames.
Runs on the CPU only.
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--face", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--npz", default="", help="also save the per-face counts (for a heat map or a later look)")
ap.add_argument("--step", type=int, default=4, help="frames between samples in every clip")
ap.add_argument("--clips", default="", help="comma list; default every action, as the audit does")
a = ap.parse_args(argv)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blender"))
from mesh_contact import Contact  # noqa: E402  (the audit's rest pose, areas, normals and dominant bones)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
meshes = [o for o in sc.objects if o.type == "MESH" and o.visible_get() and any(m.type == "ARMATURE" for m in o.modifiers)]
for o in meshes:
    for m in o.modifiers:
        if m.type == "ARMATURE":
            m.use_deform_preserve_volume = True            # the audit's default skin is dual quaternion
C = Contact(rig, meshes, 0.03)
nF = len(C.F)
live = C.live
F_json = json.load(open(a.face))
ied = float(F_json["ied_m"])
mouth = np.array(F_json["mouth"]["centre"])
chin = np.array(F_json["chin"])
cent = C.cent0


def eye_box(side):
    e = F_json["eyes"][side]
    h = max(float(e["height_m"]), 0.3 * float(e["width_m"]))           # face_rig.py's, uncapped (as HEAD)
    return np.array(e["centre"]), 1.6 * float(e["width_m"]) / 2, 2.2 * h + 0.35 * ied


def inside(C_, rx, rz):
    return (np.abs(cent[:, 0] - C_[0]) < rx) & (np.abs(cent[:, 2] - C_[2]) < rz) & (np.abs(cent[:, 1] - C_[1]) < 0.5 * ied)


mw_read = float(F_json["mouth"]["width_m"])
mw_lips = 0.6 * ied if mw_read > ied else mw_read
eye_in = np.zeros(nF, bool)
for side in ("left", "right"):
    if F_json.get("eyes_ok", True):
        eye_in |= inside(*eye_box(side))
lips_in = inside(mouth, 0.85 * mw_lips, 0.55 * mw_lips) if F_json.get("mouth_ok", True) else np.zeros(nF, bool)
read_in = inside(mouth, 0.85 * mw_read, 0.55 * mw_read) if F_json.get("mouth_ok", True) else np.zeros(nF, bool)
below = cent[:, 2] < chin[2]
near_chin = below & (cent[:, 2] > chin[2] - 1.5 * ied) & (np.abs(cent[:, 0] - chin[0]) < 1.5 * ied) \
    & (cent[:, 1] < chin[1] + ied)
group = np.full(nF, "other", dtype=object)
group[near_chin] = "neck"
group[read_in & ~lips_in & ~below] = "mouth_extra_above"
group[read_in & ~lips_in & below] = "mouth_extra_below"
group[lips_in] = "mouth_lips"
group[eye_in] = "eye"
GROUPS = ["eye", "mouth_lips", "mouth_extra_above", "mouth_extra_below", "neck", "other"]


def set_action(name):
    act = bpy.data.actions.get(name)
    if act is None:
        return None
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    return act


def bone_rotations():
    R = np.zeros((len(C.bones), 3, 3))
    for b, i in C.bidx.items():
        pb = rig.pose.bones[b]
        M = (rig.matrix_world @ pb.matrix).to_3x3().normalized()
        M0 = (rig.matrix_world @ pb.bone.matrix_local).to_3x3().normalized()
        R[i] = np.array(M @ M0.inverted())
    return R


clips = [c.strip() for c in a.clips.split(",") if c.strip()] or sorted(x.name for x in bpy.data.actions)
cnt = {k: np.zeros(nF, np.int32) for k in ("crushed", "stretched", "sheared")}
area_cnt = {k: 0.0 for k in cnt}                       # flagged rest area, summed over frames (m2)
n_frames = 0
per_clip = {}
for clip in clips:
    act = set_action(clip)
    if act is None:
        continue
    f0, f1 = (int(x) for x in act.frame_range)
    frames = list(range(f0, f1 + 1, a.step))
    got = {k: 0 for k in cnt}
    for fr in frames:
        sc.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        V, A, N = C.posed(dg, f"{clip} {fr}")
        ratio = A / np.maximum(C.A0, 1e-12)
        crushed = live & (ratio < 0.5)
        stretched = live & (ratio > 2.0)
        R = bone_rotations()
        expect = np.einsum("fij,fj->fi", R[np.maximum(C.fdom, 0)], C.N0)
        swing = np.degrees(np.arccos(np.clip((expect * N).sum(1), -1, 1)))
        sheared = live & (C.fdom >= 0) & (swing > 60)
        for k, m in (("crushed", crushed), ("stretched", stretched), ("sheared", sheared)):
            cnt[k] += m.astype(np.int32)
            area_cnt[k] += float(C.A0[m].sum())
            got[k] += int(m.sum())
        n_frames += 1
    per_clip[clip] = {"frames": len(frames), **got}
    print(f"[flags] {clip}: {len(frames)} frames, crushed {got['crushed']}, stretched {got['stretched']}, "
          f"sheared {got['sheared']} face-frames", flush=True)

live_n = int(live.sum())
out = {"blend": os.path.basename(a.blend), "faces": nF, "live_faces": live_n, "frames": n_frames,
       "step": a.step, "clips": clips, "groups": {}, "per_clip": per_clip,
       "boxes": {"mouth_lips_halfwidth_cm": round(0.85 * mw_lips * 100, 3),
                 "mouth_read_halfwidth_cm": round(0.85 * mw_read * 100, 3),
                 "eyes_ok": bool(F_json.get("eyes_ok", True)), "mouth_ok": bool(F_json.get("mouth_ok", True))}}
for g in GROUPS:
    m = live & (group == g)
    n_g = int(m.sum())
    row = {"faces": n_g}
    for k in cnt:
        s = int(cnt[k][m].sum())
        row[k] = s
        row[k + "_per_1000"] = round(1000.0 * s / max(n_g * max(n_frames, 1), 1), 3)
    out["groups"][g] = row
tot = {k: int(cnt[k][live].sum()) for k in cnt}
out["totals"] = {k: tot[k] for k in tot}
out["totals_per_1000"] = {k: round(1000.0 * tot[k] / max(live_n * max(n_frames, 1), 1), 3) for k in tot}
# the same shares weighted by rest area: the flagged share of the surface, which does not depend on how densely a
# region is meshed (a face-count share does: a finer region adds faces at its own rate)
live_area = float(C.A0[live].sum()) * max(n_frames, 1)
out["area_share_pct"] = {k: round(100.0 * area_cnt[k] / max(live_area, 1e-12), 4) for k in area_cnt}
print(f"[flags] rest-area share (%): {out['area_share_pct']}", flush=True)
with open(a.out, "w") as f:
    json.dump(out, f, indent=1)
if a.npz:
    np.savez(a.npz, cent=cent, live=live, group=group.astype(str), crushed=cnt["crushed"],
             stretched=cnt["stretched"], sheared=cnt["sheared"])
for g in GROUPS:
    r = out["groups"][g]
    print(f"[flags] {g:12s} faces {r['faces']:6d}  per 1000 face-frames: crushed {r['crushed_per_1000']:8.3f} "
          f"stretched {r['stretched_per_1000']:8.3f} sheared {r['sheared_per_1000']:8.3f}", flush=True)
print(f"[flags] totals over {n_frames} frames: {out['totals']} -> {a.out}", flush=True)
