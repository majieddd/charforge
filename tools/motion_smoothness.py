"""The roughness our post-processing adds to a clip, measured against the same clip's source.

    blender -b -noaudio --python tools/motion_smoothness.py -- --base work/<n>/motion/raw.blend \
        --state clearance=work/<n>/motion/clearance.blend --state final=work/<n>/final.blend \
        [--clips walk,run] [--json out.json] [--fps 60] [--force]

The source (--base) is a clip as retarget.py wrote it: captured motion, planted feet, no clearance and no contact
solve. Each --state is the same clip after a later step. For every clip, bone and frame the local rotation is read
from the evaluated pose (frame_set, so every F-curve, interpolation and quaternion sign is as the viewer sees it),
and the rotation between consecutive frames gives:

  omega  rotation vector of q(t+1) * conj(q(t))                  angular velocity   (deg/frame, x fps)
  alpha  first difference of omega, times fps                    angular acceleration
  jerk   first difference of alpha, times fps                    angular jerk       (deg/s^3)

A state's correction is conj(q_base) * q_state per key (local frame): its own jerk (corr_jerk) is what our step added,
with the source's motion taken out. The root (pelvis head) and each foot (midpoint of ankle head and toe) are measured
in world space: the same three differences, in m/s, m/s^2 and m/s^3.

A spike is a sample whose jerk magnitude exceeds all of: twice the source's magnitude at that sample, the source's own
99th percentile in that clip (so a source that is already rough is not blamed for its roughness), and a floor of
1000 deg/s^3 for bones or 10 m/s^3 for the root and feet (about a one-frame pop of 0.005 deg or 0.05 mm at 60 fps).

Per clip and state the report gives: jerk_ratio (RMS of the state's jerk over the source's, mean over the bones that
move), jerk_energy (sum of squared jerk over the source's), spike_pct (share of bone-frame samples flagged),
spike_frames (frames with any bone flagged), corr_jerk (RMS of the correction's jerk, deg/s^3, mean over moving
bones), touched_pct (share of bone-frames the step changed), root_spike_frames, foot_spike_frames, and the worst
frames by excess, for rendering.

Nothing here is a gate: experiments read these numbers beside the penetration audit (aberration_audit.py) and the
foot slip audit (foot_audit.py), and keep a change only if it lowers roughness without worsening those.
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="the source: retarget output, no clearance or contact")
ap.add_argument("--state", action="append", default=[], help="label=path.blend (repeatable)")
ap.add_argument("--clips", default="", help="comma list; default every action present in the source")
ap.add_argument("--fps", type=float, default=60.0)
ap.add_argument("--worst", type=int, default=5, help="worst frames listed per clip and state")
ap.add_argument("--dump", default="", help="clip:bone - also write that bone's per-frame correction (deg) to the JSON")
ap.add_argument("--json", default=None)
ap.add_argument("--force", action="store_true", help="allow replacing an existing JSON output")
a = ap.parse_args(argv)
if a.json and os.path.exists(a.json) and not a.force:
    raise SystemExit(f"[smooth] refusing to overwrite {a.json} (pass --force)")
states = []
for s in a.state:
    if "=" not in s:
        raise SystemExit(f"[smooth] --state wants label=path, got {s!r}")
    lab, path = s.split("=", 1)
    states.append((lab, path))

FOOT_BONES = [("left_ankle", "left_foot"), ("right_ankle", "right_foot")]
FLOOR_BONE = 1000.0          # deg/s^3
FLOOR_WORLD = 10.0           # m/s^3


# ---- quaternion helpers (w, x, y, z), batched -----------------------------------------------------------
def qmul(p, q):
    pw, px, py, pz = (p[..., i] for i in range(4))
    qw, qx, qy, qz = (q[..., i] for i in range(4))
    return np.stack([pw * qw - px * qx - py * qy - pz * qz,
                     pw * qx + px * qw + py * qz - pz * qy,
                     pw * qy - px * qz + py * qw + pz * qx,
                     pw * qz + px * qy - py * qx + pz * qw], -1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def rotvec(q):
    """Rotation vector (radians) of quaternions (..., 4), the short way round."""
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    ang = 2.0 * np.arctan2(s, q[..., 0])
    k = np.where(s > 1e-12, ang / np.maximum(s, 1e-12), 2.0)
    return v * k[..., None]


def jerk_of(x, fps):
    """third difference of x (T, ...) times fps^3: the jerk, one sample shorter per difference (T-3, ...)"""
    v = np.diff(x, axis=0) * fps
    acc = np.diff(v, axis=0) * fps
    return np.diff(acc, axis=0) * fps


def accel_of(x, fps):
    v = np.diff(x, axis=0) * fps
    return np.diff(v, axis=0) * fps


# ---- reading a blend's clips ----------------------------------------------------------------------------
def read(blend_path):
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    sc = bpy.context.scene
    rig = next(o for o in sc.objects if o.type == "ARMATURE")
    pbs = rig.pose.bones
    bone_names = [pb.name for pb in pbs]
    feet_ok = [all(b in bone_names for b in pair) for pair in FOOT_BONES]
    pelvis = rig.pose.bones.get("pelvis")
    out = {}
    for act in bpy.data.actions:
        rig.animation_data_create()
        rig.animation_data.action = act
        if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
            rig.animation_data.action_slot = act.slots[0]
        f0, f1 = (int(x) for x in act.frame_range)
        T = f1 - f0 + 1
        Q = np.zeros((T, len(bone_names), 4))
        root = np.zeros((T, 3))
        feet = np.zeros((T, len(FOOT_BONES), 3))
        for t, fr in enumerate(range(f0, f1 + 1)):
            sc.frame_set(fr)
            bpy.context.view_layer.update()
            for i, pb in enumerate(pbs):
                q = pb.rotation_quaternion if pb.rotation_mode == "QUATERNION" else pb.rotation_euler.to_quaternion()
                Q[t, i] = (q.w, q.x, q.y, q.z)
            if pelvis is not None:
                root[t] = np.array(rig.matrix_world @ pelvis.head)
            for k, (l, f) in enumerate(FOOT_BONES):
                if feet_ok[k]:
                    feet[t, k] = 0.5 * (np.array(rig.matrix_world @ pbs[l].head) + np.array(rig.matrix_world @ pbs[f].tail))
        rig.animation_data.action = None
        out[act.name] = {"Q": Q, "root": root, "feet": feet, "feet_ok": feet_ok, "f0": f0, "f1": f1}
    return bone_names, out


# ---- the measures ---------------------------------------------------------------------------------------
def flag(mag_s, mag_b, floor):
    """samples above the floor, twice the source at that sample, and the source's own 99th percentile"""
    p99 = float(np.percentile(mag_b, 99)) if mag_b.size else 0.0
    thr = np.maximum(np.maximum(2.0 * mag_b, p99), floor)
    return mag_s > thr, thr


def measure_clip(base, st, fps, names, worst_n, dump_bone=None):
    T = min(base["Q"].shape[0], st["Q"].shape[0])
    Qb, Qs = base["Q"][:T], st["Q"][:T]
    jb = np.degrees(jerk_of(rotvec(qmul(Qb[1:], qconj(Qb[:-1]))), fps))      # (T-3, B, 3) deg/s^3
    js = np.degrees(jerk_of(rotvec(qmul(Qs[1:], qconj(Qs[:-1]))), fps))
    mb, ms = np.linalg.norm(jb, axis=-1), np.linalg.norm(js, axis=-1)        # (T-3, B)
    rms_b = np.sqrt((mb ** 2).mean(0))
    moving = rms_b > 1e-6
    rms_s = np.sqrt((ms ** 2).mean(0))
    ratio = float((rms_s[moving] / rms_b[moving]).mean()) if moving.any() else 1.0
    energy = float((ms ** 2).sum()) / max(float((mb ** 2).sum()), 1e-12)

    flagged = np.zeros(mb.shape, bool)
    excess = np.zeros(mb.shape)
    for b in np.where(moving)[0]:
        fl, thr = flag(ms[:, b], mb[:, b], FLOOR_BONE)
        flagged[:, b] = fl
        excess[:, b] = np.where(fl, ms[:, b] / np.maximum(thr, 1e-9), 0.0)
    spike_pct = 100.0 * float(flagged[:, moving].mean()) if moving.any() else 0.0

    # the correction: what our step did to each bone (local frame), the source taken out
    C = rotvec(qmul(qconj(Qb), Qs))                                          # (T, B, 3) radians
    jc = np.degrees(jerk_of(C, fps))
    corr_jerk = float(np.sqrt((np.linalg.norm(jc, axis=-1)[:, moving] ** 2).mean())) if moving.any() else 0.0
    touched_pct = 100.0 * float((np.linalg.norm(C, axis=-1)[:, moving] > 1e-5).mean()) if moving.any() else 0.0

    # root and feet, world space
    rb = jerk_of(base["root"][:T], fps)
    rs = jerk_of(st["root"][:T], fps)
    rmb, rms_ = np.linalg.norm(rb, axis=-1), np.linalg.norm(rs, axis=-1)
    root_fl, _ = flag(rms_, rmb, FLOOR_WORLD)
    root_ratio = float(np.sqrt((rms_ ** 2).mean()) / max(np.sqrt((rmb ** 2).mean()), 1e-12))
    feet, foot_any = {}, None
    for k, (lb, _) in enumerate(FOOT_BONES):
        if not base["feet_ok"][k]:
            continue
        fb, fs = base["feet"][:T, k], st["feet"][:T, k]
        jfb, jfs = jerk_of(fb, fps), jerk_of(fs, fps)
        mjb, mjs = np.linalg.norm(jfb, axis=-1), np.linalg.norm(jfs, axis=-1)
        fl, _ = flag(mjs, mjb, FLOOR_WORLD)
        foot_any = fl if foot_any is None else (foot_any | fl)
        knee = names.index(lb.replace("ankle", "knee")) if lb.replace("ankle", "knee") in names else None
        spikes_at = np.where(fl)[0][:6]
        foot_rows = [{"frame": int(base["f0"] + t + 3), "excess": round(float(mjs[t] / max(mjb[t], FLOOR_WORLD)), 2),
                      "knee_corr_deg": (round(float(np.degrees(np.linalg.norm(C[t + 3, knee]))), 2) if knee is not None else None)}
                     for t in spikes_at]
        ab, as_ = np.linalg.norm(accel_of(fb, fps), axis=-1), np.linalg.norm(accel_of(fs, fps), axis=-1)
        feet[lb.split("_")[0]] = {
            "acc_p99_base": round(float(np.percentile(ab, 99)), 3),
            "acc_p99_state": round(float(np.percentile(as_, 99)), 3),
            "spike_frames": int(fl.sum()),
            "spikes": foot_rows,
        }
    foot_spike_frames = int(foot_any.sum()) if foot_any is not None else 0

    # worst frames by excess over the threshold, for rendering
    per_frame = excess.max(1) if excess.size else np.zeros(0)
    worst = []
    for t in np.argsort(-per_frame)[:worst_n] if per_frame.size else []:
        if per_frame[t] <= 0:
            break
        worst.append({"frame": int(base["f0"] + t + 3), "excess": round(float(per_frame[t]), 2),
                      "bones": [names[b] for b in np.where(flagged[t])[0]][:4]})
    cnt = flagged.sum(0)
    worst_bones = []
    for b in np.argsort(-cnt)[:6]:
        if cnt[b] == 0:
            break
        worst_bones.append({"bone": names[b], "spike_frames": int(cnt[b]),
                            "jerk_rms_base": round(float(rms_b[b]), 1), "jerk_rms_state": round(float(rms_s[b]), 1)})

    dump = None
    if dump_bone is not None:
        dump = {"bone": dump_bone, "correction_deg": np.degrees(C[:, dump_bone]).round(3).tolist(),
                "correction_angle_deg": np.degrees(np.linalg.norm(C[:, dump_bone], axis=-1)).round(3).tolist(),
                "f0": int(base["f0"])}
    return {
        "dump": dump,
        "frames": int(T),
        "jerk_ratio": round(ratio, 4),
        "jerk_energy": round(energy, 4),
        "spike_pct": round(spike_pct, 4),
        "spike_frames": int(flagged.any(1).sum()),
        "corr_jerk": round(corr_jerk, 2),
        "touched_pct": round(touched_pct, 2),
        "root_ratio": round(root_ratio, 4),
        "root_spike_frames": int(root_fl.sum()),
        "feet": feet,
        "foot_spike_frames": foot_spike_frames,
        "worst": worst,
        "worst_bones": worst_bones,
    }


# ---- run ------------------------------------------------------------------------------------------------
print(f"[smooth] source {a.base}", flush=True)
names, base_clips = read(a.base)
wanted = [c for c in a.clips.split(",") if c] or sorted(base_clips)
results = {"fps": a.fps, "base": a.base, "states": {}, "summary": {}}
state_clips = {}
for lab, path in states:
    print(f"[smooth] state {lab}: {path}", flush=True)
    nm, clips = read(path)
    if nm != names:
        raise SystemExit(f"[smooth] {lab}: bone list differs from the source's")
    state_clips[lab] = clips
for lab, _ in states:
    results["states"][lab] = {}
    for c in wanted:
        if c not in base_clips or c not in state_clips[lab]:
            print(f"[smooth] {lab}: {c} missing in source or state - skipped", flush=True)
            continue
        dump_bone = None
        if a.dump.startswith(c + ":"):
            dump_bone = names.index(a.dump.split(":", 1)[1])
        m = measure_clip(base_clips[c], state_clips[lab][c], a.fps, names, a.worst, dump_bone)
        results["states"][lab][c] = m
        print(f"[smooth] {lab:10s} {c:16s} jerk x{m['jerk_ratio']:.2f} energy x{m['jerk_energy']:.2f} "
              f"spikes {m['spike_pct']:.2f}% ({m['spike_frames']} frames) corr jerk {m['corr_jerk']:.0f} "
              f"| root {m['root_spike_frames']} feet {m['foot_spike_frames']} spike frames", flush=True)

for lab, _ in states:
    rows = list(results["states"][lab].values())
    if not rows:
        continue
    results["summary"][lab] = {
        "clips": len(rows),
        "jerk_ratio_mean": round(float(np.mean([r["jerk_ratio"] for r in rows])), 4),
        "jerk_energy_mean": round(float(np.mean([r["jerk_energy"] for r in rows])), 4),
        "spike_pct_mean": round(float(np.mean([r["spike_pct"] for r in rows])), 4),
        "spike_frames_total": int(sum(r["spike_frames"] for r in rows)),
        "corr_jerk_mean": round(float(np.mean([r["corr_jerk"] for r in rows])), 2),
        "touched_pct_mean": round(float(np.mean([r["touched_pct"] for r in rows])), 2),
        "root_spike_frames_total": int(sum(r["root_spike_frames"] for r in rows)),
        "foot_spike_frames_total": int(sum(r["foot_spike_frames"] for r in rows)),
    }
    s = results["summary"][lab]
    print(f"[smooth] {lab:10s} over {s['clips']} clips: jerk x{s['jerk_ratio_mean']:.3f} "
          f"energy x{s['jerk_energy_mean']:.3f} spikes {s['spike_pct_mean']:.3f}% ({s['spike_frames_total']} frames) "
          f"corr jerk {s['corr_jerk_mean']:.0f} root {s['root_spike_frames_total']} feet {s['foot_spike_frames_total']}",
          flush=True)
if a.json:
    with open(a.json, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"[smooth] -> {a.json}", flush=True)
