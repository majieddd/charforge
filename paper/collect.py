"""Collect every number the paper quotes from the measurement files the pipeline writes.

    python paper/collect.py        # reads out/, work/, results/, research/data/ -> paper/data/metrics.json

The snapshot is committed, so the paper builds anywhere (`python paper/build.py`); only this step needs
the working files (work/ and out/ are local: gigabytes, and not in git). Run it after an experiment and
the paper's tables follow. Absolute paths are never copied into the snapshot.
"""
from __future__ import annotations

import datetime
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "data" / "metrics.json"
MOVES = [("mara", "punch_combo", "Mara · punch combo"), ("mara", "roundhouse_kick", "Mara · roundhouse kick"),
         ("aoi", "spell_cast", "Aoi · spell cast"), ("pip", "victory_cheer", "Pip · victory cheer")]
SUPERSEDED = {"juno"}                          # the first Juno, remade as juno3
SUPERSEDED |= {"cadet_hq", "cadet_hq2"}        # E135's test builds of Cadet (1M faces; filtered surface)
SUPERSEDED |= {d.name for d in (ROOT / "out").glob("e1[0-9][0-9]_*")}  # an experiment's variant build, e137_cadet
GAITS = ["walk", "jog", "run", "sprint", "walk_back", "jog_back", "strafe_left", "strafe_right", "crouch_walk"]


def load(p):
    p = Path(p)
    try:
        return json.load(open(p)) if p.exists() else None
    except (json.JSONDecodeError, OSError):
        return None


def r(x, n=3):
    return None if x is None else round(float(x), n)


def roster():
    cfg = load(ROOT / "web" / "roster.json") or {"characters": []}
    shown = {c["id"]: c for c in cfg["characters"]}
    rows = []
    for d in sorted((ROOT / "out").glob("*")):
        m = load(d / f"{d.name}.json")
        if not m or d.name in SUPERSEDED:
            continue
        w = ROOT / "work" / d.name
        gen = load(w / "generate.json") or {}
        hs = load(w / "hands_spec.json") or {}
        side = next(iter((hs.get("sides") or {}).values()), {})
        lods = [x["triangles"] for x in (m.get("lods") or {}).get("fbx", [])] or [m.get("triangles")]
        src = m.get("source") or {}
        rows.append({
            "id": d.name, "name": shown.get(d.name, {}).get("name", d.name),
            "style": (m.get("style") or {}).get("name") if isinstance(m.get("style"), dict) else m.get("style"),
            "in_playground": d.name in shown, "released": shown.get(d.name, {}).get("released", d.name in shown),
            "prompt": src.get("prompt"), "from_image": bool(src.get("image")) and not src.get("prompt"),
            "height_m": m.get("height_m"), "triangles_lods": lods, "bones": (m.get("skeleton") or {}).get("bones"),
            "clips": len(m.get("clips") or []), "face": (m.get("face") or {}).get("morphs") if m.get("face") else None,
            "springs": bool(m.get("springs")), "front_iou": gen.get("front_iou"), "generate_input": gen.get("input"),
            "hand_bulk": side.get("bulk"), "forearm_bare_skin": side.get("bare_skin"),
        })
    return rows


def feet(ids):
    """The current foot audit (blender/foot_audit.py, metric v2: slip as a share of the true ground speed),
    written to work/<name>/qa/feet_v2.json. E029's older table (work/<name>/foot_audit.json, a 1 m/s
    floor under the speed) is frozen in research/data/e029_feet_archived.json and shown as archived."""
    out = {}
    for cid in ids:
        a = load(ROOT / "work" / cid / "qa" / "feet_v2.json")
        if not a:
            continue
        row = {}
        for g in GAITS:
            if g in a:
                audit = a[g]
                speed = audit.get("ground_speed_mps")
                v2_or_newer = audit.get("metric_version", 1) >= 2
                sides = [audit[s] for s in ("left", "right") if s in audit]
                valid_slips = [s["slip"] for s in sides if s.get("slip") is not None
                               and (v2_or_newer or (speed is not None and speed >= 1.0))]
                row[g] = {"slip_pct": r(100 * max(valid_slips), 1) if valid_slips else None,
                          "deepest_cm": r(min(s["deepest_cm"] for s in sides), 2)}
        out[cid] = row
    return out


def moves():
    out = []
    for cid, mv, label in MOVES:
        base = ROOT / "work" / cid / "motion_videos"
        f = load(base / f"{mv}_fidelity.json") or {}
        st = load(base / f"{mv}_stages.json") or {}
        fit = load(base / f"{mv}_fit.json") or {}
        ang = load(ROOT / "work" / cid / "qa" / f"angles_{mv}_video.json") or {}
        ref = (load(ROOT / "work" / cid / "refine.json") or {}).get(mv, {})
        spec = (load(ROOT / "work" / cid / "extra_clips.json") or {}).get(mv, {})
        fs = spec.get("fit") or {}
        stages = st.get("stages") or {}
        out.append({
            "character": cid, "move": mv, "label": label, "frames": f.get("frames"),
            "joints": r(f.get("joint_error_torso")), "joints_p90": r(f.get("joint_error_torso_p90")),
            "iou": r(f.get("silhouette_iou")), "iou_p10": r(f.get("silhouette_iou_p10")),
            "face": r(f.get("face_error_torso")), "hands": r(f.get("hand_error_palm"), 2),
            "hands_median": r(f.get("hand_error_palm_median"), 2), "palm_facing": r(f.get("palm_facing_agree"), 3),
            "pointing_deg": r(f.get("hand_direction_deg"), 1),
            "match_cost": r(spec.get("match_cost")), "image_error_clip": r(fs.get("image_error_clip")),
            "image_error_fit": r(fs.get("image_error_fit")),
            "stage_fit": r((stages.get("fit") or {}).get("joints")), "stage_fbx": r((stages.get("fbx") or {}).get("joints")),
            "stage_rig": r((stages.get("rig") or {}).get("joints")), "stage_render": r((stages.get("render") or {}).get("joints")),
            "misread_frames": (st.get("render") or {}).get("misread_frames"),
            "joints_trusted": r((st.get("render") or {}).get("joints_trusted")),
            "roundtrip_cm": r(fit.get("joint_error_cm_mean"), 2),
            "iou_refine_clip": r(ref.get("iou_points_clip")), "iou_refine_refined": r(ref.get("iou_points_refined")),
            "worst_elbow_deg": r(ang.get("worst_elbow_deg"), 1), "worst_knee_deg": r(ang.get("worst_knee_deg"), 1),
        })
    return out


def aberrations(ids):
    """Each character's aberration audit (blender/aberration_audit.py via tools/aberrations.py: DQS, every
    frame), what clearance changed (blender/clearance.py) and the joint limits it held to (joint_limits.py)."""
    out = {}
    for cid in ids:
        a = load(ROOT / "work" / cid / "qa" / "aberrations.json")
        if not a or "penetration_max_cm" not in a.get("summary", {}):
            continue
        s, clips = a["summary"], a["clips"]
        body = [max([v for k, v in e["penetration_regions_cm"].items() if "head" not in k], default=0.0) for e in clips.values()]
        total = lambda e: e["crushed_pct"] + e["stretched_pct"] + e["sheared_pct"]
        worst = sorted(clips, key=lambda c: -total(clips[c]))[:2]
        row = {"skin": a.get("skin"), "clips": len(clips), "crushed_pct": s["crushed_pct"], "stretched_pct": s["stretched_pct"],
               "sheared_pct": s["sheared_pct"], "penetration_max_cm": s["penetration_max_cm"],
               "body_penetration_cm": r(statistics.mean(body), 2), "deep_frames_pct": s["deep_frames_pct"],
               "deep_body_frames_pct": s.get("deep_body_frames_pct"), "penetration_body_max_cm": s.get("penetration_body_max_cm"),
               "popping_windows": s["popping_event_windows"],
               "exposed_crushed_pct": (s.get("normal_ray_exposed_pct_of_flagged_equal_clip_mean") or {}).get("crushed"),
               "worst_clips": worst}
        cl = load(ROOT / "work" / cid / "clearance.json")
        if cl:
            turns = [v["max_deg"] for c in cl["clips"].values() for v in c.values()]
            row["clearance_max_deg"] = max(turns, default=0.0)
            row["clearance_clips"] = sum(1 for c in cl["clips"].values() if c)
        lim = load(ROOT / "work" / cid / "joint_limits.json")
        if lim:
            j = lim["joints"]
            row["limits"] = {k: (v["limit_deg"] if v["kind"] == "hinge" else min(v["below_limit_deg"])) for k, v in j.items()}
            knees = [v for k, v in j.items() if k.endswith("_knee") and v.get("limit_deg") is not None]
            if knees:
                # the tightest knee, and the deepest bend any of the character's clips asked of a knee
                row["knee_limit_deg"] = min(v["limit_deg"] for v in knees)
                row["knee_clips_max_deg"] = max(v.get("clips_max_deg") or 0.0 for v in knees)
        out[cid] = row
    return out


def stress(ids):
    """Each character's extreme-pose battery (tools/deform_suite.py: eight held poses, audited like clips; E153), per pose."""
    out = {}
    for cid in ids:
        a = load(ROOT / "work" / cid / "qa" / "stress.json")
        if not a:
            continue
        out[cid] = {k[len("stress_"):]: {m: v[m] for m in ("crushed_pct", "stretched_pct", "sheared_pct", "penetration_max_cm")}
                    for k, v in a["clips"].items() if k.startswith("stress_")}
    return out


def weights(ids):
    """Which skin weights each build kept and how the candidates scored on the extreme-pose battery (E154, E157): the pipeline's
    own work/<id>/weights_choice.json, plus Make-It-Animatable's battery scores where they were measured (data/e157_mia.json)."""
    mia = load(ROOT / "paper" / "data" / "e157_mia.json") or {}
    out = {}
    for cid in ids:
        c = load(ROOT / "work" / cid / "weights_choice.json")
        if not c or "per_pose" not in c:
            continue
        row = {"picked": c["picked"], "sharp": r(c["scores"]["sharp"], 2), "hybrid": r(c["scores"].get("hybrid"), 2) if c["scores"].get("hybrid") is not None else None,
               "soft": r(c["scores"].get("soft"), 2) if c["scores"].get("soft") is not None else None}
        if cid in mia:
            row["mia"] = r(mia[cid]["mean"], 2)
        out[cid] = row
    return out


def multiview():
    d = load(ROOT / "research" / "data" / "e106_multiview.json")
    return d["characters"] if d else {}


def controls():
    c = load(ROOT / "research" / "data" / "e102_controls.json")
    return c["checks"] if c else []


def selftest():
    # four variants: the move in the library or left out, and the video's body the capture actor's own or
    # another ("body"), each fitted with the video's own bone lengths (the pipeline) or the capture's
    d = {}
    for k in ("in_library", "left_out", "capture_lengths_in_library", "capture_lengths_left_out",
              "body_in_library", "body_left_out", "capture_lengths_body_in_library", "capture_lengths_body_left_out"):
        s = load(ROOT / "results" / "v3" / f"fit_selftest_{k}.json")
        if s:
            fit = s.get("fit") or {}
            d[k] = {"tests": s.get("tests"), "top1": s.get("top1"), "top1_or_tie": s.get("top1_or_tie"),
                    "angle_error_deg_median": r(s.get("angle_error_deg_median"), 1),
                    "pose_error_capture": r(fit.get("pose_error_clip_mean")),
                    "pose_error_fit_gated": r(fit.get("pose_error_gated_mean"))}
    return d


def image_models():
    c = load(ROOT / "results" / "v3" / "image_models" / "compare.json") or {}
    rows = []
    for cid, models in c.items():
        if not isinstance(models, dict):
            continue
        row = {"character": cid}
        for m, v in models.items():
            if isinstance(v, dict):
                row[m] = {"seconds": v.get("seconds"), "in_frame": v.get("in_frame"), "arm_gap": r(v.get("arm_gap"), 2)}
        rows.append(row)
    return rows


def prompt_models():
    # each model's best run: a run that errored (Gemma 4 with its thinking mode on answered nothing) never
    # hides one that completed
    best = {}
    for f in sorted((ROOT / "research" / "data" / "prompt_models").glob("*.json")):
        for m, v in (load(f) or {}).items():
            if isinstance(v, dict) and "score" in v:
                errors = sum("error" in x for x in v.get("rows", []))
                row = {"score": v["score"], "of": v["of"], "secs": v["secs"], "errors": errors, "file": f.name}
                old = best.get(m)
                if old is None or (errors, -v["score"]) < (old["errors"], -old["score"]):
                    best[m] = row
    return dict(sorted(best.items(), key=lambda kv: (kv[1]["errors"] > 0, -kv[1]["score"], kv[1]["secs"])))


def turntable():
    out = {}
    for k in ("turntable_mara", "turntable_mara_heldout"):
        d = load(ROOT / "results" / "v3" / f"{k}.json") or {}
        out[k] = {m: {"iou_mean": v.get("iou_mean"), "by_quarter": v.get("by_quarter")} for m, v in d.items() if isinstance(v, dict)}
    return out


def stage_times():
    """Seconds per stage, from every build log on this Mac (the Studio's jobs and the command line's): the
    median, the 90th percentile and how many builds. A stage's time includes any wait for the GPU while
    another job held it, so these are what a user waited, not the stage's own cost (E106 times it alone)."""
    import re
    stage_re = re.compile(r"^\s*\[\s*(\d+)/(\d+)\]\s+(\w+)")
    dur_re = re.compile(r"^\s{6,}(\d+)s\s*$")
    seen = {}
    for f in list((ROOT / "work" / "_studio").glob("*.log")) + list((ROOT / "logs").glob("*.log")):
        cur = None
        for line in f.read_text(errors="replace").splitlines():
            m = stage_re.match(line)
            if m:
                cur = m.group(3)
                continue
            d = dur_re.match(line)
            if d and cur:
                seen.setdefault(cur, []).append(int(d.group(1)))
                cur = None
    out = {}
    for k, v in seen.items():
        v = sorted(v)
        out[k] = {"median_s": statistics.median(v), "p90_s": v[min(len(v) - 1, int(0.9 * len(v)))], "n": len(v)}
    return out


def main():
    ros = roster()
    ft = feet([c["id"] for c in ros])
    slips = [g["slip_pct"] for c in ft.values() for g in c.values() if g.get("slip_pct") is not None]
    data = {
        "_about": "Written by paper/collect.py from the pipeline's measurement files; read by paper/build.py.",
        "collected": datetime.date.today().isoformat(),
        "roster": ros, "feet": ft,
        "feet_summary": {"median_slip_pct": r(statistics.median(slips), 1) if slips else None,
                         "mean_slip_pct": r(statistics.mean(slips), 1) if slips else None,
                         "max_slip_pct": max(slips) if slips else None, "n": len(slips),
                         "characters": len(ft)},
        "feet_e029": load(ROOT / "research" / "data" / "e029_feet_archived.json"),
        "moves": moves(), "fit_selftest": selftest(), "image_models": image_models(),
        "prompt_models": prompt_models(), "turntable": turntable(), "stage_times": stage_times(),
        "aberrations": aberrations([c["id"] for c in ros if c["id"] not in SUPERSEDED]), "aberration_controls": controls(),
        "stress": stress([c["id"] for c in ros if c["id"] not in SUPERSEDED]),
        "weights": weights([c["id"] for c in ros if c["id"] not in SUPERSEDED]),
        "multiview": multiview(),
        "unimate": load(ROOT / "research" / "data" / "e129_unimate.json"),
        "planting": load(ROOT / "research" / "data" / "e130_planting.json"),
        "quality": load(ROOT / "research" / "data" / "v012_quality.json"),
        "faces": load(ROOT / "research" / "data" / "v013_faces.json"),
        "e134": load(ROOT / "research" / "data" / "e134_trial.json"),
        "e133": load(ROOT / "research" / "data" / "e133_delight_only.json"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(data, indent=1, ensure_ascii=False)
    assert "/Users/" not in s, "an absolute path would leak into the snapshot"
    OUT.write_text(s)
    print(f"[collect] {len(ros)} characters, {len(data['moves'])} moves, {len(ft)} foot audits -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
