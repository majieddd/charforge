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
    out = {}
    for cid in ids:
        a = load(ROOT / "work" / cid / "foot_audit.json")
        if not a:
            continue
        row = {}
        for g in GAITS:
            if g in a:
                sides = [a[g][s] for s in ("left", "right") if s in a[g]]
                row[g] = {"slip_pct": r(100 * max(s["slip"] for s in sides), 1),
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
                         "max_slip_pct": max(slips) if slips else None, "n": len(slips)},
        "moves": moves(), "fit_selftest": selftest(), "image_models": image_models(),
        "prompt_models": prompt_models(), "turntable": turntable(), "stage_times": stage_times(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(data, indent=1, ensure_ascii=False)
    assert "/Users/" not in s, "an absolute path would leak into the snapshot"
    OUT.write_text(s)
    print(f"[collect] {len(ros)} characters, {len(data['moves'])} moves, {len(ft)} foot audits -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
