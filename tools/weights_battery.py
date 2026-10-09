#!/usr/bin/env python3
"""The extreme-pose battery of one weights file, rigged with a chosen albedo (weights2, problem 1).

charforge.py's weights_variant_score rigs a weights file with the character's own albedo.png and scores it on the ten
stress poses. The auto choice of skin weights (s_weights) compares the sharp and hybrid files on that score, and Pip's
lead is 0.0946 against a margin of 0.10. This tool runs the same chain (rig, springs, frame, T, stress poses, audit)
with any albedo and any --colour-blur, so the score can be read for the 0.18-style albedo as well.

    ../.venv/bin/python tools/weights_battery.py --name pip --weights work/pip/weights_sharp.npz --albedo work/pip/texwork/v018/albedo.png --tag v018 --variant sharp
    ../.venv/bin/python tools/weights_battery.py --name pip --weights work/pip/weights_hybrid.npz --albedo ... --tag v018 --variant hybrid --colour-blur 0

Writes work/<name>/colour_<tag>/battery_<variant>.json: the mean over the poses of stretched + sheared + crushed faces (%),
and each pose's value. Intermediate .blend files are removed. Each Blender step runs under the GPU lock (Run.bl).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def battery(name: str, weights: Path, albedo: Path, tag: str, variant: str, blur: float) -> dict:
    import charforge

    r = charforge.Run(SimpleNamespace(name=name))
    work = ROOT / "work" / name
    d = work / f"colour_{tag}" / f"battery_{variant}"
    d.mkdir(parents=True, exist_ok=True)
    hands = ["--hands", str(work / "hands.json")] if (work / "hands.json").exists() else []
    s = f"{tag}_{variant}"
    r.bl(f"bat_rig_{s}", "rig_build.py", "--mesh", str(work / "retopo.glb"), "--joints", str(work / "joints_refined.json"),
         "--frame-from", str(work / "sdf.npz"), *hands, "--weights", str(weights), "--albedo", str(albedo),
         "--out", str(d / "rig.blend"), "--labels", str(work / "labels.json"), "--colour-blur", str(blur), keep=())
    r.bl(f"bat_springs_{s}", "springs.py", "--blend", str(d / "rig.blend"), "--labels", str(work / "labels.json"),
         "--out", str(d / "rig_s.blend"), "--json", str(d / "springs.json"), keep=())
    r.bl(f"bat_frame_{s}", "normalize_frame.py", "--blend", str(d / "rig_s.blend"), "--out", str(d / "rig_m.blend"),
         "--height", str(charforge.height_of(r)), keep=())
    r.bl(f"bat_tpose_{s}", "tpose.py", "--blend", str(d / "rig_m.blend"), "--out", str(d / "rig_t.blend"),
         *(["--keep-rest"] if charforge.option_of(r, "rest", "A") == "A" else []), keep=())
    r.bl(f"bat_stress_{s}", "deform_suite.py", "--blend", str(d / "rig_t.blend"), "--out", str(d / "stress.blend"), keep=())
    r.bl(f"bat_audit_{s}", "aberration_audit.py", "--blend", str(d / "stress.blend"), "--out", str(d / "stress.json"),
         "--step", "1", "--bvh-step", "3", "--skin", "dqs", "--clips", charforge.STRESS_CLIPS, "--force", keep=())
    clips = json.load(open(d / "stress.json"))["clips"]
    per = {c[len("stress_"):]: clips[c]["stretched_pct"] + clips[c]["sheared_pct"] + clips[c]["crushed_pct"]
           for c in clips if c.startswith("stress_")}
    for f in ("rig.blend", "rig_s.blend", "rig_m.blend", "rig_t.blend", "stress.blend"):
        (d / f).unlink(missing_ok=True)
    out = {"character": name, "tag": tag, "variant": variant, "weights": str(weights), "albedo": str(albedo),
           "colour_blur": blur, "mean": round(sum(per.values()) / max(len(per), 1), 4),
           "per_pose": {k: round(v, 4) for k, v in per.items()}}
    (work / f"colour_{tag}" / f"battery_{variant}.json").write_text(json.dumps(out, indent=1))
    print(f"[battery] {name} {tag} {variant}: mean {out['mean']:.4f}% (blur {blur:g})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--albedo", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--colour-blur", type=float, default=0.0)
    a = ap.parse_args()
    wpath = Path(a.weights) if Path(a.weights).is_absolute() else ROOT / a.weights
    apath = Path(a.albedo) if Path(a.albedo).is_absolute() else ROOT / a.albedo
    battery(a.name, wpath, apath, a.tag, a.variant, a.colour_blur)


if __name__ == "__main__":
    main()
