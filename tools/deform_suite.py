#!/usr/bin/env python3
"""The extreme-pose battery for built characters: skin and penetration at poses no clip reaches (experiment E153).

    python tools/deform_suite.py [--name mara,pip | all] [--out DIR] [--force]

For each character: blender/deform_suite.py writes eight held poses (arms overhead, forward, across the chest, behind;
a deep squat; a bend with a twist; a kick; the head turned hard) as actions into a copy of its rig, then
blender/aberration_audit.py measures them like any clip. Writes work/<name>/qa/stress.json (the audit) and prints, per pose,
the share of crushed / stretched / sheared faces and the deepest penetration, with the roster's median beside each so a
character's weak pose stands out. Report only - nothing is gated on it yet."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import charforge  # noqa: E402

POSES = ["arms_overhead", "arms_forward", "arms_across", "arms_behind", "squat", "bend_twist", "kick", "head_turn", "tiptoe", "heel_strike"]


def run(name: str, out_dir: Path | None, force: bool):
    work = ROOT / "work" / name
    qa = (out_dir or work / "qa")
    qa.mkdir(parents=True, exist_ok=True)
    blend, audit = qa / f"{name}_stress.blend" if out_dir else qa / "stress.blend", qa / (f"{name}_stress.json" if out_dir else "stress.json")
    if audit.exists() and not force:
        return json.load(open(audit))
    bl = charforge.blender_bin()
    with charforge.gpu(f"the stress poses of {name}"):
        r = subprocess.run([bl, "-b", "-noaudio", "--python", str(ROOT / "blender" / "deform_suite.py"), "--",
                            "--blend", str(work / "final.blend"), "--out", str(blend)], capture_output=True, text=True)
        if not blend.exists():
            print(f"[stress] {name}: no poses\n" + "\n".join((r.stdout + r.stderr).splitlines()[-6:]))
            return None
        r = subprocess.run([bl, "-b", "-noaudio", "--python", str(ROOT / "blender" / "aberration_audit.py"), "--",
                            "--blend", str(blend), "--out", str(audit), "--step", "1", "--bvh-step", "3", "--skin", "dqs",
                            "--clips", ",".join(f"stress_{p}" for p in POSES), "--force"], capture_output=True, text=True)
    blend.unlink(missing_ok=True)                                   # the poses are cheap to rebuild
    return json.load(open(audit)) if audit.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="all")
    ap.add_argument("--out", default=None, help="a folder for the audits instead of work/<name>/qa")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = ([p.parent.name for p in sorted((ROOT / "work").glob("*/final.blend")) if p.parent.name != "juno"]
             if a.name == "all" else [n.strip() for n in a.name.split(",") if n.strip()])
    res = {}
    for n in names:
        d = run(n, Path(a.out) if a.out else None, a.force)
        if d:
            res[n] = d["clips"]
            print(f"[stress] {n} done", flush=True)
    if not res:
        return
    for key, label in (("crushed_pct", "crushed faces %"), ("stretched_pct", "stretched faces %"),
                       ("sheared_pct", "sheared faces %"), ("penetration_max_cm", "deepest penetration cm")):
        print(f"\n{label}")
        print("char".ljust(10) + "".join(p[:11].rjust(12) for p in POSES))
        for n, cl in res.items():
            print(n.ljust(10) + "".join(f"{cl.get('stress_' + p, {}).get(key, float('nan')):12.2f}" for p in POSES))
        print("median".ljust(10) + "".join(f"{statistics.median(cl['stress_' + p][key] for cl in res.values() if 'stress_' + p in cl):12.2f}" for p in POSES))


if __name__ == "__main__":
    main()
