#!/usr/bin/env python3
"""Let the clips decide between two skin-weight files (E161).

The extreme-pose battery (charforge.py s_weights, E154) ranks candidate weights by how the joints bend in ten held
poses. It cannot tell the surface-relative weights (E159) apart: they took Pip's vest out of its wings and Vex's
jacket out of its fins, and put Vex's clips through the head and torso (deep frames 6.8% -> 11.6%, pops 3 -> 15).
So a candidate is built through the clips and audited the way the animate stage audits its own contact solve, and
kept only when no measure is clearly worse and at least one is clearly better.

    ../.venv/bin/python pipeline/weights_surface.py --name vex             # work/vex/weights_surface.npz (the candidate)
    ../.venv/bin/python tools/weights_trial.py --name vex --candidate work/vex/weights_surface.npz --tag surface
    ../.venv/bin/python tools/weights_trial.py --name pip --candidate work/pip/weights_surface.npz --tag surface --no-build
    (the candidate's md5 must match the trial folder's weights.npz; a regressed clip lists the body pairs whose contact
    frames rose most, from the audit's per-clip region tables)

Steps:
  1. control  the character's current weights (work/<name>/weights.npz) built from the rig stage through animate
              in work/<name>__ctl, then audited. Built once and reused; --control-audit FILE|shipped replaces the
              build with an existing audit (shipped: work/<name>/contact_audit_strict.json, which is the build
              only when the character has no refined clips and no second look was adopted).
  2. trial    the candidate, built the same way in work/<name>__<tag> (a copy-on-write clone of work/<name>).
  3. verdict  global and per-clip comparison; written to work/<name>__<tag>/weights_trial.json.

The audit is blender/aberration_audit.py with the contact stage's settings (--step 2 --bvh-step 6 --skin dqs).
Stages after animate (refine, package, web) are not run: refine only changes video-made clips of Pip, and both sides
are built with the same range, so the comparison is like for like.

Tolerances (points of frames or faces, pops as counts, penetration in cm). A regression is a rise beyond the
tolerance. The keep rule: no global regression, no clip regression, and at least one clear global gain.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# global (the audit's summary): tolerance for a rise, and the drop that counts as a clear gain
GLOBAL_TOL = {"deep_frames_pct": 1.0, "crushed_pct": 0.25, "stretched_pct": 0.25, "sheared_pct": 0.25,
              "popping_event_windows": 2, "penetration_max_cm": 0.5}
GLOBAL_GAIN = {"deep_frames_pct": 1.0, "stretched_pct": 0.25, "sheared_pct": 0.25, "popping_event_windows": 2}
# per clip: a clip may not rise beyond these (one clip's wave going 8% -> 50% is a regression however the mean reads)
CLIP_TOL = {"deep_frames_pct": 3.0, "crushed_pct": 0.5, "stretched_pct": 1.0, "sheared_pct": 1.0, "popping_event_windows": 2}
CLIP_GAIN = {"deep_frames_pct": 2.0, "stretched_pct": 0.5, "sheared_pct": 0.5, "popping_event_windows": 2}
CLIP_KEYS = list(CLIP_TOL)


def read_audit(path: Path):
    """(summary, clips) of an aberration audit JSON; the metrics sit under 'summary' and per clip, not at top level."""
    data = json.loads(Path(path).read_text())
    return data["summary"], data["clips"]


def file_md5(path) -> str:
    """Checksum of a weights file: the verdict is about the candidate that was built, so the build must carry it."""
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def pair_rises(ctl_c: dict, tr_c: dict, top: int = 3) -> dict:
    """The body pairs whose contact frames or depth rose most in one clip (the audit's own per-clip region tables):
    where a regression sits, e.g. torso x upperarm.L rising from 0 to 40 frames."""
    out = {}
    for key in ("intersecting_regions_frames", "penetration_regions_cm"):
        a, b = ctl_c.get(key) or {}, tr_c.get(key) or {}
        rising = sorted(((p, b.get(p, 0) - a.get(p, 0)) for p in set(a) | set(b)), key=lambda x: -x[1])[:top]
        out[key] = {p: {"control": a.get(p, 0), "trial": b.get(p, 0)} for p, d in rising if d > 0}
    return out


def build_and_audit(name: str, weights: Path, until: str, force_build: bool) -> Path:
    """Clone work/<name> to work/<name>__<tag>, put the weights in, rebuild rig..until, audit final.blend.
    Returns the audit JSON. The clone's own audit files are replaced by this build's."""
    import charforge  # the stage driver: the GPU lock, Blender's binary and the audit's settings

    trial = ROOT / "work" / name
    orig = ROOT / "work" / name.split("__")[0]
    if not orig.exists():
        sys.exit(f"[trial] work/{name.split('__')[0]} does not exist")
    if trial.exists() and force_build:
        shutil.rmtree(trial)
    if not trial.exists():
        print(f"[trial] cloning {orig.name} -> {name} (copy-on-write)", flush=True)
        p = subprocess.run(["cp", "-cR", str(orig), str(trial)], capture_output=True)
        if p.returncode != 0:
            sys.exit(f"[trial] clone failed: {p.stderr.decode()[:300]}")
    if weights is not None:
        target = trial / "weights.npz"
        target.unlink(missing_ok=True)                 # a clone is copy-on-write; unlink first, never write through it
        shutil.copy(weights, target)
        if file_md5(target) != file_md5(weights):
            sys.exit(f"[trial] {name}: the copied weights differ from {weights}")
        print(f"[trial] {name}: weights <- {weights}", flush=True)

    if not (trial / "final.blend").exists() or force_build or weights is not None:
        env = dict(os.environ, OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
        log = trial / "logs" / f"trial_make_{until}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "charforge.py"), "make", f"--name={name}", "--from", "rig", "--until", until]
        print(f"[trial] rebuilding {name} rig..{until}: log {log}", flush=True)
        with open(log, "w") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.flush()
            rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            sys.exit(f"[trial] rebuild of {name} failed (exit {rc}); tail of {log}:\n"
                     + "\n".join(log.read_text(errors="replace").splitlines()[-15:]))

    final = trial / "final.blend"
    if not final.exists():
        sys.exit(f"[trial] {name}: final.blend missing after rig..{until}")
    audit = trial / "contact_audit_trial.json"
    audit.unlink(missing_ok=True)
    r = charforge.Run(SimpleNamespace(name=name))           # same logging and GPU lock as the stage driver
    r.bl(f"trial_audit_{name}", "aberration_audit.py", "--blend", final, "--out", audit, "--step", 2, "--bvh-step", 6,
         "--skin", "dqs", "--force", keep=("[aberr] summary",))
    return audit


def compare(ctl_sum, ctl_clips, tr_sum, tr_clips):
    """The verdict: global and per-clip regressions and clear gains, from the two audits."""
    reg_g = {k: round(tr_sum[k] - ctl_sum[k], 3) for k in GLOBAL_TOL if tr_sum[k] - ctl_sum[k] > GLOBAL_TOL[k]}
    gain_g = {k: round(tr_sum[k] - ctl_sum[k], 3) for k in GLOBAL_GAIN if ctl_sum[k] - tr_sum[k] >= GLOBAL_GAIN[k]}
    clips = sorted(set(ctl_clips) & set(tr_clips))
    per = {}
    reg_c = []
    better = worse = 0
    for c in clips:
        d = {k: round(tr_clips[c][k] - ctl_clips[c][k], 3) for k in CLIP_KEYS}
        per[c] = {"control": {k: ctl_clips[c][k] for k in CLIP_KEYS}, "trial": {k: tr_clips[c][k] for k in CLIP_KEYS},
                  "diff": d}
        bad = [k for k in CLIP_KEYS if d[k] > CLIP_TOL[k]]
        good = [k for k in CLIP_GAIN if -d[k] >= CLIP_GAIN[k]]
        if bad:
            reg_c.append({"clip": c, "measures": {k: d[k] for k in bad}, "pairs": pair_rises(ctl_clips[c], tr_clips[c])})
            worse += 1
        elif good:
            better += 1
    if reg_g or reg_c:
        verdict = "reject"
    elif gain_g:
        verdict = "keep"
    else:
        verdict = "no change"
    return {
        "verdict": verdict,
        "global_regressions": reg_g,
        "global_gains": gain_g,
        "clip_regressions": reg_c,
        "clips_better": better,
        "clips_worse": worse,
        "clips_compared": len(clips),
        "clips_only_control": sorted(set(ctl_clips) - set(tr_clips)),
        "clips_only_trial": sorted(set(tr_clips) - set(ctl_clips)),
        "per_clip": per,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="character (pip, vex, rowan, bo)")
    ap.add_argument("--candidate", required=True, help="candidate weights .npz (relative to the worktree or absolute)")
    ap.add_argument("--tag", default=None, help="trial folder suffix (default: the candidate's file stem)")
    ap.add_argument("--until", default="animate", help="last stage built (default animate; the rig stage is the first)")
    ap.add_argument("--control-audit", default=None,
                    help="an audit JSON to use as the control, or 'shipped' (work/<name>/contact_audit_strict.json); "
                         "default: build the current weights as work/<name>__ctl")
    ap.add_argument("--no-build", action="store_true", help="audit the existing trial folder without rebuilding")
    a = ap.parse_args()

    cand = Path(a.candidate)
    cand = cand if cand.is_absolute() else ROOT / cand
    if not cand.exists():
        sys.exit(f"[trial] candidate not found: {cand}")
    tag = a.tag or cand.stem.replace("weights_", "").replace("weights", "cand")
    trial_name = f"{a.name}__{tag}"

    # control
    if a.control_audit == "shipped":
        ctl_json = ROOT / "work" / a.name / "contact_audit_strict.json"
        print(f"[trial] control: shipped audit {ctl_json} (not rebuilt)", flush=True)
    elif a.control_audit:
        ctl_json = Path(a.control_audit)
        ctl_json = ctl_json if ctl_json.is_absolute() else ROOT / ctl_json
        print(f"[trial] control: audit {ctl_json}", flush=True)
    else:
        ctl_json = ROOT / "work" / f"{a.name}__ctl" / "contact_audit_trial.json"
        if not ctl_json.exists():
            build_and_audit(f"{a.name}__ctl", None, a.until, force_build=True)
        print(f"[trial] control: current weights built as {a.name}__ctl ({ctl_json})", flush=True)
    if not ctl_json.exists():
        sys.exit(f"[trial] control audit not found: {ctl_json}")

    # trial
    if a.no_build:
        tr_json = ROOT / "work" / trial_name / "contact_audit_trial.json"
        if not tr_json.exists():
            sys.exit(f"[trial] --no-build but {tr_json} is missing")
    else:
        tr_json = build_and_audit(trial_name, cand, a.until, force_build=True)

    ctl_sum, ctl_clips = read_audit(ctl_json)
    tr_sum, tr_clips = read_audit(tr_json)
    res = compare(ctl_sum, ctl_clips, tr_sum, tr_clips)
    built = ROOT / "work" / trial_name / "weights.npz"
    if not a.no_build and file_md5(built) != file_md5(cand):
        sys.exit(f"[trial] {trial_name} does not carry the candidate weights")
    report = {
        "character": a.name,
        "candidate": str(cand.relative_to(ROOT)) if cand.is_relative_to(ROOT) else str(cand),
        "candidate_md5": file_md5(cand),
        "trial": trial_name,
        "until": a.until,
        "control_audit": str(ctl_json),
        "trial_audit": str(tr_json),
        "tolerances": {"global": GLOBAL_TOL, "global_gain": GLOBAL_GAIN, "clip": CLIP_TOL, "clip_gain": CLIP_GAIN},
        "control_summary": {k: ctl_sum.get(k) for k in GLOBAL_TOL},
        "trial_summary": {k: tr_sum.get(k) for k in GLOBAL_TOL},
        **res,
    }
    out = ROOT / "work" / trial_name / "weights_trial.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    keep = ROOT / "work" / "trials"
    keep.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, keep / f"{trial_name}.json")

    print(f"[trial] {a.name} {tag}: {res['verdict'].upper()}"
          f" | global regressions {res['global_regressions'] or 'none'}"
          f" | global gains {res['global_gains'] or 'none'}"
          f" | clip regressions {[c['clip'] for c in res['clip_regressions']] or 'none'}"
          f" | clips better {res['clips_better']}, worse {res['clips_worse']} of {res['clips_compared']}", flush=True)
    for k in GLOBAL_TOL:
        print(f"[trial]   {k:<24s} control {ctl_sum.get(k)!s:>8}  trial {tr_sum.get(k)!s:>8}", flush=True)
    for rc in res["clip_regressions"]:
        pr = rc["pairs"]["intersecting_regions_frames"]
        print(f"[trial]   regressed {rc['clip']}: {rc['measures']}; pairs rising in frames "
              + ", ".join(f"{p} {v['control']}->{v['trial']}" for p, v in pr.items()), flush=True)
    print(f"[trial] report: {out}", flush=True)
    sys.exit(0 if res["verdict"] == "keep" else (1 if res["verdict"] == "reject" else 2))


if __name__ == "__main__":
    main()
