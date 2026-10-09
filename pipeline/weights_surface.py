#!/usr/bin/env python3
"""Surface-relative skin weights for one character (E159, E161).

The weights measure each bone's distance from the body part's SURFACE, not from its line (geodesic_weights.py
--radius 1): a loose jacket's side then stays with the trunk instead of the thin upper arm. Made the way the
pipeline makes its hybrid: the default falloff and the soft one (--power 2 --smooth 30), both with the radius, then
blend_weights.py. Writes into work/<name>/ (the trial tool reads the hybrid):

  weights_surf_sharp<sfx>.npz   default falloff with the radius (the E159 candidate)
  weights_surf_soft<sfx>.npz    soft falloff with the radius
  weights_surface<sfx>.npz      the hybrid of the two (the E161 candidate: the pipeline's hybrid, surface-relative)

Variants (E161; each one is a separate candidate for tools/weights_trial.py, the clips choose):
  --groups clothing       the radius only on clothing-labelled vertices (labels.json)
  --skip-arm 0.8          vertices bound 80% or more to an arm by the default keep their bone distances
  --skip-leg 0.8          the same for the hip, knee and ankle bones
  --radius-bones pelvis,spine1,spine2,spine3   the radius only on the trunk bones
  --radius 0.5            half the radius

The E161 trial set (Vex, Pip, Rowan, Bo) rejected every one of these: the all-bone radius raises deep frames on Vex (+2.3
points) and Bo (+7.4), the trunk-only radius on Vex (+49) and Rowan (+36). Per-bone radii flip weight along a limb (the
wrist has radius 0, the forearm 2.6-3.5 cm, so the forearm took the hand's weight); see the trial's per-pair reports.

    ../.venv/bin/python pipeline/weights_surface.py --name vex
    ../.venv/bin/python pipeline/weights_surface.py --name vex --groups clothing --tag cloth
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOFT = ["--power", "2", "--smooth", "30"]          # the pipeline's soft falloff (charforge.py WEIGHTS_SOFT, E154)


def run(stage: str, cmd: list, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[surface] {stage}: {Path(cmd[1]).name}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        fh.flush()
        rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    keep = [ln for ln in log.read_text(errors="replace").splitlines() if ln.startswith("[weights]")]
    for ln in keep[-2:]:
        print(f"[surface]   {ln[:160]}", flush=True)
    if rc != 0:
        raise SystemExit(f"[surface] {stage} failed (exit {rc}); see {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--groups", default="", help="limit the radius to these label groups (comma list)")
    ap.add_argument("--tag", default="", help="variant name: the files get the suffix _<tag>")
    ap.add_argument("--skip-arm", type=float, default=0.0,
                    help="vertices the default weights bind at least this much to an arm keep their bone distances")
    ap.add_argument("--radius-bones", default="",
                    help="subtract the radius only for these bones (comma list), the others keep their bone distances")
    ap.add_argument("--skip-leg", type=float, default=0.0,
                    help="vertices the default weights bind at least this much to a leg keep their bone distances")
    a = ap.parse_args()

    w = ROOT / "work" / a.name
    if not w.exists():
        raise SystemExit(f"work/{a.name} does not exist")
    solid = w / "solid_free.npz" if (w / "solid_free.npz").exists() else w / "solid.npz"
    sfx = f"_{a.tag}" if a.tag else ""
    common = ["--mesh", w / "retopo.glb", "--solid", solid, "--sdf", w / "sdf.npz", "--joints", w / "joints_refined.json"]
    rad = ["--radius", f"{a.radius:g}"]
    if a.groups:
        rad += ["--radius-groups", a.groups, "--labels", w / "labels.json"]
    if a.skip_arm > 0:
        rad += ["--radius-skip-arm", f"{a.skip_arm:g}"]
    if a.radius_bones:
        rad += ["--radius-bones", a.radius_bones]
    if a.skip_leg > 0:
        rad += ["--radius-skip-leg", f"{a.skip_leg:g}"]
    geo = [sys.executable, ROOT / "pipeline" / "geodesic_weights.py"]
    logs = w / "logs"

    sharp = w / f"weights_surf_sharp{sfx}.npz"
    soft = w / f"weights_surf_soft{sfx}.npz"
    hybrid = w / f"weights_surface{sfx}.npz"
    run("sharp", geo + common + rad + ["--out", sharp], logs / f"surface_sharp{sfx}.log")
    run("soft", geo + common + rad + SOFT + ["--out", soft], logs / f"surface_soft{sfx}.log")
    run("hybrid", [sys.executable, ROOT / "pipeline" / "blend_weights.py", "--sharp", sharp, "--soft", soft, "--out", hybrid],
        logs / f"surface_hybrid{sfx}.log")
    print(f"[surface] {a.name}: {sharp.name}, {soft.name}, {hybrid.name}", flush=True)


if __name__ == "__main__":
    main()
