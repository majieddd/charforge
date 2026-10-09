#!/usr/bin/env python3
"""Pose renders of a built character, for the skin-weight trial (E161), under the shared GPU lock.

For one folder under work/ (a shipped build such as vex, or a trial such as vex__surf) it builds the stress blend
(blender/deform_suite.py on its final.blend, the same input tools/deform_suite.py uses), then renders the arms-overhead
held pose and any clip frames asked for (pipeline/weights_pose_render.py), from the front and the side.

    ../.venv/bin/python pipeline/weights_render_shots.py --folder vex --tag ship --out renders_E161/vex \
        --shots wave:12,sprint:20

Shots are action:frame pairs of clips in final.blend; the arms-overhead pose is always rendered (frame 13, where the
battery holds it). Writes PNGs into --out, named <tag>_<action>_f<frame>_az<view>.png.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import charforge  # noqa: E402  (the GPU lock and Blender's binary)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folder", required=True, help="a built character folder under work/")
    ap.add_argument("--tag", required=True, help="prefix for the PNG names (ship, ctl, surf, ...)")
    ap.add_argument("--out", required=True, help="folder for the PNGs")
    ap.add_argument("--shots", default="", help="clip frames to render, action:frame,... (e.g. wave:12,sprint:20)")
    ap.add_argument("--stress-frame", type=int, default=13, help="held frame of the arms-overhead pose (blender/deform_suite.py)")
    a = ap.parse_args()

    work = ROOT / "work" / a.folder
    final = work / "final.blend"
    if not final.exists():
        raise SystemExit(f"[shots] no final.blend in work/{a.folder}")
    stress = work / "qa" / "render_stress.blend"
    stress.parent.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bl = charforge.blender_bin()
    log = work / "logs" / f"render_shots_{a.tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def blender(args: list, what: str):
        with open(log, "a") as fh:
            fh.write(f"$ {' '.join(str(x) for x in args)}\n")
            fh.flush()
            rc = subprocess.run([str(x) for x in args], stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise SystemExit(f"[shots] {what} failed (exit {rc}); see {log}")

    with charforge.gpu(f"pose renders of {a.folder}"):
        blender([bl, "-b", "-noaudio", "--python", ROOT / "blender" / "deform_suite.py", "--",
                 "--blend", final, "--out", stress], "the stress poses")
        pose = ROOT / "pipeline" / "weights_pose_render.py"
        blender([bl, "-b", "-noaudio", "--python", pose, "--", "--blend", stress, "--out", out, "--tag", a.tag,
                 "--shots", f"stress_arms_overhead:{a.stress_frame}", "--views", "0,90"], "the arms-overhead render")
        if a.shots:
            blender([bl, "-b", "-noaudio", "--python", pose, "--", "--blend", final, "--out", out, "--tag", a.tag,
                     "--shots", a.shots, "--views", "0,90"], "the clip renders")
    made = sorted(p.name for p in out.glob(f"{a.tag}_*.png"))
    print(f"[shots] {a.folder} ({a.tag}): {len(made)} images in {out}", flush=True)
    for name in made:
        print(f"[shots]   {out / name}", flush=True)


if __name__ == "__main__":
    main()
