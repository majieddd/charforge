#!/usr/bin/env python3
"""Does a sharper albedo change the skin weights? (weights2, problem 1)

rig_build.py reads each vertex's garment colour from the albedo (the armhole and the hem rules), so the weights it writes
depend on the texture's sharpness, not only on the garment. This tool measures that dependence directly: it rigs a
character twice with the same weights file, once per albedo, dumps the final per-vertex weights (rig_build.py
--dump-weights), and compares the two dumps.

    ../.venv/bin/python tools/weights_colour.py run  --name pip --albedo work/pip/albedo.png --tag v019
    ../.venv/bin/python tools/weights_colour.py run  --name pip --albedo work/pip/texwork/v018/albedo.png --tag v018
    ../.venv/bin/python tools/weights_colour.py compare --name pip --a v019 --b v018

`run` writes work/<name>/colour_<tag>/ (rig.blend, rig.json, weights.npz, rig.log); the weights file is the character's
own weights.npz, so only the albedo differs between tags. `compare` reports, for the body and for the clothing, the
per-vertex L1 difference between the two weight sets (sum over bones of |w_a - w_b|, 0 to 2), the share of vertices
that differ by more than 0.05 and 0.20, and the garment rule lines the two rig logs printed (armhole, hem, loose
pieces). Blender runs under the GPU lock like every charforge stage (charforge.gpu).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")


def run(name: str, albedo: Path, tag: str, blur: float = 0.0) -> Path:
    import charforge  # the GPU lock and the Blender binary, as the stage driver uses them

    work = ROOT / "work" / name
    out = work / f"colour_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    if not albedo.exists():
        sys.exit(f"[colour] albedo not found: {albedo}")
    hands = ["--hands", str(work / "hands.json")] if (work / "hands.json").exists() else []
    cmd = [BLENDER, "-b", "-noaudio", "--python", str(ROOT / "blender" / "rig_build.py"), "--",
           "--mesh", str(work / "retopo.glb"), "--joints", str(work / "joints_refined.json"),
           "--frame-from", str(work / "sdf.npz"), *hands, "--weights", str(work / "weights.npz"),
           "--albedo", str(albedo), "--out", str(out / "rig.blend"), "--labels", str(work / "labels.json"),
           "--json", str(out / "rig.json"), "--dump-weights", str(out / "weights.npz"),
           "--colour-blur", str(blur)]
    env = dict(os.environ, OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
    with charforge.gpu(f"colour {name} {tag} (Blender)"):
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    (out / "rig.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0 or not (out / "weights.npz").exists():
        sys.exit(f"[colour] rig failed for {name} {tag}; see {out / 'rig.log'}")
    print(f"[colour] {name} {tag}: rig with {albedo} -> {out / 'weights.npz'}", flush=True)
    return out


RULE = re.compile(r"\[rig\].*(armhole|hem|loose pieces|arm: its own|vertices round the shoulder)")


def compare(name: str, a_tag: str, b_tag: str) -> dict:
    import numpy as np

    work = ROOT / "work" / name
    res = {}
    dumps = {}
    for t in (a_tag, b_tag):
        Z = np.load(work / f"colour_{t}" / "weights.npz", allow_pickle=True)
        dumps[t] = (np.asarray([str(b) for b in Z["bones"]]), Z["weight"].astype(np.float64))
    if not np.array_equal(dumps[a_tag][0], dumps[b_tag][0]):
        sys.exit("[colour] the two dumps have different bones")
    bones, Wa = dumps[a_tag]
    _, Wb = dumps[b_tag]
    d = np.abs(Wa - Wb).sum(1)
    lab = np.asarray(json.load(open(work / "labels.json"))["labels"])
    groups = {"body": 0, "clothing": 1}
    res["vertices"] = int(len(d))
    res["mean_L1"] = round(float(d.mean()), 5)
    res["max_L1"] = round(float(d.max()), 4)
    for g, gid in groups.items():
        m = lab == gid
        res[g] = {"n": int(m.sum()), "mean_L1": round(float(d[m].mean()), 5),
                  "share_gt_0.05": round(float((d[m] > 0.05).mean()), 5),
                  "share_gt_0.20": round(float((d[m] > 0.20).mean()), 5)}
    res["rules"] = {}
    for t in (a_tag, b_tag):
        lines = (work / f"colour_{t}" / "rig.log").read_text(errors="replace").splitlines()
        res["rules"][t] = [ln.split("] ", 1)[-1][:160] for ln in lines if RULE.search(ln)]
    res["a"], res["b"] = a_tag, b_tag
    out = work / "colour_compare" / f"{a_tag}_vs_{b_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--albedo", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--colour-blur", type=float, default=0.0, help="rig_build --colour-blur (sigma in 1024 texels; 0 = off)")
    c = sub.add_parser("compare")
    c.add_argument("--name", required=True)
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.name, (ROOT / a.albedo) if not Path(a.albedo).is_absolute() else Path(a.albedo), a.tag, a.colour_blur)
    else:
        res = compare(a.name, a.a, a.b)
        print(json.dumps({k: v for k, v in res.items() if k != "rules"}, indent=1))
        for t, lines in res["rules"].items():
            print(f"[colour] rule lines {t}:")
            for ln in lines:
                print("   " + ln)


if __name__ == "__main__":
    main()
