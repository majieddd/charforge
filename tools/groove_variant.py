"""Run one solid variant through the stages that shape the armpit and body surface, then count grooves on a proxy.

    ../.venv/bin/python tools/groove_variant.py --name rowan --tag base [--solidify "--min-escapes 2"] [--keep-open 3]

Reads work/<name>/ (sdf.npz, mesh.glb, parts.json, joints.json, eyes/eyes.json) and writes work/grooves_exp/<name>_<tag>/:
solidify.py -> refine_joints.py -> cut_hands.py -> free_arms.py (--keep-open) -> sdf_io.py to-mesh (projected onto
mesh.glb) -> hands.py -> groove_decimate.py (to --tris, a proxy for the 60k low-poly) -> marks_grooves.py.
Same arguments as charforge.py's s_solidify / s_joints / s_hands; the proxy is for screening variants against each
other, not for the shipped count (retopo.glb carries a head share and a bake; see the lane report).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
BLENDER = os.environ.get("BLENDER") or "/Applications/Blender.app/Contents/MacOS/Blender"

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--solidify", default="", help="extra arguments for pipeline/solidify.py, e.g. '--min-escapes 2'")
ap.add_argument("--keep-open", type=float, default=3.0, help="free_arms.py --keep-open (charforge default 3)")
ap.add_argument("--tris", type=int, default=60000)
ap.add_argument("--from-stage", default="solidify", choices=["solidify", "cut", "mesh", "proxy"],
                help="resume from a stage whose outputs already exist in the variant folder")
a = ap.parse_args()

W = ROOT / "work" / a.name
V = ROOT / "work" / "grooves_exp" / f"{a.name}_{a.tag}"
V.mkdir(parents=True, exist_ok=True)
logs = V / "logs"
logs.mkdir(exist_ok=True)
T0 = time.time()


def run(stage, cmd):
    t = time.time()
    log = logs / f"{stage}.log"
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(shlex.quote(str(c)) for c in cmd) + "\n")
        fh.flush()
        p = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2"))
    keep = [ln for ln in log.read_text(errors="replace").splitlines()
            if ln.startswith("[") and ("]" in ln)]
    for ln in keep[-6:]:
        print(f"    {ln[:160]}", flush=True)
    if p.returncode != 0:
        tail = log.read_text(errors="replace").splitlines()[-12:]
        raise SystemExit(f"[{stage}] failed ({p.returncode}); see {log}\n" + "\n".join(tail))
    print(f"  [{stage}] {time.time() - t:.0f} s", flush=True)


def blender(stage, script, *args):
    run(stage, [BLENDER, "-b", "-noaudio", "--python", str(ROOT / script), "--", *args])


eyes = W / "eyes" / "eyes.json"
eye_args = ["--eyes", str(eyes)] if eyes.exists() and any(e.get("used") for e in json.load(open(eyes)).get("eyes", [])) else []
order = ["solidify", "cut", "mesh", "proxy"]
start = order.index(a.from_stage)

if start <= 0:
    print(f"[{a.name}/{a.tag}] solidify {a.solidify or '(defaults)'}", flush=True)
    run("solidify", [PY, ROOT / "pipeline" / "solidify.py", "--sdf", W / "sdf.npz", "--mesh", W / "mesh.glb",
                     "--parts", W / "parts.json", "--out", V / "solid.npz", *eye_args, *shlex.split(a.solidify)])
    run("joints", [PY, ROOT / "pipeline" / "refine_joints.py", "--joints", W / "joints.json", "--solid", V / "solid.npz",
                   "--sdf", W / "sdf.npz", "--out", V / "joints_refined.json"])
if start <= 1:
    print(f"[{a.name}/{a.tag}] cut hands and free arms (keep-open {a.keep_open:g})", flush=True)
    run("hands", [PY, ROOT / "pipeline" / "cut_hands.py", "--solid", V / "solid.npz", "--sdf", W / "sdf.npz",
                  "--joints", V / "joints_refined.json", "--out", V / "solid_cut.npz", "--spec", V / "hands_spec.json",
                  "--parts", W / "parts.json"])
    run("arms", [PY, ROOT / "pipeline" / "free_arms.py", "--solid", V / "solid_cut.npz", "--sdf", W / "sdf.npz",
                 "--joints", V / "joints_refined.json", "--out", V / "solid_free.npz", "--report", V / "free_arms.json",
                 "--keep-open", f"{a.keep_open:g}"])
if start <= 2:
    print(f"[{a.name}/{a.tag}] to mesh, project onto the source, union the hands", flush=True)
    keep_out = ["--keep-out", str(V / "solid_eyes.json")] if (V / "solid_eyes.json").exists() else []
    blender("mesh", "blender/sdf_io.py", "to-mesh", "--sdf", str(V / "solid_free.npz"), "--out", str(V / "solid_cut.glb"),
            "--project", str(W / "mesh.glb"), *keep_out)
    blender("hands_union", "blender/hands.py", "--mesh", str(V / "solid_cut.glb"), "--spec", str(V / "hands_spec.json"),
            "--out", str(V / "solid_hands.glb"), "--json", str(V / "hands.json"))
print(f"[{a.name}/{a.tag}] proxy {a.tris} triangles and count grooves", flush=True)
blender("proxy", "tools/groove_decimate.py", "--in", str(V / "solid_hands.glb"), "--out", str(V / "proxy.glb"),
        "--tris", str(a.tris))
run("grooves", [PY, ROOT / "tools" / "marks_grooves.py", "--mesh", V / "proxy.glb", "--json", V / "grooves.json"])
g = json.load(open(V / "grooves.json"))
summary = {"name": a.name, "tag": a.tag, "solidify": a.solidify, "keep_open": a.keep_open,
           "flagged": g["flagged"], "vertices": g["vertices"], "share": round(g["share"], 5),
           "clusters": [(c["vertices"], c["centre_blender"]) for c in g["clusters"][:4]],
           "minutes": round((time.time() - T0) / 60, 1)}
json.dump(summary, open(V / "summary.json", "w"), indent=1)
print(f"[{a.name}/{a.tag}] flagged {g['flagged']} of {g['vertices']} proxy vertices ({summary['minutes']} min)", flush=True)
