"""Count what a person would see go wrong in every clip of every character (experiment E102).

    python tools/aberrations.py                       # every character with a final.blend
    python tools/aberrations.py --name pip,aoi        # some
    python tools/aberrations.py --name pip --render   # and render each clip's worst frames, flagged faces red
    python tools/aberrations.py --skin lbs            # linear skinning instead (aberrations_lbs.json)

Writes work/<name>/qa/aberrations.json (blender/aberration_audit.py) and prints one line a character,
then the clips and body regions that account for most of it. Shares are of all faces or vertices,
so characters compare directly:

  crushed    faces below half their rest area          stretched  faces past twice their rest area
  sheared    faces turned 60 deg beyond their bone     penetration  how deep one body region goes inside
  floor      frames with a vertex 1 cm under the floor               another (cm); deep = past 2 cm
  popping    frames where a vertex spikes out of its surface and back
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv, _argv = [sys.argv[0]], sys.argv
import charforge  # noqa: E402
sys.argv = _argv


def audit(name, render, step, skin="dqs", bvh_step=6, force=False):
    blend = ROOT / "work" / name / "final.blend"
    out = ROOT / "work" / name / "qa" / ("aberrations.json" if skin == "dqs" else f"aberrations_{skin}.json")
    render_dir = ROOT / "work" / name / "qa" / "aberrations"
    if out.exists() and not force:
        print(f"[aberr] {name}: refusing to overwrite {out}; pass --force to replace it", flush=True)
        return None
    if render and render_dir.exists() and any(render_dir.iterdir()) and not force:
        print(f"[aberr] {name}: refusing to overwrite renders in {render_dir}; pass --force to replace them", flush=True)
        return None
    cmd = [charforge.blender_bin(), "-b", "-noaudio", "--python", str(ROOT / "blender" / "aberration_audit.py"), "--",
           "--blend", str(blend), "--out", str(out), "--step", str(step), "--bvh-step", str(bvh_step),
           "--skin", skin]
    if force:
        cmd.append("--force")
    if render:
        cmd += ["--render", str(render_dir)]
    with charforge.gpu(f"the aberration audit of {name}"):
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        print(f"[aberr] {name}: failed\n" + "\n".join((r.stdout + r.stderr).splitlines()[-8:]), flush=True)
        return None
    return json.load(open(out))


def main(a):
    names = [n.strip() for n in a.name.split(",") if n.strip()] if a.name else sorted(
        p.parent.name for p in (ROOT / "work").glob("*/final.blend") if p.parent.name not in ("juno",))
    rows = []
    for n in names:
        d = audit(n, a.render, a.step, a.skin, a.bvh_step, a.force)
        if not d:
            continue
        s = d["summary"]
        rows.append((n, s))
        clips = sorted(d["clips"].items(), key=lambda kv: -(kv[1]["crushed_pct"] + kv[1]["stretched_pct"] + kv[1]["sheared_pct"]))
        worst = ", ".join(f"{c} {e['crushed_pct'] + e['stretched_pct'] + e['sheared_pct']:.1f}%" for c, e in clips[:3])
        pen = sorted(((k, v) for e in d["clips"].values()
                      for k, v in e["penetration_regions_cm"].items()), key=lambda kv: -kv[1])
        exposed = s.get("normal_ray_exposed_pct_of_flagged_equal_clip_mean", {})
        pct = lambda x: "n/a" if x is None else f"{x:.1f}%"
        print(f"[aberr] {n:9s} crushed {s['crushed_pct']:5.2f}%  stretched {s['stretched_pct']:5.2f}%  sheared {s['sheared_pct']:5.2f}%  "
              f"penetration max {s['penetration_max_cm']:5.2f} cm, deep {s['deep_body_frames_pct']:4.1f}% of frames ({s['deep_frames_pct']:4.1f}% with hair)  floor {s['floor_frames_pct']:4.1f}%  "
              f"popping {s['popping_event_windows']:3d} sample windows  "
              f"| normal-ray exposed of flags C/S/H {pct(exposed.get('crushed'))}/{pct(exposed.get('stretched'))}/{pct(exposed.get('sheared'))}  "
              f"| worst clips: {worst} | deepest: {pen[0][0] + f' {pen[0][1]:.1f} cm' if pen else '-'}", flush=True)
    if rows:
        keys = ("crushed_pct", "stretched_pct", "sheared_pct", "penetration_max_cm", "deep_body_frames_pct", "deep_frames_pct", "floor_frames_pct",
                "popping_event_windows")
        print("[aberr] mean over characters: " + ", ".join(f"{k} {sum(r[1][k] for r in rows) / len(rows):.2f}" for k in keys), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", default="")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--bvh-step", "--through-step", dest="bvh_step", type=int, default=6,
                    help="exact frame cadence for the intersection/penetration/exposure checks (legacy alias: --through-step)")
    ap.add_argument("--force", action="store_true", help="allow replacing an existing audit JSON/render output")
    ap.add_argument("--skin", default="dqs", choices=("dqs", "lbs"), help="dual quaternion as shown, or linear (-> aberrations_lbs.json)")
    main(ap.parse_args())
