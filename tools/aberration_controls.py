"""Check the aberration audit against a character whose aberrations are known (E102 controls).

    python tools/aberration_controls.py [--out DIR] [--skin dqs,lbs]

Builds blender/aberration_controls.py's synthetic character, audits it at every frame
(blender/aberration_audit.py --step 1 --bvh-step 1 --series) and compares each measure with what
was built in. Prints one line per check and exits non-zero if any fails. Run it after changing
the audit.
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


def blender(*args):
    r = subprocess.run([charforge.blender_bin(), "-b", "-noaudio", "--python", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("\n".join((r.stdout + r.stderr).splitlines()[-12:]))
    return r.stdout


def checks(d, exp):
    """(name, passed, detail) for every built-in expectation."""
    c = d["clips"]
    series = {k: {r["frame"]: r for r in v["series"]} for k, v in c.items()}
    out = []
    zero = ("crushed_pct", "stretched_pct", "sheared_pct", "intersecting_pairs_mean", "penetration_max_cm",
            "floor_frames_pct", "popping_event_windows")
    s = c["still"]
    out.append(("still: every measure zero", all(s[k] == 0 for k in zero), {k: s[k] for k in zero}))

    tb = 100 * exp["torso_top_bottom_triangles"] / exp["triangles"]
    out.append(("squash: torso top+bottom crushed", abs(c["squash"]["crushed_pct"] - tb) < 0.01 and c["squash"]["stretched_pct"] == 0,
                f"{c['squash']['crushed_pct']:.2f}% vs {tb:.2f}%"))
    out.append(("stretch: torso top+bottom stretched", abs(c["stretch"]["stretched_pct"] - tb) < 0.01 and c["stretch"]["crushed_pct"] == 0,
                f"{c['stretch']['stretched_pct']:.2f}% vs {tb:.2f}%"))

    sk = c["sink"]
    out.append(("sink: floor every frame, lowest -3 cm", sk["floor_frames_pct"] == 100 and abs(sk["zmin_cm"] - exp["sink_cm"]) < 0.05,
                f"{sk['floor_frames_pct']}% of frames, {sk['zmin_cm']} cm"))

    pops = [f for f, r in series["pop"].items() if r.get("popping", 0) > 0]
    out.append(("pop: popping at the popped frame only", pops == [exp["pop_frame"]], f"frames {pops}"))
    bend_pops = [f for f, r in series["bend"].items() if r.get("popping", 0) > 0]
    out.append(("bend: a fast smooth bend is not popping", not bend_pops, f"frames {bend_pops}"))

    n = exp["push_stage_frames"]
    for i, want in enumerate(exp["push_depths_cm"]):
        frames = range(1 + i * n, 1 + (i + 1) * n)
        got = [series["push"][f]["penetration_cm"] for f in frames]
        tol = max(0.1, 0.05 * want)                          # a mm, or 5%
        out.append((f"push: penetration {want:g} cm", all(abs(g - want) <= tol for g in got), f"{min(got):.2f}-{max(got):.2f} cm"))
    rigid = c["push"]
    out.append(("push: rigid, so nothing crushed/stretched/sheared/popping",
                rigid["crushed_pct"] == rigid["stretched_pct"] == rigid["sheared_pct"] == 0 and rigid["popping_event_windows"] == 0, ""))
    return out


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    blend, expf = out / "controls.blend", out / "expected.json"
    blender(str(ROOT / "blender" / "aberration_controls.py"), "--", "--out", str(blend), "--expect", str(expf))
    exp = json.load(open(expf))
    failed, results = 0, []
    for skin in a.skin.split(","):
        res = out / f"audit_{skin}.json"
        blender(str(ROOT / "blender" / "aberration_audit.py"), "--", "--blend", str(blend), "--out", str(res),
                "--step", "1", "--bvh-step", "1", "--series", "--skin", skin, "--force")
        for name, ok, detail in checks(json.load(open(res)), exp):
            failed += not ok
            results.append({"skin": skin, "check": name, "pass": bool(ok), "measured": detail if isinstance(detail, str) else json.dumps(detail)})
            print(f"[controls] {skin} {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"_about": "tools/aberration_controls.py: the aberration audit on a synthetic "
                                                      "character with known aberrations (blender/aberration_controls.py)",
                                            "triangles": exp["triangles"], "checks": results}, indent=1) + "\n")
    print(f"[controls] {'all passed' if not failed else f'{failed} failed'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT.parent / "charforge_notes" / "polish" / "e102_controls"))
    ap.add_argument("--skin", default="dqs,lbs")
    ap.add_argument("--json", default=str(ROOT / "research" / "data" / "e102_controls.json"),
                    help="where the checks' results are kept for the paper ('' for nowhere)")
    main(ap.parse_args())
