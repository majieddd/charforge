#!/usr/bin/env python3
"""Topology and UV audit of the game mesh: quads, valence, open edges, UV islands, stretch and seams (lane topology).

    python tools/topology_audit.py --name pip,vex,mara            # work/<name>/retopo_polys.obj, else retopo.glb
    python tools/topology_audit.py --name pip_quads --tag quads   # writes work/<name>/qa/topology_<tag>.json
    python tools/topology_audit.py --mesh some/retopo.glb --out audit.json

Runs blender/quad_audit.py in Blender for each mesh and prints one row per character. Polygon-preserving .obj
files (retopo.py writes retopo_polys.obj beside each retopo.glb) carry the quads; a .glb is triangulated by the
exporter, so quad counts read as triangle pairs. Report only - nothing is gated on it.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import charforge  # noqa: E402

COLUMNS = [
    ("faces", "faces"), ("tris", "triangles"), ("quads", "quads"), ("quad share", "quad_share"),
    ("verts", "verts"), ("open edges", "boundary_edges"), ("non-manifold", "nonmanifold_edges"),
    ("valence-4 interior", "valence4_share_interior"),
    ("corner dev med deg", ("shape", "quad_corner_dev_median_deg")),
    ("skinny tris <10deg", ("shape", "skinny_tri_share_under_10deg")),
    ("UV islands", ("uv", "islands")), ("islands /1k faces", ("uv", "islands_per_1k_faces")),
    ("stretch p95 log2", ("uv", "stretch_p95_log2")), ("faces >2x texel", ("uv", "share_faces_beyond_2x")),
    ("seam length rel", ("uv", "seam_length_rel")),
    ("QF folded quads", ("quadriflow", "folded_quads")), ("QF turned edges", ("quadriflow", "edges_turned_over_120deg")),
]


def get(d, key):
    if isinstance(key, tuple):
        for k in key:
            d = (d or {}).get(k)
        return d
    return d.get(key)


def audit(mesh: Path, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([charforge.blender_bin(), "-b", "-noaudio", "--python", str(ROOT / "blender" / "quad_audit.py"),
                        "--", "--mesh", str(mesh), "--out", str(out)], capture_output=True, text=True)
    if not out.exists():
        raise SystemExit(f"audit failed for {mesh}:\n" + "\n".join((r.stdout + r.stderr).splitlines()[-8:]))
    return json.load(open(out))


BATTERY_COLS = ["crushed%", "stretched%", "sheared%", "faces bad%", "intersecting", "penetration max cm"]


def battery_row(clip: dict) -> list:
    bad = clip["crushed_pct"] + clip["stretched_pct"] + clip["sheared_pct"]
    return [clip["crushed_pct"], clip["stretched_pct"], clip["sheared_pct"], bad,
            clip.get("intersecting_pairs_mean"), clip.get("penetration_max_cm")]


def battery(paths: list):
    """Per held pose (deform_suite.py's ten poses) the share of crushed, stretched and sheared faces, the
    intersecting pairs and the deepest penetration, one table per stress.json, with the battery mean last."""
    for path in paths:
        d = json.load(open(path))
        clips = {k: v for k, v in d["clips"].items() if k.startswith("stress_")}
        label = Path(path).parent.parent.name if Path(path).parent.name == "qa" else Path(path).stem
        print(f"\n{label}  ({Path(path).name}, {len(clips)} poses)")
        print("pose | " + " | ".join(BATTERY_COLS))
        rows = []
        for k in sorted(clips):
            row = battery_row(clips[k])
            rows.append(row)
            print(f"{k[len('stress_'):]} | " + " | ".join("-" if v is None else f"{v:.2f}" for v in row))
        mean = [sum(r[i] for r in rows) / len(rows) for i in range(len(BATTERY_COLS))]
        print("mean | " + " | ".join(f"{v:.2f}" for v in mean))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--battery", default="", help="comma list of stress.json files: print their per-pose tables")
    ap.add_argument("--name", default="", help="comma list of characters (work/<name>)")
    ap.add_argument("--mesh", default="", help="comma list of meshes instead (.obj polygons or .glb)")
    ap.add_argument("--tag", default="", help="suffix for the json (topology_<tag>.json)")
    ap.add_argument("--out", default="", help="with --mesh and one file: the json to write")
    a = ap.parse_args()
    if a.battery:
        battery(a.battery.split(","))
        return
    jobs = []
    if a.name:
        for name in a.name.split(","):
            w = ROOT / "work" / name
            mesh = w / "retopo_polys.obj" if (w / "retopo_polys.obj").exists() else w / "retopo.glb"
            out = w / "qa" / (f"topology_{a.tag}.json" if a.tag else "topology.json")
            jobs.append((name, mesh, out))
    for m in filter(None, a.mesh.split(",")):
        mp = Path(m).resolve()
        jobs.append((f"{mp.parent.name}/{mp.stem}", mp, Path(a.out).resolve() if a.out else mp.with_suffix(".topology.json")))
    rows = []
    for label, mesh, out in jobs:
        rows.append((label, audit(mesh, out)))
    head = ["character"] + [c for c, _ in COLUMNS]
    print(" | ".join(head))
    for label, d in rows:
        cells = [label]
        for _, key in COLUMNS:
            v = get(d, key)
            cells.append("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)))
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
