#!/usr/bin/env python3
"""One E168 variant end to end: remesh the solid to a budget, bake it with retopo.py, measure it (E168).

    ../.venv/bin/python pipeline/remesh_variant.py --name pip --variant iso60 --tris 60000
    ../.venv/bin/python pipeline/remesh_variant.py --name pip --variant default --baseline

--variant iso60 writes work/<name>/e168/iso60/: low.glb (the remesh, plus low_remesh.json), retopo.glb with its baked maps
beside it (blender/retopo.py --low low.glb, the same bake flags as charforge.py's retopo stage), detail.json
(tools/detail_retention.py), shape.json (blender/model_quality.py then tools/model_quality.py, shape measures only) and
summary.json. --baseline measures the character's shipped retopo.glb (the decimated default) the same way, into
work/<name>/e168/default/. Blender runs under charforge.gpu(), as the stages do. Remeshing is numpy only.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import charforge  # noqa: E402  (gpu lock and the Blender path)

PY = os.path.join(ROOT, "..", ".venv", "bin", "python")
if not os.path.exists(PY):
    PY = sys.executable


def run(cmd, what, keep_tail=6):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-keep_tail:])
    print(f"[variant] {what}: exit {r.returncode} in {time.time() - t0:.0f}s", flush=True)
    if r.returncode != 0:
        print(tail, flush=True)
        raise SystemExit(f"[variant] {what} failed")
    return r


def measure(work, name, low_glb, vdir, tag):
    """detail (against the solid and the generator) and the shape measures (Blender's model_quality, then ours)."""
    joints = os.path.join(work, "joints_refined.json")
    run([PY, os.path.join(ROOT, "tools", "detail_retention.py"), "--low", low_glb,
         "--cage", os.path.join(work, "solid_hands.glb"), "--generator", os.path.join(work, "mesh.glb"),
         "--joints", joints, "--labels", os.path.join(work, "labels.json"),
         "--labels-ref", os.path.join(work, "retopo.glb"), "--tag", tag, "--out", os.path.join(vdir, "detail.json")],
        "detail_retention")
    npz = os.path.join(vdir, "shape.npz")
    blender = charforge.blender_bin()
    run([blender, "-b", "-noaudio", "--python", os.path.join(ROOT, "blender", "model_quality.py"), "--",
         "--mesh", low_glb, "--height", "1.75", "--out", npz], "model_quality (Blender)")
    run([PY, os.path.join(ROOT, "tools", "model_quality.py"), "--npz", npz, "--out",
         os.path.join(vdir, "shape.json")], "model_quality (shape)")


def summarise(vdir, label):
    d = json.load(open(os.path.join(vdir, "detail.json")))
    s = json.load(open(os.path.join(vdir, "shape.json")))["shape"]     # tools/model_quality.py's shape measures
    vs = d["vs_cage"]
    n = d["normal_vs_cage_deg"]
    out = {"label": label, "triangles": d["triangles"], "two_way_p95_mm": vs["two_way_p95_mm_at_1750"],
           "two_way_p99_mm": vs["two_way_p99_mm_at_1750"], "two_way_max_mm": vs["two_way_max_mm_at_1750"],
           "low_to_cage_p95_mm": vs["low_to_cage"]["all"]["distance_mm_at_1750"]["p95"],
           "cage_to_low_p95_mm": vs["cage_to_low"]["all"]["distance_mm_at_1750"]["p95"],
           "normal_mapped_p95_deg": (n["mapped"] or {}).get("p95"), "normal_geometric_p95_deg": n["geometric"]["p95"],
           "tilt_p50_deg": s["tilt_deg"]["p50"], "tilt_p99_deg": s["tilt_deg"]["p99"],
           "skinny_tri": s["skinny_tri"], "head_share": d["shape"]["head_share"],
           "open_edges_per10k": s["open_edges_per10k"], "non_manifold_per10k": s["non_manifold_per10k"],
           "noise_deg_p50": s["noise_deg"]["p50"], "noise_deg_p90": s["noise_deg"]["p90"],
           "lump_mm_p50": s["lump_mm"]["p50"], "lump_mm_p75": s["lump_mm"]["p75"],
           "symmetry_mm_p50": s["symmetry_mm"]["p50"], "edge_mm": s["edge_mm"], "area_m2": s["area_m2"],
           "pieces": s.get("pieces"), "cuts_per10k": s.get("cuts_per10k")}
    json.dump(out, open(os.path.join(vdir, "summary.json"), "w"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="pip, vex, mara or cadet")
    ap.add_argument("--variant", required=True, help="a folder name under work/<name>/e168/")
    ap.add_argument("--baseline", action="store_true", help="measure the shipped default retopo.glb, no remesh or bake")
    ap.add_argument("--tris", type=int, default=60000)
    ap.add_argument("--head-share", type=float, default=0.28)
    ap.add_argument("--start", default=None, help="the cage to remesh (default: the shipped retopo.glb)")
    ap.add_argument("--no-bake", action="store_true", help="measure the remesh itself (no normal map, no bake)")
    ap.add_argument("--remesh-args", default="", help="extra arguments for pipeline/remesh_iso.py, e.g. '--e-rel 1e-4'")
    ap.add_argument("--measure-only", action="store_true",
                    help="measure a variant already made (its retopo.glb, or low.glb with --no-bake) again")
    ap.add_argument("--decimate", action="store_true",
                    help="no remesh: the shipped collapse decimation of the solid to --tris (retopo.py's own path), "
                         "baked and measured like a variant - the comparison that asks what decimation gives at a budget")
    a = ap.parse_args()

    work = os.path.join(ROOT, "work", a.name)
    vdir = os.path.join(work, "e168", a.variant)
    os.makedirs(vdir, exist_ok=True)
    if a.baseline:
        summarise_src = os.path.join(work, "retopo.glb")
        measure(work, a.name, summarise_src, vdir, f"{a.name}_{a.variant}")
        print(json.dumps(summarise(vdir, f"{a.name} {a.variant}")), flush=True)
        return
    if a.measure_only:
        glb = os.path.join(vdir, "low.glb" if a.no_bake else "retopo.glb")
        measure(work, a.name, glb, vdir, f"{a.name}_{a.variant}")
        print(json.dumps(summarise(vdir, f"{a.name} {a.variant}")), flush=True)
        return

    if a.decimate:
        albedo = os.path.join(work, "source_albedo.png")
        if not os.path.exists(albedo):
            run([PY, os.path.join(ROOT, "pipeline", "despeck_source.py"), "--mesh", os.path.join(work, "mesh.glb"),
                 "--out", albedo], "despeck", keep_tail=2)
        retopo = os.path.join(vdir, "retopo.glb")
        with charforge.gpu(f"{a.name} {a.variant} decimate and bake"):
            run([charforge.blender_bin(), "-b", "-noaudio", "--python", os.path.join(ROOT, "blender", "retopo.py"),
                 "--", "--source-albedo", albedo, "--mesh", os.path.join(work, "mesh.glb"), "--cage",
                 os.path.join(work, "solid_hands.glb"), "--tris", str(a.tris), "--out", retopo, "--bake-res", "4096",
                 "--cage-extrusion", "0.006", "--ray-distance", "0.014", "--joints",
                 os.path.join(work, "joints_refined.json"), "--head-share", str(a.head_share)],
                "retopo --tris (decimate, bake)", keep_tail=3)
        measure(work, a.name, retopo, vdir, f"{a.name}_{a.variant}")
        print(json.dumps(summarise(vdir, f"{a.name} {a.variant}")), flush=True)
        return

    low = os.path.join(vdir, "low.glb")
    start = a.start or os.path.join(work, "retopo.glb")
    run([PY, os.path.join(ROOT, "pipeline", "remesh_iso.py"), "--solid", os.path.join(work, "solid_hands.glb"),
         "--start", start, "--mesh", os.path.join(work, "mesh.glb"), "--joints", os.path.join(work, "joints_refined.json"),
         "--tris", str(a.tris), "--head-share", str(a.head_share), "--out", low] + a.remesh_args.split(),
        "remesh_iso", keep_tail=3)
    if a.no_bake:
        measure(work, a.name, low, vdir, f"{a.name}_{a.variant}_nomap")
        print(json.dumps(summarise(vdir, f"{a.name} {a.variant}")), flush=True)
        return
    albedo = os.path.join(work, "source_albedo.png")
    if not os.path.exists(albedo):
        run([PY, os.path.join(ROOT, "pipeline", "despeck_source.py"), "--mesh", os.path.join(work, "mesh.glb"),
             "--out", albedo], "despeck", keep_tail=2)
    retopo = os.path.join(vdir, "retopo.glb")
    with charforge.gpu(f"{a.name} {a.variant} bake"):
        run([charforge.blender_bin(), "-b", "-noaudio", "--python", os.path.join(ROOT, "blender", "retopo.py"), "--",
             "--source-albedo", albedo, "--mesh", os.path.join(work, "mesh.glb"), "--cage",
             os.path.join(work, "solid_hands.glb"), "--low", low, "--out", retopo, "--bake-res", "4096",
             "--cage-extrusion", "0.006", "--ray-distance", "0.014", "--joints",
             os.path.join(work, "joints_refined.json"), "--head-share", str(a.head_share)],
            "retopo --low (bake)", keep_tail=3)
    measure(work, a.name, retopo, vdir, f"{a.name}_{a.variant}")
    print(json.dumps(summarise(vdir, f"{a.name} {a.variant}")), flush=True)


if __name__ == "__main__":
    main()
