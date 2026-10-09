#!/usr/bin/env python3
"""The triangle budget chosen per character, swept, and the LOD ladder measured (E168 steps 3-4).

    ../.venv/bin/python tools/budget_sweep.py geometry --names pip mara cadet vex --budgets 20000 30000 40000 60000 90000
    ../.venv/bin/python tools/budget_sweep.py bake --names pip --budgets 40000        (full retopo, GPU lock held)
    ../.venv/bin/python tools/budget_sweep.py fbx --names pip mara cadet vex          (the shipped FBX's LODs)
    ../.venv/bin/python tools/budget_sweep.py lod --names pip --budgets 36000 13500   (solid-based LODs, GPU lock held)
    ../.venv/bin/python tools/budget_sweep.py table --names pip mara cadet vex

geometry: blender/retopo.py --geometry-only collapse-decimates each character's solid (solid_hands.glb, the head held
to 28% as the stage does) to each budget and writes the cleaned low as a GLB: no UVs, no maps, no GPU. The 60k
result equals the shipped low's surface (pip: 0.24 mm two-way p95 both ways), so the sweep is the shipped method.
bake: the full retopo at that budget (UVs, the normal map baked from the solid, Cycles on Metal under the GPU lock),
so the mapped normal can be read too. Only the budgets named are baked; 60k is the shipped low.
fbx: blender/lod_export.py writes the packaged FBX's _LOD0/_LOD1/_LOD2 meshes as GLBs; each is measured against the
solid (the LOD1/LOD2 surfaces carry LOD0's normal map).
lod: a solid-based LOD: retopo.py at the budget with a 2048 bake from the solid (its own UVs and normal map).
table: every measured row, written to $CHARFORGE_LANE_NOTES/e168_budget_curve.json (default work/budget_notes).

Every measure carries the shipped labels.json (body / clothing / hair / accessory) to the measured mesh by nearest
vertex. Outputs: work/<name>/budget/ (git-ignored); the table file is the lane's data, not a report.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                      # charforge.gpu(): one GPU job at a time across every lane
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import charforge  # noqa: E402
import remesh_io as R  # noqa: E402

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
NOTES = os.environ.get("CHARFORGE_LANE_NOTES", os.path.join(ROOT, "work", "budget_notes"))
ENV = dict(os.environ, OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
HEAD_SHARE = "0.28"                           # the stage's default (charforge.py: --head-share)
STAGE_BAKE = ["--cage-extrusion", "0.006", "--ray-distance", "0.014"]     # as s_retopo passes them
SHIPPED = 60000
# the solid-based LODs measured at the shipped packages' LOD1 and LOD2 triangle counts (rounded)
LOD_BUDGETS = {"pip": (36000, 13500), "mara": (30500, 11500), "cadet": (38750, 14500), "vex": (31500, 11800)}
DEFAULT_NAMES = ["pip", "mara", "cadet", "vex"]
DEFAULT_BUDGETS = [20000, 30000, 40000, 60000, 90000]


def shipped(name, f):
    return os.path.join(ROOT, "work", name, f)


def budget_dir(name):
    d = os.path.join(ROOT, "work", name, "budget")
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    return d


def run(cmd, log, what):
    """One subprocess with its output kept in a log; the console gets its bracketed summary lines only."""
    with open(log, "w") as f:
        p = subprocess.run(cmd, env=ENV, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    keep = [ln.rstrip() for ln in open(log, errors="replace")
            if ln.startswith("[") or "Error" in ln or "Traceback" in ln]
    for ln in keep[-4:]:
        print(f"      {ln[:220]}", flush=True)
    if p.returncode:
        raise SystemExit(f"{what} failed (exit {p.returncode}); see {log}")


def blender(script, args, log, what):
    run([BLENDER, "-b", "-noaudio", "--python", os.path.join(ROOT, "blender", script), "--"] + args, log, what)


def transfer_labels(name, target_glb, out_json):
    """The shipped labels.json carried to another mesh by nearest vertex: detail_retention reads a point's group from
    the nearest vertex of the mesh the labels index, so each measured mesh needs its own labels."""
    ref = R.read_mesh(shipped(name, "retopo.glb"))["V"]
    lab = np.asarray(json.load(open(shipped(name, "labels.json")))["labels"], dtype=np.int64)
    _, idx = cKDTree(ref).query(R.read_mesh(target_glb)["V"])
    json.dump({"labels": lab[idx].tolist(), "carried_from": f"work/{name}/retopo.glb (nearest vertex)"},
              open(out_json, "w"))
    return out_json


def measure(name, low, out, tag):
    # labels_<measure>.json: not a detail_ file, so the table never reads it
    labels = transfer_labels(name, low, os.path.join(os.path.dirname(out), "labels_" + os.path.basename(out)))
    cmd = [sys.executable, os.path.join(ROOT, "tools", "detail_retention.py"), "--low", low,
           "--cage", shipped(name, "solid_hands.glb"), "--generator", shipped(name, "mesh.glb"),
           "--joints", shipped(name, "joints_refined.json"), "--labels", labels, "--labels-ref", low,
           "--tag", tag, "--out", out]
    run(cmd, os.path.splitext(out)[0] + ".log", f"detail_retention {tag}")
    return out


def cut_args(name, budget):
    return ["--mesh", shipped(name, "mesh.glb"), "--cage", shipped(name, "solid_hands.glb"),
            "--joints", shipped(name, "joints_refined.json"), "--head-share", HEAD_SHARE, "--tris", str(budget)]


def geometry(name, budget, force):
    d = budget_dir(name)
    geo = os.path.join(d, f"geo_{budget}.glb")
    if force or not os.path.exists(geo):
        blender("retopo.py", cut_args(name, budget) + ["--geometry-only", "--out", geo],
                os.path.join(d, "logs", f"geo_{budget}.log"), f"geometry {name} {budget}")
    return measure(name, geo, os.path.join(d, f"detail_geo_{budget}.json"), f"{name}{budget // 1000}k geometry")


def bake(name, budget, force):
    d = budget_dir(name)
    if budget == SHIPPED:
        return measure(name, shipped(name, "retopo.glb"), os.path.join(d, f"detail_bake_{budget}.json"),
                       f"{name}60k shipped")
    out_dir = os.path.join(d, f"bake_{budget}")
    os.makedirs(out_dir, exist_ok=True)
    low = os.path.join(out_dir, "retopo.glb")
    if force or not os.path.exists(low):
        args = cut_args(name, budget) + ["--bake-res", "4096", "--out", low] + STAGE_BAKE
        with charforge.gpu(f"E168 budget bake {name} {budget}"):
            blender("retopo.py", args, os.path.join(d, "logs", f"bake_{budget}.log"), f"bake {name} {budget}")
    return measure(name, low, os.path.join(d, f"detail_bake_{budget}.json"), f"{name}{budget // 1000}k baked")


def lod(name, budget, force, res=2048):
    """A solid-based LOD: the solid collapse-decimated to the budget, its own normal map baked from the solid."""
    d = budget_dir(name)
    tag = f"{budget}" if res == 2048 else f"{budget}_r{res}"
    out_dir = os.path.join(d, f"lod_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    low = os.path.join(out_dir, "retopo.glb")
    if force or not os.path.exists(low):
        args = cut_args(name, budget) + ["--bake-res", str(res), "--out", low] + STAGE_BAKE
        with charforge.gpu(f"E168 solid LOD {name} {budget}"):
            blender("retopo.py", args, os.path.join(d, "logs", f"lod_{tag}.log"), f"lod {name} {budget}")
    return measure(name, low, os.path.join(d, f"detail_lod_{tag}.json"), f"{name} solid LOD {budget // 1000}k r{res}")


def box(path):
    """A mesh's bounding box in Blender's frame as lo x,y,z,hi x,y,z (the rig frame is the generator's, see lod_export)."""
    V = R.to_blender(R.read_mesh(path)["V"])
    return ",".join(f"{x:.6f}" for x in list(V.min(0)) + list(V.max(0)))


def fbx(name, force):
    d = budget_dir(name)
    fd = os.path.join(d, "fbx_lods")
    os.makedirs(fd, exist_ok=True)
    if force or not all(os.path.exists(os.path.join(fd, f"lod{i}.glb")) for i in range(3)):
        blender("lod_export.py", ["--fbx", os.path.join(ROOT, "out", name, f"{name}.fbx"), "--out-dir", fd,
                                  f"--gen-box={box(shipped(name, 'mesh.glb'))}",
                                  f"--cage-box={box(shipped(name, 'solid_hands.glb'))}"],
                os.path.join(d, "logs", "fbx_lods.log"), f"lod_export {name}")
    return [measure(name, os.path.join(fd, f"lod{i}.glb"), os.path.join(d, f"detail_fbx_lod{i}.json"),
                    f"{name} shipped LOD{i}") for i in range(3)]


def render_grid(name):
    """Raking-light close-ups (face and torso, blender/lod_render.py) of the shipped LOD0-2 and the solid-based LODs
    beside the shipped 60k low, each with its triangles and errors written on it: the numbers and the eye side by side."""
    from PIL import Image, ImageDraw
    d = budget_dir(name)
    rd = os.path.join(d, "renders")
    os.makedirs(rd, exist_ok=True)
    b1, b2 = LOD_BUDGETS[name]
    variants = [                                       # (label, mesh, its detail measure)
        ("shipped LOD0", os.path.join(d, "fbx_lods", "lod0.glb"), "detail_fbx_lod0.json"),
        ("shipped LOD1", os.path.join(d, "fbx_lods", "lod1.glb"), "detail_fbx_lod1.json"),
        ("shipped LOD2", os.path.join(d, "fbx_lods", "lod2.glb"), "detail_fbx_lod2.json"),
        ("solid LOD1 (4k)", os.path.join(d, f"lod_{b1}_r4096", "retopo.glb"), f"detail_lod_{b1}_r4096.json"),
        ("solid LOD2 (4k)", os.path.join(d, f"lod_{b2}_r4096", "retopo.glb"), f"detail_lod_{b2}_r4096.json"),
        ("shipped 60k low", shipped(name, "retopo.glb"), "detail_bake_60000.json"),
    ]
    tiles = {}
    for label, glb, _ in variants:
        for view in ("face", "torso"):
            out = os.path.join(rd, f"{label.replace(' ', '_').replace('(', '').replace(')', '')}_{view}.png")
            if not os.path.exists(out):
                run([BLENDER, "-b", "-noaudio", "--python", os.path.join(ROOT, "blender", "lod_render.py"), "--",
                     "--glb", glb, "--out", out, "--view", view], os.path.join(rd, f"{view}.log"), f"render {label}")
            tiles[(label, view)] = out
    rows = []
    for label, _, js in variants:
        j = json.load(open(os.path.join(d, js)))
        mapped = j["normal_vs_cage_deg"].get("mapped")
        rows.append((label, j["triangles"], j["vs_cage"]["two_way_p95_mm_at_1750"],
                     mapped["p95"] if mapped else None))
    T = 384
    sheet = Image.new("RGB", (2 * T + 260, len(rows) * T + 34), (30, 30, 30))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 10), f"{name}: face | torso. Two-way p95 at 1.75 m, mapped normal p95 vs the solid.",
            fill=(230, 230, 230))
    for r, (label, tris, p95, mp) in enumerate(rows):
        y = 34 + r * T
        for c_, view in enumerate(("face", "torso")):
            sheet.paste(Image.open(tiles[(label, view)]).convert("RGB").resize((T, T)), (c_ * T, y))
        mtxt = "n/a" if mp is None else f"{mp:.1f} deg"
        dr.multiline_text((2 * T + 12, y + 12), f"{label}\n{tris:,} tris\np95 {p95:.2f} mm\nnormal p95 {mtxt}",
                          fill=(230, 230, 230), spacing=8)
    out = os.path.join(rd, f"{name}_lod_grid.png")
    sheet.save(out)
    print(f"[budget] render grid: {out}", flush=True)
    return out


def choose(name, tol_mm, grid, force):
    """The budget for the retopo stage under --tris-rule: the smallest budget on the grid whose two-way surface p95 at
    1.75 m is within tol_mm of the solid, or the largest if none is. Geometry only (a few decimations, no bake, no
    GPU). Writes work/<name>/budget/chosen.json, which the stage reads."""
    rows = []
    for b in grid:
        j = json.load(open(geometry(name, b, force)))
        rows.append({"budget": b, "triangles": j["triangles"],
                     "two_way_p95_mm": j["vs_cage"]["two_way_p95_mm_at_1750"]})
    ok = [r for r in rows if r["two_way_p95_mm"] <= tol_mm]
    pick = min(ok, key=lambda r: r["budget"]) if ok else max(rows, key=lambda r: r["budget"])
    out = {"name": name, "tol_mm": tol_mm, "grid": rows, "chosen": pick["budget"],
           "chosen_triangles": pick["triangles"], "two_way_p95_mm": pick["two_way_p95_mm"], "met": bool(ok)}
    json.dump(out, open(os.path.join(budget_dir(name), "chosen.json"), "w"), indent=1)
    print(f"[budget] {name}: tolerance {tol_mm} mm -> budget {pick['budget']:,} ({pick['triangles']:,} triangles, "
          f"{pick['two_way_p95_mm']} mm)" + ("" if ok else " (no grid budget meets it: the largest)"), flush=True)
    return out


def table(names):
    """Every measured row as JSON for the lane notes, and a text table for the terminal."""
    rows = []
    for name in names:
        paths = []
        for kind in ("geo", "bake", "lod", "fbx"):
            paths += glob.glob(os.path.join(ROOT, "work", name, "budget", f"detail_{kind}_*.json"))
        for path in sorted(p for p in paths if "labels" not in os.path.basename(p)):
            d = json.load(open(path))
            _, kind, tag = os.path.basename(path)[:-5].split("_", 2)     # detail_<kind>_<budget|lodN>.json
            v, nv = d["vs_cage"], d["normal_vs_cage_deg"]
            groups = {g: v["low_to_cage"][g]["distance_mm_at_1750"]["p95"] for g in
                      ("body", "clothing", "hair", "accessory") if g in v["low_to_cage"]}
            rows.append({
                "name": name, "kind": kind, "tag": tag, "triangles": d["triangles"],
                "two_way_p50_mm": v["two_way_p50_mm_at_1750"], "two_way_p95_mm": v["two_way_p95_mm_at_1750"],
                "two_way_p99_mm": v["two_way_p99_mm_at_1750"], "two_way_max_mm": v["two_way_max_mm_at_1750"],
                "head_p95_mm": v["low_to_cage"]["head"]["distance_mm_at_1750"]["p95"] if "head" in v["low_to_cage"] else None,
                "hands_p95_mm": v["low_to_cage"]["hands"]["distance_mm_at_1750"]["p95"] if "hands" in v["low_to_cage"] else None,
                "group_p95_mm": groups, "geometric_normal_p95_deg": nv["geometric"]["p95"],
                "mapped_normal_p95_deg": nv["mapped"]["p95"] if nv.get("mapped") else None,
                "head_share": d["shape"]["head_share"], "tilt_p50_deg": d["shape"]["tilt_deg"]["p50"],
                "skinny_tri": d["shape"]["skinny_tri"],
                "generator_two_way_p95_mm": d.get("generator", {}).get("two_way_p95_mm_at_1750"),
                "normal_map": d.get("normal_map_size"), "measure": os.path.relpath(path, ROOT)})
    rows.sort(key=lambda r: (r["name"], r["kind"], r["triangles"]))
    os.makedirs(NOTES, exist_ok=True)
    json.dump(rows, open(os.path.join(NOTES, "e168_budget_curve.json"), "w"), indent=1)
    print(f"{'name':6}{'kind':6}{'tag':>9}{'tris':>8}{'p50':>7}{'p95':>7}{'p99':>7}{'max':>8}{'head95':>8}"
          f"{'map95':>7}{'geo95':>7}{'tilt50':>8}{'skinny':>8}", flush=True)
    for r in rows:
        m = "" if r["mapped_normal_p95_deg"] is None else f"{r['mapped_normal_p95_deg']:7.2f}"
        h = "" if r["head_p95_mm"] is None else f"{r['head_p95_mm']:8.3f}"
        print(f"{r['name']:6}{r['kind']:6}{r['tag']:>9}{r['triangles']:>8}{r['two_way_p50_mm']:7.3f}"
              f"{r['two_way_p95_mm']:7.3f}{r['two_way_p99_mm']:7.3f}{r['two_way_max_mm']:8.2f}{h:>8}{m:>7}"
              f"{r['geometric_normal_p95_deg']:7.2f}{r['tilt_p50_deg']:8.2f}{r['skinny_tri']:8.4f}", flush=True)
    return rows


def rules(names, abs_tols=(0.25, 0.30, 0.35, 0.40), rel_mm=0.10, rel_deg=1.0):
    """What each candidate rule gives each character, from the measured curves. Absolute T: the smallest budget whose
    two-way surface p95 at 1.75 m is within T. Relative: the smallest budget whose surface p95 is within rel_mm of the
    shipped 60k's and, where it was baked, whose mapped normal p95 is within rel_deg of the 60k's."""
    rows = []
    for name in names:
        rows += [r for r in table([name]) if r["kind"] in ("geo", "bake")]
    out = {}
    for name in names:
        geo = {}
        for r in rows:
            if r["name"] == name and r["kind"] == "geo":
                geo[int(r["tag"])] = r
        mapped = {int(r["tag"]): r["mapped_normal_p95_deg"] for r in rows
                  if r["name"] == name and r["kind"] == "bake" and r["mapped_normal_p95_deg"] is not None}
        base = geo[SHIPPED]
        choices = {}
        for T in abs_tols:
            ok = sorted(b for b, r in geo.items() if r["two_way_p95_mm"] <= T)
            choices[f"abs_{T:.2f}"] = ok[0] if ok else None
        rel = []
        for b in sorted(geo):
            if geo[b]["two_way_p95_mm"] > base["two_way_p95_mm"] + rel_mm:
                continue
            if b in mapped and mapped[SHIPPED] is not None and mapped[b] > mapped[SHIPPED] + rel_deg:
                continue
            rel.append(b)
        choices[f"rel_{rel_mm:.2f}mm_{rel_deg:.0f}deg"] = rel[0] if rel else None
        out[name] = {"choices": choices, "geo": {b: geo[b]["two_way_p95_mm"] for b in sorted(geo)},
                     "mapped": {b: mapped[b] for b in sorted(mapped)}}
        cells = "  ".join(f"{k}={v:,}" if v else f"{k}=-" for k, v in choices.items())
        print(f"[rules] {name}: {cells}", flush=True)
    json.dump(out, open(os.path.join(NOTES, "e168_budget_rules.json"), "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("geometry", "bake", "lod", "fbx", "render", "choose", "rules", "table"))
    ap.add_argument("--tol-mm", type=float, default=0.30, help="choose: the two-way surface p95 tolerance, mm at 1.75 m")
    ap.add_argument("--grid", nargs="+", type=int, default=[20000, 30000, 40000, 50000, 60000, 75000, 90000],
                    help="choose: the budgets considered")
    ap.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    ap.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    ap.add_argument("--force", action="store_true", help="redo a run whose output exists")
    ap.add_argument("--res", type=int, default=2048, help="lod: the normal map's size (the shipped LODs share LOD0's 4096)")
    a = ap.parse_args()
    if a.stage == "table":
        table(a.names)
        return
    if a.stage == "render":
        for name in a.names:
            render_grid(name)
        return
    if a.stage == "rules":
        rules(a.names)
        return
    if a.stage == "choose":
        for name in a.names:
            choose(name, a.tol_mm, a.grid, a.force)
        return
    for name in a.names:
        if a.stage == "fbx":
            print(f"[budget] {name} shipped FBX LODs", flush=True)
            fbx(name, a.force)
            continue
        for budget in a.budgets:
            print(f"[budget] {name} {budget:,} ({a.stage})", flush=True)
            {"geometry": geometry, "bake": bake, "lod": lambda n, b, f: lod(n, b, f, a.res)}[a.stage](name, budget, a.force)
    table(a.names)


if __name__ == "__main__":
    main()
