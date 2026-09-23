"""Re-solve skin weights per split part, with bone heat instead of proximity falloff.

The original skinning was a hand-rolled inverse-distance falloff over a bone neighbourhood,
written because Blender's automatic (bone heat) weights failed on the fused character - it left
159,259 of 170,958 vertices unweighted while reporting success. That failure had a cause worth
naming: bone heat solves a diffusion problem over the surface, and the fused marching-cubes mesh
it was handed was non-manifold, self-intersecting and full of degenerate faces. The solver
cannot build a Laplacian on that.

Everything that blocked it has since been fixed for other reasons. The parts are separated, the
garment has been retopologised into a clean surface, and the seam-split vertices are welded. So
bone heat is worth another try, and it matters because the proximity weights have a specific,
visible failure: they give armpit vertices far too much upper-arm influence, so raising the arms
drags a triangular web of jacket out of the torso. That web is what makes the retargeted arms
look broken, and no amount of retargeting maths fixes a weight problem.

Parts are handled by what they are:

  body, clothing   bone heat, then a check that nothing is left unweighted and nothing exceeds
                   the 4-influence budget an engine will accept
  hair, accessory  left exactly as they are - rigid per-island binding is already correct for
                   them and diffusion weights would be worse

Run: blender -b -noaudio --python reskin.py -- --blend cloth_ready.blend --out reskinned.blend
"""
import argparse
import json
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--parts", default="body,clothing")
ap.add_argument("--max-influences", type=int, default=4)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
if rig is None:
    raise SystemExit("[reskin] no armature")

want = [p.strip() for p in a.parts.split(",") if p.strip()]
report = {}

for key in want:
    ob = next((o for o in scene.objects if o.type == "MESH" and key in o.name.lower()), None)
    if ob is None:
        print(f"[reskin] no mesh for {key!r}", flush=True)
        continue

    before_groups = {vg.name: vg.index for vg in ob.vertex_groups}
    saved = None
    if a.json:
        saved = sum(1 for v in ob.data.vertices if v.groups)

    # Bone heat needs a clean start: leftover groups bias nothing, but stale ones linger.
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    ob.vertex_groups.clear()

    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except Exception as e:
        print(f"[reskin] {ob.name}: bone heat raised {type(e).__name__}: {e}", flush=True)

    weighted = sum(1 for v in ob.data.vertices if v.groups)
    total = len(ob.data.vertices)
    unweighted = total - weighted
    frac = unweighted / max(total, 1)
    over = sum(1 for v in ob.data.vertices if len(v.groups) > a.max_influences)

    print(f"[reskin] {ob.name}: {len(ob.vertex_groups)} groups, "
          f"{weighted:,}/{total:,} weighted ({frac:.2%} unweighted), "
          f"{over:,} vertices over {a.max_influences} influences", flush=True)

    # Bone heat is exactly the operator that failed silently before, so it is checked rather
    # than trusted. Anything above a fraction of a percent unweighted means it did not solve.
    if frac > 0.01:
        raise SystemExit(
            f"[reskin] {ob.name}: bone heat left {frac:.1%} of vertices unweighted. That is the "
            "same silent failure the proximity skinning was written to work around; refusing to "
            "write a rig whose mesh would partly not follow the skeleton.")

    # trim to the engine's influence budget, renormalising what is left
    trimmed = 0
    for v in ob.data.vertices:
        gs = sorted(v.groups, key=lambda g: g.weight, reverse=True)
        if len(gs) <= a.max_influences:
            continue
        keep, drop = gs[:a.max_influences], gs[a.max_influences:]
        s = sum(g.weight for g in keep) or 1.0
        for g in drop:
            ob.vertex_groups[g.group].remove([v.index])
        for g in keep:
            ob.vertex_groups[g.group].add([v.index], g.weight / s, "REPLACE")
        trimmed += 1
    if trimmed:
        print(f"[reskin] {ob.name}: trimmed {trimmed:,} vertices to "
              f"{a.max_influences} influences", flush=True)

    report[ob.name] = {"vertices": total, "groups": len(ob.vertex_groups),
                       "unweighted": unweighted, "unweighted_fraction": round(frac, 5),
                       "trimmed_to_budget": trimmed,
                       "weighted_before": saved}

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump(report, open(a.json, "w"), indent=2)
print(f"[reskin] -> {a.out}", flush=True)
