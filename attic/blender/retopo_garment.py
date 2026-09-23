"""Rebuild the split garment as a simulation-grade mesh.

Cloth simulation on the generated jacket diverged in every configuration tried - 13,800% edge
stretch with self-collision, 8,438% without, and the decimated-proxy route failed because
Surface Deform would not bind to the surface at all. Measuring the garment explains all three
at once: 129,961 vertices, 154,640 faces, and **32.9% of its edges are boundary edges**. Cutting
the clothing out of a fused marching-cubes character by vertex label leaves an open shell with
91,486 open borders, scattered across many disconnected fragments. A solver asked to integrate
that is not simulating a jacket, it is simulating tens of thousands of loose flaps.

So the garment gets its own retopology pass, aimed at what a solver needs rather than what a
renderer needs:

  weld          co-located vertices merged, because glTF splits them at every UV seam and the
                split inherits that (see parts.py for the same bug in the segmentation stage)
  fragments     connected components below a size threshold dropped - these are the specks the
                label flood-fill left behind
  quadriflow    a surface remesher: it keeps the garment an open shell with its neck, cuff and
                hem boundaries intact, which is what a cloth solver wants. It needs manifold
                input, and the *welded* garment is manifold-with-boundary even though the whole
                character is not - which is why it can be used here and could not be used on the
                character in retopo.py
  voxel remesh  fallback only. It is volumetric, so it closes the shell into a solid, and on a
                thin garment it also punches holes clean through: the first attempt chose a
                2.1 cm voxel for fabric thinner than that and produced a moth-eaten jacket
  weights + UV  transferred back from the original garment, so the retopologised mesh still
                skins to the same skeleton and still wears the same texture

The output is the same object name with new mesh data, so the armature modifier, the material
and the rest of the scene survive and cloth_sim.py can run on it unchanged.

Run: blender -b -noaudio --python retopo_garment.py -- --blend split_animated.blend \
         --out cloth_ready.blend --target-verts 12000
"""
import argparse
import json
import os
import sys
from collections import Counter

import bmesh
import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--part", default="clothing")
ap.add_argument("--target-verts", type=int, default=12000)
ap.add_argument("--min-island", type=float, default=0.004,
                help="drop connected components smaller than this fraction of the mesh")
ap.add_argument("--method", default="auto",
                choices=["auto", "decimate", "voxel"],
                help="auto tries quadriflow, then decimate, then voxel")
ap.add_argument("--json", default=None)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene


def stats(me):
    ec = Counter()
    for p in me.polygons:
        vs = list(p.vertices)
        for i in range(len(vs)):
            ec[tuple(sorted((vs[i], vs[(i + 1) % len(vs)])))] += 1
    n = max(len(ec), 1)
    quads = sum(1 for p in me.polygons if len(p.vertices) == 4)
    return {"vertices": len(me.vertices), "faces": len(me.polygons),
            "boundary_fraction": round(sum(1 for v in ec.values() if v == 1) / n, 4),
            "nonmanifold_fraction": round(sum(1 for v in ec.values() if v > 2) / n, 4),
            "quad_fraction": round(quads / max(len(me.polygons), 1), 4)}


src = next((o for o in scene.objects
            if o.type == "MESH" and a.part in o.name.lower()), None)
if src is None:
    raise SystemExit(f"[retopo-g] no mesh matching {a.part!r}")
before = stats(src.data)
print(f"[retopo-g] {src.name} before: {before}", flush=True)

# ---- work on a duplicate, so the original stays available as a transfer source -------------
bpy.ops.object.select_all(action="DESELECT")
src.select_set(True)
bpy.context.view_layer.objects.active = src
bpy.ops.object.duplicate()
work = bpy.context.view_layer.objects.active
work.name = f"{src.name}_retopo"
for m in list(work.modifiers):
    work.modifiers.remove(m)          # bake nothing; remesh the rest pose

# ---- weld, then drop the specks -------------------------------------------------------------
bm = bmesh.new()
bm.from_mesh(work.data)
dims = work.dimensions
scale = max(dims.x, dims.y, dims.z) or 1.0
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5 * scale)
bm.verts.ensure_lookup_table()

# connected components over the welded surface
seen = set()
islands = []
for v in bm.verts:
    if v.index in seen:
        continue
    stack, comp = [v], []
    seen.add(v.index)
    while stack:
        u = stack.pop()
        comp.append(u)
        for e in u.link_edges:
            w = e.other_vert(u)
            if w.index not in seen:
                seen.add(w.index)
                stack.append(w)
    islands.append(comp)
islands.sort(key=len, reverse=True)
cut = max(8, int(len(bm.verts) * a.min_island))
dead = [v for comp in islands if len(comp) < cut for v in comp]
print(f"[retopo-g] welded to {len(bm.verts):,} verts, {len(islands):,} islands; "
      f"dropping {len(dead):,} verts in islands smaller than {cut:,}", flush=True)
if dead:
    bmesh.ops.delete(bm, geom=dead, context="VERTS")
bm.to_mesh(work.data)
bm.free()
work.data.update()

# ---- remesh ----------------------------------------------------------------------------------
bpy.context.view_layer.objects.active = work
method = None
before_v = len(work.data.vertices)

# QuadriFlow first: it preserves the shell and its boundaries. It returns success even when it
# does nothing, so the result is checked rather than trusted.
try:
    bpy.ops.object.quadriflow_remesh(
        target_faces=max(2000, a.target_verts),
        use_mesh_symmetry=False, use_preserve_sharp=False, use_preserve_boundary=True,
        mode="FACES")
    got = len(work.data.vertices)
    if abs(got - before_v) > before_v * 0.05:
        method = "quadriflow"
        print(f"[retopo-g] quadriflow: {before_v:,} -> {got:,} verts", flush=True)
    else:
        print(f"[retopo-g] quadriflow returned success but left the mesh at {got:,} verts "
              f"(was {before_v:,}) - treating as a no-op", flush=True)
except Exception as e:
    print(f"[retopo-g] quadriflow failed: {e}", flush=True)

best = None
# Decimation before voxel: collapse keeps the garment a *surface*, so the thin shell survives
# and its neck/cuff/hem boundaries stay boundaries. It gives triangles rather than quads, which
# a cloth solver does not care about - what the solver cared about was the vertex count and the
# 91,486 loose borders, and welding plus decimation fixes both. Voxel remesh is kept only as a
# last resort because it is volumetric: it closes the shell into a solid and, at any voxel size
# coarse enough to hit the budget, punches holes through fabric thinner than one voxel.
if method is None and a.method in ("auto", "decimate"):
    bpy.context.view_layer.objects.active = work
    dec = work.modifiers.new("retopo_decimate", "DECIMATE")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = min(1.0, a.target_verts / max(len(work.data.vertices), 1))
    dec.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=dec.name)
    got = len(work.data.vertices)
    if got and abs(got - before_v) > before_v * 0.05:
        method = "decimate"
        print(f"[retopo-g] decimate: {before_v:,} -> {got:,} verts "
              f"(surface preserved, boundaries kept)", flush=True)

if method is None:
    print("[retopo-g] falling back to voxel remesh (volumetric: closes the shell)", flush=True)
if method is None and a.method in ("auto", "voxel"):
  lo, hi = scale * 0.002, scale * 0.06
  for _ in range(7):
      mid = (lo + hi) * 0.5
      work.data.remesh_voxel_size = mid
      work.data.remesh_voxel_adaptivity = 0.0
      work.data.use_remesh_fix_poles = True
      snapshot = work.data.copy()
      bpy.ops.object.voxel_remesh()
      n = len(work.data.vertices)
      print(f"[retopo-g]   voxel {mid:.4f} -> {n:,} verts", flush=True)
      if best is None or abs(n - a.target_verts) < abs(best[1] - a.target_verts):
          best = (mid, n)
      if n > a.target_verts:
          lo = mid
      else:
          hi = mid
      work.data = snapshot
      if abs(n - a.target_verts) < a.target_verts * 0.12:
          break
  work.data.remesh_voxel_size = best[0]
  work.data.remesh_voxel_adaptivity = 0.0
  work.data.use_remesh_fix_poles = True
  bpy.ops.object.voxel_remesh()
  method = "voxel"
  print(f"[retopo-g] chose voxel size {best[0]:.4f}", flush=True)

bpy.ops.object.shade_smooth()

# ---- carry the weights and the UVs back ------------------------------------------------------
# data_transfer reads from the ACTIVE object into the SELECTED ones.
bpy.ops.object.select_all(action="DESELECT")
work.select_set(True)
src.select_set(True)
bpy.context.view_layer.objects.active = src

bpy.ops.object.data_transfer(
    data_type="VGROUP_WEIGHTS", use_create=True,
    vert_mapping="POLYINTERP_NEAREST", layers_select_src="ALL", layers_select_dst="NAME")
print(f"[retopo-g] weights: {len(work.vertex_groups)} vertex groups transferred", flush=True)

if src.data.uv_layers:
    if not work.data.uv_layers:
        work.data.uv_layers.new(name=src.data.uv_layers[0].name)
    bpy.ops.object.data_transfer(
        data_type="UV", use_create=True,
        loop_mapping="POLYINTERP_NEAREST", layers_select_src="ALL", layers_select_dst="NAME")
    print(f"[retopo-g] UVs transferred ({len(work.data.uv_layers)} layer(s))", flush=True)

# material
work.data.materials.clear()
for m in src.data.materials:
    work.data.materials.append(m)

after = stats(work.data)
print(f"[retopo-g] after: {after}", flush=True)

# ---- keep the retopologised object, drop the original ----------------------------------------
# Deliberately *not* done by moving mesh data into the original object. Vertex group weights
# live in the mesh but their names live on the object, and `vertex_groups.clear()` deletes the
# weights along with the names - so the obvious "swap the data, copy the groups" version wiped
# out the weights data_transfer had just written and left a garment with 0 groups that the
# armature could not move at all. Renaming the new object and deleting the old one keeps the
# object-level groups and the mesh-level weights together, which is the only pairing that works.
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
old_name = src.name
parent, matrix = src.parent, src.matrix_world.copy()
bpy.data.objects.remove(src, do_unlink=True)

work.name = old_name
work.data.name = old_name + "_mesh"
work.parent = parent
work.matrix_world = matrix
if rig is not None and not any(m.type == "ARMATURE" for m in work.modifiers):
    am = work.modifiers.new("Armature", "ARMATURE")
    am.object = rig
    print(f"[retopo-g] re-bound {work.name} to armature {rig.name}", flush=True)

weighted = sum(1 for v in work.data.vertices if v.groups)
print(f"[retopo-g] {len(work.vertex_groups)} vertex groups, "
      f"{weighted:,}/{len(work.data.vertices):,} vertices weighted", flush=True)
if weighted == 0:
    raise SystemExit("[retopo-g] no vertex received a weight - the garment would not follow the "
                     "skeleton at all. Failing rather than writing a static jacket.")

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump({"part": a.part, "before": before, "after": after,
               "method": method,
               "voxel_size": round(best[0], 5) if best else None,
               "vertex_groups": len(work.vertex_groups),
               "weighted_vertices": weighted}, open(a.json, "w"), indent=2)
print(f"[retopo-g] -> {a.out}", flush=True)
