"""Collapse-decimate a dense solid to a triangle budget, as retopo.py's cage decimation does (E135/E140).

    blender -b -noaudio --python tools/groove_decimate.py -- --in solid_hands.glb --out proxy.glb [--tris 60000]

Groove counts depend on the edge length (tools/marks_grooves.py counts a second sheet within 4 mm across more
than three edges, which means ~9 mm of edges on the shipped 60k low-poly and ~4 mm on a 1.5 mm solid). This
brings a dense solid to the low-poly's density so the count can be compared with the shipped retopo.glb.
Writes triangles only; no UVs, normals or materials are needed for the count.
"""
import argparse
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tris", type=int, default=60000)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in meshes:
    o.select_set(False)
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
ob = bpy.context.view_layer.objects.active
n0 = len(ob.data.polygons)
ratio = min(1.0, a.tris / max(1, sum(len(p.vertices) - 2 for p in ob.data.polygons)))
md = ob.modifiers.new("dec", "DECIMATE")
md.decimate_type = "COLLAPSE"
md.ratio = ratio
md.use_collapse_triangulate = True
bpy.ops.object.modifier_apply(modifier=md.name)
print(f"[groove_decimate] {n0:,} faces -> {len(ob.data.polygons):,} faces (ratio {ratio:.4f})", flush=True)
bpy.ops.object.select_all(action="DESELECT")
ob.select_set(True)
bpy.context.view_layer.objects.active = ob
bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=True, export_materials="NONE")
print(f"[groove_decimate] -> {a.out}", flush=True)
