"""The head of a character's mesh on its own, for painting it close up (E139).

    blender -b -noaudio --python cut_head.py -- --mesh retopo.glb --joints joints_refined.json --frame sdf.npz \
        --out head.glb [--below 0.25]

Keeps the faces above the neck joint (less --below of the head's height, so the painted region overlaps the neck)
within a sphere round the head - shoulder plates and raised collars stay out. The joints are in the solid's frame
(sdf.npz's points, scaled to 2 units tall), as blender/uv_maps.py reads them for its face camera. Writes the head as
a GLB in the mesh's own coordinates, and prints its frame.
"""
import argparse
import json
import sys

import bmesh
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--joints", required=True)
ap.add_argument("--frame", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--below", type=float, default=0.25, help="how far below the neck joint, in head heights")
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
Pf = np.load(a.frame)["points"]
flo, fhi = Pf.min(0), Pf.max(0)
fk, fc = 2.0 / float(fhi[2] - flo[2]), (flo + fhi) / 2
J = json.load(open(a.joints))["joints"]
head = np.array(J["head"]) / fk + fc
neck = np.array(J.get("neck", J["head"])) / fk + fc
for o in objs:
    me = o.data
    V = np.array([o.matrix_world @ v.co for v in me.vertices])
    near = np.linalg.norm(V[:, :2] - head[:2], axis=1) < 0.12 * 2 / fk
    top = float(V[near, 2].max()) if near.any() else float(head[2] + 0.1)
    span = top - float(neck[2])
    ctr = np.array([head[0], head[1], (top + neck[2]) / 2])
    keep_v = (V[:, 2] > neck[2] - a.below * span) & (np.linalg.norm(V - ctr, axis=1) < 0.95 * span)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    drop = [f for f in bm.faces if not all(keep_v[v.index] for v in f.verts)]
    bmesh.ops.delete(bm, geom=drop, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(me)
    bm.free()
    print(f"[head] {o.name}: {len(me.polygons):,} faces kept, neck z {neck[2]:.3f}, crown z {top:.3f}, "
          f"centre {np.round(ctr, 3).tolist()}", flush=True)
bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=False)
print(f"[head] -> {a.out}", flush=True)
