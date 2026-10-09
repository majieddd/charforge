"""Per-vertex ambient occlusion by ray casting on a mesh (CPU, no GPU, no bake).

    blender -b -noaudio --python tools/groove_ao.py -- --in proxy.glb --out ao.npz [--rays 48] [--dist 0.05]

Each vertex casts --rays cosine-weighted rays over the hemisphere of its smoothed normal; a ray that hits
within --dist metres is occluded. Output: vertex positions (Blender frame, Z up) and ao = 1 - occluded share
(1 open, 0 closed). This is a yardstick for the marks lane's measurement (baked AO 11/255 at groove
vertices against 244 elsewhere) so that variants can be compared on the same numbers. It does not reproduce
the shipped bake's values: compare variants with each other, not with the shipped texture.
"""
import argparse
import math
import random
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rays", type=int, default=48)
ap.add_argument("--dist", type=float, default=0.05)
ap.add_argument("--eps", type=float, default=0.0008, help="ray start offset along the normal, metres")
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
ob = bpy.context.view_layer.objects.active
me = ob.data
# the glTF importer already maps Y-up onto Blender's Z-up through the object's matrix, so M @ co is the Blender frame
# that tools/marks_grooves.py prints: (x, -z, y) of the glTF positions.
M = ob.matrix_world
V = [M @ v.co for v in me.vertices]
N = [(M.to_3x3() @ v.normal).normalized() for v in me.vertices]
tris = [tuple(p.vertices) for p in me.polygons]
bvh = BVHTree.FromPolygons([tuple(v) for v in V], tris)
rng = random.Random(7)
# cosine-weighted hemisphere directions in a local frame, reused for every vertex
dirs = []
for k in range(a.rays):
    u1, u2 = (k + 0.5) / a.rays, rng.random()
    r = math.sqrt(u1)
    th = 2 * math.pi * u2
    dirs.append((r * math.cos(th), r * math.sin(th), math.sqrt(max(0.0, 1 - u1))))
ao = np.ones(len(V))
for i, (p, n) in enumerate(zip(V, N)):
    t = n.orthogonal().normalized()
    b = n.cross(t)
    origin = p + n * a.eps
    hit = 0
    for (x, y, z) in dirs:
        d = (t * x + b * y + n * z).normalized()
        loc, _nrm, _idx, dist = bvh.ray_cast(origin, d, a.dist)
        if loc is not None:
            hit += 1
    ao[i] = 1.0 - hit / a.rays
P = np.array([[v.x, v.y, v.z] for v in V])
np.savez(a.out, pos=P, ao=ao)
print(f"[groove_ao] {len(V):,} vertices, {a.rays} rays, {a.dist*100:.0f} cm: mean AO {ao.mean():.3f} -> {a.out}", flush=True)
