"""Audit a face rig's shape keys: how far each moves, and whether its mesh stretches, folds or tears.

    /Applications/Blender.app/Contents/MacOS/Blender -b -noaudio --python blender/face_keys_audit.py -- \
        --blend work/mara/rig_f.blend --face work/mara/face.json --out qa/face_keys/mara_audit.json

For every shape key other than Basis, at value 1.0 and alone (the others at 0), it measures
  moved        vertices displaced by more than 0.01 mm, and the largest displacement (mm, and as a
               share of the eye distance)
  stretch      edges longer than 1.5x their rest length (r_max is the longest ratio), and edges
               shorter than half their rest length (r_min is the shortest) - a tear shows as a
               stretched edge with a neighbour crushed next to it, a fold as flipped triangles
  flips        triangles whose normal turns more than 90 degrees from rest (folded through); flips_1mm2
               counts only triangles of more than 1 mm2 at rest - the slivers (median 0.02-0.4 mm2 on
               Aoi) turn on noise, the rest are the folds that show. The stretch counts marked _big use
               only edges longer than 0.5 mm at rest.
Basis is hashed, so the neutral face's unchanged-ness can be checked across builds.
Runs on the CPU only.
"""
import argparse
import hashlib
import json
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--face", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH" and o.find_armature() == rig)
me = mesh.data
ied = float(json.load(open(a.face))["ied_m"])
kbs = me.shape_keys.key_blocks
n = len(me.vertices)

base = np.empty(n * 3)
kbs[0].data.foreach_get("co", base)
base = base.reshape(-1, 3)
ne = len(me.edges)
E = np.empty(ne * 2, dtype=np.int64)
me.edges.foreach_get("vertices", E)
E = E.reshape(-1, 2)
me.calc_loop_triangles()
nt = len(me.loop_triangles)
T = np.empty(nt * 3, dtype=np.int64)
me.loop_triangles.foreach_get("vertices", T)
T = T.reshape(-1, 3)


def tri_normals(P):
    return np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])


N0 = tri_normals(base)
A0 = 0.5 * np.linalg.norm(N0, axis=1)
big_tri = A0 > 1e-6                       # more than 1 mm2 at rest
L0 = np.linalg.norm(base[E[:, 0]] - base[E[:, 1]], axis=1)
keep = L0 > 1e-6
big_edge = L0[keep] > 5e-4                  # longer than 0.5 mm at rest
out = {"n_verts": n, "n_edges": int(ne), "n_tris": int(nt), "ied_m": ied,
       "morph_bytes_each": n * 12,
       "basis_sha": hashlib.sha1(np.round(base, 7).tobytes()).hexdigest()[:16],
       "order": [], "keys": {}}
print(f"[audit] {n:,} verts, {ne:,} edges, {nt:,} tris; ied {ied * 1000:.1f} mm", flush=True)
for kb in list(kbs)[1:]:
    P = np.empty(n * 3)
    kb.data.foreach_get("co", P)
    P = P.reshape(-1, 3)
    d = np.linalg.norm(P - base, axis=1)
    moved = d > 1e-5
    L1 = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    r = L1[keep] / L0[keep]
    N1 = tri_normals(P)
    c0 = np.linalg.norm(N0, axis=1)
    ok = c0 > 1e-12
    cos = (N0[ok] * N1[ok]).sum(1) / (c0[ok] * np.linalg.norm(N1[ok], axis=1) + 1e-30)
    flip_big = int(((cos < 0) & big_tri[ok]).sum())
    rb = r[big_edge]
    out["order"].append(kb.name)
    out["keys"][kb.name] = {
        "moved": int(moved.sum()), "max_mm": round(float(d.max()) * 1000, 2),
        "max_share_ied": round(float(d.max()) / ied, 3),
        "r_max": round(float(r.max()), 2), "r_min": round(float(r.min()), 3),
        "stretch_gt1p5": int((r > 1.5).sum()), "crush_lt0p5": int((r < 0.5).sum()),
        "stretch_pct": round(100.0 * float((r > 1.5).mean()), 3),
        "flips": int((cos < 0).sum()), "flip_pct": round(100.0 * float((cos < 0).mean()), 3),
        "flips_1mm2": flip_big, "stretch_big": int((rb > 1.5).sum()), "crush_big": int((rb < 0.5).sum()),
    }
    print(f"[audit] {kb.name:18s} moved {moved.sum():6,d}  max {d.max() * 1000:6.2f} mm  "
          f"r {r.min():.2f}..{r.max():.2f}  >1.5 {int((r > 1.5).sum()):5d}  <0.5 {int((r < 0.5).sum()):5d}  "
          f"flips {int((cos < 0).sum()):4d} (>1mm2 {flip_big:3d})  big-edge stretch {int((rb > 1.5).sum()):4d}", flush=True)
with open(a.out, "w") as f:
    json.dump(out, f, indent=1)
print(f"[audit] -> {a.out}", flush=True)
