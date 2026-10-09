"""Topology and UV audit of a game mesh, measured in Blender (lane topology; tools/topology_audit.py drives it).

    blender -b -noaudio --python blender/quad_audit.py -- --mesh work/<name>/retopo_polys.obj --out audit.json

Reads an .obj (polygons as modelled, quads kept - written by retopo.py beside each retopo.glb) or a .glb
(triangulated by the exporter: quads read as triangle pairs). Measures:
  faces        triangles, quads, n-gons; quad share of the faces
  edges        non-manifold and open (boundary) edges
  valence      vertices by edge count; valence-4 share of the interior (closed-surface) vertices
  uv           islands (faces joined across an edge only where the UVs agree), stretch as the log2 of each
               face's texel scale against the median (median and 95th percentile; share beyond 2x), seam
               length (3D length of edges the UVs cut, over the square root of the surface area), seam edge share
  quadriflow   what bpy.ops.object.quadriflow_remesh would refuse (quad_remesh.quadriflow_check)
"""
import argparse
import json
import math
import os
import sys

import bpy
import bmesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quad_remesh  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
ext = os.path.splitext(a.mesh)[1].lower()
if ext == ".obj":
    bpy.ops.wm.obj_import(filepath=a.mesh)
elif ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=a.mesh)
else:
    raise SystemExit(f"unsupported mesh: {a.mesh}")
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
obj = max(meshes, key=lambda o: len(o.data.polygons))
bm = bmesh.new()
bm.from_mesh(obj.data)
split_before = len(bm.verts)
if ext in (".glb", ".gltf"):
    # glTF splits a vertex wherever its normal or UV changes: merge the copies back before counting
    # topology (the UVs stay per corner, so the seams are still measured below)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
merged_verts = split_before - len(bm.verts)
bm.faces.ensure_lookup_table()
bm.edges.ensure_lookup_table()

n_faces = len(bm.faces)
tris = sum(1 for f in bm.faces if len(f.verts) == 3)
quads = sum(1 for f in bm.faces if len(f.verts) == 4)
ngons = n_faces - tris - quads
nm = sum(1 for e in bm.edges if len(e.link_faces) > 2)
bnd_edges = [e for e in bm.edges if len(e.link_faces) == 1]
boundary_verts = {v for e in bnd_edges for v in e.verts}
interior = [v for v in bm.verts if v not in boundary_verts]
val_hist = {}
for v in bm.verts:
    k = len(v.link_edges)
    val_hist[k] = val_hist.get(k, 0) + 1
val4 = sum(1 for v in interior if len(v.link_edges) == 4)

result = {
    "mesh": os.path.basename(a.mesh), "merged_split_verts": merged_verts, "faces": n_faces, "triangles": tris, "quads": quads, "ngons": ngons,
    "quad_share": round(quads / max(1, n_faces), 4), "tri_share": round(tris / max(1, n_faces), 4),
    "verts": len(bm.verts), "edges": len(bm.edges), "nonmanifold_edges": nm, "boundary_edges": len(bnd_edges),
    "valence_hist": {str(k): v for k, v in sorted(val_hist.items())},
    "valence4_share_interior": round(val4 / max(1, len(interior)), 4),
    "quadriflow": quad_remesh.quadriflow_check(bm),
}

def corner_angles(f):
    """Interior angles of a face's corners, degrees."""
    vs = [v.co for v in f.verts]
    out = []
    for i in range(len(vs)):
        p0, p1, p2 = vs[i - 1], vs[i], vs[(i + 1) % len(vs)]
        u, w = (p0 - p1), (p2 - p1)
        if u.length < 1e-12 or w.length < 1e-12:
            out.append(0.0)
            continue
        c = max(-1.0, min(1.0, u.normalized().dot(w.normalized())))
        out.append(math.degrees(math.acos(c)))
    return out


def triangle_min_angles(f):
    """Minimum angle of each triangle once the face is triangulated the way a viewer would (quads split along
    the shorter diagonal) - the same triangle count model_quality.py measures."""
    if len(f.verts) == 3:
        return [min(corner_angles(f))]
    if len(f.verts) != 4:
        return []
    p = [v.co for v in f.verts]
    d02, d13 = (p[0] - p[2]).length, (p[1] - p[3]).length
    idx = [(0, 1, 2), (0, 2, 3)] if d02 <= d13 else [(0, 1, 3), (1, 2, 3)]
    out = []
    for t in idx:
        angs = []
        for k in range(3):
            a0, a1, a2 = p[t[k - 1]], p[t[k]], p[t[(k + 1) % 3]]
            u, w = a0 - a1, a2 - a1
            c = max(-1.0, min(1.0, u.normalized().dot(w.normalized()))) if u.length > 1e-12 and w.length > 1e-12 else 1.0
            angs.append(math.degrees(math.acos(c)))
        out.append(min(angs))
    return out


def quad_convexity(f):
    """'convex', 'concave' (one corner turns against the others), or 'bowtie' (two corners do), by the sign of
    each corner's turn about the quad's own normal."""
    p = [v.co for v in f.verts]
    n = (p[2] - p[0]).cross(p[3] - p[1])
    if n.length < 1e-14:
        return "degenerate"
    signs = []
    for i in range(4):
        a0, a1, a2 = p[i - 1], p[i], p[(i + 1) % 4]
        signs.append(1 if (a1 - a0).cross(a2 - a1).dot(n) > 0 else -1)
    pos = sum(1 for x in signs if x > 0)
    if pos in (0, 4):
        return "convex"
    return "concave" if pos in (1, 3) else "bowtie"


convexity = {"convex": 0, "concave": 0, "bowtie": 0, "degenerate": 0}
quad_dev, tri_total, skinny = [], 0, 0
for f in bm.faces:
    if len(f.verts) == 4:
        convexity[quad_convexity(f)] += 1
        quad_dev.append(max(abs(x - 90.0) for x in corner_angles(f)))
    for mn in triangle_min_angles(f):
        tri_total += 1
        skinny += 1 if mn < 10.0 else 0
quad_dev.sort()
result["shape"] = {
    "quad_corner_dev_median_deg": round(quad_dev[len(quad_dev) // 2], 2) if quad_dev else None,
    "quad_corner_dev_p95_deg": round(quad_dev[int(0.95 * (len(quad_dev) - 1))], 2) if quad_dev else None,
    "quad_share_dev_over_30deg": round(sum(1 for d in quad_dev if d > 30.0) / len(quad_dev), 4) if quad_dev else None,
    "skinny_tri_share_under_10deg": round(skinny / max(1, tri_total), 4),
    "quad_convexity": convexity,
}

uv_layer = bm.loops.layers.uv.active
if uv_layer is not None and n_faces:
    # per-face UV of each vertex; faces join across an edge when both corners agree in UV
    fuv = {}
    for f in bm.faces:
        fuv[f.index] = {l.vert.index: l[uv_layer].uv.copy() for l in f.loops}
    parent = list(range(n_faces))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def close(p, q):
        return (p - q).length < 1e-5

    seam_len, seam_edges, total_len = 0.0, 0, 0.0
    for e in bm.edges:
        total_len += e.calc_length()
        if len(e.link_faces) != 2:
            continue
        f0, f1 = e.link_faces
        a_, b_ = e.verts[0].index, e.verts[1].index
        u0, u1 = fuv[f0.index], fuv[f1.index]
        if close(u0[a_], u1[a_]) and close(u0[b_], u1[b_]):
            parent[find(f0.index)] = find(f1.index)
        else:
            seam_len += e.calc_length()
            seam_edges += 1
    islands = len({find(i) for i in range(n_faces)})

    total_area = sum(f.calc_area() for f in bm.faces)
    uv_total, scales, degenerate = 0.0, [], 0
    for f in bm.faces:
        area = f.calc_area()
        uvs = [l[uv_layer].uv for l in f.loops]
        uv_area = 0.0
        for i in range(len(uvs)):
            x0, y0 = uvs[i]
            x1, y1 = uvs[(i + 1) % len(uvs)]
            uv_area += x0 * y1 - x1 * y0
        uv_area = abs(uv_area) * 0.5
        uv_total += uv_area
        if area < 1e-14 or uv_area < 1e-14:
            degenerate += 1
            continue
        scales.append(math.sqrt(uv_area / area))
    med = sorted(scales)[len(scales) // 2] if scales else 1.0
    rel = [math.log2(s / med) for s in scales]
    absrel = sorted(abs(r) for r in rel)
    p95 = absrel[int(0.95 * (len(absrel) - 1))] if absrel else 0.0
    beyond = sum(1 for r in rel if abs(r) > 1.0) / max(1, len(rel))
    surf = math.sqrt(total_area) if total_area > 0 else 1.0
    result["uv"] = {
        "islands": islands, "islands_per_1k_faces": round(1000 * islands / max(1, n_faces), 2),
        "stretch_median_log2": round(sorted(absrel)[len(absrel) // 2] if absrel else 0.0, 4),
        "stretch_p95_log2": round(p95, 4), "share_faces_beyond_2x": round(beyond, 4),
        "seam_length_rel": round(seam_len / surf, 4), "seam_edge_share": round(seam_edges / max(1, len(bm.edges)), 4),
        "degenerate_uv_faces": degenerate, "uv_total_over_surface": round(uv_total / max(total_area, 1e-12), 4),
    }
else:
    result["uv"] = None

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as fh:
    json.dump(result, fh, indent=2)
print("[audit] " + json.dumps({k: result[k] for k in ("mesh", "faces", "quads", "quad_share", "valence4_share_interior")}))
