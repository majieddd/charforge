"""The surface of a model as tools/model_quality.py measures it: triangles in metres, smooth normals, topology faults.

    blender -b -noaudio --python model_quality.py -- --mesh work/<name>/retopo.glb --height 1.75 --out q.npz
    blender -b -noaudio --python model_quality.py -- --blend baseline.blend --objects "GEO-body_male_stylized$" \
        --height 1.75 --out q.npz

The model is seen as it would be shipped or shown: modifiers applied at render level (a professional base
mesh is a subdivision cage; its multiresolution sculpt is its detail), joined, triangulated, and scaled so
its height is --height, standing on the floor. Writes an .npz with V (n,3) metres, F (m,3), VN (n,3) the
smooth vertex normals the viewer shades with, polygon sides of the source faces, and the topology counts
that need Blender: non-manifold and boundary edges, loose parts, degenerate faces, and pairs of faces that
cut through each other.
"""
import argparse
import re
import sys

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", help="a .glb/.gltf/.obj/.fbx to import")
ap.add_argument("--blend", help="a .blend to open instead")
ap.add_argument("--objects", default="", help="with --blend: regex of mesh object names to keep (default: all visible)")
ap.add_argument("--height", type=float, default=1.75, help="scale the model to this height (m)")
ap.add_argument("--out", required=True)
a = ap.parse_args(argv)

if a.blend:
    bpy.ops.wm.open_mainfile(filepath=a.blend)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = a.mesh.lower().rsplit(".", 1)[-1]
    {"glb": bpy.ops.import_scene.gltf, "gltf": bpy.ops.import_scene.gltf, "fbx": bpy.ops.import_scene.fbx,
     "obj": bpy.ops.wm.obj_import}[ext](filepath=a.mesh)
sc = bpy.context.scene
pat = re.compile(a.objects) if a.objects else None
objs = [o for o in (bpy.data.objects if pat else sc.objects) if o.type == "MESH" and (pat.search(o.name) if pat else o.visible_get())
        and not any(c.name == "glTF_not_exported" for c in o.users_collection)]
if not objs:
    raise SystemExit(f"[quality] no mesh objects matched {a.objects!r}")
vl = {o.name for o in bpy.context.view_layer.objects}
for o in objs:                       # an asset kept out of the view layer is never evaluated: link it in
    if o.name not in vl:
        sc.collection.objects.link(o)
        o.hide_viewport = o.hide_render = False
for o in objs:
    for m in o.modifiers:
        if m.type == "ARMATURE" and m.object is not None:     # the rest pose: the model as modelled
            m.object.data.pose_position = "REST"
        # subdivision as the model is meant to be seen, capped at two levels (a full multiresolution
        # sculpt runs to millions of faces; two levels keep its forms at about a centimetre)
        if m.type == "SUBSURF":
            m.levels = min(m.render_levels, 2)
        if m.type == "MULTIRES":
            m.levels = min(m.total_levels, 2)
dg = bpy.context.evaluated_depsgraph_get()
dg.update()

V, F, VN, sides, FO = [], [], [], [], []
off = 0
for oi, o in enumerate(objs):
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    mw = o.matrix_world
    nv = len(me.vertices)
    co = np.empty(nv * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co = (np.array(mw)[:3, :3] @ co.T).T + np.array(mw)[:3, 3]
    # the smooth normal each vertex shades with: corner normals averaged (custom split normals honoured)
    cn = np.empty(len(me.loops) * 3)
    me.corner_normals.foreach_get("vector", cn) if hasattr(me, "corner_normals") else me.loops.foreach_get("normal", cn)
    cn = cn.reshape(-1, 3)
    lv = np.empty(len(me.loops), int)
    me.loops.foreach_get("vertex_index", lv)
    vn = np.zeros((nv, 3))
    np.add.at(vn, lv, cn)
    vn = (np.array(mw.to_3x3().inverted().transposed()) @ vn.T).T
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    me.calc_loop_triangles()
    tri = np.empty(len(me.loop_triangles) * 3, int)
    me.loop_triangles.foreach_get("vertices", tri)
    tp = np.empty(len(me.loop_triangles), int)
    me.loop_triangles.foreach_get("polygon_index", tp)
    ps = np.array([p.loop_total for p in me.polygons])
    V.append(co); VN.append(vn); F.append(tri.reshape(-1, 3) + off); sides.append(ps[tp])
    FO.append(np.full(len(tp), oi))
    off += nv
    ev.to_mesh_clear()
V, F, VN, sides, FO = (np.concatenate(x) for x in (V, F, VN, sides, FO))

# standing on the floor, centred, at the asked height
lo, hi = V.min(0), V.max(0)
s = a.height / (hi[2] - lo[2])
V = (V - [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]) * s

# glTF splits vertices along UV seams and hard edges: weld coincident ones (to a micron), or every seam
# counts as a hole and faces across it as cutting each other
key = np.round(V * 1e6).astype(np.int64)
_, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
inv = inv.reshape(-1)
vn = np.zeros((len(first), 3))
np.add.at(vn, inv, VN)
VN = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
V, F = V[first], inv[F]
keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
F, sides, FO = F[keep], sides[keep], FO[keep]

# topology faults, on the joined, welded triangle mesh
bm = bmesh.new()
for x in V:
    bm.verts.new(x)
bm.verts.ensure_lookup_table()
bad = 0
for t in F:
    try:
        bm.faces.new([bm.verts[i] for i in t])
    except ValueError:              # a duplicate face
        bad += 1
bm.edges.ensure_lookup_table()
non_manifold = sum(1 for e in bm.edges if len(e.link_faces) > 2)
boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
tree = BVHTree.FromBMesh(bm)
pairs = tree.overlap(tree)
# faces that share a vertex touch by construction; only pairs with no shared vertex cut through each other,
# and only within one object - separate garments layered over each other are allowed to overlap unseen
fv = [set(v.index for v in f.verts) for f in bm.faces]
cuts = sum(1 for i, j in pairs if i < j and FO[i] == FO[j] and not (fv[i] & fv[j]))
area = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1) / 2
bm.free()
np.savez_compressed(a.out, V=V.astype(np.float32), F=F.astype(np.int32), VN=VN.astype(np.float32),
                    sides=sides.astype(np.int8), FO=FO.astype(np.int16), non_manifold=non_manifold, boundary=boundary,
                    duplicate=bad, self_cuts=cuts, degenerate=int((area < 1e-10).sum()), scale=s,
                    objects=np.array([o.name for o in objs]))
print(f"[quality] {len(objs)} objects, {len(V):,} vertices, {len(F):,} triangles ({(sides == 4).mean():.0%} from quads); "
      f"non-manifold {non_manifold}, boundary {boundary}, self-cuts {cuts} -> {a.out}", flush=True)
