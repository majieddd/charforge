"""Quad-dominant topology for the game mesh (retopo.py --quads; lane topology, E135 follow-up).

Joining: bmesh.ops.join_triangles (Blender's Tris to Quads) merges neighbouring triangle pairs of the
decimated cage into quads where the pair is near-flat and the quad near-square. Nothing else changes:
the same vertices, the same surface, so the bake, UVs, weights and rig stages all follow unchanged.
Measured on pip, vex and mara in REPORT.md; the fraction of pairs it can join sets the quad share.

QuadriFlow check: bpy.ops.object.quadriflow_remesh (the built-in remesher) refuses a mesh with folded
faces, though each triangle is consistently wound. quadriflow_check() reports what it would refuse: the
folded quads and the edges whose two faces turn back on each other. E135's solid and voxel remeshes of it at
every size tried (0.4-1.2% of the height: 113-430 folded quads each) carry such folds, so QuadriFlow stays off
the table for this solid.
"""
import math

import bmesh


def join_quads(bm, face_angle=60.0, shape_angle=60.0):
    """Merge triangle pairs into quads (in place). Angles in degrees: the largest bend across the shared edge
    (face_angle) and the largest corner deviation from square (shape_angle). Blender's Tris to Quads defaults
    are 40; 60/60 joins 56% of the faces into quads on the three test characters (E118), 40/40 only 36%.
    Returns the number of quads made."""
    tris = [f for f in bm.faces if len(f.verts) == 3]
    res = bmesh.ops.join_triangles(
        bm, faces=tris, cmp_seam=False, cmp_sharp=False, cmp_uvs=False,
        cmp_vcols=False, cmp_materials=False,
        angle_face_threshold=math.radians(face_angle),
        angle_shape_threshold=math.radians(shape_angle))
    return sum(1 for f in res["faces"] if len(f.verts) == 4)


def polygon_counts(bm):
    counts = {3: 0, 4: 0, "n": 0}
    for f in bm.faces:
        k = len(f.verts)
        counts[k if k in (3, 4) else "n"] += 1
    return counts


def quadriflow_check(bm):
    """What QuadriFlow would refuse: non-manifold and open edges, folded quads (a quad whose two
    triangles turn against its own normal) and edges whose faces turn back more than 120 degrees."""
    bm.faces.ensure_lookup_table()
    nm = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    bd = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    folded = 0
    for f in bm.faces:
        if len(f.verts) != 4:
            continue
        p = [v.co for v in f.verts]
        n = (p[2] - p[0]).cross(p[3] - p[1])
        if n.length < 1e-14:
            continue
        n.normalize()
        t1 = (p[1] - p[0]).cross(p[2] - p[0])
        t2 = (p[2] - p[0]).cross(p[3] - p[0])
        if min(t1.normalized().dot(n), t2.normalized().dot(n)) < 0:
            folded += 1
    turned = sum(1 for e in bm.edges
                 if len(e.link_faces) == 2 and e.link_faces[0].normal.dot(e.link_faces[1].normal) < -0.5)
    return {"non_manifold_edges": nm, "boundary_edges": bd, "folded_quads": folded,
            "edges_turned_over_120deg": turned}
