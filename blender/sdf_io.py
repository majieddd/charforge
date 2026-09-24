"""Mesh <-> signed distance volume, through the OpenVDB that ships inside Blender.

    to-sdf:   blender -b --python sdf_io.py -- to-sdf  --mesh mesh.glb --out sdf.npz [--voxels 640]
    to-mesh:  blender -b --python sdf_io.py -- to-mesh --sdf solid.npz --out solid.glb

Generated meshes are double-walled, leaky and non-manifold, so anything that asks "what is
inside" of the triangles directly gets it wrong. OpenVDB's mesh-to-level-set does not care: it
measures unsigned distance and flood-fills the outside, which is the one operation that is
robust on this input (it is what Blender's voxel remesh uses). The volume is handed to numpy as
a dense array so the cleanup can use scipy's morphology, and handed back to OpenVDB to mesh.

Axes: the array is indexed [i, j, k] = Blender world x, y, z (Z up) divided by the voxel size,
starting at `origin_ijk`. Values are signed distance in world units, negative inside.
"""
import argparse
import os
import sys

import bpy
import numpy as np
import openvdb as vdb

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("to-sdf", "to-mesh"))
ap.add_argument("--mesh")
ap.add_argument("--sdf")
ap.add_argument("--out", required=True)
ap.add_argument("--voxels", type=int, default=640, help="voxels across the character's height")
ap.add_argument("--band", type=float, default=0.03, help="narrow band either side, as a fraction of height")
ap.add_argument("--project", default=None,
                help="to-mesh: put each vertex back on this source mesh where it is within --reach")
ap.add_argument("--reach", type=float, default=1.5, help="projection reach, in voxels")
a = ap.parse_args(argv)


def import_joined(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    ms = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    pts, tris, base = [], [], 0
    for o in ms:
        dg = bpy.context.evaluated_depsgraph_get()
        me = o.evaluated_get(dg).to_mesh()
        me.calc_loop_triangles()
        V = np.empty(len(me.vertices) * 3, np.float64)
        me.vertices.foreach_get("co", V)
        M = np.array(o.matrix_world)
        V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        T = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", T)
        pts.append(V)
        tris.append(T.reshape(-1, 3) + base)
        base += len(V)
    return np.vstack(pts), np.vstack(tris)


if a.mode == "to-sdf":
    P, T = import_joined(a.mesh)
    H = float(np.ptp(P[:, 2]))
    v = H / a.voxels
    band = max(3.0, a.band * H / v)
    xf = vdb.createLinearTransform(voxelSize=v)
    grid = vdb.FloatGrid.createLevelSetFromPolygons(P.astype(np.float32), triangles=T.astype(np.uint32),
                                                    transform=xf, exBandWidth=band, inBandWidth=band)
    lo, hi = grid.evalActiveVoxelBoundingBox()
    lo = tuple(int(x) - 2 for x in lo)
    hi = tuple(int(x) + 2 for x in hi)
    shape = tuple(h - l + 1 for l, h in zip(lo, hi))
    arr = np.full(shape, grid.background, np.float32)
    grid.copyToArray(arr, ijk=lo)
    # copyToArray fills tiles outside the band with +/-background already; the sign of the
    # background tiles is what flood fill decided
    np.savez_compressed(a.out, sdf=arr, origin_ijk=np.array(lo), voxel=v, background=grid.background,
                        points=P.astype(np.float32), height=H)
    print(f"[sdf] {len(T):,} triangles -> {shape} voxels of {v*1000:.2f} mm-units, band {band:.1f} voxels "
          f"({arr.nbytes/1e6:.0f} MB) -> {a.out}", flush=True)
else:
    d = np.load(a.sdf)
    arr = d["sdf"].astype(np.float32)
    v = float(d["voxel"])
    lo = tuple(int(x) for x in d["origin_ijk"])
    grid = vdb.FloatGrid()
    grid.transform = vdb.createLinearTransform(voxelSize=v)
    grid.gridClass = vdb.GridClass.LEVEL_SET
    grid.background = float(np.abs(arr).max())
    grid.copyFromArray(arr, ijk=lo, tolerance=0.0)
    pts, tris, quads = grid.convertToPolygons(isovalue=0.0, adaptivity=0.0)
    pts = np.asarray(pts, np.float64)
    if a.project:
        # The mesher's surface is quantised to the voxels. Where the source surface is within
        # reach, the vertex goes back onto it, which restores the sub-voxel detail (lips,
        # eyelids, hair grooves); where it is not - a filled leak, a closed hair gap - the
        # vertex stays where the solid put it.
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        sp, st = import_joined(a.project)
        bvh = BVHTree.FromPolygons(sp.tolist(), st.tolist())
        reach = a.reach * v
        moved = 0
        for i in range(len(pts)):
            hit = bvh.find_nearest(Vector(pts[i]), reach)
            if hit[0] is not None:
                pts[i] = hit[0][:]
                moved += 1
        print(f"[sdf] {moved:,} of {len(pts):,} vertices put back on the source surface "
              f"(within {a.reach:.1f} voxels)", flush=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new("solid")
    faces = [tuple(int(i) for i in q) for q in quads] + [tuple(int(i) for i in t) for t in tris]
    me.from_pydata(pts.tolist(), [], faces)
    me.update()
    ob = bpy.data.objects.new("solid", me)
    bpy.context.scene.collection.objects.link(ob)
    # Marching output winds consistently, so only the global orientation can be wrong: test it
    # with the signed volume rather than recalc_face_normals, whose connectivity heuristic
    # flips patches on meshes this size.
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    vol = bm.calc_volume(signed=True)
    if vol < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(me)
    bm.free()
    print(f"[sdf] signed volume {vol:+.5f}{' - flipped outward' if vol < 0 else ''}", flush=True)
    for p in me.polygons:
        p.use_smooth = True
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    ext = os.path.splitext(a.out)[1].lower()
    if ext == ".blend":
        bpy.ops.wm.save_as_mainfile(filepath=a.out)
    else:
        bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=True,
                                  export_materials="NONE")
    print(f"[sdf] {len(pts):,} vertices, {len(quads):,} quads + {len(tris):,} triangles -> {a.out}", flush=True)
