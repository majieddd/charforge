"""E155: a Pixal3D single-view mesh brought into pass1's convention, before it is judged or used as pass1.glb.

    /Applications/Blender.app/Contents/MacOS/Blender -b -noaudio --python blender/fix_pixal3d.py -- \
        --in work/<n>/pixal3d.glb --out work/<n>/pixal3d_fixed.glb [--turn 180] [--faces 1000000] [--height 0.985]

Pixal3D's single view leaves four things TRELLIS's pass1 does not have:
  1. The figure may face away from the camera our gates use (azimuth 0): it is turned --turn degrees about the
     vertical, through its vertices (mesh.transform: the glTF object is in quaternion rotation mode, so setting
     rotation_euler and applying it changes nothing - the bug behind E155's first two judgments). Which way it
     faces is measured by pipeline/pixal3d_generate.py (DWPose's face on both sides), not assumed.
  2. A floor sheet: a thin plane tilted about 22 degrees under the feet (Hana: 0.63 wide, from the floor at the front
     up to the shins at the back), which the boots stand in. Found in two steps. (a) The big connected parts that lie
     in the bottom quarter of the height (--floor-band), span over half the figure's width (--floor-width) and have
     over 2% of the vertices (--floor-share) are removed; the sheet's fragments are the figure's own parts' size, so
     a part test alone leaves thousands of strips behind. (b) Where a sheet was found, the plane through what was
     removed (least squares) takes the rest: vertices within --plane-eps of it, and the bottom-quarter vertices outside
     each foot's box (the two legs just above the sheet, 2-means on x, widened by --foot-margin of the height), are
     removed. The boots are kept (checked by eye and in the density plots). Without a sheet nothing else is cut.
  3. Size: 5.4M faces where pass1 has about 1M. Joined (4), then decimated (quadric, UV islands delimited) to --faces when larger.
  4. Welding: the import splits the closed surface at every UV and normal seam, and a decimation of the split copies
     leaves cracks (E155 v3 first attempt: half the edges open, 25,000 pieces). The vertices closer than --weld are
     joined after the floor is taken out (the sheet touches the boots: joined before, the parts would merge) and
     before the decimation, so the decimated surface stays one closed piece.
  5. Placement: scaled so its height is --height (pass1 stands 0.978-1.000 high, 0.989 on average, over the six
     characters) and its bounding box centred on the origin, as pass1 is.
Writes the .glb with its material as Blender reads it. Prints one [fix] line with what was done.
"""
import argparse
import math
import sys

import bmesh
import bpy
import mathutils
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--turn", type=float, default=180.0, help="degrees about the vertical axis")
ap.add_argument("--faces", type=int, default=1_000_000, help="decimate to about this many faces (0: keep)")
ap.add_argument("--height", type=float, default=0.99, help="the height the model is scaled to")
ap.add_argument("--floor-band", type=float, default=0.25, help="a floor part's top, as a share of the height")
ap.add_argument("--floor-width", type=float, default=0.5, help="a floor part's width, as a share of the figure's")
ap.add_argument("--floor-share", type=float, default=0.02, help="a floor part's vertices, as a share of all")
ap.add_argument("--foot-margin", type=float, default=0.06, help="each foot's box, as a share of the height")
ap.add_argument("--plane-eps", type=float, default=0.004, help="a vertex this close to the sheet's plane is sheet")
ap.add_argument("--weld", type=float, default=1e-6, help="join vertices this close before decimating (0: keep the glTF's split vertices): the raw output is one closed surface, but the import splits it at every UV and normal seam, and a decimation of the split copies leaves cracks (E155 v3: half the edges open)")
a = ap.parse_args(argv)


def label_parts(n, ea, eb):
    """Connected parts of a graph by min-label propagation with pointer jumping (numpy only: Blender has no scipy).
    Returns each vertex's root label."""
    lab = np.arange(n, dtype=np.int64)
    while True:
        la, lb = lab[ea], lab[eb]
        lo = np.minimum(la, lb)
        new = lab.copy()
        np.minimum.at(new, la, lo)
        np.minimum.at(new, lb, lo)
        while True:
            nn = new[new]
            if np.array_equal(nn, new):
                break
            new = nn
        if np.array_equal(new, lab):
            return new
        lab = new


def vertices(me):
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def faces(me):
    n = len(me.polygons)
    ls = np.empty(n, np.int32)
    me.polygons.foreach_get("loop_start", ls)
    vi = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", vi)
    lt = np.empty(n, np.int32)
    me.polygons.foreach_get("loop_total", lt)
    assert (lt == 3).all(), "triangulate first: the parts are measured on triangles"
    return vi[ls[:, None] + np.arange(3)].astype(np.int64)


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.inp)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
assert meshes, f"no mesh in {a.inp}"
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
o = bpy.context.view_layer.objects.active
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
me = o.data
if a.turn:
    # the vertices themselves: the glTF importer leaves the object in quaternion rotation mode, so rotation_euler is
    # ignored and transform_apply changes nothing (E155 v2 judged the raw output, facing away, as the turned one)
    me.transform(mathutils.Matrix.Rotation(math.radians(a.turn), 4, "Z"))
me.update()
if any(len(p.vertices) != 3 for p in me.polygons):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()

V = vertices(me)
F = faces(me)
n_v, n_f0 = len(V), len(F)
lo_all, hi_all = V.min(0), V.max(0)
H, W = hi_all[2] - lo_all[2], hi_all[0] - lo_all[0]          # Blender's Z is the glTF Y: vertical
ea = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
eb = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
root = label_parts(n_v, ea, eb)
_, part = np.unique(root, return_inverse=True)
K = int(part.max()) + 1
cnt = np.bincount(part, minlength=K)
plo = np.full((K, 3), np.inf)
phi = np.full((K, 3), -np.inf)
np.minimum.at(plo, part, V)
np.maximum.at(phi, part, V)
top = (phi[:, 2] - lo_all[2]) / H
wide = (phi[:, 0] - plo[:, 0]) / W
floor = (top <= a.floor_band) & (wide >= a.floor_width) & (cnt >= a.floor_share * n_v)
floor_v = floor[part]
removed_parts = int(floor.sum())
drop_plane = drop_box = 0
drop_v = np.zeros(n_v, bool)
if removed_parts:
    # the sheet's plane: least squares through the parts taken out (Blender's Z is the vertical; the normal points up)
    P = V[floor_v]
    c0 = P.mean(0)
    _, _, Vt = np.linalg.svd(P[:: max(1, len(P) // 20000)] - c0, full_matrices=False)
    nrm = Vt[-1] if Vt[-1][2] >= 0 else -Vt[-1]
    hs = (V[:, 2] - lo_all[2]) / H
    top_share = float(hs[floor_v].max())
    # the two legs just above the sheet, split by their x; a box round each foot (its boot is wider than its leg)
    leg = (~floor_v) & (hs >= top_share + 0.02) & (hs <= top_share + 0.14)
    x = V[leg, 0]
    if leg.sum() >= 100:
        c = np.array([np.percentile(x, 25), np.percentile(x, 75)])
        for _ in range(20):
            k = np.abs(x[:, None] - c[None, :]).argmin(1)
            c = np.array([x[k == j].mean() if (k == j).any() else c[j] for j in (0, 1)])
        m_abs = a.foot_margin * H
        inside = np.zeros(n_v, bool)
        for j in (0, 1):
            sel = leg.copy()
            sel[leg] = k == j
            if not sel.any():
                continue
            inside |= ((V[:, 0] >= V[sel, 0].min() - m_abs) & (V[:, 0] <= V[sel, 0].max() + m_abs) &
                       (V[:, 1] >= V[sel, 1].min() - m_abs) & (V[:, 1] <= V[sel, 1].max() + m_abs))
        low = (~floor_v) & (hs < a.floor_band)
        off_plane = np.abs((V - c0) @ nrm) < a.plane_eps * H
        drop_plane = int((low & off_plane).sum())
        drop_box = int((low & ~inside).sum())
        drop_v = low & (off_plane | ~inside)
kill_v = floor_v | drop_v
kill_f = kill_v[F].any(1)
removed_faces = int(kill_f.sum())
removed_verts = int(kill_v.sum())

bm = bmesh.new()
bm.from_mesh(me)
bm.faces.ensure_lookup_table()
kill = [bm.faces[i] for i in np.nonzero(kill_f)[0]]
bmesh.ops.delete(bm, geom=kill, context="FACES")
loose = [v for v in bm.verts if not v.link_faces]
bmesh.ops.delete(bm, geom=loose, context="VERTS")
bm.to_mesh(me)
bm.free()
me.update()
# joined after the floor is taken out: the sheet touches the boots, and joined before, it would be one part with them
if a.weld > 0:
    bm = bmesh.new()
    bm.from_mesh(me)
    n0 = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=a.weld)
    bm.to_mesh(me)
    bm.free()
    me.update()
    welded = n0 - len(me.vertices)
else:
    welded = 0
n_f1 = len(me.polygons)

decimated = None
if a.faces and n_f1 > 1.2 * a.faces:
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    mod = o.modifiers.new("e155_decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = a.faces / n_f1
    mod.delimit = {"UV"}
    bpy.ops.object.modifier_apply(modifier=mod.name)
    decimated = len(o.data.polygons)

me = o.data
V2 = vertices(me)
lo2, hi2 = V2.min(0), V2.max(0)
s = a.height / (hi2[2] - lo2[2])
centre = (lo2 + hi2) / 2
co = np.empty(len(me.vertices) * 3, np.float32)
me.vertices.foreach_get("co", co)
co = (co.reshape(-1, 3).astype(np.float64) - centre) * s
me.vertices.foreach_set("co", co.astype(np.float32).ravel())
me.update()

bpy.ops.object.select_all(action="DESELECT")
o.select_set(True)
bpy.context.view_layer.objects.active = o
bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=True)
print(f"[fix] turned {a.turn:g} deg; welded {welded:,} vertices; floor parts {removed_parts} (tops {'-' if not removed_parts else f'{top[floor].max():.3f}'} "
      f"of the height); sheet plane: {drop_plane:,} vertices on it, {drop_box:,} outside the feet; "
      f"removed {removed_faces:,} faces, {removed_verts:,} vertices; faces {n_f0:,} -> {n_f1:,}"
      f"{'' if decimated is None else f' -> {decimated:,} decimated'}; scaled x{s:.4f} to height {a.height} -> {a.out}",
      flush=True)
