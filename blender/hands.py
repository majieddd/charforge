"""Put modelled hands on the arms that cut_hands.py left capped at the wrist.

    blender -b --python hands.py -- --mesh solid_cut.glb --spec hands_spec.json \
        --out solid_hands.glb --json hands.json

Each hand is blender/hand_model.py's distance field, sampled on a grid a hundred-odd voxels
across the hand, meshed with OpenVDB, placed in the frame cut_hands.py measured (mirrored for
the left hand) and unioned onto the arm with the exact boolean. The union leaves a crease where
the wrist stub enters the arm; the few rings around it are relaxed so bare skin runs smoothly
into the hand, and a cuff stays a cuff.

hands.json records every finger joint in the mesh's frame, the positions the rig builds the
finger bones on - the model knows exactly where its knuckles are, so nothing is estimated.
"""
import argparse
import json
import os
import sys

import bmesh
import bpy
import numpy as np
import openvdb as vdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_model  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--spec", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--json", required=True)
ap.add_argument("--res", type=int, default=150, help="grid voxels along the hand's length")
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
body = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
body.name = "body"
bpy.context.view_layer.objects.active = body
body.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
spec = json.load(open(a.spec))
out = {"frame": "mesh", "sides": {}}


def wrist_of(s):
    """The modelled wrist's half-width and half-thickness before the bulk: a person's, clipped."""
    b = float(s.get("bulk", 1.0))
    return (float(np.clip(s["wrist_radius"][0] / b, 0.11, 0.17)), float(np.clip(s["wrist_radius"][1] / b, 0.075, 0.115)))


def hand_mesh(side, s):
    L = s["length"]
    b = float(s.get("bulk", 1.0))                  # broader and thicker for a bulky bare forearm (cut_hands.py)
    wy, wz = wrist_of(s)
    vox = 1.0 / a.res
    xs = np.arange(-0.34, 1.12, vox)
    ys = np.arange(-0.30 * b, 0.62 * b, vox)
    zs = np.arange(-0.26 * b, 0.20 * b, vox)
    G = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    d, chains = hand_model.sdf(G * np.array([1.0, 1.0 / b, 1.0 / b]), wrist_r=(wy, wz), stub=0.30)
    d = d * b
    chains = {n: [np.asarray(p_) * np.array([1.0, b, b]) for p_ in pts_] for n, pts_ in chains.items()}
    # close the stub's far end inside the arm
    d = np.maximum(d, -(G[:, 0] + 0.30))
    d = d.reshape(len(xs), len(ys), len(zs)).astype(np.float32)
    grid = vdb.FloatGrid()
    grid.transform = vdb.createLinearTransform(voxelSize=vox)
    grid.gridClass = vdb.GridClass.LEVEL_SET
    grid.background = 0.1
    i0 = (int(round(xs[0] / vox)), int(round(ys[0] / vox)), int(round(zs[0] / vox)))
    grid.copyFromArray(np.clip(d, -0.1, 0.1), ijk=i0, tolerance=0.0)
    pts, tris, quads = grid.convertToPolygons(isovalue=0.0, adaptivity=0.0)
    pts = np.asarray(pts, np.float64)
    B = np.stack([s["x"], s["y"], s["z"]], axis=1) * L
    C = np.array(s["cut_point"])
    world = pts @ B.T + C
    me = bpy.data.meshes.new(f"{side}_hand")
    faces = [tuple(q) for q in np.asarray(quads).tolist()] + [tuple(t) for t in np.asarray(tris).tolist()]
    me.from_pydata(world.tolist(), [], faces)
    me.update()
    bm = bmesh.new()
    bm.from_mesh(me)
    if bm.calc_volume(signed=True) < 0:             # the mirrored basis turns it inside out
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(f"{side}_hand", me)
    bpy.context.scene.collection.objects.link(ob)
    joints = {n: [(np.asarray(p) @ B.T + C).tolist() for p in pts_] for n, pts_ in chains.items()}
    return ob, joints, L


def volume(ob):
    bm_ = bmesh.new()
    bm_.from_mesh(ob.data)
    v_ = abs(bm_.calc_volume(signed=True))
    bm_.free()
    return v_


def tips_on(ob, joints, L):
    """How many of the five fingertips lie on ob's surface (within 8% of the hand's length)."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(ob, dg)
    n = 0
    for chain in joints.values():
        hit = tree.find_nearest(Vector(chain[-1]))
        n += hit[0] is not None and hit[3] < 0.08 * L
    return n


def nonmanifold(ob):
    bm_ = bmesh.new()
    bm_.from_mesh(ob.data)
    n_ = sum(1 for e in bm_.edges if not e.is_manifold)
    bm_.free()
    return n_


def try_union(hand, joints, L, how):
    """One boolean union of hand onto body, kept only if it did what a union does. The hand's wrist stub reaches back
    inside the arm, so the volume must grow by much of the hand's (at least a quarter) and by no more than all of it,
    and the fingertips must end up on the surface. The old check - only that the volume did not shrink - passed a
    union that returned the body unchanged: Mara, Rivet, Rowan, Bo and Wren shipped without a right hand."""
    n_before, vol_body, vol_hand = len(body.data.vertices), volume(body), volume(hand)
    nm_before = nonmanifold(body)
    keep = body.data.copy()
    mod = body.modifiers.new("union", "BOOLEAN")
    mod.operation = "UNION"
    mod.solver = "EXACT"
    mod.use_hole_tolerant = True                    # the solid can carry a few non-manifold edges
    mod.use_self = how == "self"
    mod.object = hand
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier=mod.name)
    vol_after = volume(body)
    tips = tips_on(body, joints, L)
    nm_added = nonmanifold(body) - nm_before
    # and it must not tear the surface: with self-intersection handling on, Mara's union added 6,732 non-manifold edges
    ok = (vol_body + 0.25 * vol_hand <= vol_after <= vol_body + 1.05 * vol_hand
          and len(body.data.vertices) > 0.8 * n_before and tips >= 4 and nm_added < 300)
    if not ok:
        body.data = keep
    return ok, (f"volume +{(vol_after - vol_body) / max(vol_hand, 1e-12):.2f} of the hand's, "
                f"{tips}/5 fingertips on the surface, {nm_added:+d} non-manifold edges")


# Every hand is unioned onto the body as it came from the solid; a hand whose union fails is joined as its own shell
# only after all the unions. A boolean run on a body already carrying an overlapping shell is what returned Mara's
# body unchanged on her second hand.
from mathutils import Vector  # noqa: E402
deferred = []
for side, s in spec["sides"].items():
    hand, joints, L = hand_mesh(side, s)
    n_before = len(body.data.vertices)
    tried = []
    ok = False
    for how in ("exact", "nudged", "self"):
        if how == "nudged":                         # a coplanar or degenerate contact: move the hand a tenth of a millimetre
            hand.location += Vector((1e-4, -7e-5, 5e-5))
            bpy.context.view_layer.update()
        if how == "self":
            hand.location = (0.0, 0.0, 0.0)
            bpy.context.view_layer.update()
        ok, why = try_union(hand, joints, L, how)
        tried.append(f"{how}: {why}")
        if ok:
            break
    if ok:
        bpy.data.objects.remove(hand, do_unlink=True)
        how_txt = "unioned on (" + "; ".join(tried) + ")"
    else:
        hand.location = (0.0, 0.0, 0.0)
        deferred.append(hand)
        how_txt = "to be JOINED as a separate shell - every union failed (" + "; ".join(tried) + ")"
    out["sides"][side] = {"length": L, "wrist": s["cut_point"], "fingers": joints,
                          "frame": {"x": s["x"], "y": s["y"], "z": s["z"], "mirror": s["mirror"]},
                          "wrist_radius": list(wrist_of(s)), "bulk": float(s.get("bulk", 1.0)),
                          "joined": not ok}
    print(f"[hands] {side}: modelled hand {L:.4f} long {how_txt} ({n_before:,} -> {len(body.data.vertices):,} vertices)",
          flush=True)
for hand in deferred:
    bpy.ops.object.select_all(action="DESELECT")
    hand.select_set(True)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
# relax the crease where each stub meets the arm: vertices within a band of the cut plane
for side, s in spec["sides"].items():
    L = out["sides"][side]["length"]
    C, x = np.array(s["cut_point"]), np.array(s["x"])
    bm = bmesh.new()
    bm.from_mesh(body.data)
    band = [vv for vv in bm.verts
            if abs((np.array(vv.co) - C) @ x) < 0.10 * L and np.linalg.norm(np.array(vv.co) - C) < 0.45 * L]
    for _ in range(8):
        bmesh.ops.smooth_vert(bm, verts=band, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(body.data)
    bm.free()
    print(f"[hands] {side}: {len(band)} vertices relaxed at the wrist", flush=True)
# the last word: both hands' fingertips on the mesh that ships on, or the stage fails
for side in spec["sides"]:
    t_ = tips_on(body, out["sides"][side]["fingers"], out["sides"][side]["length"])
    if t_ < 4:
        raise SystemExit(f"[hands] {side}: only {t_}/5 fingertips on the final mesh - the hand is missing")

bm = bmesh.new()
bm.from_mesh(body.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
nm = sum(1 for e in bm.edges if not e.is_manifold)
# Every closed piece must face outward. A boolean run on a mesh that already carries a joined,
# overlapping hand shell turned Vex's whole body inside out (82% of normals pointing in), and
# every later stage - bake, projection, shading - quietly did the wrong thing with it.
bm.faces.ensure_lookup_table()
seen, flipped = set(), 0
for f0 in bm.faces:
    if f0.index in seen:
        continue
    island, stack = [], [f0]
    seen.add(f0.index)
    while stack:
        f = stack.pop()
        island.append(f)
        for e in f.edges:
            for g in e.link_faces:
                if g.index not in seen:
                    seen.add(g.index)
                    stack.append(g)
    vol = sum(f.calc_center_median().dot(f.normal) * f.calc_area() for f in island) / 3.0
    if vol < 0:
        bmesh.ops.reverse_faces(bm, faces=island)
        flipped += 1
if flipped:
    print(f"[hands] {flipped} inside-out piece(s) turned outward", flush=True)
bm.to_mesh(body.data)
bm.free()
for poly in body.data.polygons:                 # one smooth surface: glTF splits flat faces apart
    poly.use_smooth = True
print(f"[hands] non-manifold edges after union: {nm}", flush=True)
bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=True, export_materials="NONE")
json.dump(out, open(a.json, "w"), indent=1)
print(f"[hands] -> {a.out}, {a.json}", flush=True)
