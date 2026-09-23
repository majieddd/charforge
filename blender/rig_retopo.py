"""Rig the retopologised mesh instead of the raw generated one.

This is the architectural fix behind every deformation failure in this project. The pipeline
built its skeleton and skin weights on the marching-cubes surface TRELLIS emits, and that
surface is not deformable:

                      generated raw     retopologised
    open edges             2.30%            0.00%
    non-manifold           0.48%            0.00%
    zero-area faces        0.29%            0.00%

Everything that went wrong follows from that one choice. Blender's bone-heat weighting failed
outright (100% of vertices unweighted) because it solves a diffusion problem that needs a
manifold surface. QuadriFlow silently did nothing, for the same reason. Cloth diverged at
8,438% on the raw garment and only became stable once the garment was retopologised. And the
face tore apart, because segmentation split it across two objects that a non-manifold surface
gave no clean seam between.

The retopologiser already existed and already produces what is needed - watertight, all-quad,
with albedo and normal baked from the high-resolution surface. It was only ever wired into the
display path. This wires it into the rigging path.

Two deliberate departures from the old rig:

  skinning.  Bone heat was tried again here and refused again, this time with an explicit
      message - "Bone Heat Weighting: failed to find solution for one or more bones" - while
      still returning FINISHED with every vertex unweighted. Watertight geometry and a skeleton
      verified to sit inside it were not enough. So the weights are solved directly: inverse
      distance to each bone *segment*, then Laplacian smoothing across the welded surface. The
      smoothing is the part that could never work before - the old mesh's surface graph was
      shredded into thousands of patches by glTF's seam splitting, so smoothing ran inside each
      patch and could not cross the boundaries that needed softening.
  the skin stays one mesh.  Previously body, hair and accessory were separate objects with
      different binding, and the face was split between two of them; under motion they drifted
      apart and the face opened up. Here everything that is not clothing stays in a single
      skinned mesh, so there is no seam to come apart. Only the garment is separated, because
      only the garment needs a cloth solver.

Run: python transfer_labels.py --retopo retopo.glb --source trellis_mesh.glb \
         --parts parts.json --out retopo_labels.json
     blender -b -noaudio --python rig_retopo.py -- --mesh retopo.glb \
         --labels retopo_labels.json --joints joints.json --out rig.blend
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True, help="retopologised GLB")
ap.add_argument("--labels", required=True,
                help="retopo_labels.json from transfer_labels.py")
ap.add_argument("--joints", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--max-influences", type=int, default=4)
a = ap.parse_args(argv)

# ---- import the clean mesh --------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=a.mesh)
added = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
if not added:
    raise SystemExit("[rig] no mesh imported")
if len(added) > 1:
    bpy.ops.object.select_all(action="DESELECT")
    for o in added:
        o.select_set(True)
    bpy.context.view_layer.objects.active = added[0]
    bpy.ops.object.join()
mesh = bpy.context.view_layer.objects.active or added[0]
mesh.name = "char"
scene = bpy.context.scene
print(f"[rig] retopo mesh: {len(mesh.data.vertices):,} verts, "
      f"{len(mesh.data.polygons):,} faces", flush=True)

# ---- normalise to the frame joints.json lives in ------------------------------------------------
# skeleton.py computes joint positions from orbit renders, which normalise the character to two
# units tall centred on the origin. The GLB is roughly half that, so an un-normalised mesh puts
# the armature 1.87x too large and entirely outside the body - and bone heat, having no mesh
# anywhere near a bone, returns success with every vertex unweighted. That is the same silent
# failure that made the original pipeline abandon bone heat in the first place; the solver was
# never the problem, the alignment was.
bpy.ops.object.select_all(action="DESELECT")
mesh.select_set(True)
bpy.context.view_layer.objects.active = mesh
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
vs = [v.co for v in mesh.data.vertices]
mn = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
mx = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
ctr = (mn + mx) / 2
sc_ = 2.0 / max(1e-6, mx.z - mn.z)
for v in mesh.data.vertices:
    v.co = (v.co - ctr) * sc_
mesh.data.update()
print(f"[rig] normalised: centre {tuple(round(x,3) for x in ctr)} scale {sc_:.4f} "
      f"-> height {(mx.z-mn.z)*sc_:.3f}", flush=True)

# ---- part labels, precomputed outside Blender --------------------------------------------------
# Blender's bundled Python has no trimesh or scipy, and the transfer needs a spatial query over
# 170k source vertices plus a Lab colour model. transfer_labels.py does both in under a second
# in the project venv and writes the result here.
L = json.load(open(a.labels))
dst_lab = np.array(L["labels"])
gid = L["group_ids"]
counts = L["counts"]
moved = L.get("reclassified_from_hair", 0)
if len(dst_lab) != len(mesh.data.vertices):
    raise SystemExit(f"[rig] label count {len(dst_lab)} != mesh vertices "
                     f"{len(mesh.data.vertices)}; transfer_labels.py was run against a "
                     "different mesh")
print(f"[rig] labels: {counts} ({moved:,} reclassified out of hair)", flush=True)

# ---- armature from the existing joints ----------------------------------------------------------
J = json.load(open(a.joints))
joints, parents = J["joints"], J["parents"]
bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
rig = bpy.context.view_layer.objects.active
rig.name = "rig"
arm = rig.data
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
for b in list(arm.edit_bones):
    arm.edit_bones.remove(b)
eb = {}
for name, co in joints.items():
    b = arm.edit_bones.new(name)
    b.head = Vector(co)
    b.tail = Vector(co) + Vector((0, 0, 0.05))
    eb[name] = b
children = defaultdict(list)
for name, par in parents.items():
    if par:
        children[par].append(name)
for name, b in eb.items():
    kids = children.get(name, [])
    if kids:
        b.tail = Vector(joints[kids[0]])
    elif parents.get(name):
        d = (Vector(joints[name]) - Vector(joints[parents[name]]))
        b.tail = Vector(joints[name]) + (d.normalized() * max(d.length * 0.4, 0.02)
                                         if d.length > 1e-6 else Vector((0, 0, 0.05)))
    if (b.tail - b.head).length < 1e-4:
        b.tail = b.head + Vector((0, 0, 0.02))
for name, par in parents.items():
    if par and par in eb and name in eb:
        eb[name].parent = eb[par]
bpy.ops.object.mode_set(mode="OBJECT")
vs = [mesh.matrix_world @ v.co for v in mesh.data.vertices]
mlo = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
mhi = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
jp = [Vector(c) for c in joints.values()]
inside = sum(1 for p in jp if all(mlo[i] - 0.05 <= p[i] <= mhi[i] + 0.05 for i in range(3)))
print(f"[rig] armature: {len(arm.bones)} bones, {inside}/{len(jp)} joints inside the mesh bounds",
      flush=True)
if inside < len(jp) * 0.9:
    raise SystemExit(
        f"[rig] only {inside}/{len(jp)} joints fall inside the mesh "
        f"(mesh z {mlo.z:.2f}..{mhi.z:.2f}, joints z "
        f"{min(p.z for p in jp):.2f}..{max(p.z for p in jp):.2f}). Bone heat would return "
        "success with everything unweighted; refusing to rig a misaligned skeleton.")

# ---- split clothing off; everything else stays one skinned mesh --------------------------------
cloth_id = gid.get("clothing")
import bmesh  # noqa: E402

bpy.context.view_layer.objects.active = mesh
bpy.ops.object.mode_set(mode="EDIT")
bm = bmesh.from_edit_mesh(mesh.data)
bm.verts.ensure_lookup_table()
for f in bm.faces:
    f.select = all(dst_lab[v.index] == cloth_id for v in f.verts)
n_cloth_faces = sum(1 for f in bm.faces if f.select)
bmesh.update_edit_mesh(mesh.data)
before_objs = set(bpy.data.objects)
if n_cloth_faces:
    bpy.ops.mesh.separate(type="SELECTED")
bpy.ops.object.mode_set(mode="OBJECT")
new = [o for o in bpy.data.objects if o not in before_objs]
garment = new[0] if new else None
if garment:
    garment.name = "char_clothing"
mesh.name = "char_skin"
print(f"[rig] split: skin {len(mesh.data.vertices):,} verts, "
      f"clothing {len(garment.data.vertices) if garment else 0:,} verts", flush=True)

def bone_segments(rig_obj):
    out = []
    for b in rig_obj.data.bones:
        if not b.use_deform:
            continue
        out.append((b.name, np.array((rig_obj.matrix_world @ b.head_local)[:]),
                    np.array((rig_obj.matrix_world @ b.tail_local)[:])))
    return out


def proximity_weights(ob, rig_obj, power=4.0, k=4, iters=18, lam=0.7):
    """Inverse-distance-to-segment weights, then smoothing over the welded surface."""
    segs = bone_segments(rig_obj)
    names = [s[0] for s in segs]
    V = np.array([(ob.matrix_world @ v.co)[:] for v in ob.data.vertices])
    n = len(V)
    D = np.empty((n, len(segs)), np.float64)
    for c, (_, h, t) in enumerate(segs):
        ab = t - h
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            D[:, c] = np.linalg.norm(V - h, axis=1)
            continue
        u = np.clip(((V - h) @ ab) / L2, 0.0, 1.0)[:, None]
        D[:, c] = np.linalg.norm(V - (h + u * ab), axis=1)
    D = np.maximum(D, 1e-5)

    order = np.argsort(D, axis=1)[:, :k]
    Wm = np.zeros((n, len(segs)))
    rows = np.arange(n)[:, None]
    w = 1.0 / D[rows, order] ** power
    Wm[rows, order] = w / w.sum(1, keepdims=True)

    # welded adjacency, so smoothing crosses UV seams
    scale = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    keys = np.round(V / (1e-5 * scale)).astype(np.int64)
    adj = [set() for _ in range(n)]
    for e in ob.data.edges:
        i, j = e.vertices
        adj[i].add(j)
        adj[j].add(i)
    bucket = defaultdict(list)
    for i in range(n):
        bucket[tuple(keys[i])].append(i)
    welded = 0
    for members in bucket.values():
        if len(members) < 2:
            continue
        union = set()
        for m in members:
            union |= adj[m]
        for m in members:
            adj[m] = union
            welded += 1
    nbr = [np.fromiter(adj[i], dtype=np.int64) for i in range(n)]
    for _ in range(iters):
        acc = np.empty_like(Wm)
        for i in range(n):
            nb = nbr[i]
            acc[i] = Wm[nb].mean(0) if len(nb) else Wm[i]
        Wm = (1.0 - lam) * Wm + lam * acc

    # ---- rigid feet ----------------------------------------------------------------------------
    # A shoe does not bend where a shin does. The diffusion above carries the shin's weight down
    # into the shoe: measured on the first two characters, no sole vertex had a dominant bone
    # holding even half its weight, and the shin (the `knee` bone) held a share of 1,140 of them.
    # So every ankle flex bent the sole like rubber - visible in the walk, and fatal to solving
    # floor contact, because a sole that changes shape with the ankle has no fixed point to plant.
    # Below the ankle joint, whatever the foot's own bones do not hold is handed to the foot,
    # blending over 5 cm up through the joint, so a boot shaft still follows the shin.
    heads = {nm: h for nm, h, _ in segs}
    idx = {nm: c for c, nm in enumerate(names)}
    u = float(V[:, 2].max() - V[:, 2].min()) / 1.75   # this stage works at 2 units tall: metres -> units
    for side, other in (("left", "right"), ("right", "left")):
        ank, toe = f"{side}_ankle", f"{side}_foot"
        if ank not in idx:
            continue
        pa, po = heads[ank], heads.get(f"{other}_ankle")
        dxy = np.linalg.norm(V[:, :2] - pa[:2], axis=1)
        near = dxy < 0.30 * u
        if po is not None:
            near &= dxy < np.linalg.norm(V[:, :2] - po[:2], axis=1)
        t = np.clip((pa[2] + 0.04 * u - V[:, 2]) / (0.05 * u), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        t[~near] = 0.0
        own = np.zeros(len(names), bool)
        own[idx[ank]] = True
        if toe in idx:
            own[idx[toe]] = True
        moved = t * Wm[:, ~own].sum(1)
        Wm[:, ~own] *= (1.0 - t)[:, None]
        Wm[:, idx[ank]] += moved
        print(f"[rig] {ob.name}: {side} foot rigid below the ankle - {int((t > 0.999).sum()):,} "
              f"vertices fully on the foot, {int(((t > 0) & (t <= 0.999)).sum()):,} blending", flush=True)

    # keep the k strongest, renormalise
    order = np.argsort(-Wm, axis=1)[:, :k]
    keep = np.zeros_like(Wm)
    keep[rows, order] = Wm[rows, order]
    ssum = keep.sum(1, keepdims=True)
    keep = np.divide(keep, np.maximum(ssum, 1e-9))

    ob.vertex_groups.clear()
    vgs = [ob.vertex_groups.new(name=nm) for nm in names]
    for c, vg in enumerate(vgs):
        idx = np.where(keep[:, c] > 1e-4)[0]
        for i in idx:
            vg.add([int(i)], float(keep[i, c]), "REPLACE")
    return welded, int((keep.sum(1) < 1e-6).sum())


# ---- skinning -------------------------------------------------------------------------------------
report = {"retopo_vertices": len(mesh.data.vertices) + (len(garment.data.vertices) if garment else 0),
          "labels": counts, "reclassified_from_hair": moved,
          "bones": len(arm.bones), "parts": {}}
for ob in [o for o in (mesh, garment) if o is not None]:
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    ob.vertex_groups.clear()
    bpy.ops.object.parent_set(type="ARMATURE", keep_transform=True)
    welded, unw = proximity_weights(ob, rig, k=a.max_influences)
    total = len(ob.data.vertices)
    weighted = total - unw
    frac = unw / max(total, 1)
    print(f"[rig] {ob.name}: {weighted:,}/{total:,} weighted ({frac:.2%} unweighted), "
          f"{len(ob.vertex_groups)} groups, {welded:,} welded vertices in the smoothing graph",
          flush=True)
    if frac > 0.005:
        raise SystemExit(f"[rig] {ob.name}: {frac:.1%} unweighted")
    report["parts"][ob.name] = {"vertices": total, "unweighted": unw,
                                "groups": len(ob.vertex_groups)}

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump(report, open(a.json, "w"), indent=2)
print(f"[rig] {json.dumps({k: v for k, v in report.items() if k != 'labels'})}", flush=True)
print(f"[rig] -> {a.out}", flush=True)
