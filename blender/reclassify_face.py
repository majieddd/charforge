"""Put the face back in the body mesh.

The part segmentation labels 970 of 7,332 hair vertices (13.2%) as hair when their albedo is
plainly skin - pieces of cheek, jaw and forehead. split_parts then moves that geometry into its
own object and binds it rigidly, while the face it was cut from stays skinned. At rest the two
surfaces are coincident, which is why every rest-pose check in this project passed. Under any
motion the rigid dark fragments stop matching the skinned face and poke through it: that is the
tearing, and no amount of weight repair fixes it, because the geometry is duplicated across two
objects with different textures and different binding.

This classifies the hair mesh by appearance and moves the misclassified faces back.

  colour model   skin and hair are separated in Lab, not RGB: the discriminating axis is a*
                 (red-green) plus lightness, and Lab makes that a straight threshold instead of
                 three coupled ones. The two class centres are seeded from the mesh itself -
                 the darkest decile and the most saturated-red decile of the hair mesh's own
                 texels - so nothing is hard-coded to this character's colouring.
  graph smoothing  a per-vertex colour call is noisy at texture seams, so the decision is
                 majority-voted over the welded surface for a few rounds. An isolated skin
                 vertex inside a mass of hair stays hair, and vice versa.
  surgery        faces whose vertices are now skin are separated from the hair object and joined
                 into the body, and the body's weights are re-solved over the enlarged mesh.

This is deliberately not a model call. It is ~7,000 points of numpy plus a few rounds of
neighbour voting; it runs in well under a second, and a text decision model has no way to see a
vertex's colour or its neighbours.

Run: blender -b -noaudio --python reclassify_face.py -- --blend animated.blend --out fixed.blend
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import bpy
import bmesh
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--smooth-rounds", type=int, default=4)
ap.add_argument("--min-fraction", type=float, default=0.01,
                help="fail if fewer than this fraction of hair vertices reclassify")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
hair = next((o for o in scene.objects if o.type == "MESH" and "hair" in o.name.lower()), None)
body = next((o for o in scene.objects if o.type == "MESH" and "body" in o.name.lower()), None)
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
if not (hair and body and rig):
    raise SystemExit("[reclass] need hair, body and an armature")


def albedo_image(ob):
    for m in ob.data.materials:
        if not m or not m.node_tree:
            continue
        for n in m.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image:
                return n.image
    return None


img = albedo_image(hair)
if img is None:
    raise SystemExit("[reclass] no albedo texture on the hair mesh")
W, H = img.size
px = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, img.channels)[:, :, :3]
print(f"[reclass] albedo {W}x{H}", flush=True)

me = hair.data
n = len(me.vertices)
uv_layer = me.uv_layers.active
if uv_layer is None:
    raise SystemExit("[reclass] hair mesh has no UVs")

# per-vertex colour = mean of its loops' texels
acc = np.zeros((n, 3), np.float64)
cnt = np.zeros(n, np.float64)
for poly in me.polygons:
    for li in poly.loop_indices:
        vi = me.loops[li].vertex_index
        u, v = uv_layer.data[li].uv
        x = min(W - 1, max(0, int(u * (W - 1))))
        y = min(H - 1, max(0, int(v * (H - 1))))      # Blender pixels are bottom-up already
        acc[vi] += px[y, x]
        cnt[vi] += 1
col = acc / np.maximum(cnt, 1)[:, None]


def srgb_to_lab(rgb):
    r = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = r @ M.T
    wp = np.array([0.95047, 1.0, 1.08883])
    t = xyz / wp
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.stack([116 * f[:, 1] - 16,
                     500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], 1)


lab = srgb_to_lab(np.clip(col, 0, 1))
L, A = lab[:, 0], lab[:, 1]

# seed the two class centres from the mesh's own extremes
hair_seed = lab[L <= np.percentile(L, 10)].mean(0)
skin_seed = lab[(A >= np.percentile(A, 90)) & (L >= np.percentile(L, 50))]
if len(skin_seed) < 20:
    skin_seed = lab[L >= np.percentile(L, 90)]
skin_seed = skin_seed.mean(0)
print(f"[reclass] seeds  hair L*a*b* {np.round(hair_seed,1)}  skin {np.round(skin_seed,1)}",
      flush=True)

d_hair = np.linalg.norm(lab - hair_seed, axis=1)
d_skin = np.linalg.norm(lab - skin_seed, axis=1)
is_skin = d_skin < d_hair

# ---- neighbour voting over the welded surface -------------------------------------------------
V = np.array([(hair.matrix_world @ v.co)[:] for v in me.vertices])
ext = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
tol = 1e-5 * ext
keys = np.round(V / tol).astype(np.int64)
bucket = defaultdict(list)
for i in range(n):
    bucket[tuple(keys[i])].append(i)
adj = [set() for _ in range(n)]
for e in me.edges:
    i, j = e.vertices
    adj[i].add(j)
    adj[j].add(i)
for members in bucket.values():
    if len(members) < 2:
        continue
    union = set()
    for m in members:
        union |= adj[m]
    for m in members:
        adj[m] = union

for _ in range(a.smooth_rounds):
    nxt = is_skin.copy()
    for i in range(n):
        nb = adj[i]
        if not nb:
            continue
        s = sum(1 for j in nb if is_skin[j])
        nxt[i] = s * 2 > len(nb)
    is_skin = nxt

n_skin = int(is_skin.sum())
frac = n_skin / max(n, 1)
print(f"[reclass] {n_skin:,}/{n:,} hair vertices ({frac:.1%}) reclassified as skin", flush=True)
if frac < a.min_fraction:
    raise SystemExit(f"[reclass] only {frac:.2%} reclassified - the colour model did not "
                     "separate the classes; refusing to write an unchanged file")
if frac > 0.6:
    raise SystemExit(f"[reclass] {frac:.0%} reclassified - the model has inverted the classes")

# ---- move the skin faces into the body --------------------------------------------------------
bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = hair
hair.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bm = bmesh.from_edit_mesh(me)
bm.verts.ensure_lookup_table()
for f in bm.faces:
    f.select = all(is_skin[v.index] for v in f.verts)
n_faces = sum(1 for f in bm.faces if f.select)
bmesh.update_edit_mesh(me)
print(f"[reclass] {n_faces:,} faces selected for transfer", flush=True)
if n_faces == 0:
    raise SystemExit("[reclass] no whole face is skin-only; nothing to move")

before_objs = set(bpy.data.objects)
bpy.ops.mesh.separate(type="SELECTED")
bpy.ops.object.mode_set(mode="OBJECT")
new = [o for o in bpy.data.objects if o not in before_objs]
if not new:
    raise SystemExit("[reclass] separate produced no object")
piece = new[0]
piece.name = "face_fragments"
print(f"[reclass] separated {len(piece.data.vertices):,} vertices", flush=True)

# weights first: the fragments must deform like the skin they rejoin, not like hair
bpy.ops.object.select_all(action="DESELECT")
piece.select_set(True)
body.select_set(True)
bpy.context.view_layer.objects.active = body          # active = source
for vg in list(piece.vertex_groups):
    piece.vertex_groups.remove(vg)
bpy.ops.object.data_transfer(
    data_type="VGROUP_WEIGHTS", use_create=True,
    vert_mapping="POLYINTERP_NEAREST", layers_select_src="ALL", layers_select_dst="NAME")
weighted = sum(1 for v in piece.data.vertices if v.groups)
print(f"[reclass] fragments weighted from body: {weighted:,}/{len(piece.data.vertices):,}",
      flush=True)

bpy.ops.object.select_all(action="DESELECT")
piece.select_set(True)
body.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
print(f"[reclass] body is now {len(body.data.vertices):,} vertices", flush=True)

info = {"hair_vertices_before": n, "reclassified": n_skin, "fraction": round(frac, 4),
        "faces_moved": n_faces, "body_vertices_after": len(body.data.vertices),
        "hair_vertices_after": len(hair.data.vertices)}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump(info, open(a.json, "w"), indent=2)
print(f"[reclass] {json.dumps(info)}", flush=True)
print(f"[reclass] -> {a.out}", flush=True)
