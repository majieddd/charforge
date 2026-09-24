"""Hair cards: strands laid over the solid hair, the way realistic game hair is built.

    blender -b --python hair_cards.py -- --blend rig_f.blend --out rig_h.blend [--spacing 0.012]

A generated head of hair is one solid surface: a helmet with strand grooves painted on, and a
silhouette as smooth as a bowl. Realistic game hair is a solid base like that under a layer of
cards - narrow strips textured with strands and alpha-tested - so this stage keeps the solid as
the base and lays cards over it:

  where     the hair is found by colour: head-bound vertices whose albedo is near the head's
            dominant non-skin colour (the part labels miss most of a bob). Card roots are spread
            over it about a centimetre apart.
  which way a strand runs down the head and out from the crown: gravity and the direction away
            from the top of the head, both laid into the surface. Each card is traced along that
            flow over the surface, riding a few millimetres above it, until it leaves the hair.
  looks     one small strand atlas for every card: clumps of strands with soft edges and a
            highlight, darker at the root, painted in the hair's own colour in four tones. Each
            card takes the tone of the painted hair around its root, so the cards keep the
            hair's light and dark. Alpha-tested at 0.5 (glTF alphaMode MASK), the way every
            engine draws hair cards without sorting.
  moves     each card vertex takes the skin weights of the nearest hair vertex - the head, or a
            ponytail's spring chain.

Status: EXPERIMENTAL - not a pipeline stage. Measured on juno (a black bob, 637 cards, 10k
quads): the cards soften the silhouette and read as strands up close, but they speckle inside the
hair, a few cards run off the hair across an eye or onto the collar, and alpha-tested cards
shimmer in any renderer without temporal anti-aliasing (the three.js playground has none). The
solid hair under them looked cleaner at game distance, so the pipeline ships the solid. Run it by
hand for a close-up realistic character in an engine with TAA; anime and stylised hair should
stay solid clumps in any case.
"""
import argparse
import json
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--spacing", type=float, default=0.011, help="metres between card roots")
ap.add_argument("--json", default=None)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
body = next(o for o in sc.objects if o.type == "MESH" and o.find_armature() == rig)
me = body.data
n = len(me.vertices)
M = body.matrix_world
V = np.array([(M @ v.co)[:] for v in me.vertices])
Nrm = np.array([(M.to_3x3() @ v.normal).normalized()[:] for v in me.vertices])
H = float(V[:, 2].max() - V[:, 2].min())
gi = {g.name: g.index for g in body.vertex_groups}
head_name = next(nm for nm in ("mixamorig:Head", "head") if nm in gi)
Wh = np.zeros(n)
wts = [dict() for _ in range(n)]
for v in me.vertices:
    for g in v.groups:
        wts[v.index][g.group] = g.weight
        if g.group == gi[head_name]:
            Wh[v.index] = g.weight

# per-vertex albedo, sampled from the base colour image at each vertex's UV
mat = me.materials[0]
bsdf = next(nd for nd in mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
img = bsdf.inputs["Base Color"].links[0].from_node.image
W_, H_ = img.size
px = np.empty(W_ * H_ * 4, np.float32)
img.pixels.foreach_get(px)
px = px.reshape(H_, W_, 4)
uvl = me.uv_layers.active.data
vuv = np.zeros((n, 2))
for loop in me.loops:
    vuv[loop.vertex_index] = uvl[loop.index].uv
col = px[np.clip((vuv[:, 1] * (H_ - 1)).astype(int), 0, H_ - 1), np.clip((vuv[:, 0] * (W_ - 1)).astype(int), 0, W_ - 1), :3]


def to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lab(c):
    c = to_linear(c)                     # image pixels of an 8-bit texture come back sRGB-encoded
    M3 = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M3.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


L = lab(col)
headb = rig.data.bones[head_name]
head_z = (rig.matrix_world @ headb.head_local).z
top_z = float(V[:, 2].max())
on_head = (Wh > 0.6) & (V[:, 2] > head_z - 0.02 * H)
skinlike = (L[:, 1] > 6) & (L[:, 2] > 8) & (L[:, 0] > 35)
cand = on_head & ~skinlike
if cand.sum() < 200:
    print("[hair] no hair found on the head (bald, or a hat) - no cards", flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    raise SystemExit(0)
hair_lab = np.median(L[cand], 0)
# the lower part of a bob or long hair is bound to the neck, not the head: take the hair colour on
# anything head- or neck-bound above the shoulders
neck_name = next((nm for nm in ("mixamorig:Neck", "neck") if nm in gi), None)
Wn = np.zeros(n)
if neck_name:
    for v in me.vertices:
        for g in v.groups:
            if g.group == gi[neck_name]:
                Wn[v.index] = g.weight
sh_z = head_z - 0.09 * H
upper = ((Wh + Wn) > 0.6) & (V[:, 2] > sh_z)
hair = upper & (np.linalg.norm(L - hair_lab, axis=1) < 22)
# hanging hair: bound to a hair spring chain
for nm, gidx in gi.items():
    if nm.startswith("spring_hair"):
        hair |= np.array([w.get(gidx, 0) > 0.3 for w in wts])
print(f"[hair] {int(hair.sum()):,} hair vertices, colour Lab ({hair_lab[0]:.0f}, {hair_lab[1]:.0f}, {hair_lab[2]:.0f})",
      flush=True)

bm = bmesh.new()
bm.from_mesh(me)
bm.transform(M)
bm.faces.ensure_lookup_table()
bvh = BVHTree.FromBMesh(bm)
kd = KDTree(n)
for i in range(n):
    kd.insert(V[i], i)
kd.balance()
hkd_idx = np.nonzero(hair)[0]
hkd = KDTree(len(hkd_idx))
for j, i in enumerate(hkd_idx):
    hkd.insert(V[i], j)
hkd.balance()

# roots: Poisson-disk over the hair vertices (a grid hash of cells one spacing wide)
rng = np.random.default_rng(11)
cell = a.spacing
grid = {}
roots = []
for i in rng.permutation(hkd_idx):
    key = tuple((V[i] // cell).astype(int))
    clash = False
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for r in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                    if np.linalg.norm(V[r] - V[i]) < a.spacing:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if clash:
            break
    if not clash:
        roots.append(i)
        grid.setdefault(key, []).append(i)
# a crown: the top of the hair
crown = V[hkd_idx[np.argmax(V[hkd_idx, 2])]]


def flow_at(p, nrm):
    g = np.array([0, 0, -1.0])
    out = p - crown
    out[2] = 0
    d = g + 0.8 * (out / (np.linalg.norm(out) + 1e-6)) * max(0.0, 1.0 - (top_z - p[2]) / (0.12 * H))
    d = d - (d @ nrm) * nrm
    ln = np.linalg.norm(d)
    return d / ln if ln > 1e-6 else None


cards = []
lift = 0.0035
for i in roots:
    p = V[i].copy()
    pts, nrms = [], []
    length = rng.uniform(0.07, 0.15) * H / 1.7
    step = 0.006 * H / 1.7
    for k in range(int(length / step)):
        loc, nrm, fi, dist = bvh.find_nearest(Vector(p), 0.03)
        if loc is None:
            break
        loc, nrm = np.array(loc), np.array(nrm)
        # still on hair?
        co, j, d = hkd.find(Vector(loc))
        if d > 0.012 * H / 1.7:
            break
        pts.append(loc + nrm * lift)
        nrms.append(nrm)
        f = flow_at(loc, nrm)
        if f is None:
            break
        p = loc + f * step
    if len(pts) < 4:
        continue
    cards.append((i, np.array(pts), np.array(nrms)))
bm.free()
print(f"[hair] {len(cards)} cards from {len(roots)} roots", flush=True)

# ---- the strand atlas (shared by every card) ----------------------------------------------------
# Four strand layouts side by side, each a few clumps of hair with soft edges, a highlight down
# each clump, darker roots and staggered tips. The hair's own colour is painted into the atlas -
# not left to vertex colours, which Unity's and Unreal's stock shaders ignore - and the four
# columns step in tone (0.8 to 1.25 of the hair's colour), so each card picks the column that
# matches the painted hair under its root and the cards keep the hair's own light and dark.
lin_hair = np.median(to_linear(col[hair]), 0)
TONES = np.array([0.88, 0.96, 1.04, 1.12])
NV, CW, TH = len(TONES), 128, 512
tex = np.zeros((TH, NV * CW, 4), np.float32)
xs = np.arange(CW, dtype=np.float32)
ys = np.arange(TH, dtype=np.float32) / TH                      # 0 = root, 1 = tip
for v_ in range(NV):
    alpha = np.zeros((TH, CW), np.float32)
    shade = np.zeros((TH, CW), np.float32)
    clumps = [(rng.uniform(0.18, 0.82) * CW, rng.uniform(5, 10), rng.uniform(0.7, 1.0), rng.uniform(0.92, 1.05))
              for _ in range(7)]
    clumps += [(rng.uniform(0.1, 0.9) * CW, rng.uniform(1.5, 2.6), rng.uniform(0.8, 1.0), rng.uniform(0.9, 1.15))
               for _ in range(5)]                                 # a few loose strands
    for x0, w0, end, tone in clumps:
        wig = rng.uniform(-4, 4)
        for yi in range(int(TH * end)):
            t = ys[yi]
            xc = x0 + wig * math.sin(t * 3.0)
            w = w0 * (1.0 - 0.55 * t / end)                       # clumps taper to the tip
            d = np.abs(xs - xc) / w
            a_ = np.clip(1.2 - d, 0, 1) * np.clip((end - t) / 0.12, 0, 1)
            hi = np.clip(1 - d / 0.45, 0, 1)                     # the highlight down the clump
            keep = a_ > alpha[yi]
            alpha[yi] = np.where(keep, a_, alpha[yi])
            shade[yi] = np.where(keep, tone * (0.9 + 0.18 * hi), shade[yi])
    shade *= (0.85 + 0.15 * np.clip(ys / 0.35, 0, 1))[:, None]  # roots in shadow
    edge = np.clip(np.minimum(xs, CW - 1 - xs) / (CW * 0.08), 0, 1)
    alpha *= edge[None, :]
    rgb = np.clip(lin_hair[None, None, :] * TONES[v_] * shade[..., None], 0, 1)
    rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)   # sRGB image
    # colour bleeds under transparent texels, so filtering at a clump edge never pulls in black
    tex[:, v_ * CW:(v_ + 1) * CW, :3] = np.where(alpha[..., None] > 0.02, rgb,
                                                 np.median(rgb[alpha > 0.5], 0) if (alpha > 0.5).any() else rgb)
    tex[:, v_ * CW:(v_ + 1) * CW, 3] = alpha
TW = NV * CW
tex = np.concatenate([tex, np.repeat(tex[-1:], TW - TH, 0)], 0) if TW > TH else tex     # a square image
timg = bpy.data.images.new("hair_strands", TW, tex.shape[0], alpha=True)
timg.pixels.foreach_set(np.flipud(tex).ravel())
timg.filepath_raw = os.path.join(os.path.dirname(os.path.abspath(a.out)), "hair_strands.png")
timg.file_format = "PNG"
timg.save()
v_scale = TH / tex.shape[0]                                     # the strands fill the top TH rows

# each card's tone column: the painted hair's brightness around its root, against the whole
def lum(c):
    return float(c @ np.array([0.2126, 0.7152, 0.0722]))


lum_hair = max(lum(lin_hair), 1e-4)
card_tone = []
for root_i, pts, nrms in cards:
    near = [hkd_idx[j] for _, j, _ in hkd.find_range(Vector(V[root_i]), 0.018 * H / 1.7)]
    local = to_linear(col[near]).mean(0) if near else lin_hair
    card_tone.append(int(np.argmin(np.abs(TONES - np.clip(lum(local) / lum_hair, 0.8, 1.2)))))

# ---- card mesh ---------------------------------------------------------------------------------
cm = bpy.data.meshes.new("hair_cards")
verts, faces, uvs, owners = [], [], [], []
for (root_i, pts, nrms), tone in zip(cards, card_tone):
    width = rng.uniform(0.011, 0.017) * H / 1.7
    u0 = (tone + 0.02) / NV
    u1 = (tone + 0.98) / NV
    m = len(pts)
    start = len(verts)
    for k in range(m):
        t = k / (m - 1)
        fwd = pts[min(k + 1, m - 1)] - pts[max(k - 1, 0)]
        fwd /= np.linalg.norm(fwd) + 1e-9
        side = np.cross(nrms[k], fwd)
        side /= np.linalg.norm(side) + 1e-9
        w = width * (1 - 0.5 * t)
        for sgn, u in ((-1, u0), (1, u1)):
            verts.append(pts[k] + side * (sgn * w / 2))
            uvs.append((u, 1.0 - t * v_scale))
            owners.append(root_i)
    for k in range(m - 1):
        a0 = start + 2 * k
        faces.append((a0, a0 + 1, a0 + 3, a0 + 2))
Minv = M.inverted()
cm.from_pydata([(Minv @ Vector(v))[:] for v in verts], [], faces)
cm.update()
uvlay = cm.uv_layers.new(name="UVMap")
for li, loop in enumerate(cm.loops):
    uvlay.data[li].uv = uvs[loop.vertex_index]
for poly in cm.polygons:
    poly.use_smooth = True
co = bpy.data.objects.new("hair_cards", cm)
sc.collection.objects.link(co)
co.matrix_world = body.matrix_world.copy()

# the material: strands as base colour, alpha tested at 0.5. The cutoff is written as
# 1 - (alpha < 0.5), the node pattern the glTF exporter turns into alphaMode MASK.
hm = bpy.data.materials.new("hair_cards")
if not hm.node_tree:
    hm.use_nodes = True
nt = hm.node_tree
bs = next(nd for nd in nt.nodes if nd.type == "BSDF_PRINCIPLED")
ti = nt.nodes.new("ShaderNodeTexImage")
ti.image = timg
lt = nt.nodes.new("ShaderNodeMath")
lt.operation = "LESS_THAN"
lt.inputs[1].default_value = 0.5
inv = nt.nodes.new("ShaderNodeMath")
inv.operation = "SUBTRACT"
inv.inputs[0].default_value = 1.0
nt.links.new(ti.outputs["Color"], bs.inputs["Base Color"])
nt.links.new(ti.outputs["Alpha"], lt.inputs[0])
nt.links.new(lt.outputs["Value"], inv.inputs[1])
nt.links.new(inv.outputs["Value"], bs.inputs["Alpha"])
bs.inputs["Roughness"].default_value = 0.78
if "Specular IOR Level" in bs.inputs:
    bs.inputs["Specular IOR Level"].default_value = 0.35
if hasattr(hm, "surface_render_method"):
    hm.surface_render_method = "DITHERED"
hm.use_backface_culling = False
cm.materials.append(hm)

# skinning: each card vertex copies the nearest hair vertex's weights
for g in body.vertex_groups:
    co.vertex_groups.new(name=g.name)
for vi, owner in enumerate(owners):
    _, j, _ = hkd.find(Vector(verts[vi]))
    src = hkd_idx[j]
    for gidx, w in wts[src].items():
        co.vertex_groups[body.vertex_groups[gidx].name].add([vi], w, "REPLACE")
mod = co.modifiers.new("Armature", "ARMATURE")
mod.object = rig
co.parent = rig
co.matrix_parent_inverse = rig.matrix_world.inverted()
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump({"cards": len(cards), "quads": len(faces), "vertices": len(verts),
               "tones": np.bincount(card_tone, minlength=NV).tolist()}, open(a.json, "w"))
print(f"[hair] {len(cards)} cards, {len(faces):,} quads -> {a.out}", flush=True)
