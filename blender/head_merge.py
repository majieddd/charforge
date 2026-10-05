"""The generated body with its head replaced by one generated on its own, aligned to it (E139).

    blender -b -noaudio --python head_merge.py -- --body mesh.glb --head head_aligned.glb --out mesh_head.glb \
        [--overlap 0.012]

The new head (pipeline/head_align.py) lies on the old one. The neck is where the head's horizontal cross-section is
smallest between the chin and the shoulders; the body keeps everything below it and a band of --overlap above (as a
fraction of the body's height), the head everything above it and the same band below, so the two shells overlap
and the solidify stage closes them into one. Only the body's surface inside the head's footprint is cut - a hand or a
shoulder plate beside the neck stays.

The new head has no texture of its own: each of its vertices takes the colour of the old surface nearest it (the
body's base colour texture, sampled at the nearest old vertex), as a colour attribute its material reads. The front
of the face is painted over by the reference later, which the new head's shape now matches; the colour carried here
is for the sides, back and hair.
"""
import argparse
import sys

import bmesh
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--body", required=True)
ap.add_argument("--head", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--overlap", type=float, default=0.012, help="the overlap band, as a fraction of the body's height")
ap.add_argument("--report", default=None, help="write the new head's region here (neck height, axis, radius; Blender frame)")
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.body)
body = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if len(body) > 1:
    bpy.ops.object.select_all(action="DESELECT")
    for o in body:
        o.select_set(True)
    bpy.context.view_layer.objects.active = body[0]
    bpy.ops.object.join()
body = bpy.context.view_layer.objects.active if len(body) > 1 else body[0]
before = set(bpy.context.scene.objects)
bpy.ops.import_scene.gltf(filepath=a.head)
head = next(o for o in bpy.context.scene.objects if o not in before and o.type == "MESH")
for o in (body, head):
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    o.select_set(False)


def verts(o):
    co = np.empty(len(o.data.vertices) * 3)
    o.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


VB, VH = verts(body), verts(head)
Hb = float(VB[:, 2].max() - VB[:, 2].min())
band = a.overlap * Hb

# the neck: the narrowest horizontal section of the new head between 10% and 50% of its height from its bottom
z0, z1 = VH[:, 2].min(), VH[:, 2].max()
zs = np.linspace(z0 + 0.10 * (z1 - z0), z0 + 0.50 * (z1 - z0), 30)
width = []
for z in zs:
    s_ = VH[np.abs(VH[:, 2] - z) < 0.004 * Hb]
    width.append(np.ptp(s_[:, 0]) * np.ptp(s_[:, 1]) if len(s_) > 30 else np.inf)
z_cut = float(zs[int(np.argmin(width))])
axis = VH[VH[:, 2] > z_cut][:, :2].mean(0)
radius = float(np.quantile(np.linalg.norm(VH[VH[:, 2] > z_cut][:, :2] - axis, axis=1), 0.98)) * 1.15

# colour for the new head: the old surface's, nearest vertex by nearest vertex
from mathutils.kdtree import KDTree  # noqa: E402
me_b = body.data
uv_layer = me_b.uv_layers.active
img = None
for slot in body.material_slots:
    if slot.material and slot.material.use_nodes:
        bsdf = next((n for n in slot.material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        lk = bsdf.inputs["Base Color"].links if bsdf else None
        if lk and lk[0].from_node.type == "TEX_IMAGE":
            img = lk[0].from_node.image
            break
if img is None or uv_layer is None:
    raise SystemExit("[head] the body has no base colour texture to carry onto the new head")
px = np.empty(img.size[0] * img.size[1] * 4, np.float32)
img.pixels.foreach_get(px)
px = px.reshape(img.size[1], img.size[0], 4)
loop_uv = np.empty(len(me_b.loops) * 2)
uv_layer.data.foreach_get("uv", loop_uv)
loop_uv = loop_uv.reshape(-1, 2)
loop_v = np.empty(len(me_b.loops), np.int64)
me_b.loops.foreach_get("vertex_index", loop_v)
vuv = np.zeros((len(VB), 2))
vuv[loop_v] = loop_uv
iu = np.clip((vuv[:, 0] * img.size[0]).astype(int), 0, img.size[0] - 1)
iv = np.clip((vuv[:, 1] * img.size[1]).astype(int), 0, img.size[1] - 1)
vcol = px[iv, iu, :3]
vcol = np.where(vcol <= 0.04045, vcol / 12.92, ((vcol + 0.055) / 1.055) ** 2.4)   # the image's sRGB, made linear
near = np.linalg.norm(VB[:, :2] - axis, axis=1) < radius * 1.3
cand = np.flatnonzero(near & (VB[:, 2] > z_cut - 0.5 * band))      # the old head and neck, not the collar below
kd = KDTree(len(cand))
for i_, c_ in enumerate(cand):
    kd.insert(VB[c_], i_)
kd.balance()
# blurred over ~2 cm first: the old face's eyes, brows and mouth sit where the old shape had them, not the new one,
# and carried over sharp they showed as a second, pale pair of eyes beside the reference's; blurred, the face is skin
# and the hair hair, and the texture stage paints the features on the new shape
blur_r = 0.012 * Hb
cc = vcol[cand]
blur = np.empty_like(cc)
for i_, c_ in enumerate(cand):
    near_ = [k for _, k, _ in kd.find_range(VB[c_], blur_r)]
    blur[i_] = cc[near_].mean(0) if near_ else cc[i_]

# cut both
def cut(o, drop_v):
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.verts.ensure_lookup_table()
    gone = [f for f in bm.faces if all(drop_v[v.index] for v in f.verts)]
    bmesh.ops.delete(bm, geom=gone, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(o.data)
    bm.free()
    return len(gone)


# the old head goes wholly - its nose and ears stood out through the new one when only the new head's own radius was
# cut - but not a shoulder plate beside it (Cadet's are two head-widths out)
# the old head is cut a band below the new one's neck and the new neck kept a little further down over the stump: cut
# at the same height, the old chin stood out under the new one as a second chin; a long overlap left two near-coincident
# surfaces at the neck, which the solid's snap back to the source folded, and the hands' booleans then failed
n_b = cut(body, (VB[:, 2] > z_cut - band) & (np.linalg.norm(VB[:, :2] - axis, axis=1) < 1.3 * radius))
sl = VH[np.abs(VH[:, 2] - z_cut) < 0.004 * Hb]
neck_r = float(np.quantile(np.linalg.norm(sl[:, :2] - sl[:, :2].mean(0), axis=1), 0.95)) if len(sl) > 30 else radius
neck_c = sl[:, :2].mean(0) if len(sl) > 30 else axis
below = VH[:, 2] < z_cut
n_h = cut(head, (VH[:, 2] < z_cut - 1.6 * band)
          | (below & (np.linalg.norm(VH[:, :2] - neck_c, axis=1) > 1.15 * neck_r)))   # below the neck: the neck alone
VH = verts(head)                                      # what is left of the new head, coloured after the cut
j = np.array([kd.find(v)[1] for v in VH])
# blurred where the picture will paint (the head's front), sharp where nothing else will (sides and back: the old
# hair's strands lie close enough to the new hair's to be worth keeping)
head.data.update()
nrm = np.empty(len(head.data.vertices) * 3)
head.data.vertices.foreach_get("normal", nrm)
front = np.clip((-nrm.reshape(-1, 3)[:, 1] - 0.15) / 0.45, 0, 1)[:, None]
hcol = front * blur[j] + (1 - front) * cc[j]
attr = head.data.color_attributes.new("head_colour", "FLOAT_COLOR", "POINT")
attr.data.foreach_set("color", np.concatenate([hcol, np.ones((len(hcol), 1))], 1).astype(np.float32).ravel())
# the colour into a texture of the head's own: smart-projected UVs, an emission bake of the colour attribute (glTF
# carries a texture through every later stage; a colour attribute came back unread)
bpy.ops.object.select_all(action="DESELECT")
head.select_set(True)
bpy.context.view_layer.objects.active = head
uv_name = body.data.uv_layers.active.name
while head.data.uv_layers:
    head.data.uv_layers.remove(head.data.uv_layers[0])
head.data.uv_layers.new(name=uv_name)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.003)
bpy.ops.object.mode_set(mode="OBJECT")
tex = bpy.data.images.new("head_albedo", 2048, 2048, alpha=False)
mat = bpy.data.materials.new("head")
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
out_n = nt.nodes["Material Output"]
attr_n = nt.nodes.new("ShaderNodeAttribute")
attr_n.attribute_name = "head_colour"
emit = nt.nodes.new("ShaderNodeEmission")
nt.links.new(attr_n.outputs["Color"], emit.inputs["Color"])
nt.links.new(emit.outputs["Emission"], out_n.inputs["Surface"])
img_n = nt.nodes.new("ShaderNodeTexImage")
img_n.image = tex
nt.nodes.active = img_n
head.data.materials.clear()
head.data.materials.append(mat)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = 1
sc.render.bake.margin = 8
bpy.ops.object.bake(type="EMIT")
# saved and read back before packing: packing an image made in this session stores it as it was made, blank
import os  # noqa: E402
import tempfile  # noqa: E402
png = os.path.join(tempfile.mkdtemp(), "head_albedo.png")
tex.filepath_raw = png
tex.file_format = "PNG"
tex.save()
tex.source = "FILE"
tex.filepath = png
tex.reload()
tex.colorspace_settings.name = "sRGB"
tex.pack()
nt.links.new(img_n.outputs["Color"], bsdf.inputs["Base Color"])
nt.links.new(bsdf.outputs["BSDF"], out_n.inputs["Surface"])
bsdf.inputs["Roughness"].default_value = 0.7
for p_ in head.data.polygons:
    p_.use_smooth = True
head.data.color_attributes.remove(head.data.color_attributes["head_colour"])
n_head = len(head.data.polygons)
# one object, as every later stage reads mesh.glb
bpy.ops.object.select_all(action="DESELECT")
head.select_set(True)
body.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=False)
if a.report:
    import json  # noqa: E402
    json.dump({"neck_z": z_cut, "band": band, "axis": [float(axis[0]), float(axis[1])], "radius": radius,
               "frame": "Blender (Z up), the mesh's own units"}, open(a.report, "w"), indent=1)
print(f"[head] neck at z {z_cut:.4f} (body height {Hb:.3f}); body: {n_b:,} faces of the old head cut; new head: "
      f"{n_head:,} faces kept ({n_h:,} below the neck cut); its colour from {len(cand):,} old "
      f"vertices -> {a.out}", flush=True)
