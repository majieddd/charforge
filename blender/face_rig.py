"""A face rig: a jaw bone and the handful of shapes that make a game character alive.

    blender -b --python face_rig.py -- --blend rig_t.blend --face face.json --out rig_f.blend

Input: the T-posed rig (metres) and pipeline/face_landmarks.py's landmarks. Output adds

  jaw        a bone from the jaw hinge (just in front of the ear, level with the nose tip) to the
             chin, owning the lower face: below the mouth line, in front of the hinge, fading
             toward the ears and down the neck. Rotating it opens the mouth for speech (a/o
             visemes).
  mouth      the mesh is cut along the seam between the lips, corner to corner, and a dark
             pouch (its own material, "mouth") is built behind the cut. The lips of a generated
             face are painted on one closed surface, and before the cut an open jaw stretched the
             lower lip into a pink band; now the lips part and the mouth behind them shows.
  blink_L/R  shape keys: the upper lid slides down to the lower lid line and the painted eye
             folds into it - the way a painted-eye character blinks, realistic or stylised, with
             the lashes closing into a line.
  smile      mouth corners up and out, cheeks lifted.
  brows_up   brows raised.
  pucker     mouth corners in, lips forward (the o/u visemes and a kiss).

Shapes are built from the landmarks, measured in units of the eye distance, so the same rule
makes a realistic face and an anime one. They are glTF morph targets in the package; the clips
never key them - a game drives blinks and speech on top of any clip (the playground blinks, and
talks while T is held).
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--face", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH" and o.find_armature() == rig)
F = json.load(open(a.face))
ied = float(F["ied_m"])
# face_landmarks.py's order-and-size checks: without a trusted mouth there is no mouth cut, no jaw
# and no mouth shapes - only the blinks and brows, whose landmarks stand on their own
MOUTH_OK = bool(F.get("mouth_ok", True))
EYES_OK = bool(F.get("eyes_ok", True))
me = mesh.data

# ---- density where the face moves ----------------------------------------------------------
# The mesh was decimated for the whole body; around an eye that left about twenty vertices, too
# few to fold a lid. The faces around each eye and the mouth are subdivided in place - UVs and
# skin weights are interpolated with them, so the texture and the rig are unchanged.
import bmesh  # noqa: E402
bm = bmesh.new()
bm.from_mesh(me)
Mw0 = np.array(mesh.matrix_world)
regions = []
for side in ("left", "right") if F.get("eyes_ok", True) else ():
    e = F["eyes"][side]
    regions.append((np.array(e["centre"]), 1.6 * e["width_m"] / 2, 2.2 * max(e["height_m"], 0.3 * e["width_m"]) + 0.35 * ied))
mouth0 = np.array(F["mouth"]["centre"])
mw0 = float(F["mouth"]["width_m"])
if F.get("mouth_ok", True):
    regions.append((mouth0, 0.85 * mw0, 0.55 * mw0))
for rep in range(2):
    bm.faces.ensure_lookup_table()
    sel = set()
    for f_ in bm.faces:
        c = Mw0[:3, :3] @ np.array(f_.calc_center_median()) + Mw0[:3, 3]
        for C, rx, rz in regions:
            if abs(c[0] - C[0]) < rx and abs(c[2] - C[2]) < rz and abs(c[1] - C[1]) < 0.5 * ied:
                sel.add(f_)
                break
    edges = list({e_ for f_ in sel for e_ in f_.edges})
    bmesh.ops.subdivide_edges(bm, edges=edges, cuts=1, use_grid_fill=True, smooth=0.0)
bm.to_mesh(me)
bm.free()
me.update()
print(f"[face] eye and mouth regions subdivided twice: mesh now {len(me.vertices):,} vertices", flush=True)

# ---- the mouth opens -----------------------------------------------------------------------------
# A generated face is one closed surface with the lips painted on it, so opening the jaw stretched
# the lower lip into a long pink band. The mesh is cut along the seam between the lips, corner to
# corner - the corners stay joined, as a mouth's do - and a dark pouch is built behind the cut: the
# mouth a game face opens onto. The cut side above the seam stays on the head, the side below goes
# with the jaw (see the jaw weights), and the pouch's roof, back and floor are split between them,
# so an open jaw shows a dark mouth instead of a stretched lip.
seam_w = np.array(F["mouth"].get("seam") or F["mouth"]["centre"])
mcl_w, mcr_w = np.array(F["mouth"]["left"]), np.array(F["mouth"]["right"])
mw_w = float(np.linalg.norm(mcl_w - mcr_w))
n_w = np.cross(mcr_w - mcl_w, seam_w - mcl_w)
if np.linalg.norm(n_w) < 1e-9 or abs(n_w[2]) < 0.5 * np.linalg.norm(n_w):
    n_w = np.array([0.0, 0.0, 1.0])                     # a degenerate or steep plane: cut level
n_w = n_w / np.linalg.norm(n_w) * (1.0 if n_w[2] >= 0 else -1.0)
Minv0 = np.linalg.inv(Mw0)
co_loc = Minv0[:3, :3] @ seam_w + Minv0[:3, 3]
no_loc = Mw0[:3, :3].T @ n_w
no_loc /= np.linalg.norm(no_loc)
sdist_w = lambda P: (P - seam_w) @ n_w
bm = bmesh.new()
bm.from_mesh(me)
to_w = lambda co: Mw0[:3, :3] @ np.array(co) + Mw0[:3, 3]
back_y = max(mcl_w[1], mcr_w[1]) + 0.15 * mw_w
region = [f_ for f_ in bm.faces
          if (lambda c: abs(c[0] - seam_w[0]) < 0.5 * mw_w and abs(sdist_w(c)) < 0.3 * mw_w and c[1] < back_y)
          (to_w(f_.calc_center_median()))]
mouth_ok = False
if region and MOUTH_OK:
    geom = list({*region, *[e_ for f_ in region for e_ in f_.edges], *[v_ for f_ in region for v_ in f_.verts]})
    cut = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-7, plane_co=co_loc, plane_no=no_loc)["geom_cut"]
    slit = [e_ for e_ in cut if isinstance(e_, bmesh.types.BMEdge)
            and abs(to_w((e_.verts[0].co + e_.verts[1].co) / 2)[0] - seam_w[0]) < 0.46 * mw_w]
    if len(slit) >= 4:
        bmesh.ops.split_edges(bm, edges=slit)
        mouth_ok = True
if mouth_ok:
    # the two lips of the cut: boundary edges along the seam, each side a chain corner to corner
    bm.verts.ensure_lookup_table()
    on_cut = [v_ for v_ in bm.verts if abs(sdist_w(to_w(v_.co))) < 1e-5
              and abs(to_w(v_.co)[0] - seam_w[0]) < 0.52 * mw_w and to_w(v_.co)[1] < back_y
              and any(e_.is_boundary for e_ in v_.link_edges)]

    def side_of(v_):
        return float(np.mean([sdist_w(to_w(f_.calc_center_median())) for f_ in v_.link_faces])) if v_.link_faces else 0.0
    # the cut duplicated every vertex along it but the two at its ends: those are the corners
    key = lambda v_: tuple(np.round(to_w(v_.co) / 1e-5).astype(int))
    from collections import Counter
    n_at = Counter(key(v_) for v_ in on_cut)
    corners = [v_ for v_ in on_cut if n_at[key(v_)] == 1]
    split_ = [v_ for v_ in on_cut if n_at[key(v_)] > 1]
    upper = sorted([v_ for v_ in split_ if side_of(v_) > 0], key=lambda v_: to_w(v_.co)[0])
    lower = sorted([v_ for v_ in split_ if side_of(v_) < 0], key=lambda v_: to_w(v_.co)[0])
    # pair each upper vertex with the lower copy at the same place
    low_at = {key(v_): v_ for v_ in lower}
    pairs = [(u_, low_at[key(u_)]) for u_ in upper if key(u_) in low_at and low_at[key(u_)] is not u_]
    dl = bm.verts.layers.deform.active
    inward = Mw0[:3, :3].T @ np.array([0.0, 1.0, 0.0])            # into the head: the face looks down -Y
    inward /= np.linalg.norm(inward)
    d1, d2 = 0.22 * mw_w / np.linalg.norm(Mw0[:3, 0]), 0.45 * mw_w / np.linalg.norm(Mw0[:3, 0])
    bag_t = {}

    def new_vert(co, src, t):
        nv = bm.verts.new(co)
        if dl is not None:
            for gk, gw in src[dl].items():
                nv[dl][gk] = gw
        bag_t[nv] = t
        return nv
    chain = []
    for u_, l_ in pairs:
        cu, cl = np.array(u_.co), np.array(l_.co)
        chain.append((u_, new_vert(cu + inward * d1, u_, 0.0), new_vert((cu + cl) / 2 + inward * d2, u_, 0.5),
                      new_vert(cl + inward * d1, l_, 1.0), l_))
    chain.sort(key=lambda row: to_w(row[0].co)[0])
    mat = bpy.data.materials.new("mouth")
    mat.diffuse_color = (0.2, 0.07, 0.07, 1.0)
    if not mat.node_tree:
        mat.use_nodes = True
    bs_ = next((nd for nd in mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED"), None)
    if bs_ is not None:
        bs_.inputs["Base Color"].default_value = (0.2, 0.07, 0.07, 1.0)
        bs_.inputs["Roughness"].default_value = 0.6
    mat.use_backface_culling = False
    me.materials.append(mat)
    mi = len(me.materials) - 1
    # Closed, the pouch is flat - both lips' rows coincide - so it faces by row: the roof (the two
    # strips hung from the upper lip) looks down into the mouth, the floor looks up.
    nb = 0

    def orient(f_, k_):
        f_.material_index = mi
        f_.smooth = True
        f_.normal_update()
        want = -no_loc if k_ < 2 else no_loc
        if np.array(f_.normal) @ want < 0:
            f_.normal_flip()
    for a_, b_ in zip(chain[:-1], chain[1:]):
        for k_ in range(4):
            quad = [a_[k_], b_[k_], b_[k_ + 1], a_[k_ + 1]]
            try:
                f_ = bm.faces.new(quad)
            except ValueError:
                continue
            orient(f_, k_)
            nb += 1
    # the ends of the pouch close on the mouth's corners
    for row, corner_pick in ((chain[0], min), (chain[-1], max)):
        if not corners:
            break
        cv = corner_pick(corners, key=lambda v_: to_w(v_.co)[0])
        for k_ in range(4):
            try:
                f_ = bm.faces.new([row[k_], row[k_ + 1], cv])
            except ValueError:
                continue
            orient(f_, k_)
            nb += 1
    bm.verts.index_update()
    bag_idx = {v_.index: t for v_, t in bag_t.items()}
    upper_idx = {v_.index for v_ in upper}
    lower_idx = {v_.index for v_ in lower}
    bm.to_mesh(me)
    me.update()
    print(f"[face] mouth cut along the lip seam: {len(pairs)} vertices each side, a mouth of {nb} faces behind it",
          flush=True)
else:
    bag_idx, upper_idx, lower_idx = {}, set(), set()
    print("[face] no lip seam to cut - the mouth stays closed (jaw opening will stretch the lips)", flush=True)
bm.free()
n = len(me.vertices)
Mw = np.array(mesh.matrix_world)
Minv = np.linalg.inv(Mw)
V = np.empty(n * 3)
me.vertices.foreach_get("co", V)
V = V.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
N = np.empty(n * 3)
me.vertices.foreach_get("normal", N)
N = N.reshape(-1, 3) @ Mw[:3, :3].T
N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
g = {vg.name: vg.index for vg in mesh.vertex_groups}
head_name = next(nm for nm in ("mixamorig:Head", "head") if nm in g)
Whead = np.zeros(n)
for v in me.vertices:
    for gg in v.groups:
        if gg.group == g[head_name]:
            Whead[v.index] = gg.weight

mouth, nose, chin = (np.array(F["mouth"]["centre"]), np.array(F["nose"]), np.array(F["chin"]))
mcl, mcr = np.array(F["mouth"]["left"]), np.array(F["mouth"]["right"])
mw = float(np.linalg.norm(mcl - mcr))
ss = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
face_front = (Whead > 0.3) & (N[:, 1] < 0.2)            # head-bound, not facing backwards

# ---- jaw ---------------------------------------------------------------------------------------
hinge = np.array([mouth[0], mouth[1] + 1.35 * ied, nose[2]])
if not MOUTH_OK:
    print("[face] no jaw: the mouth landmarks were not trusted", flush=True)
else:
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = rig.data.edit_bones
    jb = eb.new("jaw")
    RW_inv = np.linalg.inv(np.array(rig.matrix_world))
    to_local = lambda p: Vector((RW_inv[:3, :3] @ p + RW_inv[:3, 3]).tolist())
    jb.head, jb.tail = to_local(hinge), to_local(chin)
    jb.parent = eb[head_name]
    bpy.ops.object.mode_set(mode="OBJECT")
    f_vert = ss((mouth[2] + 0.04 * ied - V[:, 2]) / (0.10 * ied))      # 0 above the upper lip, 1 below
    f_front = ss((hinge[1] + 0.2 * ied - V[:, 1]) / (0.6 * ied))       # the face, not the back of the head
    f_side = ss((1.35 * ied - np.abs(V[:, 0] - mouth[0])) / (0.45 * ied))
    f_neck = ss((V[:, 2] - (chin[2] - 0.7 * ied)) / (0.5 * ied))        # fades down the neck
    # the cut's two sides and the pouch: above the seam stays, below goes, in the middle of the mouth
    # (toward the joined corners the old smooth blend takes over, so the corners stretch, not tear)
    if upper_idx or lower_idx:
        g_mid = ss((0.5 * mw_w - np.abs(V[:, 0] - seam_w[0])) / (0.2 * mw_w))
        near_mouth = (np.abs(sdist_w(V)) < 0.45 * mw_w) & (V[:, 1] < back_y)
        sd_all = sdist_w(V)
        f_cut = np.where(sd_all > 0, 0.0, 1.0)
        for i_ in upper_idx:
            f_cut[i_] = 0.0
        for i_ in lower_idx:
            f_cut[i_] = 1.0
        g_mid = np.where(near_mouth, g_mid, 0.0)
        f_vert = g_mid * f_cut + (1.0 - g_mid) * f_vert
        for i_, t_ in bag_idx.items():
            f_vert[i_] = t_
            f_front[i_] = 1.0
    wj = f_vert * f_front * f_side * f_neck * Whead
    if bag_idx:
        for i_, t_ in bag_idx.items():
            wj[i_] = t_ * max(Whead[i_], 1e-3) if Whead[i_] > 0 else t_
    jv = mesh.vertex_groups.new(name="jaw")
    hv = mesh.vertex_groups[head_name]
    moved = 0
    for i in np.nonzero(wj > 1e-3)[0]:
        jv.add([int(i)], float(wj[i]), "REPLACE")
        hv.add([int(i)], float(max(Whead[i] - wj[i], 0.0)), "REPLACE")
        moved += 1
    print(f"[face] jaw: hinge {np.linalg.norm(hinge - chin) * 100:.1f} cm above-behind the chin, "
          f"{moved:,} vertices share its weight", flush=True)

# ---- shape keys --------------------------------------------------------------------------------
if me.shape_keys is None:
    mesh.shape_key_add(name="Basis", from_mix=False)


made = []


def add_key(name, D):
    needs = {"blink_L": EYES_OK, "blink_R": EYES_OK, "brows_up": EYES_OK, "smile": MOUTH_OK, "pucker": MOUTH_OK}
    if not needs.get(name, True):
        print(f"[face] no {name}: its landmarks were not trusted", flush=True)
        return
    made.append(name)
    k = mesh.shape_key_add(name=name, from_mix=False)
    k.value = 0.0                    # Blender 5 creates keys switched on (1.0); a face rests neutral
    Dl = D @ Minv[:3, :3].T                                           # world offsets -> mesh space
    base = np.empty(n * 3)
    me.vertices.foreach_get("co", base)
    k.data.foreach_set("co", (base.reshape(-1, 3) + Dl).ravel())
    moved_ = int((np.linalg.norm(D, axis=1) > 1e-5).sum())
    print(f"[face] shape {name}: {moved_:,} vertices, max {np.linalg.norm(D, axis=1).max() * 1000:.1f} mm",
          flush=True)


for side, tag in (("left", "L"), ("right", "R")):
    e = F["eyes"][side]
    E = np.array(e["centre"])
    w, h = float(e["width_m"]), float(e["height_m"])
    h = max(h, 0.28 * w)
    u = (V[:, 0] - E[0]) / (0.5 * w)
    v = (V[:, 2] - E[2]) / (0.5 * h)
    near = face_front & (np.abs(V[:, 1] - E[1]) < 0.45 * ied) & (np.abs(u) < 1.35) & (v > -1.8) & (v < 3.4)
    fu = np.clip(1 - (np.abs(u) / 1.15) ** 2, 0, 1) ** 0.7
    v_low = -0.55
    D = np.zeros((n, 3))
    inside = near & (v >= v_low) & (v <= 1.0)
    above = near & (v > 1.0)
    D[inside, 2] = -(v[inside] - v_low) * 0.5 * h * fu[inside]
    D[above, 2] = -(1 - v_low) * 0.5 * h * fu[above] * ss((3.4 - v[above]) / 2.4)
    # the closing lid rides a little forward, over the eye's curve, rather than into the head
    bump = np.clip(1 - np.abs(v - 0.3) / 1.6, 0, 1)
    D[near, 1] -= 0.10 * h * fu[near] * bump[near]
    add_key(f"blink_{tag}", D)

D = np.zeros((n, 3))
for C, sgn in ((mcl, 1.0), (mcr, -1.0)):
    r = 0.62 * mw
    d = np.linalg.norm((V - C)[:, [0, 2]], axis=1)
    fall = np.clip(1 - (d / r) ** 2, 0, 1) ** 2 * face_front
    D[:, 0] += sgn * 0.09 * mw * fall
    D[:, 2] += 0.11 * mw * fall
    cheek = face_front & (np.linalg.norm((V - (C + np.array([sgn * 0.15 * mw, 0, 0.35 * mw])))[:, [0, 2]], axis=1) < 0.55 * mw)
    D[cheek, 2] += 0.03 * mw
    D[cheek, 1] -= 0.02 * mw
add_key("smile", D)

D = np.zeros((n, 3))
for side in ("left", "right"):
    e = F["eyes"][side]
    E = np.array(e["centre"])
    lo_ = E[2] + 0.45 * e["height_m"]
    t = (V[:, 2] - lo_) / (0.75 * ied)
    fall = np.clip(1 - np.abs(t - 0.45) / 0.55, 0, 1) * ss((0.6 * ied - np.abs(V[:, 0] - E[0])) / (0.2 * ied))
    D[:, 2] += 0.07 * ied * fall * face_front
add_key("brows_up", D)

D = np.zeros((n, 3))
for C, sgn in ((mcl, 1.0), (mcr, -1.0)):
    d = np.linalg.norm((V - C)[:, [0, 2]], axis=1)
    fall = np.clip(1 - (d / (0.55 * mw)) ** 2, 0, 1) ** 2 * face_front
    D[:, 0] -= sgn * 0.16 * mw * fall
d = np.linalg.norm((V - mouth)[:, [0, 2]], axis=1)
lips = np.clip(1 - (d / (0.7 * mw)) ** 2, 0, 1) ** 2 * face_front
D[:, 1] -= 0.10 * mw * lips
add_key("pucker", D)

bpy.ops.wm.save_as_mainfile(filepath=a.out)
json.dump({"jaw": "jaw" if MOUTH_OK else None, "morphs": made,
           "jaw_open_deg": 18, "ied_m": ied},
          open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "face_rig.json"), "w"), indent=1)
print(f"[face] -> {a.out}", flush=True)
