"""Build the skeleton and skin the mesh: body bones from the refined joints, finger bones from the
modelled hands, weights from geodesic_weights.py.

    blender -b --python rig_build.py -- --mesh retopo.glb --joints joints_refined.json \
        --frame-from sdf.npz --hands hands.json --weights weights.npz --out rig.blend \
        [--albedo albedo.png] [--json rig.json]

What changed from rig_retopo.py, and why:

* Weights come from distance measured through the solid (pipeline/geodesic_weights.py), not
  straight-line distance to the nearest bone - which bound the side of a jacket to the arm
  hanging beside it and made "bat wings" when the arms rose.
* The hands have fingers. hand_model.py knows where its knuckles are, so the finger bones are
  built exactly there: three per finger and the thumb, plus the leaf bone each chain ends in, in
  the Mixamo layout (the hand bone points at the middle finger's knuckle). A hand vertex is
  weighted to the finger bones by distance - fingers are modelled apart, so the nearest bone is
  the finger's own - blending into the geodesic weights across the wrist.
* The mesh is placed in the joints' frame from the generated mesh's bounding box (sdf.npz), not
  its own: the modelled hands change the bounding box, and normalising by it put the skeleton
  a centimetre off.

Kept from rig_retopo.py, because each was measured: shoes rigid below the ankle, and below
mid-thigh each connected leg piece bound to its own leg only.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_model  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--joints", required=True)
ap.add_argument("--frame-from", required=True)
ap.add_argument("--hands", default=None)
ap.add_argument("--weights", required=True)
ap.add_argument("--albedo", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--max-influences", type=int, default=4)
ap.add_argument("--labels", default=None, help="labels.json - accessories bind to the torso only")
a = ap.parse_args(argv)

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
mesh = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
mesh.name = "char"
bpy.context.view_layer.objects.active = mesh
mesh.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
if a.albedo and os.path.exists(a.albedo):
    for m in mesh.data.materials:
        bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf and bsdf.inputs["Base Color"].links:
            node = bsdf.inputs["Base Color"].links[0].from_node
            new = bpy.data.images.load(os.path.abspath(a.albedo))
            new.name = node.image.name if node.image else "albedo"
            new.pack()
            node.image = new
    print(f"[rig] base colour from {os.path.basename(a.albedo)}", flush=True)

# ---- the joints' frame ------------------------------------------------------------------------
P = np.load(a.frame_from)["points"]
lo, hi = P.min(0), P.max(0)
ctr = (lo + hi) / 2
k = 2.0 / float(hi[2] - lo[2])
co = np.empty(len(mesh.data.vertices) * 3)
mesh.data.vertices.foreach_get("co", co)
co = (co.reshape(-1, 3) - ctr) * k
mesh.data.vertices.foreach_set("co", co.ravel())
mesh.data.update()
V = co

Jd = json.load(open(a.joints))
J = {n: Vector(p) for n, p in Jd["joints"].items()}
PAR = Jd["parents"]
hands = json.load(open(a.hands))["sides"] if a.hands and os.path.exists(a.hands) else {}

# ---- armature ---------------------------------------------------------------------------------
arm = bpy.data.armatures.new("rig")
rig = bpy.data.objects.new("rig", arm)
bpy.context.scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
children = defaultdict(list)
for n, p in PAR.items():
    if p:
        children[p].append(n)
eb = {}
skip = {f"{s}_hand" for s in hands}                 # the old single stub, replaced by fingers
for n, p in J.items():
    if n in skip:
        continue
    b = arm.edit_bones.new(n)
    b.head = p
    kids = [c for c in children.get(n, []) if c not in skip]
    if n == "spine3":
        kids = ["neck"]
    if n.endswith("_wrist") and n.split("_")[0] in hands:
        side = n.split("_")[0]
        b.tail = (Vector(hands[side]["fingers"]["middle"][0]) - Vector(ctr)) * k
    elif kids:
        b.tail = J[kids[0]]
    elif PAR.get(n):
        d = p - J[PAR[n]]
        b.tail = p + d.normalized() * max(d.length * 0.4, 0.02)
    else:
        b.tail = p + Vector((0, 0, 0.05))
    if (b.tail - b.head).length < 1e-4:
        b.tail = b.head + Vector((0, 0, 0.02))
    eb[n] = b
for n, b in eb.items():
    p = PAR.get(n)
    if p and p in eb:
        b.parent = eb[p]
        b.use_connect = False
finger_bones = []
for side, hd in hands.items():
    wrist = eb[f"{side}_wrist"]
    for f in FINGERS:
        pts = [(Vector(q) - Vector(ctr)) * k for q in hd["fingers"][f]]
        parent = wrist
        for i in range(3):
            b = arm.edit_bones.new(f"{side}_{f}{i + 1}")
            b.head, b.tail = pts[i], pts[i + 1]
            b.parent = parent
            b.use_connect = i > 0
            parent = b
            finger_bones.append(b.name)
        leaf = arm.edit_bones.new(f"{side}_{f}4")
        leaf.head = pts[3]
        leaf.tail = pts[3] + (pts[3] - pts[2]).normalized() * (pts[3] - pts[2]).length * 0.5
        leaf.parent = parent
        leaf.use_connect = True
        leaf.use_deform = False
bpy.ops.object.mode_set(mode="OBJECT")
print(f"[rig] skeleton: {len(arm.bones)} bones ({len(finger_bones)} finger bones)", flush=True)

# ---- weights ------------------------------------------------------------------------------------
Wd = np.load(a.weights)
wnames = [str(x) for x in Wd["bones"]]
I, Wt = Wd["index"], Wd["weight"]
n = len(V)
if len(I) != n:
    raise SystemExit(f"[rig] weights for {len(I)} vertices, mesh has {n}")
names = [b.name for b in arm.bones if b.use_deform]
col = {nm: i for i, nm in enumerate(names)}
W = np.zeros((n, len(names)))
for kk in range(I.shape[1]):
    for bi, bn in enumerate(wnames):
        m = I[:, kk] == bi
        if bn in col:
            W[m, col[bn]] += Wt[m, kk]
        elif bn in skip:                             # old stub weight goes to the hand bone
            W[m, col[bn.replace("_hand", "_wrist")]] += Wt[m, kk]


def seg_dist(X, p0, p1):
    ab = p1 - p0
    t = np.clip(((X - p0) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)[:, None]
    return np.linalg.norm(X - (p0 + t * ab), axis=1)


for side, hd in hands.items():
    C = (np.array(hd["wrist"]) - ctr) * k
    L = float(hd["length"]) * k
    mid0 = (np.array(hd["fingers"]["middle"][0]) - ctr) * k
    x = (mid0 - C) / np.linalg.norm(mid0 - C)
    along = (V - C) @ x
    region = (along > -0.12 * L) & (np.linalg.norm(V - C, axis=1) < 1.4 * L)
    if "frame" in hd:
        # The hand region is the modelled hand's own surface, not "everything near the wrist":
        # Wren's satchel hung against her hand, and the distance rule gave the bag finger
        # weights - raising the arm stretched its strap into two ropes. The hand is a known
        # distance field, so membership is exact.
        fr = hd["frame"]
        B = np.stack([fr["x"], fr["y"], fr["z"]], axis=1) * L
        q = np.linalg.solve(B, (V[region] - C).T).T                  # unit hand frame (mirror included)
        dh, _ = hand_model.sdf(q, wrist_r=tuple(hd.get("wrist_radius", (0.15, 0.10))))
        keep_ = dh < 0.035
        idx_ = np.nonzero(region)[0]
        region = np.zeros(n, bool)
        region[idx_[keep_]] = True
        along = (V - C) @ x
    if not region.any():
        continue
    Xr = V[region]
    segs, sn = [], []
    # the palm: the hand bone, as the fan from the wrist to each knuckle
    for f in ("index", "middle", "ring", "pinky"):
        segs.append((C, (np.array(hd["fingers"][f][0]) - ctr) * k))
        sn.append(f"{side}_wrist")
    for f in FINGERS:
        pts = [(np.array(q) - ctr) * k for q in hd["fingers"][f]]
        for i in range(3):
            segs.append((pts[i], pts[i + 1]))
            sn.append(f"{side}_{f}{i + 1}")
    D = np.stack([seg_dist(Xr, p0, p1) for p0, p1 in segs], 1)
    D = np.maximum(D, 0.004 * L)
    Wh = np.zeros((len(Xr), len(names)))
    raw = D ** -6.0
    for j, bn in enumerate(sn):
        Wh[:, col[bn]] = np.maximum(Wh[:, col[bn]], raw[:, j])
    Wh /= Wh.sum(1, keepdims=True)
    # across the wrist, from the geodesic weights (forearm) to the hand's own
    t = np.clip((along[region] + 0.12 * L) / (0.20 * L), 0, 1)
    t = t * t * (3 - 2 * t)
    W[region] = (1 - t)[:, None] * W[region] + t[:, None] * Wh
    print(f"[rig] {side} hand: {int(region.sum()):,} vertices weighted to the palm and "
          f"{len([s for s in sn if 'wrist' not in s])} finger bones", flush=True)

# Accessories hang from the torso. A bag resting against a hand or a thigh in the A-pose is
# geodesically close to them through the fused surface, and took their weights; a bag rides on the
# body, so its vertices keep only the spine and pelvis share of their weights (spring chains, when
# a stage adds them, take over from there).
if a.labels and os.path.exists(a.labels):
    Lj = json.load(open(a.labels))
    lab = np.asarray(Lj["labels"])
    acc = lab == Lj["group_ids"].get("accessory", -1)
    if len(lab) == n and acc.any():
        torso = [col[b] for b in ("pelvis", "spine1", "spine2", "spine3", "neck", "left_collar", "right_collar")
                 if b in col]
        Wt_ = W[acc][:, torso]
        s_ = Wt_.sum(1, keepdims=True)
        fallback = np.zeros_like(Wt_)
        fallback[:, torso.index(col["spine3"]) if "spine3" in col else 0] = 1.0
        Wt_ = np.where(s_ > 1e-6, Wt_ / np.maximum(s_, 1e-9), fallback)
        W[acc] = 0.0
        W[np.ix_(np.nonzero(acc)[0], torso)] = Wt_
        print(f"[rig] {int(acc.sum()):,} accessory vertices bound to the torso only", flush=True)

# ---- the head is one rigid body down to the jaw line --------------------------------------------
# The head joint is where the pose model puts the ears - the pivot a nod turns about - and the
# face hangs below it: on juno the nose was 1 cm below, the mouth 4 cm, the chin 8 cm. Through the
# solid those are nearer the neck bone than the head bone, and the geodesic weights gave the nose
# 52% neck, the mouth 61%, the chin 73%: every nod bent the face at the lips, and the jaw (which
# the face stage cuts out of the head's weight) got a fifth of the chin. A head is also too big
# for a 1/d^4 falloff to stay rigid - even the crown kept 22% neck. A game head is one rigid body
# down to the jaw line, with the neck blending in below it. So everything above a plane from
# under the chin (front) to the nape (back) goes to the head, fading back to the geodesic weights
# over ~2 cm below the plane. There are no face landmarks yet at this stage, so the plane comes
# from the ear-to-crown distance, as proportions measured on juno (chin 1.45x that distance
# below the crown, mouth 1.25x). Clothing keeps its own weights - a collar or a hood lying on the
# shoulders must not turn with the head - and anything else above the plane (skin, hair, a hat,
# glasses, or hair the parser took for a hat - Wren's crown was 76% neck) belongs to the head.
if "head" in col:
    Eh = np.array(arm.bones["head"].head_local[:])
    u_h = float(V[:, 2].max() - V[:, 2].min()) / 1.75
    axis_near = np.linalg.norm(V[:, :2] - Eh[:2], axis=1) < 0.13 * u_h
    top_h = float(V[axis_near & (V[:, 2] > Eh[2]), 2].max()) if (axis_near & (V[:, 2] > Eh[2])).any() else Eh[2] + 0.1 * u_h
    span = top_h - Eh[2]
    lab_h = None
    if a.labels and os.path.exists(a.labels):
        Lh = json.load(open(a.labels))
        lab_h = np.asarray(Lh["labels"]) if len(Lh["labels"]) == n else None
    # The chin, measured: down from the ears the skin at the front of the head stays forward -
    # lips, chin - until the throat, where it steps back. (The crown-to-ear proportion alone put
    # the line at juno's chin, but hair decides where the crown is: a crop would lift the line
    # above the chin, a big mane drop it into the neck.) Skin only - bangs, a braid over the
    # shoulder or a collar would hide the step.
    skin_ = np.ones(n, bool) if lab_h is None else (lab_h == Lh["group_ids"].get("body", 0))
    col_ = skin_ & (np.abs(V[:, 0] - Eh[0]) < 0.03 * u_h) & (V[:, 1] < Eh[1])
    z_chin = None
    ref_band = col_ & (V[:, 2] < Eh[2] - 0.03 * u_h) & (V[:, 2] > Eh[2] - 0.06 * u_h)
    if ref_band.sum() > 5:
        y_ref = float(np.percentile(V[ref_band, 1], 10))
        last_skin, gap, behind = None, 0, []
        for zz in np.arange(Eh[2] - 0.04 * u_h, Eh[2] - 0.17 * u_h, -0.004 * u_h):
            sl_ = col_ & (np.abs(V[:, 2] - zz) < 0.003 * u_h)
            if sl_.sum() < 2:
                gap += 1
                if last_skin is not None and gap >= 2:              # a collar took over: the chin was above
                    z_chin = behind[0] if behind else last_skin
                    break
                continue
            gap = 0
            # 1.2 cm behind the lower face and staying there for 1.2 cm down: the chin has ended. A
            # realistic chin steps back at once (rowan's 3.6 cm), an anime chin slopes into the neck and
            # never steps; the crease between the lips or under the lower lip dips back for a bin or two
            # and comes forward again (a one-bin test stopped at rowan's mouth).
            if float(V[sl_, 1].min()) > y_ref + 0.012 * u_h:
                behind.append(float(zz + 0.004 * u_h))
                if len(behind) >= 3:
                    z_chin = behind[0]
                    break
            else:
                behind = []
            last_skin = float(zz)
    how_chin = "measured from the face's profile" if z_chin is not None else "from the ear-to-crown distance"
    if z_chin is None:
        z_chin = top_h - 1.5 * span
    z_chin -= 0.008 * u_h                                           # the line runs just under the chin
    z_nape = Eh[2] - 0.35 * span
    y_f, y_b = Eh[1] - 0.5 * span, Eh[1] + 0.5 * span               # the character faces -Y
    zp = z_chin + (np.clip(V[:, 1], y_f, y_b) - y_f) * (z_nape - z_chin) / (y_b - y_f)
    th = np.clip((V[:, 2] - zp) / (0.12 * span) + 1.0, 0.0, 1.0)
    th = th * th * (3.0 - 2.0 * th)
    th[np.linalg.norm(V[:, :2] - Eh[:2], axis=1) > 1.3 * span] = 0.0      # raised hands, wide hats
    if lab_h is not None:
        th[lab_h == Lh["group_ids"].get("clothing", -1)] = 0.0
    before_h = float(np.median(W[th > 0.999, col["head"]])) if (th > 0.999).any() else 0.0
    W *= (1.0 - th)[:, None]
    W[:, col["head"]] += th
    print(f"[rig] head rigid above the jaw line ({(Eh[2] - z_chin) / u_h * 100:.1f} cm below the ears at the "
          f"front, chin {how_chin}): {int((th > 0.999).sum()):,} vertices (their head share was "
          f"{before_h:.2f} median), {int(((th > 0) & (th <= 0.999)).sum()):,} blending into the neck", flush=True)

# rigid feet below the ankle (see rig_retopo.py for the measurement behind it)
u = float(V[:, 2].max() - V[:, 2].min()) / 1.75
heads = {b.name: np.array(b.head_local[:]) for b in arm.bones}
for side, other in (("left", "right"), ("right", "left")):
    ank, toe = f"{side}_ankle", f"{side}_foot"
    if ank not in col:
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
    own[col[ank]] = True
    if toe in col:
        own[col[toe]] = True
    moved = t * W[:, ~own].sum(1)
    W[:, ~own] *= (1.0 - t)[:, None]
    W[:, col[ank]] += moved
    print(f"[rig] {side} foot rigid below the ankle: {int((t > 0.999).sum()):,} vertices", flush=True)

# one leg each below mid-thigh, by connected piece
me = mesh.data
nbr = [[] for _ in range(n)]
E = np.empty(len(me.edges) * 2, np.int64)
me.edges.foreach_get("vertices", E)
for i, j in E.reshape(-1, 2):
    nbr[i].append(j)
    nbr[j].append(i)
legset = {sd: [x for x in (f"{sd}_hip", f"{sd}_knee", f"{sd}_ankle", f"{sd}_foot") if x in col]
          for sd in ("left", "right")}
zc = min((heads[f"{sd}_hip"][2] + heads[f"{sd}_knee"][2]) / 2 for sd in ("left", "right"))
xm = (heads["left_hip"][0] + heads["right_hip"][0]) / 2
left_pos = heads["left_hip"][0] > xm
low_v = V[:, 2] < zc
seen = np.zeros(n, bool)
split = 0
for s0 in np.where(low_v)[0]:
    if seen[s0]:
        continue
    piece, stack = [], [s0]
    seen[s0] = True
    while stack:
        q = stack.pop()
        piece.append(q)
        for w_ in nbr[q]:
            if low_v[w_] and not seen[w_]:
                seen[w_] = True
                stack.append(w_)
    piece = np.array(piece)
    xs = V[piece, 0] - xm
    if (xs > 0).mean() > 0.1 and (xs < 0).mean() > 0.1:
        continue
    sd = "left" if (xs.mean() > 0) == left_pos else "right"
    oth = "right" if sd == "left" else "left"
    W[np.ix_(piece, [col[b] for b in legset[oth]])] = 0.0
    split += 1
print(f"[rig] {split} leg pieces below mid-thigh bound to their own leg", flush=True)

# keep the strongest few, renormalise, write the groups
kmax = a.max_influences
order = np.argsort(-W, axis=1)[:, :kmax]
keep = np.zeros_like(W)
rows = np.arange(n)[:, None]
keep[rows, order] = W[rows, order]
keep /= np.maximum(keep.sum(1, keepdims=True), 1e-9)
unweighted = int((keep.sum(1) < 1e-6).sum())
mesh.vertex_groups.clear()
for c, nm in enumerate(names):
    vg = mesh.vertex_groups.new(name=nm)
    idx = np.nonzero(keep[:, c] > 1e-4)[0]
    for wv in np.unique(np.round(keep[idx, c], 4)):
        sel = idx[np.round(keep[idx, c], 4) == wv]
        vg.add(sel.tolist(), float(wv), "REPLACE")
mesh.parent = rig
mod = mesh.modifiers.new("Armature", "ARMATURE")
mod.object = rig
print(f"[rig] {n:,} vertices, {unweighted} unweighted, {len(names)} deforming bones", flush=True)
if unweighted > 0.005 * n:
    raise SystemExit(f"[rig] {unweighted} vertices unweighted")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump({"bones": len(arm.bones), "deforming": len(names), "finger_bones": len(finger_bones),
               "vertices": n, "unweighted": unweighted}, open(a.json, "w"), indent=1)
print(f"[rig] -> {a.out}", flush=True)
