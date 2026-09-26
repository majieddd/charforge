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


# ---- the arm is only the arm, below the armpit ------------------------------------------------------
# free_arms.py cut each arm free of whatever it was generated against below the armpit - a puffy
# vest's side, a hip. The geodesic weights still reach across the armhole into what was on the other
# side of the cut: Pip's vest kept a third of the upper arm's weight down its side, and baking the
# T-pose alone pulled it into a web from his waist to his elbow. So below the armpit the arm's weight
# stays on the arm's own surface - everything the mesh reaches from the sleeve's outer side without
# going back above the armpit, which the cut made a closed piece - and whatever else carried it lets
# go. A radius round the arm decided "the arm's own surface" first, and took the back of Pip's sleeve
# (wider front to back than side to side) for vest: it let go of the arm and tore. Before the cut this
# tore the surface everywhere (REVIEW, Tried and dropped); after it there is nothing to tear, and where
# a bridge survives the flood crosses it and nothing changes.
# each vertex's colour, for telling two garments apart where the geometry cannot (below)
vlab = None
if a.albedo and os.path.exists(a.albedo) and mesh.data.uv_layers:
    smp = bpy.data.images.load(os.path.abspath(a.albedo), check_existing=False)
    smp.scale(1024, 1024)
    px_ = np.empty(1024 * 1024 * 4, np.float32)
    smp.pixels.foreach_get(px_)
    px_ = px_.reshape(1024, 1024, 4)[..., :3]
    bpy.data.images.remove(smp)
    luv_ = np.empty(len(mesh.data.loops) * 2)
    mesh.data.uv_layers.active.data.foreach_get("uv", luv_)
    lvi_ = np.empty(len(mesh.data.loops), np.int64)
    mesh.data.loops.foreach_get("vertex_index", lvi_)
    vuv_ = np.zeros((len(V), 2))
    vuv_[lvi_] = luv_.reshape(-1, 2)
    xi_ = np.clip((vuv_[:, 0] * 1024).astype(int), 0, 1023)
    yi_ = np.clip((vuv_[:, 1] * 1024).astype(int), 0, 1023)
    rgb_ = np.clip(px_[yi_, xi_], 0, 1).astype(np.float64)
    lin_ = np.where(rgb_ <= 0.04045, rgb_ / 12.92, ((rgb_ + 0.055) / 1.055) ** 2.4)
    xyz_ = lin_ @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz_ /= np.array([0.95047, 1.0, 1.08883])
    f_ = np.where(xyz_ > 0.008856, np.cbrt(xyz_), 7.787 * xyz_ + 16 / 116)
    vlab = np.stack([116 * f_[:, 1] - 16, 500 * (f_[:, 0] - f_[:, 1]), 200 * (f_[:, 1] - f_[:, 2])], 1)

me_ = mesh.data
ed = np.empty(len(me_.edges) * 2, np.int64)
me_.edges.foreach_get("vertices", ed)
ed = ed.reshape(-1, 2)
# the surface as one piece: the imported mesh is split along every UV seam into islands, and a flood
# over its edges stopped at the first seam - it left the rest of Pip's sleeve and his hands behind
_, weld = np.unique(np.round(V, 5), axis=0, return_inverse=True)
weld = weld.ravel()
ed = np.unique(np.sort(weld[ed], axis=1), axis=0)
ed = ed[ed[:, 0] != ed[:, 1]]
nw = int(weld.max()) + 1
def smooth_on_surface(val, iters=6):
    """A per-vertex value averaged over its welded neighbours: a ramp across a seam, not a step."""
    vw_ = np.zeros(nw); cn_ = np.zeros(nw)
    np.add.at(vw_, weld, val); np.add.at(cn_, weld, 1)
    vw_ /= np.maximum(cn_, 1)
    for _ in range(iters):
        acc_ = vw_.copy(); c2_ = np.ones(nw)
        np.add.at(acc_, ed[:, 0], vw_[ed[:, 1]]); np.add.at(acc_, ed[:, 1], vw_[ed[:, 0]])
        np.add.at(c2_, ed[:, 0], 1); np.add.at(c2_, ed[:, 1], 1)
        vw_ = acc_ / c2_
    return vw_[weld]



SPN = [np.array(J[b]) for b in ("pelvis", "spine1", "spine2", "spine3", "neck") if b in J]
for side in ("left", "right"):
    if not all(f"{side}_{b}" in J and f"{side}_{b}" in col for b in ("shoulder", "elbow")):
        continue
    S_, E_, W_ = (np.array(J[f"{side}_{b}"]) for b in ("shoulder", "elbow", "wrist"))
    best_d, best_t, best_c = np.full(n, np.inf), np.zeros(n), np.zeros((n, 3))
    for p0, p1, t0_ in ((S_, E_, 0.0), (E_, W_, 1.0)):
        ab = p1 - p0
        u = np.clip(((V - p0) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
        c = p0 + u[:, None] * ab
        d = np.linalg.norm(V - c, axis=1)
        better = d < best_d
        best_d[better], best_t[better], best_c[better] = d[better], t0_ + u[better], c[better]
    zs = np.array([p[2] for p in SPN])
    ki = np.clip(np.searchsorted(zs, best_c[:, 2]) - 1, 0, len(SPN) - 2)
    fz = np.clip((best_c[:, 2] - zs[ki]) / np.maximum(zs[ki + 1] - zs[ki], 1e-9), 0, 1)
    mid = np.array(SPN)[ki] * (1 - fz)[:, None] + np.array(SPN)[ki + 1] * fz[:, None]
    ax = np.where((best_t < 1.0)[:, None], E_ - S_, W_ - E_)
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    rad = V - best_c
    rad -= (rad * ax).sum(1, keepdims=True) * ax
    med = mid - best_c
    med -= (med * ax).sum(1, keepdims=True) * ax
    cosphi = (rad * med).sum(1) / np.maximum(np.linalg.norm(rad, axis=1) * np.linalg.norm(med, axis=1), 1e-9)
    r_out = np.median(best_d[(best_t > 0.4) & (best_t < 1.6) & (cosphi < -0.7) & (best_d < 0.2)]) \
        if ((best_t > 0.4) & (best_t < 1.6) & (cosphi < -0.7) & (best_d < 0.2)).sum() > 20 else 0.08
    # the region below the armpit, and the arm's surface in it: flooded from the sleeve's outer side
    # as far down as free_arms.py cuts: past it the cuff can still touch a pocket, and the flood went down
    # the sleeve, across the pocket and up the vest's side. And from a little below where the cut opens
    # (0.25 of the upper arm), not from the opening itself: there the sleeve and the vest are still one
    # surface, and the flood walked onto the vest's side on both of Pip's arms - the left let go of
    # nothing, the right of 5 vertices, and raising his arms lifted the whole vest. From 0.30 the flood
    # stays on the arm, and 556 and 854 vertices of the vest let go.
    region = (best_t >= 0.30) & (best_t <= 1.7) & (best_d < 4 * r_out)
    seed = region & (best_t > 0.4) & (best_t < 1.6) & (cosphi < -0.6) & (best_d < 1.4 * r_out)
    reg_w = np.zeros(nw, bool)
    reg_w[weld[region]] = True
    on_w = np.zeros(nw, bool)
    on_w[weld[seed]] = True
    ok_e = reg_w[ed[:, 0]] & reg_w[ed[:, 1]]
    ea, eb2 = ed[ok_e, 0], ed[ok_e, 1]
    for _ in range(4000):
        grow = np.zeros(nw, bool)
        grow[eb2[on_w[ea] & ~on_w[eb2]]] = True
        grow[ea[on_w[eb2] & ~on_w[ea]]] = True
        if not grow.any():
            break
        on_w |= grow
    on_arm = on_w[weld]
    if os.environ.get("CF_DEBUG_ARMS"):
        np.savez(os.environ["CF_DEBUG_ARMS"] + f"_{side}.npz", V=V, weld=weld, ed=ed, t=best_t, d=best_d,
                 cosphi=cosphi, r_out=r_out, seed=seed, region=region, on_arm=on_arm)
    arm_cols = [col[f"{side}_{b}"] for b in ("shoulder", "elbow", "wrist") if f"{side}_{b}" in col]
    # The armhole. Round the shoulder a vest's armhole is the arm's surface - the sleeve goes in under
    # it and the solid has only the outside - so it takes half the arm's weight as any shoulder does,
    # and raising the arms lifted Pip's vest into flaps. No distance tells the vest from the sleeve
    # there; their colours can: where the torso's garment beside the arm and the sleeve differ, what is
    # the torso's garment round the shoulder moves with the collarbone instead of the arm. Below the
    # armpit, where the flood above reached the vest's side through what the cut left joined (both of
    # Pip's arms: nothing let go on the left), the same colour lets go of it past the arm's radius.
    # One colour for both - a jacket - and nothing here changes.
    vest_s = None
    if vlab is not None:
        sl_m = (best_t > 0.5) & (best_t < 0.9) & (cosphi < -0.3) & (best_d < 1.4 * r_out)
        to_m = (best_t > 0.3) & (best_t < 0.7) & (cosphi > 0.3) & (best_d > 1.6 * r_out) & (best_d < 3.5 * r_out)
        if sl_m.sum() > 50 and to_m.sum() > 50:
            c_sl, c_to = np.median(vlab[sl_m], 0), np.median(vlab[to_m], 0)
            gap_c = float(np.linalg.norm(c_sl - c_to))
            if gap_c > 20:
                vest = np.clip((np.linalg.norm(vlab - c_sl, axis=1) - np.linalg.norm(vlab - c_to, axis=1)) / gap_c
                               + 0.5, 0, 1)
                vest_s = smooth_on_surface(vest)
                band = np.clip(best_t / 0.05, 0, 1) * np.clip((0.4 - best_t) / 0.1, 0, 1) * (best_d < 2.5 * r_out)
                rel = np.clip((vest_s - 0.35) / 0.3, 0, 1) * band
                aw_ = W[:, arm_cols].sum(1)
                mv_ = (rel > 0) & (aw_ > 1e-4)
                if mv_.any():
                    to_col = col.get(f"{side}_collar", col.get("spine3", 0))
                    moved = W[np.ix_(mv_, arm_cols)] * rel[mv_, None]
                    W[np.ix_(mv_, arm_cols)] -= moved
                    W[mv_, to_col] += moved.sum(1)
                print(f"[rig] {side} armhole: the torso's garment {tuple(int(x) for x in c_to)} and the sleeve "
                      f"{tuple(int(x) for x in c_sl)} (Lab, {gap_c:.0f} apart): {int((mv_ & (rel > 0.5)).sum()):,} "
                      f"vertices round the shoulder moved from the arm to the collarbone", flush=True)
            else:
                print(f"[rig] {side} armhole: sleeve and torso the same colour ({gap_c:.0f} apart) - left as weighted",
                      flush=True)
    # from just inside where the cut starts (0.25 of the upper arm): above it the vest and the arm are
    # still one surface
    f = np.clip((best_t - 0.28) / 0.08, 0, 1) * np.clip((1.7 - best_t) / 0.1, 0, 1)   # the wrist and hand have rules of their own
    f = f * f * (3 - 2 * f)
    before_ = W[:, arm_cols].sum(1)
    hit = region & ~on_arm & (f > 0) & (before_ > 1e-4)
    leaked = int((on_arm & (best_d > 2.2 * r_out) & (cosphi > 0.3)).sum())
    if not hit.any():
        print(f"[rig] {side} arm: nothing beyond it below the armpit carries its weight", flush=True)
        continue
    W[np.ix_(hit, arm_cols)] *= (1 - f[hit])[:, None]
    lost = W[hit].sum(1) < 1e-4                                         # nothing left: the chest carries it
    if lost.any():
        W[np.nonzero(hit)[0][lost], col["spine3"] if "spine3" in col else 0] = 1.0
    print(f"[rig] {side} arm: its own surface below the armpit {int(on_arm.sum()):,} vertices; "
          f"{int((hit & (f > 0.5)).sum()):,} beyond it let go of the arm"
          + (f" ({leaked} reached across a surviving bridge)" if leaked else ""), flush=True)


# ---- the hem: a jacket over trousers does not follow the thighs --------------------------------------
# The weights reach from the thighs up over the hips into whatever hangs there: the lowest part of Pip's
# vest carried half its weight on the thighs (median 0.50, up to 0.70), and spreading the legs in a
# jump pulled its hem into a W; Juno's and Wren's jackets wrapped round their thighs in a squat. The
# upper garment's surface near the hips follows the pelvis instead, so the hem is where the stretch
# goes. Which surface is the upper garment is its colour - the torso's colours against the upper
# thighs' - and where those are the same (one colour head to foot, or skin above and below) nothing
# here changes.
def lab_clusters(X, k=3, iters=8):
    """Up to k colours that each cover a tenth of X, by k-means."""
    if len(X) < k * 10:
        return X.mean(0, keepdims=True)
    X = X[:: max(1, len(X) // 4000)]
    C = X[np.linspace(0, len(X) - 1, k).astype(int)].copy()
    for _ in range(iters):
        lb = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
        C = np.array([X[lb == j].mean(0) if (lb == j).any() else C[j] for j in range(k)])
    share = np.bincount(lb, minlength=k) / len(X)
    return C[share > 0.1]


leg_names = [b for b in ("left_hip", "right_hip", "left_knee", "right_knee") if b in col]
if vlab is not None and len(leg_names) == 4 and all(b in J for b in ("left_hip", "right_hip", "left_knee",
                                                                         "right_knee", "spine1", "spine3")):
    legc = [col[b] for b in leg_names]
    armc = [col[b] for b in col if b.startswith(("left_", "right_"))
            and not b.endswith(("_hip", "_knee", "_ankle", "_foot", "_toe"))]
    z_hip = float(J["left_hip"][2] + J["right_hip"][2]) / 2
    z_knee = float(J["left_knee"][2] + J["right_knee"][2]) / 2
    z_s1, z_s3 = float(J["spine1"][2]), float(J["spine3"][2])
    lw_ = W[:, legc].sum(1)
    aw_ = W[:, armc].sum(1) if armc else np.zeros(n)
    up_band = (V[:, 2] > z_s1) & (V[:, 2] < z_s3) & (aw_ < 0.2) & (lw_ < 0.05)
    lo_band = (V[:, 2] > z_knee + 0.3 * (z_hip - z_knee)) & (V[:, 2] < z_hip - 0.05 * (z_hip - z_knee)) & (lw_ > 0.7)
    y_hip = float(J["left_hip"][1] + J["right_hip"][1]) / 2

    def seat(rel_, upper_s_):
        """A long jacket rides on the seat: behind the hip joints and more than 0.15 of the way from them to
        the knees, the garment keeps most of the thighs' pull - let go, it stayed with the pelvis while the
        seat dropped with the thighs, and the stretch between showed the hem's pale underside (Mara, Juno
        and Rowan crouching). Pip's vest ends 0.12 of the way down and lets go all over. (Measuring how far
        each garment reaches failed: colour took in Pip's shorts, the parser gave the back of Mara's jacket
        to her trousers.)"""
        L_ = z_hip - z_knee
        s_back = np.clip((V[:, 1] - y_hip) / 0.03 + 0.5, 0, 1)            # the front is -y
        s_below = np.clip((z_hip - 0.15 * L_ - V[:, 2]) / (0.1 * L_), 0, 1)
        k_ = s_back * s_below
        return rel_ * (1 - 0.8 * k_), bool(((k_ > 0.5) & (rel_ > 0)).any())

    # (The human parser's own classes - "top" over "pants" - were tried first and dropped: it gave the
    # back of Mara's, Juno's and Rowan's jackets to their trousers, the rule split each jacket across
    # the seat, and crouching stretched it into a pale band.)
    if up_band.sum() > 200 and lo_band.sum() > 200:
        Cu, Cl = lab_clusters(vlab[up_band]), lab_clusters(vlab[lo_band])
        # a colour both have (skin above and below, one fabric throughout) tells nothing: left out of both
        shared_u = np.array([np.linalg.norm(Cl - c, axis=1).min() < 15 for c in Cu])
        shared_l = np.array([np.linalg.norm(Cu - c, axis=1).min() < 15 for c in Cl])
        Cu_, Cl_ = Cu[~shared_u], Cl[~shared_l]
        if len(Cu_) and len(Cl_):
            gap_h = float(min(np.linalg.norm(Cl_ - c, axis=1).min() for c in Cu_))
            du_ = np.min(np.linalg.norm(vlab[:, None, :] - Cu_[None], axis=2), 1)
            dl_ = np.min(np.linalg.norm(vlab[:, None, :] - np.vstack([Cl_, Cu[shared_u]])[None], axis=2), 1)
            upper_s = smooth_on_surface(np.clip((dl_ - du_) / max(gap_h, 1e-6) + 0.5, 0, 1))
            zb = np.clip((V[:, 2] - (z_knee + 0.55 * (z_hip - z_knee))) / (0.15 * (z_hip - z_knee)), 0, 1) \
                * (V[:, 2] < z_s1)
            rel = np.clip((upper_s - 0.6) / 0.25, 0, 1) * zb * (aw_ < 0.5)
            rel, long_ = seat(rel, upper_s)
            hit_h = (rel > 0) & (lw_ > 1e-4)
            if hit_h.any():
                moved = W[np.ix_(hit_h, legc)] * rel[hit_h, None]
                W[np.ix_(hit_h, legc)] -= moved
                W[hit_h, col["pelvis"] if "pelvis" in col else 0] += moved.sum(1)
            fmt = lambda C_: "/".join(f"({int(c[0])},{int(c[1])},{int(c[2])})" for c in C_)
            print(f"[rig] hem: the torso's garment {fmt(Cu_)} over {fmt(Cl_)} (Lab, {gap_h:.0f} apart): "
                  f"{int((hit_h & (rel > 0.5)).sum()):,} vertices near the hips let go of the thighs"
                  + (" (below the seat line it still rides on the seat)" if long_ else ""), flush=True)
        else:
            fmt0 = lambda C_: "/".join(f"({int(c[0])},{int(c[1])},{int(c[2])})" for c in C_)
            print(f"[rig] hem: the torso {fmt0(Cu)} and the thighs {fmt0(Cl)} share their colours - left as weighted",
                  flush=True)


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
        b_ = float(hd.get("bulk", 1.0))
        dh, _ = hand_model.sdf(q * np.array([1.0, 1.0 / b_, 1.0 / b_]), wrist_r=tuple(hd.get("wrist_radius", (0.15, 0.10))))
        dh = dh * b_
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
        # ...but only what hangs from the body. A piece sewn onto a leg - Pip's cargo pocket, labelled an
        # accessory - bound to the torso stayed behind when the thigh swung and stretched into a plank.
        # Bags hang from above the hips (Wren's satchel's top 0.22 of her height above the hip joints,
        # Vex's the same); a pocket sits wholly below them, so a piece whose top is below the hip
        # joints keeps its leg's weights.
        _, wld = np.unique(np.round(V, 5), axis=0, return_inverse=True)
        wld = wld.ravel()
        acc_w = np.zeros(int(wld.max()) + 1, bool)
        acc_w[wld[acc]] = True
        me0 = mesh.data
        e0 = np.empty(len(me0.edges) * 2, np.int64)
        me0.edges.foreach_get("vertices", e0)
        e0 = wld[e0.reshape(-1, 2)]
        e0 = e0[(e0[:, 0] != e0[:, 1]) & acc_w[e0[:, 0]] & acc_w[e0[:, 1]]]
        parent_ = np.arange(len(acc_w))

        def root(x):
            while parent_[x] != x:
                parent_[x] = parent_[parent_[x]]
                x = parent_[x]
            return x
        for i_, j_ in e0:
            ri, rj = root(i_), root(j_)
            if ri != rj:
                parent_[ri] = rj
        comp = np.array([root(x) for x in wld])
        z_hip = (J["left_hip"][2] + J["right_hip"][2]) / 2 if "left_hip" in J and "right_hip" in J else -1e9
        on_leg = np.zeros(n, bool)
        for c_ in np.unique(comp[acc]):
            vv = np.nonzero(acc & (comp == c_))[0]
            if V[vv, 2].max() < z_hip - 0.02 * 2.0:
                on_leg[vv] = True
        if on_leg.any():
            print(f"[rig] {int(on_leg.sum()):,} accessory vertices below the hip joints left on their leg (a pocket, not a bag)", flush=True)
        acc = acc & ~on_leg
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

# ---- loose pieces ride with the body part they sit on ------------------------------------------
# A piece of the mesh not joined to the body - Kaito's sandal soles, a hair spike, a button - has no
# path through the body for the weights to follow, and was left behind: his soles stayed on the floor
# when he jumped. Each rides rigidly with the body part nearest it (the weights of the body vertex
# closest to it). The modelled hands are pieces of their own and have their own rule, left alone.
from mathutils.kdtree import KDTree as _KD                         # Blender's Python has no scipy
lab_w = np.arange(nw)
while True:                                   # connected pieces: labels spread over the welded edges
    m_ = np.minimum(lab_w[ed[:, 0]], lab_w[ed[:, 1]])
    nxt = lab_w.copy()
    np.minimum.at(nxt, ed[:, 0], m_)
    np.minimum.at(nxt, ed[:, 1], m_)
    nxt = nxt[nxt]
    if np.array_equal(nxt, lab_w):
        break
    lab_w = nxt
_, lab_w = np.unique(lab_w, return_inverse=True)
_nc = int(lab_w.max()) + 1
lab_v = lab_w[weld]
sizes_ = np.bincount(lab_v, minlength=_nc)
main_ = int(np.argmax(sizes_))
finger_cols = [col[b] for b in col if any(f in b for f in FINGERS)]
body_v = np.nonzero(lab_v == main_)[0]
moved_p = moved_v = 0
if len(body_v) and _nc > 1:
    kd_ = _KD(len(body_v))
    for i_, v_ in enumerate(V[body_v]):
        kd_.insert(v_.tolist(), i_)
    kd_.balance()
    for c_ in range(_nc):
        if c_ == main_:
            continue
        piece = np.nonzero(lab_v == c_)[0]
        if not len(piece) or (finger_cols and W[np.ix_(piece, finger_cols)].sum() > 1e-3):
            continue                                                   # a modelled hand
        best_ = min((kd_.find(v_.tolist()) for v_ in V[piece]), key=lambda r_: r_[2])
        src_ = body_v[best_[1]]
        W[piece] = W[src_][None, :]
        moved_p += 1
        moved_v += len(piece)
if moved_p:
    print(f"[rig] {moved_p} loose pieces ({moved_v:,} vertices) ride with the body part nearest each", flush=True)

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
