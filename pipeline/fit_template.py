"""A professional body template fitted under the generated character (E149): NAVER's Anny (Apache-2.0 code, CC0
MakeHuman data) - a parametric body with quad-able topology, 104 bones with fingers and eyes, UVs and the 52 ARKit
facial blendshapes - posed and shaped so that it lies on the character's bare skin and inside its clothes.

    .venv/bin/python pipeline/fit_template.py --work work/<n> --height 1.68 --out work/<n>/template [--iters 400]

Why. The generator makes one fused surface, and most of the pipeline's repairs exist to recover structure it never had
(hands, eyes, mouth, arms apart from the body, a face rig). A template brings that structure with it; what lies outside
the fitted body is clothing and hair - layers, not a fused shell.

How. Everything in metres (the mesh's units times height / mesh height), Blender's frame (Z up, facing -Y), which is
Anny's. Parameters: a global rotation, translation and scale; Anny's six phenotypes (gender, age, muscle, weight,
height, proportions; squashed into 0-1) and its 254 local shape changes (kept small); a rotation for each main bone.
Losses, in order of weight:
  joints    our refined joints on Anny's matching bone heads (pelvis, neck, head, shoulders, elbows, wrists, hips,
            knees, ankles) - the pose
  inside    no body vertex outside the character's clean solid (solid_free.npz, signed: negative inside)
  skin      every point of the generated surface the parser calls bare skin (face, arms, legs, torso) on the body
  depth     body vertices not much deeper than DEPTH under the surface (clothes are a few centimetres thick, not 10)
  prior     small local changes and small bone rotations
Hands and feet are left out of the surface terms: the generated ones are paddles and shoes. Writes template.glb (the
fitted body), layers.glb (the generated mesh coloured by distance to the body: on it, near it, outside it),
fit.json (parameters and measures).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import roma
import torch
import trimesh
from scipy.spatial import cKDTree

import anny

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--height", type=float, required=True, help="the character's height in metres")
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--depth", type=float, default=0.03, help="how deep under the surface the body may lie (m)")
ap.add_argument("--wrap", type=int, default=0, help="then wrap the template onto the skin and tight clothing: iterations")
ap.add_argument("--tight", type=float, default=0.03, help="surface within this distance of the fitted body is wrapped (m)")
ap.add_argument("--resume", action="store_true", help="start from <out>/fit_params.pt instead of fitting again")
ap.add_argument("--landmarks", default=None,
                help="landmarks.json: the template's face-point vertices and the character's face points in metres (E149); "
                     "the wrap lays one on the other, and keeps the eyes' and mouth's insides as the template has them")
ap.add_argument("--hands", default=None,
                help="hands.json: fit the template's hands (finger bones, hand and finger size) to the character's "
                     "modelled knuckles and fingertips; without it the template keeps its own hands (E149)")
ap.add_argument("--face-data", type=float, default=0.25, help="the surface term's weight on the face, against 1 elsewhere")
a = ap.parse_args()
W, OUT = Path(a.work), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
torch.set_default_dtype(torch.float64)
t0 = time.time()

# ---- the character, in metres ------------------------------------------------------------------
P = np.load(W / "sdf.npz")["points"]
lo, hi = P.min(0), P.max(0)
kk, cc = 2.0 / float(hi[2] - lo[2]), (lo + hi) / 2
k_m = a.height / float(hi[2] - lo[2])                     # mesh units -> metres
J = {k: (np.array(v) / kk + cc) * k_m for k, v in json.load(open(W / "joints_refined.json"))["joints"].items()}
S = np.load(W / ("solid_free.npz" if (W / "solid_free.npz").exists() else "solid.npz"))
sdf = torch.tensor(S["sdf"], dtype=torch.float64).permute(2, 1, 0)[None, None]  # grid_sample wants (z, y, x)
vox, org = float(S["voxel"]), np.asarray(S["origin_ijk"], float)
shape_ijk = np.array(S["sdf"].shape, float)


def sdf_at(X):
    """The solid's signed distance (metres) at points X (metres): trilinear, positive outside."""
    ijk = X / k_m / vox - torch.tensor(org)                # voxel coordinates
    g = 2.0 * ijk / torch.tensor(shape_ijk - 1) - 1.0      # -> [-1, 1], x y z order for grid_sample
    v = torch.nn.functional.grid_sample(sdf, g[None, None, None], align_corners=True, padding_mode="border")
    return v.reshape(-1) * k_m


gm = trimesh.load(W / "mesh.glb", force="mesh", process=False)
GV = np.asarray(gm.vertices) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)   # glTF Y-up -> Blender Z-up
GV = GV * k_m
parts = json.load(open(W / "parts.json"))
names = {int(k): v for k, v in parts["id2label"].items()}
cls = np.array([names.get(c, "background") for c in parts["class_labels"]])
skin_cls = {"face", "arms", "legs", "torso"}
# bare skin, away from the hands and feet (paddles and shoes)
wristL, wristR, ankL, ankR = J["left_wrist"], J["right_wrist"], J["left_ankle"], J["right_ankle"]
far_ext = ((np.linalg.norm(GV - wristL, axis=1) > 0.06) & (np.linalg.norm(GV - wristR, axis=1) > 0.06)
           & (GV[:, 2] > max(ankL[2], ankR[2]) + 0.05))
skin_idx = np.nonzero(np.isin(cls, list(skin_cls)) & far_ext)[0]
rng = np.random.default_rng(0)
skin_pts = torch.tensor(GV[rng.choice(skin_idx, min(len(skin_idx), 20000), replace=False)]) if len(skin_idx) else None

# ---- the template ------------------------------------------------------------------------------
m = anny.Anny(local_changes="default", facial_actions="all")
L = list(m.bone_labels)
B = m.bone_count
MAP = {"pelvis": "root", "neck": "neck01", "head": "head", "left_shoulder": "upperarm01.L", "right_shoulder": "upperarm01.R",
       "left_elbow": "lowerarm01.L", "right_elbow": "lowerarm01.R", "left_wrist": "wrist.L", "right_wrist": "wrist.R",
       "left_hip": "upperleg01.L", "right_hip": "upperleg01.R", "left_knee": "lowerleg01.L", "right_knee": "lowerleg01.R",
       "left_ankle": "foot.L", "right_ankle": "foot.R"}
jn = [k for k in MAP if k in J]
J_t = torch.tensor(np.stack([J[k] for k in jn]))
jb = [L.index(MAP[k]) for k in jn]
POSED = [b for b in L if any(b.startswith(p) for p in ("spine", "neck", "head", "clavicle", "shoulder01", "upperarm", "lowerarm",
                                                       "wrist", "upperleg", "lowerleg", "foot", "pelvis")
                             + (("finger", "metacarpal") if a.hands else ()))]
pidx = [L.index(b) for b in POSED]
# vertices that belong to hands, feet and eyes, from the skinning weights: left out of the surface terms
bvi, bvw = m.vertex_bone_indices, m.vertex_bone_weights      # (V, k) bones and weights per vertex
ext_bones = [i for i, b in enumerate(L) if b.startswith(("finger", "metacarpal", "wrist", "toe", "foot", "eye"))]
ext = torch.zeros(B, dtype=torch.bool)
ext[ext_bones] = True
w_ext = (bvw * ext[bvi.long()].double()).sum(-1)          # (V,) weight on hands, feet, eyes
body_v = w_ext < 0.3

# the eyes, where eye_fill.py found the generated model's openings (x and z only: their depth is the lids')
eye_t, eye_b = [], []
ej = W / "eyes" / "eyes.json"
if ej.exists():
    E = [e for e in json.load(open(ej)).get("eyes", []) if e.get("used")]
    for e in E:
        x, z = np.array(e["centre_xz"]) * k_m                                    # Blender frame, mesh units
        eye_t.append([x, z])
        eye_b.append(L.index("eye.L" if x > J["head"][0] else "eye.R"))
eye_t = torch.tensor(eye_t) if eye_t else None

# the character's hands: wrist, knuckles and fingertips (hands.json, the generated mesh's glTF frame and units)
hand_b, hand_t = [], []
if a.hands:
    HJ = json.load(open(a.hands))["sides"]
    def _fit_frame(q):
        q = np.asarray(q, float)
        return np.array([q[0], -q[2], q[1]]) * k_m
    for side, sfx in (("left", "L"), ("right", "R")):
        if side not in HJ:
            continue
        hand_b.append(L.index(f"wrist.{sfx}"))
        hand_t.append(_fit_frame(HJ[side]["wrist"]))
        for f, nm in enumerate(("thumb", "index", "middle", "ring", "pinky"), start=1):
            pts = HJ[side]["fingers"][nm]
            for k_ in (1, 2, 3):
                hand_b.append(L.index(f"finger{f}-{k_}.{sfx}"))
                hand_t.append(_fit_frame(pts[k_ - 1]))
hand_t = torch.tensor(np.array(hand_t)) if hand_b else None

fit_lm_idx, fit_lm_tgt = None, None
if a.landmarks:
    LM0 = json.load(open(a.landmarks))
    ok0 = [i for i, (t_, g_) in enumerate(zip(LM0["template_vertex"], LM0["character_m"])) if t_ is not None and g_ is not None]
    fit_lm_idx = torch.tensor([LM0["template_vertex"][i] for i in ok0])
    fit_lm_tgt = torch.tensor([LM0["character_m"][i] for i in ok0])

n_ph = len(m.phenotype_labels)
ph = torch.zeros(n_ph, requires_grad=True)                 # logits; sigmoid 0.5 = Anny's average
lc = torch.zeros(len(m.local_change_labels), requires_grad=True)
rv = torch.zeros(len(pidx), 3, requires_grad=True)         # bone rotations, axis-angle
g_rot = torch.zeros(3, requires_grad=True)
g_t = torch.tensor(J["pelvis"], requires_grad=True)
g_s = torch.zeros(1, requires_grad=True)                   # log scale


def run():
    pose = torch.eye(4)[None, None].repeat(1, B, 1, 1)
    R = roma.rotvec_to_rotmat(rv)
    pose = pose.clone()
    pose[0, pidx, :3, :3] = R
    out = m(pose_parameters=pose, phenotype_kwargs={k: torch.sigmoid(ph[i])[None] for i, k in enumerate(m.phenotype_labels)},
            local_changes_kwargs=torch.tanh(lc)[None])
    Rg = roma.rotvec_to_rotmat(g_rot)
    s = torch.exp(g_s)
    V = (out["vertices"][0] @ Rg.T) * s + g_t
    H = (out["bone_poses"][0][:, :3, 3] @ Rg.T) * s + g_t
    return V, H


if a.resume and (OUT / "fit_params.pt").exists():
    st_ = torch.load(OUT / "fit_params.pt")
    with torch.no_grad():
        for t_, k_ in ((ph, "ph"), (lc, "lc"), (rv, "rv"), (g_rot, "g_rot"), (g_t, "g_t"), (g_s, "g_s")):
            if k_ == "rv" and st_[k_].shape != t_.shape:
                # more posed bones now (the fingers, --hands): the saved ones keep their turns
                old_names = st_.get("posed")
                if old_names:
                    for i_, b_ in enumerate(old_names):
                        if b_ in POSED:
                            t_[POSED.index(b_)] = st_[k_][i_]
                continue
            t_.copy_(st_[k_])
    a.iters = 0 if not a.landmarks else max(a.iters // 2, 100)
    if a.hands:
        a.iters = max(a.iters, 200)
opt = torch.optim.Adam([{"params": [g_rot, g_t, g_s], "lr": 0.01}, {"params": [rv], "lr": 0.02},
                        {"params": [ph], "lr": 0.05}, {"params": [lc], "lr": 0.02}])
hist = []
for it in range(a.iters):
    stage2 = it >= a.iters // 3                              # pose first, then the surface
    V, H = run()
    l_j = ((H[jb] - J_t) ** 2).sum(-1).mean()
    if eye_t is not None and stage2:
        l_eye = ((H[eye_b][:, [0, 2]] - eye_t) ** 2).sum(-1).mean()
    else:
        l_eye = torch.zeros(())
    loss = 100.0 * l_j + 300.0 * l_eye + 0.02 * (rv ** 2).sum() + 0.002 * (lc ** 2).sum()
    if hand_t is not None:
        loss = loss + 100.0 * ((H[hand_b] - hand_t) ** 2).sum(-1).mean()
    if stage2:
        d = sdf_at(V[body_v])
        l_in = torch.relu(d).pow(2).mean()
        l_deep = torch.relu(-d - a.depth).pow(2).mean()
        loss = loss + 400.0 * l_in + 40.0 * l_deep
        if fit_lm_idx is not None:
            # the face's features where the character's are, through the template's own face shape changes (its
            # local changes), not by moving vertices one at a time - which crumpled lids and chin (E149)
            loss = loss + 300.0 * ((V[fit_lm_idx] - fit_lm_tgt) ** 2).sum(-1).mean()
        if skin_pts is not None:
            Vb = V[body_v]
            idx_ = torch.cdist(skin_pts[::4], Vb.detach()).argmin(1)     # nearest body vertex for each skin point
            l_skin = ((Vb[idx_] - skin_pts[::4]) ** 2).sum(-1).mean()
            loss = loss + 200.0 * l_skin
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 50 == 0 or it == a.iters - 1:
        hist.append({"it": it, "loss": float(loss), "joints_cm": float(l_j.sqrt() * 100)})
        print(f"[template] it {it:4d} loss {float(loss):.4f} joints {float(l_j.sqrt()) * 100:.1f} cm", flush=True)

torch.save({"ph": ph.detach(), "lc": lc.detach(), "rv": rv.detach(), "g_rot": g_rot.detach(), "g_t": g_t.detach(),
            "g_s": g_s.detach(), "posed": POSED}, OUT / "fit_params.pt")
with torch.no_grad():
    Vfit, _ = run()
trimesh.Trimesh(Vfit.numpy() @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), m.get_triangular_faces().numpy(),
                process=False).export(OUT / "fitted.glb")

# ---- the wrap: the template's own topology laid onto the skin and tight clothing -------------------
wrap_info = None
if a.wrap:
    import scipy.sparse as sp
    with torch.no_grad():
        V0, H0 = run()
        d0 = sdf_at(V0)
    Fn = m.get_triangular_faces().numpy()
    nV = V0.shape[0]
    ed = np.concatenate([Fn[:, [0, 1]], Fn[:, [1, 2]], Fn[:, [2, 0]]])
    ed = np.unique(np.sort(ed, 1), axis=0)
    A = sp.coo_matrix((np.ones(2 * len(ed)), (np.r_[ed[:, 0], ed[:, 1]], np.r_[ed[:, 1], ed[:, 0]])), shape=(nV, nV)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    Lap = (sp.diags(np.ones(nV)) - sp.diags(1.0 / np.maximum(deg, 1)) @ A).tocoo()
    Lt = torch.sparse_coo_tensor(np.vstack([Lap.row, Lap.col]), Lap.data, (nV, nV))
    # which vertices follow the surface: near it (tight), not hands, feet or eyes - the template's own are kept
    eyes_hands = w_ext >= 0.3
    # not onto hair: the scalp under a hair shell keeps the template's shape
    gtree = cKDTree(GV)
    _, gi = gtree.query(V0.numpy())
    on_hair = torch.tensor(np.isin(cls[gi], ["hair", "hat"]))
    # and not the head: its shape is the template's own, fitted by landmarks above
    above_neck = torch.tensor(V0[:, 2].numpy() > J["neck"][2] + 0.02)
    w_d = ((d0.abs() < a.tight) & ~eyes_hands & ~on_hair & ~above_neck).double()
    lm_idx, lm_tgt = None, None
    if a.landmarks:
        LM = json.load(open(a.landmarks))
        ok_ = [i for i, (t_, g_) in enumerate(zip(LM["template_vertex"], LM["character_m"])) if t_ is not None and g_ is not None]
        lm_idx = torch.tensor([LM["template_vertex"][i] for i in ok_])
        lm_tgt = torch.tensor([LM["character_m"][i] for i in ok_])
        lm_lab = [i for i in ok_]
        # the face: the surface term eased there, and none inside the eyes' and the mouth's outlines (the generated
        # eye openings and lip seam are what the template replaces)
        fc = lm_tgt.mean(0)
        face_r = float((lm_tgt[:, [0, 2]] - fc[[0, 2]]).norm(dim=1).max()) * 1.15
        in_face = ((V0[:, [0, 2]] - fc[[0, 2]]).norm(dim=1) < face_r) & (V0[:, 1] < fc[1] + 0.03)
        w_d = torch.where(in_face, w_d * a.face_data, w_d)
        P_all = torch.tensor(LM["template_vertex_all"]) if "template_vertex_all" in LM else None
        for grp in (range(36, 42), range(42, 48), range(60, 68)):
            gv_ = [LM["template_vertex"][i] for i in grp if LM["template_vertex"][i] is not None]
            if len(gv_) >= 3:
                c_ = V0[gv_].mean(0)
                r_ = float((V0[gv_] - c_).norm(dim=1).max()) * 1.3
                w_d = torch.where((V0 - c_).norm(dim=1) < r_, torch.zeros_like(w_d), w_d)
    # feathered, so the edge of a wrapped region is not a step: three rounds of averaging with the neighbours
    Ac = A.tocoo()
    At = torch.sparse_coo_tensor(np.vstack([Ac.row, Ac.col]), Ac.data, (nV, nV))
    inv_deg = torch.tensor(1.0 / np.maximum(deg, 1))
    for _ in range(3):
        w_d = 0.5 * w_d + 0.5 * torch.sparse.mm(At, w_d[:, None]).ravel() * inv_deg
    LV0 = torch.sparse.mm(Lt, V0)
    D = torch.zeros_like(V0, requires_grad=True)
    opt_w = torch.optim.Adam([D], lr=0.002)
    for it in range(a.wrap):
        Vw = V0 + D
        d = sdf_at(Vw)
        l_data = (w_d * d ** 2).sum() / w_d.sum()
        l_lap = ((torch.sparse.mm(Lt, Vw) - LV0) ** 2).sum(-1).mean()
        loss_w = l_data * 1e4 + l_lap * 2e5 + (D ** 2).sum(-1).mean() * 10

        opt_w.zero_grad()
        loss_w.backward()
        opt_w.step()
        if it % 50 == 0 or it == a.wrap - 1:
            print(f"[template] wrap {it:4d} data {float(l_data.sqrt()) * 1000:.2f} mm  lap {float(l_lap.sqrt()) * 1000:.3f} mm",
                  flush=True)
    with torch.no_grad():
        Vw = (V0 + D)
        dw = sdf_at(Vw)
    sel = w_d > 0.9
    wrap_info = {"wrapped_vertices": int(sel.sum()), "of": nV,
                 "landmark_mm": (round(float((Vw[lm_idx] - lm_tgt).norm(dim=1).median()) * 1000, 1) if lm_idx is not None else None),
                 "residual_mm_median": round(float(dw[sel].abs().median()) * 1000, 2),
                 "residual_mm_p90": round(float(torch.quantile(dw[sel].abs(), 0.9)) * 1000, 2)}
    tw = trimesh.Trimesh(Vw.numpy() @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), Fn, process=False)
    tw.export(OUT / "wrapped.glb")
    print("[template] wrapped: " + json.dumps(wrap_info), flush=True)

# ---- measures and outputs ----------------------------------------------------------------------
with torch.no_grad():
    V, H = run()
    d_body = sdf_at(V[body_v]).numpy()
Vn = V.detach().numpy()
F = m.get_triangular_faces().numpy()
tm = trimesh.Trimesh(Vn @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), F, process=False)   # back to glTF Y-up
tm.export(OUT / "template.glb")
# the generated surface's distance to the fitted body: on it (skin), near it (tight clothing), outside it (layers)
body_tm = trimesh.Trimesh(Vn, F, process=False)
surf = body_tm.sample(400000, return_index=False) if hasattr(body_tm, "sample") else Vn
tree = cKDTree(np.vstack([surf, Vn[body_v.numpy()]]))
dg, _ = tree.query(GV)                                     # to the body's surface, not its vertices
col = np.zeros((len(GV), 4), np.uint8)
col[:, 3] = 255
col[dg < 0.01] = (210, 170, 140, 255)                       # on the body (within 1 cm)
col[(dg >= 0.01) & (dg < 0.03)] = (90, 140, 220, 255)       # near (1-3 cm): tight clothing
col[dg >= 0.03] = (230, 90, 60, 255)                        # outside: layers (loose clothing, hair, gear)
lay = trimesh.Trimesh(np.asarray(gm.vertices), np.asarray(gm.faces), vertex_colors=col, process=False)
lay.export(OUT / "layers.glb")
skin_d = None
if skin_pts is not None:
    sd, _ = tree.query(skin_pts.numpy())
    skin_d = {"median_mm": round(float(np.median(sd)) * 1000, 1), "p90_mm": round(float(np.percentile(sd, 90)) * 1000, 1)}
res = {"character": W.name, "height_m": a.height, "seconds": round(time.time() - t0, 1),
       "joints_cm": round(float(((H[jb] - J_t) ** 2).sum(-1).sqrt().mean()) * 100, 2),
       "skin_to_body": skin_d,
       "body_outside_pct": round(float((d_body > 0.002).mean()) * 100, 2),
       "body_deeper_than_depth_pct": round(float((d_body < -a.depth).mean()) * 100, 2),
       "surface_share": {"on_body_lt1cm": round(float((dg < 0.01).mean()), 3), "near_1_3cm": round(float(((dg >= 0.01) & (dg < 0.03)).mean()), 3),
                         "outside_gt3cm": round(float((dg >= 0.03).mean()), 3)},
       "phenotypes": {k: round(float(torch.sigmoid(ph[i])), 3) for i, k in enumerate(m.phenotype_labels)},
       "scale": round(float(torch.exp(g_s)), 4), "wrap": wrap_info, "history": hist}
json.dump(res, open(OUT / "fit.json", "w"), indent=1)
print("[template] " + json.dumps({k: v for k, v in res.items() if k not in ("history",)}), flush=True)
