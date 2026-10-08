"""Contact-aware arm solve (experiment E151): the smallest, smoothest turns of each shoulder, elbow and wrist that keep
the arm's surface out of the torso and the legs, frame by frame, measured against the character's own skinned mesh.

    python pipeline/contact_solve.py --dir <blender/export_pose.py dir> [--clips a,b] [--margin 1.0]

The idea is MeshRet's (NeurIPS 2024): judge a retargeted motion by the meshes' interaction, not by joint angles.
MeshRet learns it and watches arm x torso, arm x head and leg x leg only (E150); here it is solved directly, and the
hands and forearms are tested against the thighs and shins as well - the pairs that carry most deep frames (E148).

Per frame: the torso (pelvis, spine), thighs and shins are posed by linear skinning from the clip and do not move;
about 1,500 surface points of each arm are posed from the clip with a turn d added to each of shoulder, elbow and
wrist (in the bone's own frame; fingers follow the wrist), skinned by their four strongest bones. Both ways round:
an arm point is inside the body when it lies behind the normal of its nearest body vertex, (q - v) . n < margin,
within 12 cm and less than 2.5 cm to the side of that normal's line; a body vertex is inside the arm (a puffy vest
through a sleeve) by the same test against its nearest arm point and that point's posed normal. Not tested: an arm
point within 3 cm of that body part at rest, or with more than 0.10 of its skin weight on it (or a collar) - the
fold where arm and torso are one surface, which the audit does not count either - and a body vertex with that much
arm or collar weight, or within 3 cm at rest of the arm point it is measured against. Minimised over all frames at
once (degrees and centimetres; Adam, 300 steps, nearest points re-found every 25):
    sum relu(margin - s)^2  +  1 * |d|^2 (twist about the bone 3x, an elbow's sideways turn 5x)
                            + 20 * |d(t+1) - d(t)|^2
Each clip works only on the arm points and body vertices that come within 8 cm of each other somewhere in it
(a quarter of the time on a crouch, same result); a clip with nothing within the margin of a surface is left as it was. Writes <dir>/contact_<clip>.npz: d (T, 6, 3)
degrees, bones, and before/after counts of frames with a point deeper than 2 cm.
"""
import argparse
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--clips", default="")
ap.add_argument("--margin", type=float, default=1.0, help="cm the arm is kept outside the surface")
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--lr", type=float, default=0.5, help="Adam step, degrees")
ap.add_argument("--pen-weight", type=float, default=10.0, help="weight of the squared depth against the squared turns")
ap.add_argument("--points", type=int, default=1500, help="arm surface points per side")
ap.add_argument("--upper-gap", type=float, default=0.03, help="m an upper-arm point must be from the torso at rest")
ap.add_argument("--shared", type=float, default=0.10, help="skin weight on the obstacle (or collar) that rules a point out")
ap.add_argument("--shared-rev", type=float, default=0.10, help="the same, for body vertices tested against the arm (their arm and collar weight)")
ap.add_argument("--tag", default="", help="suffix for the output files")
ap.add_argument("--upper-legs", type=int, default=1, help="test the upper arm against the thighs too")
ap.add_argument("--reverse", type=int, default=1, help="also test body vertices against the arm's surface")
a = ap.parse_args()
torch.set_default_dtype(torch.float32)            # centimetres need no more; float64 doubled the time

R = np.load(os.path.join(a.dir, "rest.npz"))
names = list(R["names"])
ix = {n: i for i, n in enumerate(names)}
V, N, W, REST = (R[k].astype(np.float32) for k in ("verts", "normals", "weights", "rest"))
dom = W.argmax(1)
rng = np.random.default_rng(0)

SIDES = ("left", "right")
CHAIN = [f"{s}_{b}" for s in SIDES for b in ("shoulder", "elbow", "wrist")]
FINGERS = {s: [n for n in names if n.startswith(f"{s}_") and any(f in n for f in ("thumb", "index", "middle", "ring", "pinky"))]
           for s in SIDES}
TORSO = [ix[n] for n in ("pelvis", "spine1", "spine2", "spine3") if n in ix]
THIGH = {s: ix[f"{s}_hip"] for s in SIDES}
SHIN = {s: ix[f"{s}_knee"] for s in SIDES}
OBST = {"torso": np.isin(dom, TORSO), "thigh.L": dom == THIGH["left"], "thigh.R": dom == THIGH["right"],
        "shin.L": dom == SHIN["left"], "shin.R": dom == SHIN["right"]}
# which arm parts are tested against which obstacle
LEGS = ("thigh.L", "thigh.R", "shin.L", "shin.R")
TESTS = {"upper": ("torso",) + LEGS[:2] if a.upper_legs else ("torso",), "fore": ("torso",) + LEGS, "hand": ("torso",) + LEGS}

moving, part_of = [], []
for s in SIDES:
    parts = {"upper": [ix[f"{s}_shoulder"]], "fore": [ix[f"{s}_elbow"]],
             "hand": [ix[f"{s}_wrist"]] + [ix[n] for n in FINGERS[s]]}
    for p, bones in parts.items():
        idx = np.where(np.isin(dom, bones))[0]
        k = min(len(idx), a.points * {"upper": 3, "fore": 4, "hand": 3}[p] // 10)
        sel = rng.choice(idx, k, replace=False) if len(idx) > k else idx
        moving.append(sel)
        part_of += [p] * len(sel)
moving = np.concatenate(moving)
part_of = np.array(part_of)

# obstacles subsampled; per (moving point, obstacle) whether it is tested (more than 3 cm apart at rest)
obst_idx = {o: rng.choice(np.where(m)[0], min(int(m.sum()), 6000), replace=False) for o, m in OBST.items()}
# the arm and torso are one surface that folds at the armpit: a point that shares skin weight with the obstacle or
# the collar, or an upper-arm point within 6 cm of the torso at rest, is part of that fold, not a penetration
OBST_BONES = {"torso": TORSO + [ix[f"{s}_collar"] for s in SIDES], "thigh.L": [THIGH["left"]], "thigh.R": [THIGH["right"]],
              "shin.L": [SHIN["left"]], "shin.R": [SHIN["right"]]}
tested = {}
for o, oi in obst_idx.items():
    d, _ = cKDTree(V[oi]).query(V[moving])
    allowed = np.array([o in TESTS[p] for p in part_of])
    shared = W[moving][:, OBST_BONES[o]].sum(1) > a.shared
    gap = np.where(part_of == "upper", a.upper_gap, 0.03)
    tested[o] = allowed & (d > gap) & ~shared

# the other way round: a body vertex inside the arm (a puffy vest through a sleeve). Not a vertex that carries arm or
# collar weight itself, nor one within 3 cm of the arm point it is measured against at rest (checked per pair)
ARM_BONES = [ix[n] for s in SIDES for n in (f"{s}_collar", f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist") + tuple(FINGERS[s])]
REV = ("torso", "thigh.L", "thigh.R")
REV_N = 6000                                                  # body vertices per region tested this way (all of them)
rev_ok = {o: W[obst_idx[o][:REV_N]][:, ARM_BONES].sum(1) < a.shared_rev for o in REV}

inv_rest = np.linalg.inv(REST)
chain_ix = [ix[n] for n in CHAIN]


def axis_angle(v):
    """(..., 3) degrees -> (..., 3, 3)"""
    th = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-9) * (np.pi / 180)
    k = v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)
    K = torch.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3) + s * K + (1 - c) * K @ K


def skin(Mt, idx):
    """Linear skinning of rest vertices idx with world bone matrices Mt (T, B, 4, 4) -> (T, n, 3)."""
    S = Mt @ torch.from_numpy(inv_rest)                       # (T, B, 4, 4)
    w = torch.from_numpy(W[idx])                               # (n, B)
    A = torch.einsum("nb,tbij->tnij", w, S)
    p = torch.from_numpy(V[idx])
    return torch.einsum("tnij,nj->tni", A[..., :3, :3], p) + A[..., :3, 3]


REACH = 0.08                                                  # m: what can meet within a clip
TOPK = np.argsort(-W[moving], 1)[:, :4]                      # each arm point's four strongest bones
TOPW = np.take_along_axis(W[moving], TOPK, 1)
TOPW /= TOPW.sum(1, keepdims=True)
LOCAL = np.einsum("nkij,nj->nki", inv_rest[TOPK], np.c_[V[moving], np.ones(len(moving), np.float32)])   # point in each bone's rest frame


LOCALN = np.einsum("nkij,nj->nki", inv_rest[TOPK][..., :3, :3], N[moving])   # its normal in each bone's rest frame


def skin_moving(Mt, normals=False):
    """The arm points skinned by their four strongest bones (T, n, 3) - the solve's hot path - and their normals."""
    G = Mt[:, torch.from_numpy(TOPK)]                          # (T, n, 4, 4, 4)
    w = torch.from_numpy(TOPW)[None, :, :, None]
    q = (torch.einsum("tnkij,nkj->tnki", G[..., :3, :], torch.from_numpy(LOCAL)) * w).sum(2)
    if not normals:
        return q
    n = (torch.einsum("tnkij,nkj->tnki", G[..., :3, :3], torch.from_numpy(LOCALN)) * w).sum(2)
    return q, n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def posed_normals(M, idx):
    S = M @ inv_rest
    A = np.einsum("nb,tbij->tnij", W[idx], S)[..., :3, :3]
    n = np.einsum("tnij,nj->tni", A, N[idx])
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def arm_matrices(M, d):
    """World matrices with the turns d (T, 6, 3 degrees) on shoulder/elbow/wrist; fingers ride the wrist."""
    cols = [M[:, i] for i in range(M.shape[1])]               # built anew: no in-place writes under autograd
    for side in range(2):
        sh, el, wr = chain_ix[side * 3:side * 3 + 3]
        rel_el = torch.linalg.inv(M[:, sh]) @ M[:, el]
        rel_wr = torch.linalg.inv(M[:, el]) @ M[:, wr]
        fing = [ix[n] for n in FINGERS[SIDES[side]]]
        rel_f = [torch.linalg.inv(M[:, wr]) @ M[:, f] for f in fing]
        Rot = axis_angle(d[:, side * 3:side * 3 + 3])          # (T, 3, 3, 3)

        def turn(Mb, r):
            top = torch.cat([r, torch.zeros(r.shape[0], 3, 1)], 2)
            T4 = torch.cat([top, torch.tensor([[[0.0, 0, 0, 1]]]).expand(r.shape[0], 1, 4)], 1)
            return Mb @ T4
        Msh = turn(M[:, sh], Rot[:, 0])
        Mel = turn(Msh @ rel_el, Rot[:, 1])
        Mwr = turn(Mel @ rel_wr, Rot[:, 2])
        cols[sh], cols[el], cols[wr] = Msh, Mel, Mwr
        for f, rf in zip(fing, rel_f):
            cols[f] = Mwr @ rf
    return torch.stack(cols, 1)


wanted = [c for c in a.clips.split(",") if c]
clips = sorted(f[5:-4] for f in os.listdir(a.dir) if f.startswith("clip_") and f.endswith(".npz"))
for clip in clips:
    if wanted and clip not in wanted:
        continue
    t0 = time.time()
    C = np.load(os.path.join(a.dir, f"clip_{clip}.npz"))
    Mnp = C["mats"].astype(np.float32)
    T = Mnp.shape[0]
    M = torch.from_numpy(Mnp)
    obst = {o: (skin(M, oi).numpy(), posed_normals(Mnp, oi)) for o, oi in obst_idx.items()}
    trees = {o: [cKDTree(obst[o][0][t]) for t in range(T)] for o in obst}

    # only what comes near: arm points within REACH of a body part they are tested against at some frame of the clip,
    # and body vertices within REACH of the arm - nothing further can end up inside after turns of a few degrees
    q_all = skin_moving(M).numpy()
    near_m = np.zeros(len(moving), bool)
    for o in obst:
        cand = np.where(tested[o])[0]
        if len(cand):
            dmin = np.min([trees[o][t].query(q_all[t, cand], distance_upper_bound=REACH)[0] for t in range(T)], 0)
            near_m[cand[dmin < REACH]] = True
    sel = np.where(near_m)[0]
    if len(sel) == 0:
        before = (0, 0.0)
        np.savez(os.path.join(a.dir, f"contact{a.tag}_{clip}.npz"), d=np.zeros((T, 6, 3)), bones=np.array(CHAIN),
                 f0=C["f0"], f1=C["f1"], before=np.array(before), after=np.array(before))
        print(f"[contact] {clip}: no arm point within {REACH * 100:.0f} cm of the body - unchanged, {time.time() - t0:.0f}s", flush=True)
        continue
    tk, tw_ = torch.from_numpy(TOPK[sel]), torch.from_numpy(TOPW[sel])[None, :, :, None]
    lc, lcn = torch.from_numpy(LOCAL[sel]), torch.from_numpy(LOCALN[sel])
    tested_c = {o: tested[o][sel] for o in obst}
    Vm_c = V[moving][sel]
    rsel, rpos = {}, {}
    for o in REV:
        rv = obst[o][0][:, :REV_N]
        dmin = np.min([cKDTree(q_all[t, sel]).query(rv[t], distance_upper_bound=REACH)[0] for t in range(T)], 0)
        rsel[o] = np.where((dmin < REACH) & rev_ok[o])[0]
        rpos[o] = np.ascontiguousarray(rv[:, rsel[o]])

    def skin_c(Mt, normals=False):
        """skin_moving for this clip's points only"""
        G = Mt[:, tk]
        q = (torch.einsum("tnkij,nkj->tnki", G[..., :3, :], lc) * tw_).sum(2)
        if not normals:
            return q
        n = (torch.einsum("tnkij,nkj->tnki", G[..., :3, :3], lcn) * tw_).sum(2)
        return q, n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    def match(q_np):
        """nearest obstacle vertex per (frame, point, obstacle) -> positions and normals, as tensors"""
        out = {}
        for o in obst:
            I = np.stack([trees[o][t].query(q_np[t])[1] for t in range(T)])
            P = np.take_along_axis(obst[o][0], I[..., None], 1)
            Nn = np.take_along_axis(obst[o][1], I[..., None], 1)
            out[o] = (torch.from_numpy(P), torch.from_numpy(Nn))
        return out

    def penetration(q, nn):
        """(T, n) worst signed depth in cm over the tested obstacles (positive = outside), and the per-obstacle terms"""
        terms = []
        for o, (P, Nn) in nn.items():
            diff = q - P
            s = (diff * Nn).sum(-1) * 100.0
            # inside a surface means behind its nearest vertex, roughly along that vertex's normal: a point off to
            # the side of a far vertex is past an edge or a fold, not in the volume
            lateral = (diff - (diff * Nn).sum(-1, keepdim=True) * Nn).norm(dim=-1)
            near = (diff.norm(dim=-1) < 0.12) & (lateral < 0.025) & torch.from_numpy(tested_c[o])[None]
            terms.append(torch.where(near, s, torch.full_like(s, 1e3)))
        return torch.stack(terms).min(0).values

    def match_rev(q_np):
        """per body vertex, its nearest arm point now, and whether the pair is tested"""
        out = {}
        for o in REV:
            if len(rsel[o]) == 0:
                continue
            J = np.stack([cKDTree(q_np[t]).query(rpos[o][t])[1] for t in range(T)])     # (T, m)
            apart = np.linalg.norm(V[obst_idx[o][:REV_N]][rsel[o]][None] - Vm_c[J], axis=-1) > 0.03
            ok = apart & tested_c[o][J]
            out[o] = (torch.from_numpy(J), torch.from_numpy(ok))
        return out

    def penetration_rev(q, nq, rm):
        """(T, m) signed depth in cm of each tested body vertex below the arm's surface (positive = outside it)"""
        terms = []
        for o, (J, ok) in rm.items():
            Q = torch.gather(q, 1, J[..., None].expand(-1, -1, 3))
            Nq = torch.gather(nq, 1, J[..., None].expand(-1, -1, 3))
            diff = torch.from_numpy(rpos[o]) - Q
            s = (diff * Nq).sum(-1) * 100.0
            lateral = (diff - (diff * Nq).sum(-1, keepdim=True) * Nq).norm(dim=-1)
            near = (diff.norm(dim=-1) < 0.12) & (lateral < 0.025) & ok
            terms.append(torch.where(near, s, torch.full_like(s, 1e3)))
        return torch.cat(terms, 1) if terms else torch.full((T, 1), 1e3)

    def depths(Mt, nn=None, rm=None):
        """every tested signed depth (cm) of a pose: arm points in the body, and body vertices in the arm"""
        q, nq = skin_c(Mt, normals=True)
        qn = q.detach().numpy()
        nn = nn if nn is not None else match(qn)
        terms = [penetration(q, nn)]
        if a.reverse:
            terms.append(penetration_rev(q, nq, rm if rm is not None else match_rev(qn)))
        return torch.cat(terms, 1)

    def deep_frames(Mt):
        with torch.no_grad():
            s = depths(Mt).numpy()
        return int((s.min(1) < -2.0).sum()), float(-s.min())

    q0 = skin_c(M).numpy()
    before = deep_frames(M)
    nn, rm = match(q0), (match_rev(q0) if a.reverse else None)
    with torch.no_grad():
        idle = not bool((depths(M, nn, rm) < a.margin).any())
    if idle:
        # nothing inside anything: the optimum is no turn at all
        np.savez(os.path.join(a.dir, f"contact{a.tag}_{clip}.npz"), d=np.zeros((T, 6, 3)), bones=np.array(CHAIN),
                 f0=C["f0"], f1=C["f1"], before=np.array(before), after=np.array(before))
        print(f"[contact] {clip}: nothing within {a.margin} cm of a surface - unchanged, {time.time() - t0:.0f}s", flush=True)
        continue
    d = torch.zeros(T, 6, 3, requires_grad=True)
    wts = torch.tensor([[1.0, 3, 1], [1, 3, 5], [1, 3, 1]] * 2)   # x, y (twist), z per bone
    opt = torch.optim.Adam([d], lr=a.lr)
    for it in range(a.iters):
        if it and it % 25 == 0:
            with torch.no_grad():
                qn = skin_c(arm_matrices(M, d)).numpy()
                nn, rm = match(qn), (match_rev(qn) if a.reverse else None)
        s_ = depths(arm_matrices(M, d), nn, rm)
        pen = a.pen_weight * torch.relu(a.margin - s_).pow(2).sum()
        reg = (wts * d.pow(2)).sum()
        smooth = (d[1:] - d[:-1]).pow(2).sum() * 20.0
        loss = pen + reg + smooth
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = deep_frames(arm_matrices(M, d))
    dd = d.detach().numpy()
    mag = np.linalg.norm(dd, axis=-1)
    np.savez(os.path.join(a.dir, f"contact{a.tag}_{clip}.npz"), d=dd, bones=np.array(CHAIN), f0=C["f0"], f1=C["f1"],
             before=np.array(before), after=np.array(after))
    print(f"[contact] {clip}: deep frames {before[0]}/{T} -> {after[0]}/{T}, deepest {before[1]:.1f} -> "
          f"{after[1]:.1f} cm, turns mean {mag.mean():.1f} max {mag.max():.0f} deg, {time.time() - t0:.0f}s", flush=True)
