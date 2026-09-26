"""Re-pose a character, frame by frame, until its own mesh lies on the video it was made from:
the silhouette on the figure, the joints on the pose model's joints, the face turned as the face is.

    python refine_pose.py --skin skin.npz --masks move_masks.npz --kp move_kp17.npz --yaw -17.8 \
        --out corrections.npz

The clip made from a video (pipeline/video_motion.py --fit, blender/retarget.py) is fitted to 13
points on the pose model's reading, through a stand-in skeleton and back; what reaches the render
misses the video by what 13 points cannot say - how a torso turns between shoulders and hips, where
a sleeve or a vest actually is. This step closes the loop on the character itself: its skinned mesh
(blender/export_skin.py - linear blend skinning, as Blender deforms it) is posed in PyTorch,
projected by the render's orthographic camera, and every frame's pose is nudged - small turns of
twenty bones, on top of the clip - to minimise

  inside    points of the mesh that fall outside the video's figure, by their distance from it
  cover     pixels of the figure no point of the mesh comes near, by their distance to the nearest
  joints    the skeleton's 13 joints against the pose model's (confidence-weighted, robust)
  face      the head's nose, eyes and ears (tools/calibrate_joints.py) against the pose model's
  hands     with --hands (tools/dwpose.py): each hand's 20 points about its wrist - the finger joints
            and tips - against DWPose's, in palm lengths, where DWPose is sure of the hand
  prior     how far each bone turns from the clip, and how fast the turns change frame to frame

The camera's angle, one scale and each frame's placement in the image are fitted with the pose.
With --hands the fingers turn too, each joint only as a finger bends - toward the palm and, at the
knuckles, apart - within a finger's range, and a turn of the wrist about the forearm is shared with
the forearm, as the two bones of a real forearm share it: linear skinning pinches a wrist twisted
alone. Writes the turns (frames x bones x axis-angle, in each bone's own frame) for
blender/apply_pose_corrections.py, and the fitted camera.
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import torch

BONES = ["pelvis", "spine1", "spine2", "spine3", "neck", "head", "left_collar", "right_collar",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
         "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
ORDER13 = [5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]            # COCO -> the 12 body joints
PARENT17 = {7: 5, 9: 7, 8: 6, 10: 8, 13: 11, 15: 13, 14: 12, 16: 14}   # elbow <- shoulder, ankle <- knee...
FINGERS = [f"{s}_{f}{i}" for s in ("left", "right") for f in ("thumb", "index", "middle", "ring", "pinky")
           for i in (1, 2, 3)]
HAND_SURE = 0.6                  # a hand pulls where DWPose's median confidence on it is at least this
CURL = (-0.35, 1.75)             # a finger joint's bend toward the palm from straight, radians (-20 to 100 deg)
SPREAD = 0.35                    # how far a knuckle's correction may spread a finger (20 deg)
ELBOW_MAX = np.radians(150.0)    # an elbow's bend past which it costs: the motion library's elbows pass 150 deg
                                 # in 0.05% of frames (159 at most); a video can put a forearm back through the
                                 # upper arm (the image is the same either way round) - Aoi's cast did, to 176


def rotvec(R):
    """(..., 3, 3) rotations -> (..., 3) axis-angle (for turns short of 180 degrees)."""
    th = np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)[..., None]
    return np.where(s > 1e-6, w / np.maximum(2 * s, 1e-9) * th[..., None], w / 2)


def fill_gaps17(kp, sc, lo=0.5, reach=4, filled=0.5):
    """The pose model's points where it is unsure for a few frames, drawn between the sure frames either
    side, relative to the parent joint (pipeline/video_motion.fill_gaps, on the 17-point layout): at the
    top of Mara's kick it put the ankle at her hip on the frames around a correct one, and the refine
    pulled the leg there."""
    K, C = kp.copy(), sc.copy()
    T = len(K)
    for j in range(17):
        par = PARENT17.get(j)
        rel = K[:, j] - (K[:, par] if par is not None else 0.0)
        good = C[:, j] >= lo
        for t in np.nonzero(~good)[0]:
            prv = [u for u in range(max(0, t - reach), t) if good[u]]
            nxt = [u for u in range(t + 1, min(T, t + reach + 1)) if good[u]]
            if not prv or not nxt:
                continue
            t0, t1 = prv[-1], nxt[0]
            f = (t - t0) / (t1 - t0)
            K[t, j] = (K[t, par] if par is not None else 0.0) + (1 - f) * rel[t0] + f * rel[t1]
            C[t, j] = max(C[t, j], filled)
    return K, C


def rodrigues(r):
    """(..., 3) axis-angle -> (..., 3, 3)."""
    th = torch.sqrt((r * r).sum(-1, keepdim=True) + 1e-12)
    k = r / th
    K = torch.zeros(r.shape[:-1] + (3, 3), device=r.device, dtype=r.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    eye = torch.eye(3, device=r.device, dtype=r.dtype).expand(K.shape)
    return eye + s * K + (1 - c) * (K @ K)


class Model:
    def __init__(self, skin, dev, hands=False):
        self.dev = dev
        t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
        self.V = t(np.concatenate([skin["verts"], np.ones((len(skin["verts"]), 1))], 1))
        self.bi = torch.tensor(skin["bone_idx"], dtype=torch.long, device=dev)
        self.bw = t(skin["bone_w"])
        self.parents = [int(p) for p in skin["parents"]]
        names = [str(b) for b in skin["bones"]]
        self.inv_rest = t(np.linalg.inv(skin["rest"]))
        pose = skin["pose"].astype(np.float64)
        self.M = t(pose)                                              # (T, B, 4, 4)
        L = np.empty_like(pose)
        for j, p in enumerate(self.parents):
            L[:, j] = pose[:, j] if p < 0 else np.linalg.inv(pose[:, p]) @ pose[:, j]
        self.L = t(L)
        self.corr = {names.index(b): k for k, b in enumerate(BONES) if b in names}
        # the hands: DWPose's 21 points a hand on the rig, and each finger joint's two ways to bend
        self.hands = hands and "hand_bone" in skin and all(f in names for f in FINGERS)
        self.fingers = FINGERS if self.hands else []
        if self.hands:
            rest = skin["rest"].astype(np.float64)
            for i, f in enumerate(FINGERS):
                self.corr[names.index(f)] = len(BONES) + i
            self.hand_bi = torch.tensor(skin["hand_bone"].reshape(-1), dtype=torch.long, device=dev)
            self.hand_loc = t(np.concatenate([skin["hand_local"].reshape(-1, 3), np.ones((42, 1))], 1))
            self.palm_m = float(np.mean([np.linalg.norm(rest[names.index(f"{s}_middle1"), :3, 3]
                                                        - rest[names.index(f"{s}_wrist"), :3, 3]) for s in ("left", "right")]))
            ax = np.zeros((len(FINGERS), 2, 3))
            curl0 = np.zeros((len(pose), len(FINGERS)))
            for i, f in enumerate(FINGERS):
                j = names.index(f)
                Rr = rest[j, :3, :3] / np.linalg.norm(rest[j, :3, :3], axis=0)
                d, n = Rr[:, 1], skin["palm_normal"][0 if f.startswith("left") else 1].astype(np.float64)
                curl = np.cross(d, n)
                curl /= np.linalg.norm(curl)                           # turning about it bends the finger to the palm
                spread = n - (n @ d) * d
                spread /= np.linalg.norm(spread)
                ax[i, 0] = Rr.T @ curl
                if "thumb" in f or f.endswith("1"):                   # knuckles spread; the thumb moves both ways
                    ax[i, 1] = Rr.T @ spread
                # how far the clip already bends it, from its rest
                p = self.parents[j]
                rl = np.linalg.inv(rest[p]) @ rest[j]
                Q = np.linalg.inv(rl[:3, :3])[None] @ L[:, j, :3, :3]
                curl0[:, i] = rotvec(Q) @ ax[i, 0]
            self.faxes = t(ax)
            self.curl0 = t(curl0)
            self.thumb = torch.tensor(["thumb" in f for f in FINGERS], device=dev)
            self.twist = [(BONES.index(f"{s}_elbow"), BONES.index(f"{s}_wrist")) for s in ("left", "right")]
        self.joints = [int(j) for j in skin["joints"]]
        # each arm's shoulder, elbow and wrist - the heads of the upper arm, forearm and hand bones
        self.arms = [tuple(names.index(f"{s_}_{b}") for b in ("shoulder", "elbow", "wrist"))
                     for s_ in ("left", "right") if all(f"{s_}_{b}" in names for b in ("shoulder", "elbow", "wrist"))]
        self.head = names.index("head")
        self.face = t(np.concatenate([skin["face_local"], np.ones((5, 1))], 1)) if "face_local" in skin else None
        # the joints as the pose model sees them, where calibrated and trusted: shoulders, hips, knees
        # and ankles (a T-pose misleads it on stylised arms, so elbows and wrists stay the bones' heads)
        jl = np.zeros((13, 4), np.float32)
        jl[:, 3] = 1
        self.use_calib = False
        if "joint_local" in skin:
            for j in (1, 4, 7, 8, 9, 10, 11, 12):
                jl[j, :3] = skin["joint_local"][j]
            self.use_calib = True
        self.jl = t(jl)
        self.T, self.B = pose.shape[:2]

    def fk(self, delta, frames):
        """World matrices (F, B, 4, 4) with each corrected bone turned by delta (F, nb, 3) in its own frame."""
        R = rodrigues(delta)
        Rh = torch.zeros(R.shape[:-2] + (4, 4), device=self.dev)
        Rh[..., :3, :3] = R
        Rh[..., 3, 3] = 1
        L = self.L[frames]
        out = [None] * self.B
        for j, p in enumerate(self.parents):
            m = L[:, j] if p < 0 else out[p] @ L[:, j]
            k = self.corr.get(j)
            out[j] = m @ Rh[:, k] if k is not None else m
        return torch.stack(out, 1)

    def points(self, Mw, idx):
        """Skinned points (F, n, 3) for point indices idx."""
        A = Mw @ self.inv_rest                                        # (F, B, 4, 4)
        bi, bw, V = self.bi[idx], self.bw[idx], self.V[idx]
        P = 0
        for k in range(4):
            P = P + bw[None, :, k, None] * (A[:, bi[:, k]] @ V[None, :, :, None])[..., 0]
        return P[..., :3]


def project(P, yaw, scale, off):
    """(F, n, 3) world -> (F, n, 2) pixels: level orthographic camera at `yaw` (render_match.py)."""
    side = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)])
    u = P @ side
    return torch.stack([u, -P[..., 2]], -1) * scale + off[:, None, :]


def main(a):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    skin = dict(np.load(a.skin))
    masks = np.load(a.masks)["masks"].astype(bool)
    kz = np.load(a.kp)
    kp17, sc17 = fill_gaps17(kz["kp17"].astype(np.float64), kz["sc17"].astype(np.float64))
    kp17, sc17 = kp17.astype(np.float32), sc17.astype(np.float32)
    m = Model(skin, dev, hands=bool(a.hands))
    T = min(m.T, len(masks), len(kp17))
    if m.hands:
        hz = np.load(a.hands)
        T = min(T, len(hz["kp133"]))
    elif a.hands:
        print("[refine] --hands given but the skin has no hand points (blender/export_skin.py); body only", flush=True)
    H, W = masks.shape[1:]
    # the figure: distance outside it (zero inside), and pixels to cover (edge and inside)
    dt = np.stack([cv2.distanceTransform((~mk).astype(np.uint8), cv2.DIST_L2, 5) for mk in masks[:T]])
    rng = np.random.default_rng(0)
    cover = []
    for mk in masks[:T]:
        edge = mk & ~cv2.erode(mk.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ey, ex = np.nonzero(edge)
        iy, ix = np.nonzero(mk)
        e = rng.choice(len(ey), min(a.cover // 2, len(ey)), replace=False)
        i = rng.choice(len(iy), a.cover - len(e), replace=False)
        cover.append(np.stack([np.concatenate([ex[e], ix[i]]), np.concatenate([ey[e], iy[i]])], 1))
    cover = torch.tensor(np.array(cover), dtype=torch.float32, device=dev)
    dtt = torch.tensor(dt, dtype=torch.float32, device=dev)[:, None]         # (T, 1, H, W)
    kp = torch.tensor(kp17[:T], dtype=torch.float32, device=dev)
    # only points the pose model is fairly sure of pull (a limb it misreads at 0.3-0.4 dragged the leg away)
    kc = torch.tensor(sc17[:T], dtype=torch.float32, device=dev) * (torch.tensor(sc17[:T], device=dev) >= 0.5)
    body_t, body_c = kp[:, ORDER13], kc[:, ORDER13]
    face_t, face_c = kp[:, :5], kc[:, :5]
    torso_px = float(np.nanmedian(np.linalg.norm((kp17[:T, 5] + kp17[:T, 6]) / 2 - (kp17[:T, 11] + kp17[:T, 12]) / 2, axis=1)))

    frames = torch.arange(T, device=dev)
    nb, nf = len(BONES), len(m.fingers)
    delta = torch.zeros(T, nb, 3, device=dev, requires_grad=True)
    fing = torch.zeros(T, nf, 2, device=dev, requires_grad=True)     # each finger joint: bend, spread

    def full(d, f):
        return torch.cat([d, torch.einsum("tfk,fkd->tfd", f, m.faxes)], 1) if nf else d
    yaw = torch.tensor(np.radians(a.yaw), dtype=torch.float32, device=dev, requires_grad=a.fit_yaw)
    # scale and placement: the mesh's height onto the figure's, feet on feet, to start
    with torch.no_grad():
        P0 = m.points(m.fk(torch.zeros(T, nb + nf, 3, device=dev), frames), torch.arange(len(m.V), device=dev)[::7])
        uv0 = project(P0, yaw, torch.tensor(1.0, device=dev), torch.zeros(T, 2, device=dev))
        hm = torch.tensor([np.ptp(np.nonzero(mk)[0]) for mk in masks[:T]], dtype=torch.float32, device=dev)
        hp = uv0[..., 1].amax(1) - uv0[..., 1].amin(1)
        s0 = float(torch.median(hm / hp))
        cx = torch.tensor([np.nonzero(mk)[1].mean() for mk in masks[:T]], dtype=torch.float32, device=dev)
        by = torch.tensor([np.nonzero(mk)[0].max() for mk in masks[:T]], dtype=torch.float32, device=dev)
        off0 = torch.stack([cx - s0 * uv0[..., 0].mean(1), by - s0 * uv0[..., 1].amax(1)], 1)
    log_s = torch.tensor(np.log(s0), dtype=torch.float32, device=dev, requires_grad=True)
    off = off0.clone().requires_grad_(True)
    params = [{"params": [delta], "lr": a.lr}, {"params": [log_s], "lr": 0.003}, {"params": [off], "lr": 0.5}]
    if nf:
        params.append({"params": [fing], "lr": a.lr * 2})
        # DWPose's hands, each point about its wrist; a hand pulls where DWPose is sure of it
        HK = torch.tensor(hz["kp133"][:T, 91:133], dtype=torch.float32, device=dev).view(T, 2, 21, 2)
        hsc = hz["sc133"][:T, 91:133].reshape(T, 2, 21).astype(np.float32)
        hw = hsc[..., 1:] * (hsc[..., 1:] >= 0.5) * (hsc[..., :1] >= 0.5) * (np.median(hsc, -1) >= HAND_SURE)[..., None]
        HW = torch.tensor(hw, dtype=torch.float32, device=dev)          # (T, 2, 20)
        palm_px = s0 * m.palm_m
        print(f"[refine] hands: DWPose sure of {int((hw.sum(-1) > 0).sum())} of {2 * T} hand-frames; "
              f"palm {palm_px:.0f} px", flush=True)
    if a.fit_yaw:
        params.append({"params": [yaw], "lr": 0.003})
    opt = torch.optim.Adam(params)
    npts = len(m.V)

    def hand_error(Mw, per_point=False):
        """Each hand's 20 points about its wrist against DWPose's, in palm lengths (T, 2, 20)."""
        HP = (Mw[:, m.hand_bi] @ m.hand_loc[None, :, :, None])[..., :3, 0]
        uvh = project(HP, yaw, torch.exp(log_s).detach(), off.detach()).view(T, 2, 21, 2)
        return (((uvh[:, :, 1:] - uvh[:, :, :1]) - (HK[:, :, 1:] - HK[:, :, :1])).pow(2).sum(-1) + 1e-8).sqrt() / palm_px

    def elbow_bends(Mw):
        """Each arm's bend at the elbow, radians (F, arms): 0 straight, pi folded back on itself."""
        out = []
        for sh, el, wr in m.arms:
            u, f = Mw[:, el, :3, 3] - Mw[:, sh, :3, 3], Mw[:, wr, :3, 3] - Mw[:, el, :3, 3]
            c = (u * f).sum(-1) / (u.norm(dim=-1) * f.norm(dim=-1)).clamp(min=1e-9)
            out.append(torch.acos(c.clamp(-0.9999, 0.9999)))
        return torch.stack(out, 1)

    def losses(sub, full_pts=False):
        Mw = m.fk(full(delta, fing), frames)
        idx = torch.arange(npts, device=dev) if full_pts else torch.randint(0, npts, (sub,), device=dev)
        uv = project(m.points(Mw, idx), yaw, torch.exp(log_s), off)
        # inside: the outside-distance of each point, sampled bilinearly
        g = torch.stack([uv[..., 0] / (W - 1) * 2 - 1, uv[..., 1] / (H - 1) * 2 - 1], -1)[:, :, None]
        d_out = torch.nn.functional.grid_sample(dtt, g, align_corners=True, padding_mode="border")[:, 0, :, 0]
        inside = (d_out / torso_px).pow(2).mean()
        # cover: each figure pixel to its nearest mesh point
        dmin = torch.cdist(cover, uv).amin(-1)
        cov = (dmin / torso_px).pow(2).mean()
        # joints and face
        if a.calib_joints and m.use_calib:
            J = (Mw[:, m.joints] @ m.jl[None, :, :, None])[..., :3, 0]
        else:
            J = Mw[:, m.joints, :3, 3]
        uvj = project(J, yaw, torch.exp(log_s), off)[:, 1:]
        ej = ((uvj - body_t) / torso_px).pow(2).sum(-1)
        ej = torch.nn.functional.huber_loss(ej, torch.zeros_like(ej), reduction="none", delta=0.05)
        joints = (ej * body_c).sum() / body_c.sum().clamp(min=1)
        face = torch.tensor(0.0, device=dev)
        if m.face is not None:
            F = (Mw[:, m.head] @ m.face.T).transpose(1, 2)[..., :3]
            uvf = project(F, yaw, torch.exp(log_s), off)
            # about their centre: the head's turn, not where the body puts it
            cf = (uvf * face_c[..., None]).sum(1, keepdim=True) / face_c.sum(1, keepdim=True)[..., None].clamp(min=1e-6)
            ct = (face_t * face_c[..., None]).sum(1, keepdim=True) / face_c.sum(1, keepdim=True)[..., None].clamp(min=1e-6)
            ef = (((uvf - cf) - (face_t - ct)) / torso_px).pow(2).sum(-1)
            face = (ef * face_c).sum() / face_c.sum().clamp(min=1)
        prior = delta.pow(2).sum(-1).mean()
        smooth = (delta[1:] - delta[:-1]).pow(2).sum(-1).mean() + ((off[1:] - off[:-1]) / torso_px).pow(2).sum(-1).mean()
        out = {"inside": inside, "cover": cov, "joints": joints, "face": face, "prior": prior, "smooth": smooth}
        if m.arms:
            out["elbows"] = torch.relu(elbow_bends(Mw) - ELBOW_MAX).pow(2).mean()
        if nf:
            # the hands turn the wrists, the forearms about their length and the fingers - nothing else: the
            # arm is placed by the body's terms, and a hand pulling on it swung the elbows off the video's
            dh = delta.detach().clone()
            for e, w_ in m.twist:
                dh[:, w_] = delta[:, w_]
                dh[:, e, 1] = delta[:, e, 1]
            eh = torch.nn.functional.huber_loss(hand_error(m.fk(full(dh, fing), frames)), torch.zeros_like(HW),
                                                reduction="none", delta=0.3)
            out["hands"] = (eh * HW).sum() / HW.sum().clamp(min=1)
            # within a finger's range (the clip's own bend plus the correction); the thumb is left to the prior
            curl = m.curl0[:T] + fing[..., 0]
            lim = (torch.relu(CURL[0] - curl).pow(2) + torch.relu(curl - CURL[1]).pow(2)) * (~m.thumb)
            out["limits"] = lim.mean() + torch.relu(fing[..., 1].abs() - SPREAD).pow(2).mean()
            out["fingers"] = fing.pow(2).sum(-1).mean()
            out["smooth"] = out["smooth"] + (fing[1:] - fing[:-1]).pow(2).sum(-1).mean()
            # a wrist's turn about the forearm shared with the forearm (Y is along each bone)
            out["twist"] = sum((delta[:, e, 1] - delta[:, w_, 1]).pow(2).mean() for e, w_ in m.twist)
        return out

    wts = {"inside": a.w_inside, "cover": a.w_cover, "joints": a.w_joints, "face": a.w_face,
           "prior": a.w_prior, "smooth": a.w_smooth, "hands": a.w_hands, "limits": a.w_limits,
           "fingers": a.w_fingers, "twist": a.w_twist, "elbows": a.w_elbows}
    t0 = time.time()
    hist = []
    for it in range(a.iters + 1):
        opt.zero_grad()
        L = losses(a.points)
        tot = sum(wts[k] * v for k, v in L.items())
        if it % 50 == 0 or it == a.iters:
            hist.append({k: round(float(v), 6) for k, v in L.items()})
            print(f"[refine] it {it:4d}  " + "  ".join(f"{k} {float(v):.5f}" for k, v in L.items())
                  + f"  scale {float(torch.exp(log_s)):.2f} yaw {float(np.degrees(yaw.item())):.1f}  {time.time() - t0:.0f}s", flush=True)
        if it == a.iters:
            break
        tot.backward()
        opt.step()
    # a silhouette check at the video's resolution: every point splatted, closed into a figure
    with torch.no_grad():
        ious, hands_err, bends = {}, {}, {}
        for name, d_ in (("clip", full(torch.zeros_like(delta), torch.zeros_like(fing))), ("refined", full(delta, fing))):
            Mw = m.fk(d_, frames)
            if m.arms:
                bends[name] = np.degrees(elbow_bends(Mw).cpu().numpy())
            if nf:
                hands_err[name] = float((hand_error(Mw) * HW).sum() / HW.sum().clamp(min=1))
            uv = project(m.points(Mw, torch.arange(npts, device=dev)), yaw, torch.exp(log_s), off).cpu().numpy()
            iou = []
            for t in range(T):
                img = np.zeros((H, W), np.uint8)
                p = uv[t].round().astype(int)
                ok = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
                img[p[ok, 1], p[ok, 0]] = 1
                img = cv2.morphologyEx(cv2.dilate(img, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE,
                                       np.ones((5, 5), np.uint8)).astype(bool)
                iou.append((img & masks[t]).sum() / max((img | masks[t]).sum(), 1))
            ious[name] = float(np.mean(iou))
    turn = np.degrees(np.linalg.norm(delta.detach().cpu().numpy(), axis=-1))
    print(f"[refine] silhouette IoU (points, same placement): clip {ious['clip']:.3f} -> refined {ious['refined']:.3f}; "
          f"mean turn {turn.mean():.1f} deg, largest per bone: "
          + ", ".join(f"{b} {turn[:, k].mean():.1f}" for k, b in sorted(enumerate(BONES), key=lambda kv: -turn[:, kv[0]].mean())[:6]),
          flush=True)
    rep = {"iou_points_clip": ious["clip"], "iou_points_refined": ious["refined"], "history": hist,
           "yaw_deg": float(np.degrees(yaw.item())), "weights": wts}
    if bends:
        for name, b_ in bends.items():
            rep[f"elbow_max_deg_{name}"] = round(float(b_.max()), 1)
            rep[f"elbow_frames_past_150_{name}"] = int((b_ > 150).any(1).sum())
        print(f"[refine] elbows: clip bends to {rep['elbow_max_deg_clip']:.0f} deg ({rep['elbow_frames_past_150_clip']} frames "
              f"past 150) -> refined {rep['elbow_max_deg_refined']:.0f} deg ({rep['elbow_frames_past_150_refined']})", flush=True)
    if nf:
        bend = np.degrees(np.abs(fing.detach().cpu().numpy()[..., 0]))
        wr = [np.degrees(np.linalg.norm(delta.detach().cpu().numpy()[:, BONES.index(f"{s_}_wrist")], axis=-1)) for s_ in ("left", "right")]
        print(f"[refine] hands (points about the wrist): clip {hands_err['clip']:.3f} -> refined {hands_err['refined']:.3f} "
              f"palm lengths; wrists turned {wr[0].mean():.0f}/{wr[1].mean():.0f} deg on average, finger joints bent "
              f"{bend.mean():.0f} deg (p90 {np.percentile(bend, 90):.0f})", flush=True)
        rep.update(hands_clip=hands_err["clip"], hands_refined=hands_err["refined"])
    np.savez_compressed(a.out, delta=full(delta, fing).detach().cpu().numpy(), bones=np.array(BONES + m.fingers),
                        yaw_deg=float(np.degrees(yaw.item())), scale=float(torch.exp(log_s)),
                        offset=off.detach().cpu().numpy(), fps=a.fps)
    json.dump(rep, open(a.out.replace(".npz", ".json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skin", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--kp", required=True, help="npz with kp17 (frames, 17, 2; pixels, y down) and sc17")
    ap.add_argument("--yaw", type=float, required=True, help="the camera's angle, degrees (the fit's preview yaw)")
    ap.add_argument("--fit-yaw", action="store_true", help="refine the camera angle too")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--points", type=int, default=5000, help="mesh points per step")
    ap.add_argument("--cover", type=int, default=600, help="figure pixels to cover, per frame")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--calib-joints", action="store_true",
                    help="hold the pose model's joints against where it sees them on this character (the skin's "
                         "joint_local) rather than against the bones' heads")
    ap.add_argument("--w-inside", type=float, default=1.0)
    ap.add_argument("--w-cover", type=float, default=1.0)
    ap.add_argument("--w-joints", type=float, default=0.6)
    ap.add_argument("--w-face", type=float, default=0.6)
    ap.add_argument("--w-prior", type=float, default=0.01)
    ap.add_argument("--w-smooth", type=float, default=1.0)
    ap.add_argument("--hands", default=None, help="DWPose's reading of the video (tools/dwpose.py: kp133, sc133)")
    ap.add_argument("--w-hands", type=float, default=0.2)
    ap.add_argument("--w-limits", type=float, default=1.0)
    ap.add_argument("--w-fingers", type=float, default=0.0005)
    ap.add_argument("--w-twist", type=float, default=0.002)
    ap.add_argument("--w-elbows", type=float, default=1.0, help="an elbow bent past 150 deg, squared (radians)")
    main(ap.parse_args())
