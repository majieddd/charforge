"""How close a character's clip is to the video it was made from, frame by frame, in the video's pixels.

    python tools/motion_fidelity.py --name mara --move punch_combo [--performer mara] [--yaw-search]

1. The character's clip is rendered at the video's timing from the video's camera angle
   (blender/render_match.py; the angle is the fit's preview yaw, or with --yaw-search the one of
   seven around it that scores best on a sample of frames).
2. The same pose model (ViTPose) reads the same joints off every render and every video frame -
   a pose model's joints are not a skeleton's, and reading both with one model cancels its bias.
3. The renders are laid over the video: one scale for the whole clip (the torso's length) and,
   per frame, the hips on the video's hips - a clip ships in place, the person in the video may
   drift, and the drift is the game's to add.
4. Each frame is scored: the joints' mean distance from the video's, in the video's torso lengths,
   and the overlap (IoU) of the character's silhouette with the video's figure.
5. The hands, which the 17 points do not reach: DWPose (tools/dwpose.py, when its model is there)
   reads 21 points a hand on the video and on every render, and each hand the video shows clearly is
   scored by its points about the wrist (in palm lengths), which way its palm faces and where it
   points - with close-ups of both hands in <move>_hands.png.

Writes work/<name>/motion_videos/<move>_fidelity.json (per joint, per frame, the summary) and
<move>_fidelity.png: six frames of the video with the character's outline and both skeletons.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from align import image_mask  # noqa: E402

BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
NAMES = ["head", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist",
         "l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle"]
DW_PY = ROOT / "vendor" / "trellis2mlx" / ".venv" / "bin" / "python"   # tools/dwpose.py runs on onnxruntime there
DW_MODEL = ROOT / "models" / "dwpose" / "dw-ll_ucoco_384.onnx"
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
HAND_SURE = 0.6                  # a hand counts where DWPose's median confidence on it is at least this


def dwpose(out, video=None, masks=None, frames=None):
    """DWPose's 133 points (tools/dwpose.py) for a video or a folder of renders, cached in `out`."""
    src = [Path(frames) / p for p in os.listdir(frames) if p.endswith(".png")] if frames else [Path(video)]
    if not out.exists() or out.stat().st_mtime < max(p.stat().st_mtime for p in src):
        args = ["--frames", str(frames)] if frames else ["--video", str(video), "--masks", str(masks)]
        subprocess.run([str(DW_PY), str(ROOT / "tools" / "dwpose.py"), *args, "--out", str(out)],
                       check=True, capture_output=True)
    d = np.load(out)
    return d["kp133"], d["sc133"]


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def hand_scores(vk, vc, rk, rc, s, off):
    """How each hand lies, per frame, from DWPose's 21 points a hand on the video and on the render (laid
    on the video at the clip's scale and each frame's placement, as the joints are): the 20 points about
    the wrist, in the video's palm lengths (wrist to middle knuckle); which way the palm faces - which
    side of the wrist the index knuckle turns to from the little finger's - where the video shows it
    clearly; and the angle between where the two hands point (wrist to middle knuckle). A hand counts
    where DWPose is sure of it on both (median confidence HAND_SURE on the video, 0.4 on the render).
    (T, 2) arrays, left hand then right; NaN where a hand does not count."""
    T = len(vk)
    err, facing, direc = (np.full((T, 2), np.nan) for _ in range(3))
    rk = rk * s + off[:, None, :]
    palms = []
    for h, s0 in enumerate((91, 112)):
        V, Vc, R, Rc = vk[:, s0:s0 + 21], vc[:, s0:s0 + 21], rk[:, s0:s0 + 21], rc[:, s0:s0 + 21]
        sure = (Vc[:, 0] >= HAND_SURE) & (Vc[:, 9] >= HAND_SURE)
        palm = float(np.percentile(np.linalg.norm(V[sure, 9] - V[sure, 0], axis=1), 90)) if sure.sum() >= 3 else np.nan
        palms.append(palm)
        if not np.isfinite(palm):
            continue
        for t in range(T):
            if np.median(Vc[t]) < HAND_SURE or np.median(Rc[t]) < 0.4:
                continue
            ok = (Vc[t, 1:] >= 0.5) & (Rc[t, 1:] >= 0.3)
            if ok.sum() < 10:
                continue
            a_, b_ = V[t, 1:] - V[t, 0], R[t, 1:] - R[t, 0]
            err[t, h] = np.linalg.norm(a_ - b_, axis=1)[ok].mean() / palm
            zv = cross2(V[t, 5] - V[t, 0], V[t, 17] - V[t, 0]) / palm ** 2
            zr = cross2(R[t, 5] - R[t, 0], R[t, 17] - R[t, 0]) / palm ** 2
            if abs(zv) >= 0.2:
                facing[t, h] = float(np.sign(zv) == np.sign(zr))
            u, w = V[t, 9] - V[t, 0], R[t, 9] - R[t, 0]
            direc[t, h] = abs(np.degrees(np.arctan2(cross2(u, w), u @ w)))
    return err, facing, direc, palms


class Pose:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(self.dev).eval()

    def read17(self, rgb, mask):
        """The model's own 17 points (COCO: nose, eyes, ears, then the body), image y down, and their
        confidences - read inside the figure's box (the mask's, padded 10%)."""
        from PIL import Image
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            return np.full((17, 2), np.nan), np.zeros(17)
        pad = 0.1 * max(np.ptp(xs), np.ptp(ys))
        x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
        x1, y1 = min(xs.max() + pad, rgb.shape[1]), min(ys.max() + pad, rgb.shape[0])
        box = [[[float(x0), float(y0), float(x1 - x0), float(y1 - y0)]]]
        inp = self.proc(Image.fromarray(rgb), boxes=box, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            res = self.proc.post_process_pose_estimation(self.model(**inp), boxes=box)[0][0]
        return res["keypoints"].cpu().numpy(), res["scores"].cpu().numpy()

    def __call__(self, rgb, mask):
        """13 points (head + 12 joints), image y down, and their confidences."""
        return to13(*self.read17(rgb, mask))


def to13(kp, sc):
    """The pose model's 17 points -> the 13 the fit works in: the head (nose and ears, by confidence)
    and the twelve joints."""
    if not np.isfinite(kp).all():
        return np.full((13, 2), np.nan), np.zeros(13)
    hw = sc[[0, 3, 4]]
    head = (kp[[0, 3, 4]] * hw[:, None]).sum(0) / max(hw.sum(), 1e-6)
    order = [5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]
    return np.vstack([head, kp[order]]), np.concatenate([[hw.mean()], sc[order]])


def face_error(vk17, vc17, rk17, rc17, s, tv):
    """How the face lies, per frame: the nose, eyes and ears about their own centre - render against
    video, the render at the clip's scale - in the video's torso lengths. The head's turn and tilt,
    apart from where the body puts it."""
    out = np.full(len(vk17), np.nan)
    for t in range(len(vk17)):
        w = np.minimum(vc17[t, :5], rc17[t, :5])
        w = w * (w > 0.3)
        if (w > 0).sum() < 4:
            continue
        a_ = vk17[t, :5] - (vk17[t, :5] * w[:, None]).sum(0) / w.sum()
        b_ = s * (rk17[t, :5] - (rk17[t, :5] * w[:, None]).sum(0) / w.sum())
        out[t] = float((np.linalg.norm(a_ - b_, axis=1) * w).sum() / w.sum() / tv)
    return out


def video_frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    return out


def render(blend, clip, frames, yaw, out, every=1):
    done = sorted(out.glob("[0-9]*.png")) if out.exists() else []
    if len(done) == len(range(0, frames, every)) and (out / "camera.txt").exists() and \
            f"yaw {yaw:.1f}" in (out / "camera.txt").read_text() and \
            (out / "camera.txt").stat().st_mtime > blend.stat().st_mtime:
        return done                                                    # rendered already, same rig
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([BLENDER, "-b", "-noaudio", "--python", str(ROOT / "blender" / "render_match.py"), "--",
                    "--blend", str(blend), "--clip", clip, "--frames", str(frames), "--yaw", f"{yaw:.1f}",
                    "--out", str(out), "--every", str(every)], check=True, capture_output=True)
    (out / "camera.txt").write_text((out / "camera.txt").read_text().replace(f"yaw {float(f'{yaw:.1f}')}", f"yaw {yaw:.1f}"))
    return sorted(out.glob("[0-9]*.png"))


def load_render(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB)
    alpha = im[..., 3] > 127
    # the pose model reads it on the video's grey, as it reads the video
    grey = np.full_like(rgb, 205)
    a = (im[..., 3:4].astype(np.float32) / 255)
    comp = (rgb * a + grey * (1 - a)).astype(np.uint8)
    return comp, alpha


def score(vid_kp, vid_conf, vid_masks, ren, pose, idx):
    """Joint error (torso lengths) and silhouette IoU per frame, after one scale for the clip and the
    hips laid on the video's hips per frame."""
    rk, rc, rm, rk17, rc17 = [], [], [], [], []
    for p in ren:
        rgb, alpha = load_render(p)
        k17, c17 = pose.read17(rgb, alpha)
        k, c = to13(k17, c17)
        rk.append(k), rc.append(c), rm.append(alpha), rk17.append(k17), rc17.append(c17)
    rk, rc = np.array(rk), np.array(rc)
    score.last17 = (np.array(rk17), np.array(rc17))
    vk, vc = vid_kp[idx], vid_conf[idx]
    hip = lambda k: (k[:, 7] + k[:, 10]) / 2                          # noqa: E731
    sh = lambda k: (k[:, 1] + k[:, 4]) / 2                            # noqa: E731
    tv = np.nanmedian(np.linalg.norm(sh(vk) - hip(vk), axis=1))
    tr = np.nanmedian(np.linalg.norm(sh(rk) - hip(rk), axis=1))
    s = tv / max(tr, 1e-6)
    off = hip(vk) - s * hip(rk)                                        # per frame
    rk_al = rk * s + off[:, None, :]
    ok = (vc > 0.3) & (rc > 0.3)
    err = np.linalg.norm(rk_al - vk, axis=-1) / tv
    err[~ok] = np.nan
    ious = []
    for j, i in enumerate(idx):
        M = np.float32([[s, 0, off[j, 0]], [0, s, off[j, 1]]])
        h, w = vid_masks[i].shape
        warped = cv2.warpAffine(rm[j].astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
        inter = (warped & vid_masks[i]).sum()
        ious.append(inter / max((warped | vid_masks[i]).sum(), 1))
    score.last_scale = (s, tv)
    # the silhouettes alone: one scale for the clip (the best of a few about the torso's), each frame
    # placed feet on feet and centre on centre - where a game puts the character is its own business,
    # and the pose model's hips put the render a few percent large on stylised figures
    best = None
    for k in np.linspace(0.9, 1.06, 9):
        sc_ = s * k
        iou_k = []
        for j, i in enumerate(idx):
            m_ = vid_masks[i]
            ys, xs = np.nonzero(rm[j])
            vy, vx = np.nonzero(m_)
            if len(ys) < 20 or len(vy) < 20:
                iou_k.append(0.0)
                continue
            M = np.float32([[sc_, 0, (vx.min() + vx.max()) / 2 - sc_ * (xs.min() + xs.max()) / 2],
                            [0, sc_, vy.max() - sc_ * ys.max()]])
            wp = cv2.warpAffine(rm[j].astype(np.uint8), M, (m_.shape[1], m_.shape[0]), flags=cv2.INTER_NEAREST) > 0
            iou_k.append((wp & m_).sum() / max((wp | m_).sum(), 1))
        if best is None or np.mean(iou_k) > np.mean(best):
            best = iou_k
    score.last_shape_iou = np.array(best)
    return err, np.array(ious), rk_al, s, off, rm, rk, rc


def main(a):
    performer = a.performer or a.name
    vids = ROOT / "work" / performer / "motion_videos"
    mine = ROOT / "work" / a.name / "motion_videos"
    mp4 = vids / f"{a.move}.mp4"
    spec = json.load(open(vids / f"{a.move}_match.json"))["clip_spec"]
    yaw0 = (spec.get("fit") or {}).get("preview_yaw_deg", 0.0)
    blend = Path(a.blend) if a.blend else ROOT / "work" / a.name / "final.blend"
    out = Path(a.out) if a.out else mine
    out.mkdir(parents=True, exist_ok=True)
    frames = video_frames(mp4)
    n = len(frames)
    pose = Pose()
    print(f"[fidelity] {a.name} {a.move}: {n} video frames; reading the video", flush=True)
    # the video's figure, cut out by the generation stage's background remover (cached): a colour
    # threshold took half of H3's studio backdrop - darker in the middle - for the figure
    mfile = vids / f"{a.move}_masks.npz"
    if not mfile.exists():
        subprocess.run([str(ROOT / "vendor" / "trellis2mlx" / ".venv" / "bin" / "python"),
                        str(ROOT / "pipeline" / "foreground.py"), "--video", str(mp4), "--out", str(mfile)],
                       check=True, capture_output=True)
    masks = np.load(mfile)["masks"]
    vid_kp, vid_conf, vid_masks, vid17, vidc17 = [], [], [], [], []
    for fr, m in zip(frames, masks):
        k17, c17 = pose.read17(fr, m)
        k, c = to13(k17, c17)
        vid_kp.append(k), vid_conf.append(c), vid_masks.append(m), vid17.append(k17), vidc17.append(c17)
    vid_kp, vid_conf, vid17, vidc17 = np.array(vid_kp), np.array(vid_conf), np.array(vid17), np.array(vidc17)
    work = Path(a.renders) if a.renders else mine / f"{a.move}_match_renders"
    yaw = yaw0 if a.yaw is None else a.yaw
    if a.yaw_search and a.yaw is None:
        sample = list(range(0, n, 8))
        best = None
        for dy in (-30, -20, -10, 0, 10, 20, 30):
            ren = render(blend, a.move, n, yaw0 + dy, work / f"yaw{dy:+d}", every=8)
            err, iou, *_ = score(vid_kp, vid_conf, vid_masks, ren, pose, sample)
            e = float(np.nanmean(err))
            print(f"[fidelity]   yaw {yaw0 + dy:6.1f}: joints {e:.3f} torso lengths, IoU {iou.mean():.3f}", flush=True)
            if best is None or e < best[0]:
                best = (e, yaw0 + dy)
        yaw = best[1]
    ren = render(blend, a.move, n, yaw, work / "final")
    idx = list(range(len(ren)))
    err, iou, rk_al, s, off, rm, rk, rc = score(vid_kp, vid_conf, vid_masks, ren, pose, idx)
    fe = face_error(vid17[idx], vidc17[idx], *score.last17, *score.last_scale)
    # what the pose model read on both sides, for tools/motion_stages.py
    np.savez_compressed(out / f"{a.move}_fidelity_kp.npz", vid_kp=vid_kp, vid_conf=vid_conf, ren_kp=rk,
                        ren_conf=rc, yaw=yaw, vid17=vid17, vidc17=vidc17, ren17=score.last17[0], renc17=score.last17[1])
    per_joint = {nm: round(float(np.nanmean(err[:, j])), 4) for j, nm in enumerate(NAMES)}
    summary = {"character": a.name, "move": a.move, "performer": performer, "frames": n, "yaw_deg": yaw,
               "joint_error_torso": round(float(np.nanmean(err)), 4),
               "joint_error_torso_p90": round(float(np.nanpercentile(err, 90)), 4),
               "silhouette_iou": round(float(np.mean(iou)), 4), "silhouette_iou_p10": round(float(np.percentile(iou, 10)), 4),
               "face_error_torso": round(float(np.nanmean(fe)), 4),
               "silhouette_iou_placed": round(float(np.mean(score.last_shape_iou)), 4),
               "per_joint": per_joint,
               "per_frame": {"joint_error": [None if np.isnan(x) else round(float(x), 4) for x in np.nanmean(err, 1)],
                             "iou": [round(float(x), 4) for x in iou]}}
    # the hands: DWPose on the video and on the renders (ViTPose's 17 points stop at the wrists)
    hands_txt = ""
    if DW_MODEL.exists():
        vdk, vdc = dwpose(vids / f"{a.move}_dw.npz", video=mp4, masks=mfile)
        rdk, rdc = dwpose(work / "final" / "dw.npz", frames=work / "final")
        he, hf, hd, palms = hand_scores(vdk[idx], vdc[idx], rdk, rdc, s, off)
        summary.update({"hand_error_palm": round(float(np.nanmean(he)), 4) if np.isfinite(he).any() else None,
                        "palm_facing_agree": round(float(np.nanmean(hf)), 4) if np.isfinite(hf).any() else None,
                        "hand_direction_deg": round(float(np.nanmean(hd)), 1) if np.isfinite(hd).any() else None,
                        # the median too: the render's hand misread in a few frames (DWPose finding it on a dark
                        # sleeve 1.2-1.5 palm lengths off) moved Aoi's mean 0.25 -> 0.33 at a median of 0.17
                        "hand_error_palm_median": round(float(np.nanmedian(he)), 4) if np.isfinite(he).any() else None,
                        "hand_direction_deg_median": round(float(np.nanmedian(hd)), 1) if np.isfinite(hd).any() else None,
                        "hand_frames": int(np.isfinite(he).sum()), "palm_px": [round(p, 1) for p in palms],
                        "per_frame_hands": {"error": [[None if np.isnan(x) else round(float(x), 3) for x in r] for r in he],
                                            "facing": [[None if np.isnan(x) else int(x) for x in r] for r in hf]}})
        if summary["hand_error_palm"] is not None:
            hands_txt = (f", hands {summary['hand_error_palm']:.2f} palm lengths, palm facing "
                         f"{summary['palm_facing_agree']:.0%}, pointing {summary['hand_direction_deg']:.0f} deg off "
                         f"({summary['hand_frames']} hand-frames)")
        hand_sheet(frames, vdk[idx], vdc[idx], work / "final", rdk, rdc, he, out / f"{a.move}_hands.png")
    json.dump(summary, open(out / f"{a.move}_fidelity.json", "w"), indent=1)
    # the sheet: six frames, the character's outline and both skeletons over the video
    bones = [(1, 2), (2, 3), (4, 5), (5, 6), (1, 4), (7, 10), (1, 7), (4, 10), (7, 8), (8, 9), (10, 11), (11, 12)]
    tiles = []
    for i in np.linspace(0, n - 1, 6).astype(int):
        im = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR).copy()
        M = np.float32([[s, 0, off[i, 0]], [0, s, off[i, 1]]])
        h, w = im.shape[:2]
        warped = cv2.warpAffine(rm[i].astype(np.uint8) * 255, M, (w, h), flags=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cnts, -1, (255, 140, 40), 2)
        for kp, col in ((vid_kp[i], (60, 200, 60)), (rk_al[i], (220, 60, 220))):
            for p, q in bones:
                if np.isfinite(kp[[p, q]]).all():
                    cv2.line(im, tuple(int(v) for v in kp[p]), tuple(int(v) for v in kp[q]), col, 2, cv2.LINE_AA)
        e = np.nanmean(err[i])
        cv2.putText(im, f"f{i}  joints {e:.2f}  IoU {iou[i]:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(im, f"f{i}  joints {e:.2f}  IoU {iou[i]:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (250, 250, 250), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(im, (int(w * 360 / h), 360)))
    cv2.imwrite(str(out / f"{a.move}_fidelity.png"), np.hstack(tiles))
    print(f"[fidelity] {a.name} {a.move} at yaw {yaw:.0f}: joints {summary['joint_error_torso']:.3f} torso lengths "
          f"(p90 {summary['joint_error_torso_p90']:.3f}), silhouette IoU {summary['silhouette_iou']:.3f} "
          f"(p10 {summary['silhouette_iou_p10']:.3f}; placed by silhouette {summary['silhouette_iou_placed']:.3f}), "
          f"face {summary['face_error_torso']:.3f}{hands_txt}; worst joints: "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(per_joint.items(), key=lambda kv: -kv[1])[:3]), flush=True)


def hand_sheet(frames, vk, vc, ren_dir, rk, rc, err, path, n=4):
    """Close-ups of both hands at n frames DWPose is sure of: the video's, then the character's, each with
    DWPose's points (thumb red, index green, middle blue, ring magenta, little finger yellow)."""
    col = {1: (60, 60, 240), 5: (60, 200, 60), 9: (240, 160, 40), 13: (200, 60, 200), 17: (40, 200, 230)}
    good = [t for t in range(len(vk)) if np.isfinite(err[t]).all()]
    if not good:
        return
    tiles = []

    def tile(img, P, txt):
        size = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]), 24) * 1.6
        c = P.mean(0)
        M = np.float32([[130 / size, 0, 65 - 130 / size * c[0]], [0, 130 / size, 65 - 130 / size * c[1]]])
        im = cv2.warpAffine(img, M, (130, 130), flags=cv2.INTER_CUBIC, borderValue=(205, 205, 205))
        Q = np.c_[P, np.ones(21)] @ M.T
        for p, q in HAND_EDGES:
            cv2.line(im, tuple(np.int32(Q[p])), tuple(np.int32(Q[q])), col[max(k for k in col if k <= max(p, q))], 1, cv2.LINE_AA)
        for c_, w_ in (((0, 0, 0), 2), ((255, 255, 255), 1)):
            cv2.putText(im, txt, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, c_, w_, cv2.LINE_AA)
        return im

    for t in [good[i] for i in np.linspace(0, len(good) - 1, n).astype(int)]:
        vim = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        rim = cv2.imread(str(sorted(ren_dir.glob("[0-9]*.png"))[t]), cv2.IMREAD_UNCHANGED)
        al = rim[..., 3:4].astype(np.float32) / 255
        rim = (rim[..., :3] * al + 205 * (1 - al)).astype(np.uint8)
        for h, s0 in enumerate((91, 112)):
            tiles += [tile(vim, vk[t, s0:s0 + 21], f"f{t} {'LR'[h]} video"),
                      tile(rim, rk[t, s0:s0 + 21], f"rig  {err[t, h]:.2f} palm"), np.full((130, 4, 3), 255, np.uint8)]
    cv2.imwrite(str(path), np.hstack(tiles[:-1]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--move", required=True)
    ap.add_argument("--performer", default=None)
    ap.add_argument("--yaw-search", action="store_true")
    ap.add_argument("--yaw", type=float, default=None, help="the camera angle to render from (skips the search)")
    ap.add_argument("--blend", default=None, help="the character's blend (default work/<name>/final.blend)")
    ap.add_argument("--renders", default=None, help="where to keep the renders (default next to the video)")
    ap.add_argument("--out", default=None, help="where to write the scores (default next to the video)")
    main(ap.parse_args())
