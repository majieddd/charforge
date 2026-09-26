"""DWPose whole-body keypoints (133: body 17, feet 6, face 68, hands 2 x 21) for a video's figure.

    vendor/trellis2mlx/.venv/bin/python tools/dwpose.py --video move.mp4 --masks move_masks.npz --out move_dw.npz
    vendor/trellis2mlx/.venv/bin/python tools/dwpose.py --frames renders/ --out renders/dw.npz

ViTPose's 17 points stop at the wrists and ankles, and it misreads poses it rarely saw - at the top of
Mara's high kick it put her ankle at her hip. DWPose (Yang et al. 2023, distilled RTMPose-l on
COCO-WholeBody; models/dwpose/dw-ll_ucoco_384.onnx from huggingface.co/yzd-v/DWPose, Apache-2.0)
reads the whole body, hands and feet included. Top-down: each frame is cropped to its figure's box
(pipeline/foreground.py --video cut the figure out, so no person detector is needed), padded 25% and
brought to its 288 x 384 input; its SimCC output (two 1-D distributions per point, at half-pixel
resolution) is decoded by argmax and mapped back. Runs with onnxruntime (the TRELLIS environment has it).

Writes kp133 (frames, 133, 2; pixels, y down) and sc133 (frames, 133) - the COCO-WholeBody order:
0-16 body (COCO), 17-22 feet (left big toe, small toe, heel; right the same), 23-90 face, 91-111 left
hand, 112-132 right hand.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "dwpose" / "dw-ll_ucoco_384.onnx"
IN_W, IN_H = 288, 384
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


class DWPose:
    def __init__(self, model=MODEL):
        import onnxruntime as ort
        prov = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        try:
            self.sess = ort.InferenceSession(str(model), providers=prov)
        except Exception:                                            # noqa: BLE001 - CoreML can refuse a graph
            self.sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, rgb, mask):
        """133 points (pixels, y down) and their confidences for the figure in `mask`."""
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            return np.full((133, 2), np.nan), np.zeros(133)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        w, h = (xs.max() - xs.min()) * 1.25, (ys.max() - ys.min()) * 1.25
        if w > h * IN_W / IN_H:                                      # the input's aspect, 3:4
            h = w * IN_H / IN_W
        else:
            w = h * IN_W / IN_H
        sx, sy = IN_W / w, IN_H / h
        M = np.float32([[sx, 0, IN_W / 2 - sx * cx], [0, sy, IN_H / 2 - sy * cy]])
        crop = cv2.warpAffine(rgb, M, (IN_W, IN_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        x = ((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        sim_x, sim_y = self.sess.run(None, {self.inp: x})
        px = sim_x[0].argmax(1) / 2.0
        py = sim_y[0].argmax(1) / 2.0
        conf = np.minimum(sim_x[0].max(1), sim_y[0].max(1))
        kp = np.stack([(px - IN_W / 2) / sx + cx, (py - IN_H / 2) / sy + cy], 1)
        return kp, conf


def video_frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    return out


def render_frames(folder):
    """A folder of RGBA renders (blender/render_match.py): each laid on the video's grey, as
    tools/motion_fidelity.py shows them to its pose model, and its alpha for the figure."""
    frames, masks = [], []
    for p in sorted(Path(folder).glob("[0-9]*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        al = im[..., 3:4].astype(np.float32) / 255
        frames.append((rgb * al + 205 * (1 - al)).astype(np.uint8))
        masks.append(im[..., 3] > 127)
    return frames, masks


def main(a):
    if a.frames:
        frames, masks = render_frames(a.frames)
    else:
        frames, masks = video_frames(a.video), np.load(a.masks)["masks"]
    net = DWPose()
    K, C = [], []
    for fr, m in zip(frames, masks):
        k, c = net(fr, m)
        K.append(k), C.append(c)
    K, C = np.array(K), np.array(C)
    np.savez_compressed(a.out, kp133=K.astype(np.float32), sc133=C.astype(np.float32))
    print(f"[dwpose] {len(K)} frames -> {a.out}; median confidence body {np.median(C[:, :17]):.2f}, "
          f"feet {np.median(C[:, 17:23]):.2f}, hands {np.median(C[:, 91:]):.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="the video (with --masks, its figure per frame)")
    ap.add_argument("--masks")
    ap.add_argument("--frames", help="or a folder of RGBA renders, the figure from their alpha")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not (a.frames or (a.video and a.masks)):
        ap.error("give --video and --masks, or --frames")
    main(a)
