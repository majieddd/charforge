"""How far a reference image's hands hang clear of the body, read by a pose model.

A reference whose hands touch the hips or thighs fuses them to the body in the 3D model: Bo, a
cartoon chef drawn with his fists on his hips, came out of the whole pipeline with his hands in
shreds at his belt. The reference stage measures each new image with this and draws it again when
the arms do not clear the body.

    arm_gap   each wrist's sideways distance from its own hip, in torso lengths (shoulders to hips),
              the smaller of the two. An A-pose clears the body at 0.58-0.72 (Krea 2 and Qwen-Image
              2.1 with the angle spelled out, the pipeline's own references 0.60-0.79); Bo's fists
              on his hips read 0.33.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_MODEL = {}


def _pose():
    if "run" in _MODEL:
        return _MODEL["run"]
    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(dev).eval()

    def run(rgb, mask):
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            return None, None
        box = [[[float(xs.min()), float(ys.min()), float(np.ptp(xs)), float(np.ptp(ys))]]]
        inp = proc(Image.fromarray(rgb), boxes=box, return_tensors="pt").to(dev)
        with torch.no_grad():
            res = proc.post_process_pose_estimation(model(**inp), boxes=box)[0][0]
        return res["keypoints"].cpu().numpy(), res["scores"].cpu().numpy()
    _MODEL["run"] = run
    return run


def arm_gap(path_or_rgba, mask=None) -> dict:
    """{'arm_gap': float or None, 'wrist_conf': float} for an RGBA cut-out or an RGB image and its mask."""
    im = Image.open(path_or_rgba) if isinstance(path_or_rgba, (str, bytes)) or hasattr(path_or_rgba, "__fspath__") \
        else path_or_rgba
    arr = np.asarray(im.convert("RGBA") if im.mode in ("RGBA", "LA") else im.convert("RGB"))
    if arr.shape[2] == 4 and mask is None:
        mask = arr[..., 3] > 127
        bg = np.full(arr.shape[:2] + (3,), 205, np.uint8)
        a = arr[..., 3:4].astype(np.float32) / 255
        rgb = (arr[..., :3] * a + bg * (1 - a)).astype(np.uint8)
    else:
        rgb = arr[..., :3]
        if mask is None:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from align import image_mask
            mask = image_mask(rgb)
    kp, sc = _pose()(rgb, mask)
    if kp is None:
        return {"arm_gap": None, "wrist_conf": 0.0}
    sh, hip = (kp[5] + kp[6]) / 2, (kp[11] + kp[12]) / 2
    torso = float(np.linalg.norm(sh - hip)) + 1e-6
    gap = [abs(kp[9][0] - kp[11][0]) / torso, abs(kp[10][0] - kp[12][0]) / torso]
    return {"arm_gap": round(float(min(gap)), 3), "wrist_conf": round(float(min(sc[9], sc[10])), 3)}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="how far a reference's hands hang clear of its body")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None, help="write the result here as JSON")
    a = ap.parse_args()
    res = arm_gap(a.image)
    print(f"[gate] arms clear the body by {res['arm_gap']} torso lengths (wrists found at {res['wrist_conf']})",
          flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))
