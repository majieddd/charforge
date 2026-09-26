"""Two motions next to each other in one video: a source video and a clip rendered as frames.

    python tools/side_by_side.py --left move.mp4 --right frames_dir --fps 24 --out compare.mp4 \
        [--labels "video" "character"] [--height 512]

The right side is a directory of PNGs (blender/render_clip_seq.py) played at --fps; the left
video is sampled at the same rate and both are scaled to --height. Written as H.264 (ffmpeg) so
browsers play it.
"""
import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--left", required=True)
ap.add_argument("--right", required=True)
ap.add_argument("--fps", type=float, default=24.0)
ap.add_argument("--out", required=True)
ap.add_argument("--labels", nargs=2, default=None)
ap.add_argument("--height", type=int, default=512)
a = ap.parse_args()


def fit_h(im, h):
    return cv2.resize(im, (int(round(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture(a.left)
vfps = cap.get(cv2.CAP_PROP_FPS) or a.fps
left, t, k = [], 0.0, 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    while k / vfps + 1e-9 >= t:                  # a 24 fps video against 30 fps renders repeats a frame
        left.append(fit_h(fr, a.height))         # now and then; an `if` took each frame once, and the
        t += 1.0 / a.fps                         # video ran 25% fast beside the rig, then stood still
    k += 1
right = [fit_h(cv2.imread(os.path.join(a.right, f)), a.height)
         for f in sorted(x for x in os.listdir(a.right) if x.lower().endswith(".png"))]
n = max(len(left), len(right))
tmp = tempfile.mkdtemp()
for i in range(n):
    L = left[min(i, len(left) - 1)].copy()
    R = right[min(i, len(right) - 1)].copy()
    if a.labels:
        for im, txt in ((L, a.labels[0]), (R, a.labels[1])):
            cv2.putText(im, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(im, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(tmp, f"{i:04d}.png"), np.hstack([L, R]))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "%04d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                a.out], check=True)
print(f"[compare] {n} frames at {a.fps:g} fps -> {a.out}")
