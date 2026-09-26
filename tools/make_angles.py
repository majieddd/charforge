"""One clip from several cameras at once, in one video: the same frame from the front, the side and above,
side by side, with the joints read out under them - work/<name>/qa/angles_<clip>.mp4.

    python tools/make_angles.py --name aoi --clip spell_cast                  # front, left side, top
    python tools/make_angles.py --name aoi --clip spell_cast --video          # a move from video: the video,
                                                                              # its camera, the side, the top
    python tools/make_angles.py --name pip --clip jump --views front,left,back,top --res 320

One camera hides depth: Aoi's spell cast looked right from the video's angle while her elbow was folded
back through her arm (176 degrees). Every panel is orthographic and at one scale - a 10 cm grid on the
floor and on the wall behind - so what lies in front of what, and how far, reads straight off the
picture. The strip under the panels gives each elbow's and knee's bend (red past 155 and 165 degrees,
further than the motion library ever bends them) and each shoe's lowest point above the floor (red
below -1 cm: through it). Needs Blender and ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv, _argv = [sys.argv[0]], sys.argv
import charforge  # noqa: E402
sys.argv = _argv

LABELS = {"front": "front", "back": "back", "left": "left side", "right": "right side", "top": "from above"}
ELBOW_MAX, KNEE_MAX = 155.0, 165.0          # the motion library: elbows 159 at most (150 in 0.05% of frames), a sprint's knee 156
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    SMALL = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    BOLD = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15)
except OSError:
    FONT = SMALL = BOLD = ImageFont.load_default()
INK, DIM, BAD, GOOD = (236, 240, 245), (150, 160, 172), (255, 92, 92), (120, 220, 150)


def tag(d, xy, text, font=FONT):
    x, y = xy
    w = d.textlength(text, font=font)
    d.rounded_rectangle((x - 5, y - 3, x + w + 6, y + font.size + 5), radius=5, fill=(14, 17, 22))
    d.text((x, y), text, fill=INK, font=font)


def video_frames(mp4, fps, n):
    """The video sampled at the render's rate (as tools/side_by_side.py does), n frames."""
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    out, t, k = [], 0.0, 0
    while len(out) < n:
        ok, fr = cap.read()
        if not ok:
            break
        while k / vfps + 1e-9 >= t and len(out) < n:     # repeat a frame when the render runs faster
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            t += 1.0 / fps
        k += 1
    return out


def readout(row):
    """The strip's text, as (text, colour) runs."""
    runs = [(f"{row['t']:5.2f} s", INK), ("   elbows ", DIM)]
    for s_ in ("left", "right"):
        v = row["elbow"].get(s_)
        runs.append((f"{s_[0].upper()} {v:.0f}°  " if v is not None else f"{s_[0].upper()} -  ", BAD if v and v > ELBOW_MAX else INK))
    runs.append(("  knees ", DIM))
    for s_ in ("left", "right"):
        v = row["knee"].get(s_)
        runs.append((f"{s_[0].upper()} {v:.0f}°  " if v is not None else f"{s_[0].upper()} -  ", BAD if v and v > KNEE_MAX else INK))
    runs.append(("  shoes above the floor ", DIM))
    for s_ in ("left", "right"):
        v = (row.get("foot") or {}).get(s_)
        runs.append((f"{s_[0].upper()} {v * 100:.1f} cm  " if v is not None else f"{s_[0].upper()} -  ",
                     BAD if v is not None and v < -0.01 else INK))
    return runs


def main(a):
    blend = ROOT / "work" / a.name / "final.blend"
    if not blend.exists():
        raise SystemExit(f"[angles] no finished character at {blend}")
    ff = shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if Path(p).exists()), None)
    if not ff:
        raise SystemExit("[angles] ffmpeg is needed to make the video: brew install ffmpeg")
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    mp4, yaw = None, 0.0
    if a.video:
        spec = (json.load(open(ROOT / "work" / a.name / "extra_clips.json")) if (ROOT / "work" / a.name / "extra_clips.json").exists() else {}).get(a.clip)
        if not spec or not str(spec.get("from_video", "")).endswith(".mp4"):
            raise SystemExit(f"[angles] {a.clip} is not a move from video for {a.name}")
        src = Path(spec["from_video"])
        mp4 = ROOT / "work" / src.parent.parent.name / "motion_videos" / f"{a.clip}.mp4"
        yaw = float((spec.get("fit") or {}).get("preview_yaw_deg", 0.0))
        views = [f"angle:{yaw:.1f}"] + [v for v in views if v != "front"]
    out = Path(a.out) if a.out else ROOT / "work" / a.name / "qa" / f"angles_{a.clip}{'_video' if a.video else ''}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cmd = [charforge.blender_bin(), "-b", "-noaudio", "--python", str(ROOT / "blender" / "render_angles.py"), "--",
               "--blend", str(blend), "--clip", a.clip, "--out", str(tmp / "r"), "--views", ",".join(views),
               "--res", str(a.res)]
        with charforge.gpu("rendering the angles"):
            r = subprocess.run(cmd, capture_output=True, text=True)
        said = [ln for ln in (r.stdout + r.stderr).splitlines() if "[angles]" in ln]
        for ln in said:
            print(ln, flush=True)
        meta_p = tmp / "r" / "angles.json"
        if r.returncode != 0 or not meta_p.exists():
            raise SystemExit("[angles] the render failed:\n" + "\n".join((r.stdout + r.stderr).splitlines()[-15:]))
        meta = json.load(open(meta_p))
        rows, fps, res = meta["frames"], meta["fps_out"], meta["res"]
        vid = video_frames(mp4, fps, len(rows)) if mp4 else []
        strip = 34
        cm_px = 0.01 / meta["metres_per_px"]
        frames_dir = tmp / "f"
        frames_dir.mkdir()
        worst = {"elbow": 0.0, "knee": 0.0}
        for k, row in enumerate(rows):
            panels = []
            if mp4:
                v = vid[min(k, len(vid) - 1)] if vid else Image.new("RGB", (res, res))
                v = v.resize((max(1, round(v.width * res / v.height)), res), Image.LANCZOS)
                d = ImageDraw.Draw(v)
                tag(d, (10, 8), "the video")
                panels.append(v)
            for vw in views:
                im = Image.open(tmp / "r" / vw.replace(":", "_") / f"f{k:04d}.png").convert("RGB")
                d = ImageDraw.Draw(im)
                tag(d, (10, 8), "the video's camera" if vw.startswith("angle:") and mp4 else LABELS.get(vw, vw.replace("angle:", "") + "°"))
                # a 50 cm scale bar at the bottom left
                x0, y0 = 12, res - 16
                d.line((x0, y0, x0 + 50 * cm_px, y0), fill=INK, width=3)
                d.text((x0, y0 - 18), "50 cm", fill=INK, font=SMALL)
                panels.append(im)
            W = sum(p.width for p in panels) + 4 * (len(panels) - 1)
            fr = Image.new("RGB", (W + (W % 2), res + strip + (res + strip) % 2), (10, 12, 16))
            x = 0
            for p in panels:
                fr.paste(p, (x, 0))
                x += p.width + 4
            d = ImageDraw.Draw(fr)
            x = 12
            for text, col in readout(row):
                d.text((x, res + 8), text, fill=col, font=BOLD)
                x += d.textlength(text, font=BOLD)
            fr.save(frames_dir / f"{k:04d}.png")
            for j in ("elbow", "knee"):
                worst[j] = max([worst[j]] + [v for v in row[j].values() if v is not None])
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", str(frames_dir / "%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-movflags", "+faststart",
                        str(out)], check=True)
    flags = [f"elbow to {worst['elbow']:.0f}°" + (" (past 155)" if worst["elbow"] > ELBOW_MAX else ""),
             f"knee to {worst['knee']:.0f}°" + (" (past 165)" if worst["knee"] > KNEE_MAX else "")]
    json.dump({"clip": a.clip, "views": views, "video": str(mp4) if mp4 else None, "frames": len(rows),
               "worst_elbow_deg": worst["elbow"], "worst_knee_deg": worst["knee"]},
              open(out.with_suffix(".json"), "w"), indent=1)
    print(f"[angles] {a.clip}: {len(rows)} frames, {', '.join(flags)} -> {out} ({out.stat().st_size / 1e6:.1f} MB)",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--views", default="front,left,top", help="front, back, left, right, top, angle:<deg>")
    ap.add_argument("--video", action="store_true", help="a move from video: its video and its camera first")
    ap.add_argument("--res", type=int, default=360)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
