"""A short video of a finished character: its clips back to back from one camera, as an .mp4.

    python tools/make_reel.py --name bo                    # work/bo/qa/reel.mp4
    python tools/make_reel.py --name bo --clips idle:1.0,wave,walk:2 --res 640 --out reel.mp4

By default: a second of idle, a wave, two walk cycles, two jog cycles, a jump, a turn, and every move
the character has from video. Needs Blender and ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv, _argv = [sys.argv[0]], sys.argv                  # charforge parses nothing at import
import charforge  # noqa: E402
sys.argv = _argv

STANDARD = {"idle", "walk", "jog", "run", "sprint", "walk_back", "jog_back", "strafe_left", "strafe_right",
            "turn_left", "turn_right", "turn_180", "crouch_idle", "crouch_walk", "jump", "fall", "land", "wave"}


def main(a):
    blend = ROOT / "work" / a.name / "final.blend"
    if not blend.exists():
        raise SystemExit(f"[reel] no finished character at {blend}")
    ff = shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
                                          if Path(p).exists()), None)
    if not ff:
        raise SystemExit("[reel] ffmpeg is needed to make the video: brew install ffmpeg")
    man = json.load(open(ROOT / "out" / a.name / f"{a.name}.json")) if (ROOT / "out" / a.name / f"{a.name}.json").exists() else {}
    have = [c.get("name") for c in man.get("clips", [])]
    if a.clips:
        clips = a.clips
    else:
        want = ["idle:1.0", "wave", "walk:2", "jog:2", "jump", "turn_180"]
        want = [w for w in want if w.split(":")[0] in have] if have else want
        want += [c for c in have if c and c not in STANDARD]                      # its moves from video
        clips = ",".join(want)
    out = Path(a.out) if a.out else ROOT / "work" / a.name / "qa" / "reel.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        with charforge.gpu("recording a reel"):
            r = subprocess.run([charforge.blender_bin(), "-b", "-noaudio", "--python", str(ROOT / "blender" / "render_reel.py"),
                                "--", "--blend", str(blend), "--clips", clips, "--out", tmp, "--res", str(a.res),
                                "--fps", str(a.fps)], capture_output=True, text=True)
        said = [ln for ln in (r.stdout + r.stderr).splitlines() if "[reel]" in ln]
        for ln in said:
            print(ln, flush=True)
        if r.returncode != 0 or not any(Path(tmp).glob("*.png")):
            raise SystemExit("[reel] the render failed:\n" + "\n".join((r.stdout + r.stderr).splitlines()[-15:]))
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(Path(tmp) / "%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-movflags", "+faststart",
                        str(out)], check=True)
    print(f"[reel] {clips} -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--clips", default=None, help="clip[:loops or seconds.],... (default: a tour of the clips)")
    ap.add_argument("--res", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=24)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
