#!/usr/bin/env python3
"""Raking renders of E168 variants, side by side (E132, E135 style): the shape with its normal map, grey, low light.

    ../.venv/bin/python pipeline/remesh_render.py --name pip --out work/pip/e168/renders \
        --item default=work/pip/retopo.glb --item iso60=work/pip/e168/iso60/retopo.glb

For each view (torso: front and back of the chest and arms; face: a close-up of the head) it runs
blender/render_raking.py on every item (under charforge.gpu(), EEVEE is the GPU) and stacks the items into one PNG per
view, labelled, so the variants can be compared in one image. Writes <out>/<view>_<item>_<az>.png and
<out>/compare_<view>_<az>.png.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import charforge  # noqa: E402

# view name -> (render_raking arguments, azimuths)
VIEWS = {
    "torso": (["--z", "0.70", "--span", "0.50"], ["0", "180"]),
    "face": (["--z", "0.90", "--span", "0.22", "--res", "700"], ["0"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--item", action="append", required=True, help="label=path/to/retopo.glb (repeatable)")
    ap.add_argument("--views", default="torso,face")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(out, exist_ok=True)
    items = []
    for it in a.item:
        label, path = it.split("=", 1)
        items.append((label, path if os.path.isabs(path) else os.path.join(ROOT, path)))
    from PIL import Image, ImageDraw
    blender = charforge.blender_bin()
    for view in a.views.split(","):
        extra, azs = VIEWS[view]
        for az in azs:
            paths = []
            for label, glb in items:
                prefix = os.path.join(out, f"{view}_{label}")
                with charforge.gpu(f"raking {a.name} {view} {label}"):
                    r = subprocess.run([blender, "-b", "-noaudio", "--python",
                                        os.path.join(ROOT, "blender", "render_raking.py"), "--", "--mesh", glb,
                                        "--out", prefix, "--az", az] + extra, capture_output=True, text=True)
                p = f"{prefix}_{int(az):03d}.png"            # render_raking.py's name for azimuth az
                if r.returncode != 0 or not os.path.exists(p):
                    print("\n".join((r.stdout + r.stderr).splitlines()[-8:]))
                    raise SystemExit(f"[render] {view} {label} az {az} failed")
                paths.append((label, p))
                print(f"[render] {view} {label} az {az}: {p}", flush=True)
            imgs = [Image.open(p).convert("RGB") for _, p in paths]
            w = sum(i.width for i in imgs) + 8 * (len(imgs) - 1)
            h = max(i.height for i in imgs) + 28
            sheet = Image.new("RGB", (w, h), (255, 255, 255))
            x = 0
            d = ImageDraw.Draw(sheet)
            for (label, _), im in zip(paths, imgs):
                sheet.paste(im, (x, 28))
                d.text((x + 6, 6), f"{label}  ({view}, az {az})", fill=(0, 0, 0))
                x += im.width + 8
            sheet.save(os.path.join(out, f"compare_{view}_{az}.png"))
            print(f"[render] wrote {os.path.join(out, f'compare_{view}_{az}.png')}", flush=True)


if __name__ == "__main__":
    main()
