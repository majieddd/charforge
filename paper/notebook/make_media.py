"""Images for the lab book, from the audit renders and screenshots. Rerun after the final rebuilds.

    python make_media.py            # writes media/*.webp for whatever sources exist
"""
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import os
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # the repository
# the audit renders these figures are cut from live outside the repository (charforge_notes/ beside it)
S = Path(os.environ.get("CF_NOTES", ROOT.parent / "charforge_notes"))
M = HERE.parent / "media"
AUD0, AUD1 = S / "polish" / "audit", S / "polish" / "audit_final"   # this morning / after the last rebuild
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    SMALL = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
except OSError:
    FONT = SMALL = ImageFont.load_default()
BG = (236, 240, 245)


def save(im, name, q=82):
    im.convert("RGB").save(M / name, "WEBP", quality=q, method=6)
    print(name, im.size)


def label(im, text, xy=(10, 8), font=None):
    d = ImageDraw.Draw(im)
    x, y = xy
    f = font or FONT
    w = d.textlength(text, font=f)
    d.rounded_rectangle((x - 5, y - 3, x + w + 7, y + f.size + 6), radius=5, fill=(20, 25, 32))
    d.text((x + 1, y), text, fill=(240, 244, 248), font=f)
    return im


def tile(path, size, crop=None):
    im = Image.open(path).convert("RGB")
    if crop:
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    im.thumbnail((size, size), Image.LANCZOS)
    out = Image.new("RGB", (size, size), BG)
    out.paste(im, ((size - im.size[0]) // 2, (size - im.size[1]) // 2))
    return out


def grid(rows, size, gap=6):
    """rows: list of lists of PIL images (all size x size)."""
    h = len(rows) * size + (len(rows) - 1) * gap
    w = max(len(r) for r in rows) * size + (max(len(r) for r in rows) - 1) * gap
    out = Image.new("RGB", (w, h), (255, 255, 255))
    for i, r in enumerate(rows):
        for j, t in enumerate(r):
            out.paste(t, (j * (size + gap), i * (size + gap)))
    return out


def studio():
    shots = S / "shots"
    for src, dst in (("studio_grid.png", "studio_grid.webp"), ("studio_char_final.png", "studio_char.webp"),
                     ("studio_new.png", "studio_new.webp"), ("studio_jobs.png", "studio_jobs.webp"),
                     ("studio_angles.png", "studio_angles.webp")):
        p = shots / src
        if p.exists():
            save(Image.open(p), dst, 80)


def wings():
    shots = ["fall_1.00s_+210", "jump_1.20s_+210", "land_0.50s_+210", "crouch_walk_0.40s_+210", "wave_0.50s_+210"]
    if not all((AUD1 / "pip" / f"{s}.png").exists() for s in shots):
        return
    names = ["fall", "jump", "land", "crouch walk", "wave"]
    before = [label(tile(AUD0 / "pip" / f"{s}.png", 240), f"{n} - before", font=SMALL) for s, n in zip(shots, names)]
    after = [label(tile(AUD1 / "pip" / f"{s}.png", 240), f"{n} - after", font=SMALL) for s, n in zip(shots, names)]
    save(grid([before, after], 240), "polish_wings.webp")


def cheer():
    b = [AUD0 / "pip" / "victory_cheer_2.20s_+30.png", AUD0 / "pip" / "victory_cheer_2.20s_+210.png"]
    a = [AUD1 / "pip" / "victory_cheer_2.20s_+30.png", AUD1 / "pip" / "victory_cheer_2.20s_+210.png"]
    if not all(p.exists() for p in b + a):
        return
    row = [label(tile(b[0], 300), "before"), label(tile(a[0], 300), "after"),
           label(tile(b[1], 300), "before"), label(tile(a[1], 300), "after")]
    save(grid([row], 300), "polish_cheer.webp")


def uv():
    b = [S / "polish" / "pip_sleeves" / f"idle_0.50s_+{z}.png" for z in (210, 270)]
    a = [S / "polish" / "final_sleeves" / f"idle_0.50s_+{z}.png" for z in (210, 270)]
    if not all(p.exists() for p in b + a):
        return
    crop = (0.25, 0.25, 0.95, 0.95)
    row = [label(tile(b[0], 300, crop), "before"), label(tile(a[0], 300, crop), "after"),
           label(tile(b[1], 300, crop), "before"), label(tile(a[1], 300, crop), "after")]
    save(grid([row], 300), "polish_uv.webp")


def aoi():
    p = S / "polish" / "aoi_fix_cmp.png"
    if p.exists():
        im = Image.open(p).convert("RGB")
        w, h = im.size
        left, right = im.crop((0, 0, w // 2 - 5, h)), im.crop((w // 2 + 5, 0, w, h))
        g = grid([[label(left.resize((400, 600)), "before"), label(right.resize((400, 600)), "after")]], 600)
        g = g.crop((0, 0, 806, 600))
        out = Image.new("RGB", (806, 600), (255, 255, 255))
        out.paste(left.resize((400, 600)), (0, 0)); out.paste(right.resize((400, 600)), (406, 0))
        label(out, "before", (10, 8)); label(out, "after", (416, 8))
        save(out, "polish_aoi.webp")


def hands():
    b = S / "polish" / "pip_sleeves" / "idle_0.50s_+150.png"
    a = S / "polish" / "final_sleeves" / "idle_0.50s_+150.png"
    if b.exists() and a.exists():
        crop = (0.12, 0.66, 0.46, 1.0)
        save(grid([[label(tile(b, 360, crop), "before"), label(tile(a, 360, crop), "after")]], 360), "polish_hands.webp")


def hem():
    p = S / "polish" / "squat_zoom.png"
    if p.exists():
        im = Image.open(p).convert("RGB")
        top = im.crop((0, 0, im.size[0], im.size[1] // 2))            # Juno: old 30, old 200, new 30, new 200
        w = top.size[0] // 4
        parts = [top.crop((k * w, 0, (k + 1) * w, top.size[1])) for k in range(4)]
        for pt in parts:                                           # the source's own labels, under the new ones
            ImageDraw.Draw(pt).rectangle((0, 0, 150, 34), fill=pt.getpixel((6, 60)))
        row = [label(parts[0].resize((360, 270)), "before"), label(parts[2].resize((360, 270)), "after"),
               label(parts[1].resize((360, 270)), "before"), label(parts[3].resize((360, 270)), "after")]
        out = Image.new("RGB", (4 * 360 + 18, 270), (255, 255, 255))
        for k, t in enumerate(row):
            out.paste(t, (k * 366, 0))
        save(out, "polish_hem.webp")


def bo():
    ref0 = S / "bo_before" / "reference.png"
    mod0 = S / "bo_before" / "r" / "idle_0.50s_+20.png"
    ref1 = HERE.parent.parent / "charforge_bo_ref_new.png"
    ref1 = ROOT / "work" / "bo" / "reference.png"
    if not (ref0.exists() and mod0.exists()):
        return
    tiles = [label(tile(ref0, 360), "first draw: 0.33"), label(tile(mod0, 360), "its model")]
    if ref1.exists() and ref1.stat().st_mtime > ref0.stat().st_mtime:
        tiles.append(label(tile(ref1, 360), "drawn again"))
    save(grid([tiles], 360), "polish_bo.webp")


def webs():
    b = [S / "polish" / "knight_look" / f"{x}.png" for x in ("wave_0.50s_+30", "jump_1.20s_+30")]
    a = [AUD1 / "knight" / f"{x}.png" for x in ("wave_0.50s_+30", "jump_1.20s_+30")]
    if all(p.exists() for p in b + a):
        row = [label(tile(b[0], 300), "before"), label(tile(a[0], 300), "after"),
               label(tile(b[1], 300), "before"), label(tile(a[1], 300), "after")]
        save(grid([row], 300), "polish_webs.webp")


def loose():
    b = S / "polish" / "squat_kaito_cur_30.png"
    a = S / "polish" / "squat_kaito_new_30.png"
    if b.exists() and a.exists():
        # the whole figure: the hair spike left in the air above him and the boot toes left on the floor
        save(grid([[label(tile(b, 360), "before"), label(tile(a, 360), "after")]], 360), "polish_loose.webp")


def elbow():
    """Aoi's spell cast at 1.5 and 2.0 s: the video's frame, the elbow folded through the arm, and now."""
    B, A = S / "polish" / "elbow_before", S / "polish" / "elbow_after"
    shots = ["spell_cast_1.50s_+0.png", "spell_cast_2.00s_+0.png"]
    if not all((B / f).exists() and (A / f).exists() for f in shots):
        return
    import cv2
    cap = cv2.VideoCapture(str(ROOT / "work" / "aoi" / "motion_videos" / "spell_cast.mp4"))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    rows = []
    for f, t in zip(shots, (1.5, 2.0)):
        fr = frames[min(int(round(t * 24)), len(frames) - 1)]
        h, w = fr.shape[:2]
        v = Image.fromarray(cv2.cvtColor(fr[int(0.03 * h):int(0.03 * h) + w, :], cv2.COLOR_BGR2RGB))   # the upper body, square
        rows.append([label(tile_im(v, 300), f"video {t:.1f} s"), label(tile(B / f, 300), "before"), label(tile(A / f, 300), "after")])
    save(grid(rows, 300), "polish_elbow.webp")


def tile_im(im, size):
    im = im.copy()
    im.thumbnail((size, size), Image.LANCZOS)
    out = Image.new("RGB", (size, size), BG)
    out.paste(im, ((size - im.size[0]) // 2, (size - im.size[1]) // 2))
    return out


def board():
    """TRELLIS's first pass seen from the front: a board built in with the figure, and a clean one."""
    B = S / "polish" / "board"
    iou = {"gray": 0.44, "knight": 0.39, "kaito": 0.35, "bo": 0.93}
    who = [n for n in ("gray", "knight", "kaito", "bo") if (B / f"{n}_pass1_front.png").exists()]
    if len(who) < 4:
        return
    tiles = []
    for n in who:
        im = Image.open(B / f"{n}_pass1_front.png").convert("RGBA")
        bg = Image.new("RGBA", im.size, (98, 112, 134, 255))
        bg.alpha_composite(im)
        t = tile_im(bg.convert("RGB"), 300)
        tiles.append(label(t, f"{n.capitalize()}: {iou[n]:.2f}", font=SMALL))
    save(grid([tiles], 300), "prompts_board.webp")


def prompts():
    """The characters made from a few words, as they came out and as they come out now."""
    U = S / "user_before"
    CF = ROOT
    rows = []
    for n, what in (("boyscout", "a boy scout"), ("gray", "gray alien, extremely muscular")):
        b_ref, b_mod = U / n / "reference.png", U / n / f"{n}_thumb.png"
        a_ref, a_mod = CF / "work" / n / "reference.png", CF / "out" / n / f"{n}_thumb.png"
        if not (b_ref.exists() and b_mod.exists()):
            continue
        row = [label(tile(b_ref, 300), "before: picture", font=SMALL), label(tile_rgba(b_mod, 300), "before: model", font=SMALL)]
        if a_ref.exists() and a_ref.stat().st_mtime > b_ref.stat().st_mtime + 60:
            row += [label(tile(a_ref, 300), "now: picture", font=SMALL)]
            if a_mod.exists() and a_mod.stat().st_mtime > a_ref.stat().st_mtime:
                row += [label(tile_rgba(a_mod, 300), "now: model", font=SMALL)]
        rows.append(row)
    if rows:
        w = max(len(r) for r in rows)
        rows = [r + [Image.new("RGB", (300, 300), BG)] * (w - len(r)) for r in rows]
        save(grid(rows, 300), "prompts_before_after.webp")
    shot = S / "shots" / "studio_describe.png"
    if shot.exists():
        save(Image.open(shot), "studio_describe.webp", 80)


def tile_rgba(path, size):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, BG + (255,))
    bg.alpha_composite(im)
    return tile_im(bg.convert("RGB"), size)


def feet():
    """Aoi's spell cast from the side: after the refine, both shoes off the floor; now planted."""
    B, A = S / "recovery" / "angles_test3" / "left", S / "polish" / "feet_after" / "left"
    if not (B.exists() and A.exists()):
        return
    frames = ["f0000.png", "f0015.png", "f0030.png"]
    if not all((B / f).exists() and (A / f).exists() for f in frames):
        return
    def feet_crop(p):
        """The boots and the floor line, enlarged: the gap is centimetres."""
        im = Image.open(p).convert("RGB")
        w, h = im.size
        return im.crop((int(0.22 * w), int(0.72 * h), int(0.78 * w), h)).resize((280, 140), Image.LANCZOS)
    row = []
    for f, t in zip(frames, ("0 s", "1 s", "2 s")):
        row += [label(feet_crop(B / f), f"{t} before", font=SMALL), label(feet_crop(A / f), f"{t} now", font=SMALL)]
    out = Image.new("RGB", (len(row) * 286 - 6, 140), (255, 255, 255))
    for k, t in enumerate(row):
        out.paste(t, (k * 286, 0))
    save(out, "polish_feet.webp")


def angles():
    """The four moves from four sides at once (tools/make_angles.py --video), for the page."""
    import shutil
    CF = ROOT
    for n, mv in (("aoi", "spell_cast"), ("mara", "punch_combo"), ("mara", "roundhouse_kick"), ("pip", "victory_cheer")):
        src = CF / "work" / n / "qa" / f"angles_{mv}_video.mp4"
        if src.exists():
            shutil.copy(src, M / f"angles_{n}_{mv}.mp4")
            # a still for the page at rest (and for players that hold autoplay back): the middle frame
            png = M / f"angles_{n}_{mv}.png"
            dur = float(subprocess.run(["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries", "format=duration",
                                        "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout or 0)
            subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-ss", f"{dur * 0.45:.2f}", "-i", str(src),
                            "-frames:v", "1", str(png)], check=True)
            Image.open(png).convert("RGB").save(M / f"angles_{n}_{mv}.webp", quality=82)
            png.unlink()
            print(f"angles_{n}_{mv}.mp4", round(src.stat().st_size / 1e6, 2), "MB")


def compares():
    """The moves' side-by-side videos as the rigs play them now (the page's moves section)."""
    import shutil
    CF = ROOT
    for n, mv in (("aoi", "spell_cast"), ("mara", "punch_combo"), ("mara", "roundhouse_kick"), ("pip", "victory_cheer")):
        src = CF / "work" / n / "motion_videos" / f"{mv}_compare.mp4"
        if src.exists():
            shutil.copy(src, M / f"{n}_{mv}.mp4")
            print(f"{n}_{mv}.mp4", round(src.stat().st_size / 1e6, 2), "MB")


def cast():
    C = S / "polish" / "cast"
    for n in ("knight", "kaito", "bo"):
        for part in ("ref", "rest", "jump"):
            p = C / f"{n}_{part}.png"
            if p.exists():
                save(tile(p, 400), f"cast_{n}_{part}.webp", 85)


if __name__ == "__main__":
    M.mkdir(exist_ok=True)
    for f in (studio, wings, cheer, uv, aoi, hands, hem, bo, webs, loose, elbow, board, prompts, feet, angles, compares, cast):
        f()
