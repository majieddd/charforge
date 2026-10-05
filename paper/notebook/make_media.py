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
                     ("studio_angles.png", "studio_angles.webp"), ("studio_words.png", "studio_words.webp")):
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


def e130_floor():
    """Cadet's boots against the floor, the same frames before and after E130: the aberration audit's renders, every
    face more than 1 cm under the floor amber. Before: polish/cadet/floor_render (the rigid sole, 30 September);
    after: polish/e130/floor_after (aberration_audit.py --render-at walk:floor:41,crouch_walk:floor:59)."""
    B, A = S / "polish" / "cadet" / "floor_render", S / "polish" / "e130" / "floor_after"
    shots = [("walk_floor_f41", 30), ("crouch_walk_floor_f59", 30), ("walk_floor_f41", 210), ("crouch_walk_floor_f59", 210)]
    if not all((d / f"{n}_{az}.png").exists() for d in (B, A) for n, az in shots):
        return
    size, gap, head = 360, 6, 34
    out = Image.new("RGB", (len(shots) * size + (len(shots) - 1) * gap, 2 * (size + head)), (255, 255, 255))
    for row, (d, title) in enumerate(((B, "Before: a rigid sole on the ankle bone"), (A, "After: the shoe posed as the viewer sees it"))):
        y = row * (size + head)
        ImageDraw.Draw(out).text((4, y + 7), title, fill=(20, 25, 32), font=FONT)
        for i, (n, az) in enumerate(shots):
            im = Image.open(d / f"{n}_{az}.png").convert("RGB").resize((size, size), Image.LANCZOS)
            if row == 0:
                clip = n.split("_floor")[0].replace("_", " ")
                label(im, f"{clip} · {'front' if az == 30 else 'back'}", font=SMALL)
            out.paste(im, (i * (size + gap), y + head))
    save(out, "e130_cadet_floor.webp", 85)


def cast():
    C = S / "polish" / "cast"
    for n in ("knight", "kaito", "bo", "cadet"):
        for part in ("ref", "rest", "jump"):
            p = C / f"{n}_{part}.png"
            if p.exists():
                save(tile(p, 400), f"cast_{n}_{part}.webp", 85)


# ---- v0.12: toward a professionally made model (E132-E139) ---------------------------------------------------------
Q = S / "polish"


def rows_figure(rows, name, size=330, head=34, q=85):
    """rows: [(title, [(label, path), ...]), ...]; every path must exist, or nothing is written."""
    if not all(Path(pth).exists() for _, cells in rows for _, pth in cells):
        missing = [str(pth) for _, cells in rows for _, pth in cells if not Path(pth).exists()]
        print(f"{name}: skipped, missing {missing[:2]}")
        return
    ncol = max(len(c) for _, c in rows)
    gap = 6
    out = Image.new("RGB", (ncol * size + (ncol - 1) * gap, len(rows) * (size + head)), (255, 255, 255))
    for r, (title, cells) in enumerate(rows):
        y = r * (size + head)
        ImageDraw.Draw(out).text((4, y + 7), title, fill=(20, 25, 32), font=FONT)
        for c, (lab, pth) in enumerate(cells):
            im = Image.open(pth).convert("RGB").resize((size, size), Image.LANCZOS)
            if lab:
                label(im, lab, font=SMALL)
            out.paste(im, (c * (size + gap), y + head))
    save(out, name, q)


def q_rake():
    """The shipped low-poly with its normal map under a raking light, v0.11 against the v0.12 rebuild (E135)."""
    R = Q / "v012_rebuild" / "renders"
    rows = [(f"{n.capitalize()}: v0.11 (200k faces) and v0.12 (1M faces, filtered surface)",
             [("v0.11 front", R / f"{n}_before_rake_000.png"), ("v0.12 front", R / f"{n}_after_rake_000.png"),
              ("v0.11 chest", R / f"{n}_before_chest_025.png"), ("v0.12 chest", R / f"{n}_after_chest_025.png")])
            for n in ("knight2", "cadet", "gray")]
    rows_figure(rows, "q_rake.webp")


def q_rest():
    """Cadet's shoulder with the T baked into the rest pose against the modelled pose kept (E137)."""
    R = Q / "e137" / "run"
    rows = [(f"{lab}: rest, idle, walk",
             [(None, R / f"cpu_{b}_left_shoulder_{c}.png") for c in ("rest", "idle_030", "walk_025")])
            for b, lab in (("cadet", "Baked to a T (v0.11)"), ("e137_cadet", "Kept as modelled (v0.12)"))]
    rows_figure(rows, "q_rest.webp")


def q_hands():
    """Modelled hands shaded from the generated mesh's colour bake, and from their own occlusion (E138)."""
    R = Q / "inventory" / "handfix"
    rows_figure([("Cadet's hand: shaded from the generated mesh's bake, then from its own occlusion",
                  [("before", R / "before_top_rest.png"), ("after", R / "after_top_rest.png"),
                   ("before", R / "before_idle_idle_030.png"), ("after", R / "after_idle_idle_030.png")])], "q_hands.webp")


def q_inventory():
    """What close-ups of the v0.11 builds showed first (E138)."""
    I = Q / "inventory"
    rows_figure([("Close-ups of v0.11: blotched faces, torn armpits, a bell at the wrist, two-tone hands",
                  [("Mara", I / "mara_v011" / "face34_rest.png"), ("Boyscout", I / "boyscout_v011" / "face34_rest.png"),
                   ("Bo's back", I / "bo_v011" / "back_rest.png"), ("Pip's wrist", I / "pip_v011" / "lhand_top_rest.png"),
                   ("Knight2's back", I / "knight2_v011" / "back_rest.png")])], "q_inventory.webp", size=300)


def q_views():
    """Hunyuan3D-Paint's sides and backs on the shipped meshes (E134): better on Pip, worse by eye on six of eight."""
    T = Q / "e134" / "trial"
    rows_figure([("Shipped against Hunyuan3D-Paint 2.1's views (unlit albedo)",
                  [("Boyscout shipped", T / "boyscout_shipped_180.png"), ("Boyscout views", T / "boyscout_trial_180.png"),
                   ("Bo shipped", T / "bo_shipped_090.png"), ("Bo views", T / "bo_trial_090.png"),
                   ("Knight2 shipped", T / "knight2_shipped_180.png"), ("Knight2 views", T / "knight2_trial_180.png")])],
                "q_views.webp", size=280)


def q_head():
    """The head from the whole figure against one made on its own (E139), shape under raking light."""
    E = Q / "e139"
    rows_figure([("Boyscout's head: from the whole figure (TRELLIS.2), and made on its own from a crop (Hunyuan3D 2.1)",
                  [("whole figure", E / "rake_trellis_head_000.png"), ("whole figure", E / "rake_trellis_head_035.png"),
                   ("on its own", E / "rake_hyhead_000.png"), ("on its own", E / "rake_hyhead_035.png")])], "q_head.webp")


def q_debris():
    """Knight2's back under a raking light, v0.11 against v0.12: the shards and the noise gone (E135, E138)."""
    R = Q / "v012_rebuild" / "renders"
    rows_figure([("Knight2's back: v0.11 (95 loose pieces) and v0.12 (one)",
                  [("v0.11", R / "knight2_before_rake_180.png"), ("v0.12", R / "knight2_after_rake_180.png")])],
                "q_debris.webp", size=420)


def q_skin():
    """One skin tone at the 7 mm scale (E138): the picture's shading out of the skin, its detail kept."""
    K = Q / "inventory" / "skin"
    rows_figure([("Faces as shipped and with one skin tone (three-quarter, lit)",
                  [(f"{n} before", K / f"{n}_orig_face_rest.png") for n in ("mara", "boyscout")]
                  + [(f"{n} after", K / f"{n}_fine_face_rest.png") for n in ("mara", "boyscout")])], "q_skin.webp")


def q_eyes():
    """Mara's eyes in grey at every stage, and rebuilt as lids over a ball (E140)."""
    F = Q / "faces" / "e140" / "fig"
    rows_figure([("Mara's eyes, the shape alone: generated, solid, v0.12's low-poly; rebuilt in the solid and decimated",
                  [("generated", F / "eye_1_generated.png"), ("solid", F / "eye_2_solid.png"),
                   ("v0.12 low-poly", F / "eye_3_v012.png"), ("rebuilt solid", F / "eye_4_rebuilt_solid.png"),
                   ("v0.13 low-poly", F / "eye_5_rebuilt_low.png")])], "q_eyes.webp", size=300)


def q_faceparts():
    """Mara's face, one part at a time (E140): the picture, v0.12, the eyes rebuilt (face flow), the face laid on in
    parts, and with the face left to the front view and out of the skin tone."""
    F = Q / "faces" / "e140" / "fig"
    rows_figure([("Mara's face, lit: the picture; v0.12; eyes rebuilt; laid on in parts; the face the picture's alone",
                  [("picture", F / "face_0_picture.png"), ("v0.12", F / "face_1_v012.png"),
                   ("eyes rebuilt, face flow", F / "face_2_flow.png"), ("in parts", F / "face_3_parts.png"),
                   ("v0.13", F / "face_4_final.png")])], "q_faceparts.webp", size=300)


def q_crackle():
    """The generator's own colours on Knight's head: as generated, mended in the atlas, rebuilt through the mesh (E140)."""
    src = Q / "faces" / "e140" / "fig" / "crackle_knight.png"
    if src.exists():
        im = Image.open(src).convert("RGB")
        for i, lab in enumerate(("as generated", "mended in the atlas", "rebuilt through the mesh")):
            label(im, lab, xy=(10 + i * im.width // 3, 8), font=SMALL)
        save(im, "q_crackle.webp")


def q_faces_bench():
    """The benchmark set's faces before and after v0.13 (E140): front, soft light, from tools/face_review.sh."""
    V = Q / "v013"
    rows = []
    for row in (("mara", "juno3", "rowan", "knight"), ("vex", "rivet", "pip", "cadet"), ("aoi", "hana", "kaito", "ren")):
        cells = []
        for n in row:
            b = V / "faces_before" / "face_review_parts" / n / "az0_rest.png"
            a = V / "faces" / "face_review_parts" / n / "az0_rest.png"
            name = {"juno3": "Juno"}.get(n, n.capitalize())
            if a.exists():
                if b.exists():
                    cells += [(f"{name} v0.12", b), (f"{name} v0.13", a)]
                else:                            # new in this round: no v0.12 build, its picture instead
                    cells += [(f"{name}: picture", ROOT / "work" / n / "texproj" / "face_src.png"), (f"{name} v0.13", a)]
        if cells:
            rows.append(("", cells))
    rows_figure(rows, "q_faces_bench.webp", size=220, head=4)


if __name__ == "__main__":
    M.mkdir(exist_ok=True)
    for f in (studio, wings, cheer, uv, aoi, hands, hem, bo, webs, loose, elbow, board, prompts, feet, angles, compares, cast,
              e130_floor, q_rake, q_rest, q_hands, q_inventory, q_views, q_head, q_skin, q_debris, q_eyes, q_faceparts,
              q_faces_bench, q_crackle):
        f()
