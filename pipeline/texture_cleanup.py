"""Remove skin that the generator painted onto clothing.

TRELLIS's multi-view pass decodes texture from several views at once, and where those views
disagree it sometimes fills the gap with the colour of the body underneath the garment. On Juno
the second pass - and only the second; the first was clean - put hand-sized patches of skin on
the outside and back of both trouser legs. The bake then faithfully carried them into the atlas.

The part labels say which texels are clothing, so the fix is local and conservative. A clothing
texel is a candidate when it is close to this character's own skin tone (measured on its body
texels) AND far from the fabric around it. Candidates are grouped into connected patches, and
a patch is only filled if it is an island in the fabric: the ring around it must be clothing,
with no body texel in it. That rule is what separates a painted-on patch from the edge of a
wrist or a neckline, where the labels and the texture disagree by a few texels - the first
version filled those too, and painted jacket onto every cuff. Filled patches take the colour of
the fabric around them, feathered in. If more than a few percent of the clothing would change,
the garment is probably skin-toned - a tan coat, a nude slip - and nothing is changed.

Run: python texture_cleanup.py --retopo retopo.glb --labels labels.json \
         --albedo baked_base_color.png --out albedo_clean.png [--mask mask.png]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--retopo", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--albedo", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mask", default=None, help="write the flagged texels here, for inspection")
ap.add_argument("--work-res", type=int, default=1024)
ap.add_argument("--skin-de", type=float, default=18.0, help="max colour distance to the skin tone")
ap.add_argument("--local-de", type=float, default=22.0, help="min colour distance to the fabric around")
ap.add_argument("--max-share", type=float, default=0.04,
                help="flag more than this share of the clothing and nothing is changed")
a = ap.parse_args()


def srgb_to_lab(rgb):
    """rgb in 0..1, any leading shape -> CIE Lab (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def box_mean(img, mask, r):
    """Mean of img over the masked texels within a (2r+1) square - summed-area tables."""
    def box(x):
        c = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0)) + ((0, 0),) * (x.ndim - 2)), 0), 1)
        H, W = x.shape[:2]
        y0 = np.clip(np.arange(H) - r, 0, H); y1 = np.clip(np.arange(H) + r + 1, 0, H)
        x0 = np.clip(np.arange(W) - r, 0, W); x1 = np.clip(np.arange(W) + r + 1, 0, W)
        return (c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0])
    m = mask.astype(np.float64)
    num = box(img * m[..., None]) if img.ndim == 3 else box(img * m)
    den = box(m)
    return num / np.maximum(den, 1e-9)[..., None] if img.ndim == 3 else num / np.maximum(den, 1e-9), den


def hair_pass(full):
    """Dark gaps in the hair. solidify.py closes the gaps between generated hair flakes, but the
    bake still reads each closed gap's colour from the flakes' shadowed insides, which left dark
    streaks through Vex's pink hair. A hair texel much darker than the hair around it takes the
    local hair colour, feathered in. Only hair texels move."""
    hair = labels_px == gid.get("hair", -99)
    if hair.sum() < 400:
        return full
    sm = np.asarray(Image.fromarray((full * 255).astype(np.uint8)).resize((S, S), Image.BILINEAR),
                    dtype=np.float64) / 255.0
    Lh = srgb_to_lab(sm)[..., 0]
    rr = max(3, S // 90)
    mean_L, _ = box_mean(Lh, hair, rr)
    dark = hair & (Lh < mean_L - 14)
    if not dark.any():
        return full
    col, den = box_mean(sm, hair & ~dark, rr * 2)
    ok = dark & (den > 3)
    fill = np.where(ok[..., None], col, sm)
    up_ = lambda x: np.asarray(Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).resize((R, R), Image.BILINEAR),
                               dtype=np.float64) / 255.0
    wgt = up_(ndimage.binary_dilation(ok, iterations=1).astype(np.float64))[..., None]
    fill_full = np.stack([up_(fill[..., c]) for c in range(3)], axis=-1)
    print(f"[texclean] hair: {int(ok.sum()):,} dark gap texels ({ok.sum() / hair.sum():.1%} of the hair) "
          "filled from the hair around them", flush=True)
    return full * (1 - wgt) + fill_full * wgt


def finish(full):
    full = hair_pass(full)
    Image.fromarray(np.clip(full * 255 + 0.5, 0, 255).astype(np.uint8)).save(a.out)
    raise SystemExit(0)


mesh = trimesh.load(a.retopo, force="mesh", process=False)
L = json.load(open(a.labels))
lab = np.array(L["labels"])
gid = L["group_ids"]
if len(lab) != len(mesh.vertices):
    raise SystemExit(f"[texclean] {len(lab)} labels for {len(mesh.vertices)} vertices")
uv = np.asarray(mesh.visual.uv)
faces = np.asarray(mesh.faces)

albedo = Image.open(a.albedo).convert("RGB")
R = albedo.size[0]
S = a.work_res
# per-face label by majority of its vertices, rasterised into the atlas at working resolution
face_lab = np.array([np.bincount(lab[f], minlength=4).argmax() for f in faces])
label_img = Image.new("L", (S, S), 255)
face_img = Image.new("I", (S, S), -1)                  # which face each texel belongs to
draw, fdraw = ImageDraw.Draw(label_img), ImageDraw.Draw(face_img)
pix = np.stack([uv[:, 0] * S, (1.0 - uv[:, 1]) * S], axis=1)     # trimesh uv is v-up; PNG rows run down
for fi, (f, l) in enumerate(zip(faces, face_lab)):
    poly = [tuple(pix[i]) for i in f]
    draw.polygon(poly, fill=int(l))
    fdraw.polygon(poly, fill=fi)
labels_px = np.asarray(label_img)
face_px = np.asarray(face_img)
cloth = labels_px == gid["clothing"]
body = labels_px == gid["body"]

small = np.asarray(albedo.resize((S, S), Image.BILINEAR), dtype=np.float64) / 255.0
L_ab = srgb_to_lab(small)
# this character's skin: body texels with a skin-like chroma (shoes and eyes are body too)
sk = body & (L_ab[..., 1] > 4) & (L_ab[..., 2] > 4) & (L_ab[..., 0] > 25) & (L_ab[..., 0] < 90)
if sk.sum() < 200:
    print("[texclean] too little skin to take a tone from - skin pass skipped", flush=True)
    finish(np.asarray(albedo, dtype=np.float64) / 255.0)
skin = np.median(L_ab[sk], axis=0)

r = max(4, S // 40)
local, _ = box_mean(L_ab, cloth, r)
d_skin = np.linalg.norm(L_ab - skin, axis=-1)
d_local = np.linalg.norm(L_ab - local, axis=-1)
flag = cloth & (d_skin < a.skin_de) & (d_local > a.local_de)
share = flag.sum() / max(cloth.sum(), 1)
print(f"[texclean] skin tone Lab ({skin[0]:.0f}, {skin[1]:.0f}, {skin[2]:.0f}); {int(flag.sum()):,} of "
      f"{int(cloth.sum()):,} clothing texels flagged ({share:.2%})", flush=True)
if share > a.max_share:
    print(f"[texclean] more than {a.max_share:.0%} of the clothing matches the skin - taking the "
          "garment to be skin-toned and leaving it alone", flush=True)
    finish(np.asarray(albedo, dtype=np.float64) / 255.0)
if not flag.any():
    finish(np.asarray(albedo, dtype=np.float64) / 255.0)

# A patch's own colour drags the fabric average around it, and its paler or darker parts are
# not skin-toned, so the skin-toned texels are only seeds: the fabric is re-estimated without
# them, and each seed then grows through every clothing texel that does not match that fabric.
seeds = flag
for _ in range(2):
    fabric, _ = box_mean(L_ab, cloth & ~ndimage.binary_dilation(seeds, iterations=3), 2 * r)
    unlike = cloth & (np.linalg.norm(L_ab - fabric, axis=-1) > 0.75 * a.local_de)
    comp_u, _ = ndimage.label(unlike, structure=np.ones((3, 3)))
    hit = np.unique(comp_u[seeds & unlike])
    seeds = np.isin(comp_u, hit[hit > 0])

# Keep only patches that are islands in the fabric - asked on the mesh, not in the atlas. A UV
# seam can put a cuff's edge next to empty atlas while the hand it meets is millimetres away on
# the body, and a painted patch can sit at the edge of its island. So: where are the patch's
# faces in 3D, and how far is the nearest body vertex? Next to skin, it is a label edge.
body_tree = cKDTree(np.asarray(mesh.vertices)[lab == gid["body"]])
reach = 0.025 * float(np.ptp(np.asarray(mesh.vertices), axis=0).max())   # ~4 cm on a person
centres = np.asarray(mesh.triangles_center)
comp, n = ndimage.label(seeds, structure=np.ones((3, 3)))
keep = np.zeros_like(flag)
kept = 0
why = {}
for i, sl in enumerate(ndimage.find_objects(comp), start=1):
    if sl is None:
        continue
    c = comp[sl] == i
    if c.sum() < 6:                                # specks are texture, not a patch
        continue
    # a painted skin patch is skin-coloured on the whole, even where its edges are paler; a
    # zipper or piping that grew from a few beige texels is not - growth alone turned Vex's
    # zipper line teal. Measured: Juno's real patches average 13.7-14.6 from her skin tone
    # (one only 30% seeds), Vex's false ones 24-56. The seed threshold applies to the mean.
    if os.environ.get("TEXCLEAN_DEBUG"):
        hh, ww = c.shape
        print(f"[texclean-debug] patch at ({sl[1].start},{sl[0].start}) {hh}x{ww} area {int(c.sum())} "
              f"seed {flag[sl][c].mean():.2f} dSkin {d_skin[sl][c].mean():.1f} thick {c.sum()/max(hh,ww):.1f} "
              f"fill {c.sum()/(hh*ww):.2f}", flush=True)
    if d_skin[sl][c].mean() > a.skin_de:
        why["not skin-coloured"] = why.get("not skin-coloured", 0) + 1
        continue
    fs = np.unique(face_px[sl][c])
    fs = fs[fs >= 0]
    if len(fs) == 0:
        continue
    d = float(body_tree.query(centres[fs])[0].min())
    if d < reach:
        why["next to skin"] = why.get("next to skin", 0) + 1
        continue
    keep[sl] |= c
    kept += 1
print(f"[texclean] {n} candidate patches, {kept} are islands in the fabric"
      + (f" (rejected: {why})" if why else ""), flush=True)
if a.mask:
    Image.fromarray((seeds * 120 + keep * 135).astype(np.uint8)).save(a.mask.replace(".png", "_candidates.png"))
if not keep.any():
    finish(np.asarray(albedo, dtype=np.float64) / 255.0)

# grow the patches a little so the fill covers their soft edges
grow = ndimage.binary_dilation(keep, iterations=2) & cloth
# fabric colour from the unflagged clothing around each flagged texel, widening until found
fill = np.zeros_like(small)
todo = grow.copy()
for rr in (r, 2 * r, 4 * r, 8 * r):
    m, den = box_mean(small, cloth & ~grow, rr)
    ok = todo & (den > 4)
    fill[ok] = m[ok]
    todo &= ~ok
    if not todo.any():
        break
grow &= ~todo                                     # nothing to fill from: leave as baked

# apply at full resolution, feathered
up = lambda x: np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize((R, R), Image.BILINEAR),
                          dtype=np.float64) / 255.0
w = up(grow.astype(np.float64))[..., None]
full = np.asarray(albedo, dtype=np.float64) / 255.0
fill_full = np.stack([up(fill[..., c]) for c in range(3)], axis=-1)
out = full * (1 - w) + fill_full * w
if a.mask:
    Image.fromarray((grow * 255).astype(np.uint8)).save(a.mask)
print(f"[texclean] filled {int(grow.sum()):,} texels at {S} ({grow.sum() / max(cloth.sum(), 1):.2%} of "
      f"the clothing) from the fabric around them -> {a.out}", flush=True)
finish(out)
