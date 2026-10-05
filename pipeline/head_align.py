"""Put a head generated on its own onto the generated body: the same place, size and turn (E139).

    python pipeline/head_align.py --work work/<name> --head head_hy.glb --crop head_rgba.png --box x0,y0,x1,y1 \
        --out head_aligned.glb [--debug overlay.png]

A head generated from a close crop of the reference (Hunyuan3D 2.1's shape model, tools/hunyuan_shape.py) has a
sculpted face where the whole-figure model's is a few rough facets; it lives in the generator's own frame. Both
models were made from the same picture, so the picture ties them together:

1. front, across and up: the head's front silhouette is fitted to the crop's cut-out (a similarity on the masks),
   the crop's box is a known window of the reference, and the body's front render was fitted to the reference by
   the generate stage's check (gate/<tag>/meta.json: its normalisation and orthographic camera). Composed, they
   carry the head's x and z into the body's frame, scale included.
2. depth: the head's front surface is moved to the body's head's, over the face region (the median of the
   front-most points of each, row by row of the face).
3. a few rounds of closest points over the head region (the worst quarter left out) settle its depth; across and up
   it stays where the picture puts it (--icp full also fits the turn, to the old head - which moved the face off the
   picture's by enough to paint a second eye on the cheek).

Frames: GLBs are read with trimesh (glTF, Y up) and worked in Blender's (Z up, the figure facing -Y), as the
renders were; the result is written back as a GLB in the body's own coordinates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import image_mask, similarity  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True, help="the character's work dir: reference.png, reference_mask.png, gate/, the mesh")
ap.add_argument("--body", default="pass1.glb", help="the body mesh in --work whose gate render is used")
ap.add_argument("--gate", default=None, help="gate/<tag> holding the body's front render (default: by --body's stem)")
ap.add_argument("--head", required=True)
ap.add_argument("--crop", required=True, help="the RGBA crop the head was generated from")
ap.add_argument("--box", required=True, help="the crop's window of the reference, x0,y0,x1,y1 (px)")
ap.add_argument("--out", required=True)
ap.add_argument("--debug", default=None)
ap.add_argument("--json", default=None, help="write the fit's measures here")
ap.add_argument("--icp", choices=("depth", "full"), default="depth",
                help="depth (default): the closest-point fit moves the head in depth only - across and up it stays where the "
                     "picture puts it, which the texture stage projects the picture by; full: a rigid fit to the old head")
a = ap.parse_args()
W = Path(a.work)


def to_blender(V):
    return np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)


def to_gltf(V):
    return np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1)


def load(p):
    m = trimesh.load(p, force="mesh", process=False)
    return m, to_blender(np.asarray(m.vertices, float))


def raster(P2, res, lo, hi):
    """A silhouette from projected points (x right, z up) on a res x res grid spanning lo..hi."""
    u = ((P2[:, 0] - lo[0]) / (hi[0] - lo[0]) * (res - 1)).round().astype(int)
    v = ((hi[1] - P2[:, 1]) / (hi[1] - lo[1]) * (res - 1)).round().astype(int)
    ok = (u >= 0) & (u < res) & (v >= 0) & (v < res)
    m = np.zeros((res, res), bool)
    m[v[ok], u[ok]] = True
    m = ndimage.binary_closing(ndimage.binary_dilation(m, iterations=2), iterations=3)
    return ndimage.binary_fill_holes(m)


# ---- the body: its gate render and that render's fit to the reference ----------------------------------------
body_m, VB = load(W / a.body)
gate = Path(a.gate) if a.gate else W / "gate" / Path(a.body).stem
meta = json.load(open(gate / "meta.json"))
vw = meta["views"][0]
nrm = meta["normalize"]
c_n, s_n, R_px, ortho = np.array(nrm["center"]), float(nrm["scale"]), int(vw["res"]), float(vw["ortho_scale"])
ref = np.asarray(Image.open(W / "reference.png").convert("RGB"))
mask_p = W / "reference_mask.png"
imask = (np.asarray(Image.open(mask_p).convert("L").resize(ref.shape[1::-1])) > 127) if mask_p.exists() else image_mask(ref)
imask = ndimage.binary_fill_holes(imask)
rmask = np.asarray(Image.open(gate / "view_00.png").convert("RGBA"))[..., 3] > 127
(s_g, tx_g, ty_g), iou_g = similarity(rmask, imask)              # reference px = render px * s_g + t_g


def body_px_to_xz(px):
    """reference pixels -> the body's x, z (Blender frame)."""
    rp = (np.asarray(px, float) - [tx_g, ty_g]) / s_g                 # render px
    xn = (rp[..., 0] / R_px - 0.5) * ortho
    zn = (0.5 - rp[..., 1] / R_px) * ortho
    return np.stack([xn / s_n + c_n[0], zn / s_n + c_n[2]], -1)


# ---- the head: its silhouette against the crop's cut-out ----------------------------------------------------
head_m, VH = load(a.head)
crop = Image.open(a.crop).convert("RGBA")
cmask = np.asarray(crop.getchannel("A")) > 127
res = 512
cmask_r = np.asarray(Image.fromarray(cmask.astype(np.uint8) * 255).resize((res, res))) > 127
lo, hi = VH[:, [0, 2]].min(0), VH[:, [0, 2]].max(0)
side = float((hi - lo).max()) * 1.1
mid = (lo + hi) / 2
lo_, hi_ = mid - side / 2, mid + side / 2
hmask = raster(VH[:, [0, 2]], res, lo_, hi_)
(s_h, tx_h, ty_h), iou_h = similarity(hmask, cmask_r)             # crop px (at res) = head px * s_h + t_h
x0, y0, x1, y1 = [float(v) for v in a.box.split(",")]
k_crop = (x1 - x0) / res                                           # crop px (at res) -> reference px


def head_xz_to_body_xz(XZ):
    u = (XZ[:, 0] - lo_[0]) / (hi_[0] - lo_[0]) * (res - 1)
    v = (hi_[1] - XZ[:, 1]) / (hi_[1] - lo_[1]) * (res - 1)
    cu, cv = u * s_h + tx_h, v * s_h + ty_h                         # crop px at res
    return body_px_to_xz(np.stack([x0 + cu * k_crop, y0 + cv * k_crop], -1))


# the x, z map is a similarity: fit it on the head's own points
src = VH[:, [0, 2]]
dst = head_xz_to_body_xz(src)
sc = float(np.sqrt(((dst - dst.mean(0)) ** 2).sum(1).mean() / ((src - src.mean(0)) ** 2).sum(1).mean()))
VA = VH * sc
VA[:, [0, 2]] += (dst.mean(0) - (src * sc).mean(0))

# ---- the body's head region: what lies inside the crop's window, above where its cut-out fades --------------
alpha = np.asarray(crop.getchannel("A"), np.float32) / 255
rows = np.nonzero(alpha.max(1) > 0.5)[0]
fade_row = rows.max() / alpha.shape[0] if len(rows) else 0.86   # the lowest fully kept row of the crop, 0..1
win = body_px_to_xz(np.array([[x0, y0], [x1, y0 + fade_row * (y1 - y0)]]))
bx = (VB[:, 0] > min(win[:, 0])) & (VB[:, 0] < max(win[:, 0]))
bz = (VB[:, 2] < max(win[:, 1])) & (VB[:, 2] > min(win[:, 1]))
region_b = bx & bz
hz_cut = np.quantile(VA[:, 2], 0.12)                                # the head's own faded bottom left out
region_h = VA[:, 2] > hz_cut

# ---- depth: front surfaces matched, row by row -------------------------------------------------------------
def front_profile(P, zs):
    out = []
    for z0, z1 in zip(zs[:-1], zs[1:]):
        s_ = P[(P[:, 2] >= z0) & (P[:, 2] < z1)]
        out.append(np.quantile(s_[:, 1], 0.03) if len(s_) > 20 else np.nan)
    return np.array(out)


PB, PH = VB[region_b], VA[region_h]
ztop = min(PB[:, 2].max(), PH[:, 2].max())
zbot = max(PB[:, 2].min(), PH[:, 2].min())
zs = np.linspace(zbot + 0.2 * (ztop - zbot), ztop - 0.15 * (ztop - zbot), 12)   # the face's band
d = front_profile(PB, zs) - front_profile(PH, zs)
VA[:, 1] += float(np.nanmedian(d))

# ---- ICP over the head region --------------------------------------------------------------------------------
tree_b = cKDTree(VB[region_b])
rng = np.random.default_rng(0)
idx_h = np.flatnonzero(VA[:, 2] > hz_cut)
pick = rng.choice(idx_h, min(40_000, len(idx_h)), replace=False)
R_tot, t_tot = np.eye(3), np.zeros(3)
for it in range(12):
    P = VA[pick] @ R_tot.T + t_tot
    dd, j = tree_b.query(P)
    keep = dd < np.quantile(dd, 0.75)
    A_, B_ = P[keep], VB[region_b][j[keep]]
    ca, cb = A_.mean(0), B_.mean(0)
    H = (A_ - ca).T @ (B_ - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Ri = Vt.T @ D @ U.T
    if a.icp == "depth":
        Ri = np.eye(3)
        R_tot, t_tot = R_tot, t_tot + np.array([0.0, float(np.median(B_[:, 1] - A_[:, 1])), 0.0])
        continue
    R_tot, t_tot = Ri @ R_tot, Ri @ t_tot + (cb - Ri @ ca)
final = VA @ R_tot.T + t_tot
dd, _ = tree_b.query(final[pick])
ang = np.degrees(np.arccos(np.clip((np.trace(R_tot) - 1) / 2, -1, 1)))
H_body = float(VB[:, 2].max() - VB[:, 2].min())
print(f"[head] silhouettes: body on the reference IoU {iou_g:.3f}, head on its crop IoU {iou_h:.3f}; scale {sc:.4f}; "
      f"ICP turned {ang:.2f} deg; head to body surface median {np.median(dd) / H_body * 1750:.1f} mm, "
      f"p90 {np.quantile(dd, 0.9) / H_body * 1750:.1f} mm (at 1.75 m)", flush=True)
if a.json:
    json.dump({"body_iou": round(float(iou_g), 3), "head_iou": round(float(iou_h), 3), "scale": round(sc, 5),
               "icp_turn_deg": round(float(ang), 2), "surface_median_mm": round(float(np.median(dd) / H_body * 1750), 1),
               "surface_p90_mm": round(float(np.quantile(dd, 0.9) / H_body * 1750), 1)}, open(a.json, "w"), indent=1)
out = trimesh.Trimesh(to_gltf(final), np.asarray(head_m.faces), process=False)
out.export(a.out)
print(f"[head] -> {a.out}", flush=True)

if a.debug:
    from PIL import ImageDraw
    img = Image.new("RGB", (1000, 500), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    zlo, zhi = min(PB[:, 2].min(), final[:, 2].min()), max(PB[:, 2].max(), final[:, 2].max())
    span = zhi - zlo
    for k, (ax, label) in enumerate(((0, "front: x across, z up"), (1, "side: y across, z up"))):
        c0 = (VB[region_b][:, ax].mean())
        for P, col, step in ((VB[region_b], (40, 90, 200), 3), (final, (220, 60, 40), 25)):
            u = ((P[::step, ax] - c0) / span * 0.9 + 0.5) * 480 + k * 500 + 10
            v = (1 - ((P[::step, 2] - zlo) / span * 0.9 + 0.05)) * 480 + 10
            for x_, y_ in zip(u, v):
                dr.point((x_, y_), fill=col)
        dr.text((k * 500 + 14, 4), label + "  (blue body, red head)", fill=(0, 0, 0))
    img.save(a.debug)
