"""The generator's own colours without their crackle: the base colour texture of mesh.glb, rebuilt through the mesh,
for retopo.py to bake from.

    python pipeline/despeck_source.py --mesh mesh.glb --out source_albedo.png

TRELLIS.2 bakes its colour into an atlas of thousands of small charts - at 1024 px for a million faces, two texels in
three lie on a chart's edge, and most triangles are smaller than a texel. Along the charts' edges it leaves texels
sampled off the surface, grey or dark, and baked onto the low-poly they were a crackle of short lines over every
surface the picture does not paint: the sides of the face, the ears, the hair, the armour (E140). They are not dark
specks a 2D filter can find - grey on skin, light on dark cloth - and they are not holes in the mesh (a plain white
render of it has none); nor can a texel be mended from the texels round it in the atlas, which belong to other charts
(tried: tiny triangles took a neighbouring chart's colour and became grey blobs).

So the colour is rebuilt on the mesh. Each triangle is sampled at its centroid; the copies of a vertex the atlas split
at its seams are joined by position; each vertex takes the median of its triangles' colours (a mean let one triangle
sampled off the surface darken it), and a vertex far from the median of its neighbours along the mesh (more than
ESCAPE in CIELAB) takes that median, twice over. The atlas is then painted again from these vertex colours at four times the
resolution (so that nearly every triangle holds a texel's centre), each texel from the triangle its centre lies in, and
every texel no triangle's interior reaches takes the nearest painted one. The
alpha channel is kept.
"""
import argparse

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

ESCAPE = 12.0

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def srgb_to_lab(x):
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = x @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


m = trimesh.load(a.mesh, force="mesh", process=False)
tex = getattr(m.visual.material, "baseColorTexture", None)
if tex is None:
    raise SystemExit("[despeck] the mesh carries no base colour texture")
rgba = np.asarray(tex.convert("RGBA")).astype(np.float32)
R = rgba.shape[0]
F = np.asarray(m.faces)
uv = np.asarray(m.visual.uv, np.float64)
P = np.stack([uv[:, 0] * R, (1 - uv[:, 1]) * R], 1)                # texel units, texel (i, j)'s centre at i+.5, j+.5
T = P[F]


def sample(q):
    """Bilinear texture lookup at texel coordinates q (N, 2)."""
    return np.stack([ndimage.map_coordinates(rgba[..., c], [q[:, 1] - 0.5, q[:, 0] - 0.5], order=1, mode="nearest")
                     for c in range(3)], 1)


cen = T.mean(1)
tri_col = sample(cen)
# the vertices the atlas split at its seams, joined by position
V = np.asarray(m.vertices)
key = np.round(V / (np.ptp(V, 0).max() * 1e-6)).astype(np.int64)
_, uid = np.unique(key, axis=0, return_inverse=True)
uid = uid.ravel()
U = uid.max() + 1
Fu = uid[F]
# each vertex: the median of its triangles' colours - a triangle sampled off the surface is outvoted
vf = np.concatenate([np.stack([Fu[:, k], np.arange(len(F))], 1) for k in range(3)])
vf = vf[np.argsort(vf[:, 0], kind="stable")]
vstart = np.searchsorted(vf[:, 0], np.arange(U))
vcount = np.bincount(vf[:, 0], minlength=U)
KV = 10
vi = vstart[:, None] + np.arange(KV)[None]
vvalid = np.arange(KV)[None] < np.minimum(vcount, KV)[:, None]
vtri = vf[np.minimum(vi, len(vf) - 1), 1]
vcol = np.empty((U, 3))
for c in range(3):
    vals = np.where(vvalid, tri_col[vtri, c], np.nan)
    vcol[:, c] = np.nanmedian(vals, 1)
# neighbours along the mesh, and the outliers among them
E_ = np.concatenate([Fu[:, [0, 1]], Fu[:, [1, 2]], Fu[:, [2, 0]]])
E_ = np.unique(np.sort(E_, 1), axis=0)
E_ = E_[E_[:, 0] != E_[:, 1]]
nb = np.concatenate([E_, E_[:, ::-1]])
order = np.argsort(nb[:, 0], kind="stable")
nb = nb[order]
starts = np.searchsorted(nb[:, 0], np.arange(U))
counts = np.bincount(nb[:, 0], minlength=U)
fixed = 0
for _ in range(2):
    lab = srgb_to_lab(np.clip(vcol / 255.0, 0, 1))
    # the median of the neighbours' colours, channel by channel, over at most 12 neighbours a vertex
    K = 12
    idx = starts[:, None] + np.arange(K)[None]
    valid = np.arange(K)[None] < np.minimum(counts, K)[:, None]
    nbr = nb[np.minimum(idx, len(nb) - 1), 1]
    med = np.empty_like(vcol)
    for c in range(3):
        vals = np.where(valid, vcol[nbr, c], np.nan)
        med[:, c] = np.nanmedian(np.where(valid.any(1, keepdims=True), vals, vcol[:, c:c + 1]), 1)
    out = np.linalg.norm(lab - srgb_to_lab(np.clip(med / 255.0, 0, 1)), axis=1) > ESCAPE
    out &= counts > 2
    vcol[out] = med[out]
    fixed += int(out.sum())
# the atlas painted again, at UP times its resolution so that nearly every triangle holds a texel's centre: each texel
# from the triangle its centre lies in (at 1024 px most triangles held none and took a neighbouring chart's colour)
UP = 4
RO = R * UP
To = T * UP
img = np.zeros((RO, RO, 3), np.float32)
painted = np.zeros((RO, RO), bool)
a_, b_, c_ = To[:, 0], To[:, 1], To[:, 2]
den = (b_[:, 1] - c_[:, 1]) * (a_[:, 0] - c_[:, 0]) + (c_[:, 0] - b_[:, 0]) * (a_[:, 1] - c_[:, 1])
ok = np.abs(den) > 1e-12
lo = np.floor(To.min(1) - 0.5).astype(int)
hi = np.floor(To.max(1) - 0.5).astype(int)
span = (hi - lo + 1).max(1)
cols = vcol[Fu]                                                       # (F, 3 corners, 3)
prev = 0
for cap in (2, 4, 8, 16, 32, 64, 128):
    grp = np.nonzero(ok & (span > prev) & (span <= cap))[0]
    prev = cap
    if not len(grp):
        continue
    ga, gb, gc, gd, gl, gh = a_[grp], b_[grp], c_[grp], den[grp], lo[grp], hi[grp]
    for dx in range(cap):
        for dy in range(cap):
            x, y = gl[:, 0] + dx, gl[:, 1] + dy
            sel = (x <= gh[:, 0]) & (y <= gh[:, 1]) & (x >= 0) & (y >= 0) & (x < RO) & (y < RO)
            if not sel.any():
                continue
            px, py = x + 0.5, y + 0.5
            l1 = ((gb[:, 1] - gc[:, 1]) * (px - gc[:, 0]) + (gc[:, 0] - gb[:, 0]) * (py - gc[:, 1])) / gd
            l2 = ((gc[:, 1] - ga[:, 1]) * (px - gc[:, 0]) + (ga[:, 0] - gc[:, 0]) * (py - gc[:, 1])) / gd
            l3 = 1 - l1 - l2
            ins = sel & (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
            if not ins.any():
                continue
            cc = cols[grp[ins]]
            img[y[ins], x[ins]] = cc[:, 0] * l1[ins, None] + cc[:, 1] * l2[ins, None] + cc[:, 2] * l3[ins, None]
            painted[y[ins], x[ins]] = True
near = ndimage.distance_transform_edt(~painted, return_distances=False, return_indices=True)
img = img[near[0], near[1]]
alpha = np.asarray(Image.fromarray(rgba[..., 3].astype(np.uint8)).resize((RO, RO), Image.BILINEAR), np.float32)
img = np.concatenate([img, alpha[..., None]], -1)
Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8)).save(a.out)
print(f"[despeck] the generator's {R} px colour rebuilt through its {U:,} vertices ({fixed:,} outliers taken to their "
      f"neighbours'), painted again at {RO} px ({painted.mean():.1%} from triangles) -> {a.out}", flush=True)
