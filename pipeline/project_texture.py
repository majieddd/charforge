"""Sharpen the baked texture by projecting the source images back onto the mesh.

The generator's texture is its own reconstruction of the reference image, decoded at a
resolution where a face is a few dozen texels: Juno's eyes came out as grey smudges and her
skin carried dark "cracks" where its views were fused. The images it was made from are sharp -
the reference (front) and the multi-view pass's enhanced side and back views - so every texel a
view camera sees squarely takes its colour from that view.

Per view (blender/uv_maps.py rendered the mesh from each camera and baked every texel's world
position and normal):

1. Align. The reference's camera is unknown and the side/back views were made from the first
   pass's mesh, so the mesh's own render is aligned to each image: a similarity transform that
   maximises silhouette overlap, then dense optical flow between the mesh's render (its baked
   albedo) and the image for what remains - perspective, and small differences in shape.
2. Project. Each texel goes through the view's camera to a pixel of the mesh render, through the
   alignment to a pixel of the image. It is used only if the render shows that same surface point
   there (so arms do not paint onto the torso behind them), and weighted by how squarely it
   faces the camera and how far it is from the silhouette's edge.
3. Blend. Views are averaged by weight with the baked texture underneath at a low weight, so
   texels no camera sees keep the bake and seams between views fade over the grazing angles.
   Each view is first colour-matched to the bake over the texels both are confident about.

The modelled hands are left out - the source images show the generator's hands, not these - and
take the character's median skin tone from the bake.

Run: python project_texture.py --dir texproj --base baked_base_color.png --out albedo_proj.png \
         --view 000:reference.png --view 090:mv/side.png --view 180:mv/back.png [--hands hands_spec.json]
"""
from __future__ import annotations

import argparse
import json
import os

import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import image_mask, similarity  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="uv_maps.py output directory")
ap.add_argument("--base", required=True, help="the baked albedo")
ap.add_argument("--out", required=True)
ap.add_argument("--view", action="append", default=[],
                help="TAG:image[:mask[:weight]] - a source image seen by view TAG (an azimuth, or 'face' "
                     "for uv_maps.py's close-up), its foreground mask (pipeline/foreground.py) when its "
                     "background is not a plain field, and how much it counts against the others (1)")
ap.add_argument("--min-iou", type=float, default=0.75,
                help="a view whose silhouette overlaps the mesh's less than this is not used")
ap.add_argument("--hands", default=None, help="hands_spec.json - modelled hands, excluded")
ap.add_argument("--hands-solid", default=None,
                help="solid_cut.npz - the solid the modelled hands were joined to: a texel near a wrist is the "
                     "hand's only if it stands outside that solid (the region round Aoi's wrist took in her hip, "
                     "which came out in skin tone)")
ap.add_argument("--mesh", default=None,
                help="retopo.glb, the mesh labels.json indexes (per-part colour matching); with --solid, an opened "
                     "surface takes its colour from the old surface nearest along it rather than through the air")
ap.add_argument("--labels", default=None, help="labels.json - colour is matched per part when given")
ap.add_argument("--base-weight", type=float, default=0.08)
ap.add_argument("--face-detail", default=None,
                help="a detailed image of the reference's face region (pipeline/face_detail.py): the "
                     "front view samples it wherever it lands inside that region")
ap.add_argument("--face-box", default=None, help="face_src.json - where that region is in the reference")
ap.add_argument("--face-flow-max", type=float, default=0.025,
                help="the largest face flow applied, as a share of the figure's height")
ap.add_argument("--keep-base-face", action="store_true",
                help="where the face cannot be aligned (the face flow is refused), keep the generated face")
ap.add_argument("--no-face-flow", action="store_true",
                help="leave out the finer flow on the head (for faces that are not photographic)")
ap.add_argument("--debug", default=None, help="write per-view alignment overlays here")
ap.add_argument("--edge-guard", type=float, default=0.0,
                help="in the views after the first, fade out paint within this fraction of the view's height "
                     "of the outline of a nearer surface (the pipeline passes 0.02): a generated view and the "
                     "mesh never line up exactly, and the texels just past an arm's outline read the arm - "
                     "Aoi's hands streaked her trousers from the side view")
ap.add_argument("--old-hands", default=None,
                help="solid.npz,solid_cut.npz: the generated hands cut_hands.py removed; the pixels each source "
                     "image shows them in paint nothing (they painted Aoi's trousers and Wren's satchel)")
ap.add_argument("--solid", default=None,
                help="the generated solid (solid.npz) when free_arms.py cut the arms free of it: the surfaces "
                     "the cut opened lie inside it, and take their colour from their own side of the cut")
a = ap.parse_args()

meta = json.load(open(os.path.join(a.dir, "views.json")))
ctr, scale = np.array(meta["center"]), float(meta["scale"])
base = np.asarray(Image.open(a.base).convert("RGB"), np.float32) / 255.0
R = base.shape[0]
pos = np.load(os.path.join(a.dir, "uv_position.npy")).astype(np.float32)
nrm = np.load(os.path.join(a.dir, "uv_normal.npy")).astype(np.float32)
if pos.shape[0] != R:
    raise SystemExit(f"[texproj] uv maps are {pos.shape[0]}^2 but the base texture is {R}^2")
cover = np.isfinite(pos[..., 0])
ti = np.nonzero(cover.ravel())[0]                    # texels that carry a surface
P = pos.reshape(-1, 3)[ti]
N = nrm.reshape(-1, 3)[ti]
N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
print(f"[texproj] {len(ti):,} texels on the surface ({cover.mean():.1%} of {R}^2)", flush=True)


def srgb_to_lab(rgb):
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    c = np.clip(xyz @ M.T, 0, None)
    return np.clip(np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055), 0, 1)


def camera(v):
    loc = np.array(v["cam_location"], float)
    tgt = np.array(v.get("target", (0.0, 0.0, 0.0)), float)       # the face camera looks off-centre
    f = (tgt - loc) / np.linalg.norm(tgt - loc)
    up = np.array([0, 0, 1.0])
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return f, right, up


def to_render_px(X, v):
    f, right, up = camera(v)
    Xn = (X - ctr) * scale - np.array(v.get("target", (0.0, 0.0, 0.0)), float)
    half = v["ortho_scale"] / 2
    res = v["res"]
    u = ((Xn @ right) / half * 0.5 + 0.5) * res
    w = (1.0 - ((Xn @ up) / half * 0.5 + 0.5)) * res
    return np.stack([u, w], 1)


def gray(img, mask):
    g = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    g[~mask] = g[mask].mean() if mask.any() else 128
    return cv2.GaussianBlur(g, (0, 0), 0.8).astype(np.uint8)


def face_flow(warped, img, wmask, imask):
    """A finer flow for the head. The body's flow is smoothed hard because a blurry render and a
    sharp photo only agree on large shapes; a face needs its features to land on the geometry's
    features, and the generator's own texture (aligned with its geometry) shows where those are.
    Matched on an upscaled crop, on lightness for eyes and brows and on redness for the lips,
    with the photo blurred toward the render's softness. Returns (flow, weight) at full size."""
    H_, W_ = img.shape[:2]
    ys, xs = np.nonzero(wmask)
    top = ys.min()
    fig_h = ys.max() - top
    head_rows = (ys < top + 0.13 * fig_h)
    if head_rows.sum() < 50:
        return None, None
    cx = int(np.median(xs[head_rows]))
    half = int(0.07 * fig_h)
    y0, y1 = max(0, top - 5), min(H_, top + int(0.15 * fig_h))
    x0, x1 = max(0, cx - half), min(W_, cx + half)
    k = 2
    def prep(im, blur):
        c = cv2.resize(im[y0:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        if blur:
            c = cv2.GaussianBlur(c, (0, 0), blur)
        lab = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_RGB2LAB)
        out = []
        for ch in (0, 1):
            x = lab[..., ch].astype(np.float32)
            x = (x - x.mean()) / (x.std() + 1e-6)
            out.append(np.clip(x * 40 + 128, 0, 255).astype(np.uint8))
        return out
    A = prep(warped, 0)
    B = prep(img, 2.5)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setPatchSize(12)
    dis.setPatchStride(3)
    dis.setVariationalRefinementIterations(10)
    fl = [dis.calc(A[c], B[c], None) for c in range(2)]
    # Only skin says where facial features are. A fringe that falls differently in the photo and
    # on the mesh drags the flow along its edge and tore Juno's eye; so the flow is kept on skin
    # in both images and filled in elsewhere from the skin around it.
    def skin(im):
        c = cv2.resize(im[y0:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        return (lab[..., 0] > 0.35 * 255) & (lab[..., 1] > 133) & (lab[..., 2] > 130)
    sk = ndimage.binary_erosion(skin(warped) & skin(img), iterations=3).astype(np.float32)
    if sk.sum() > 200:
        wsk = cv2.GaussianBlur(sk, (0, 0), 12)
        for c in range(2):
            fl[c] = np.dstack([cv2.GaussianBlur(fl[c][..., j] * sk, (0, 0), 12) / np.maximum(wsk, 1e-3)
                               for j in range(2)])
    # redness drives the lower face (lips), lightness everything else
    hh = fl[0].shape[0]
    wy = np.clip((np.arange(hh) / hh - 0.55) / 0.15, 0, 1)[:, None, None]
    f = fl[0] * (1 - wy) + fl[1] * wy
    f = np.dstack([cv2.GaussianBlur(f[..., c], (0, 0), 4.0) for c in range(2)]) / k
    f = cv2.resize(f, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    mag = np.linalg.norm(f, axis=2, keepdims=True)
    face_flow.p95 = float(np.percentile(mag, 95)) / fig_h
    f = f * np.minimum(1.0, 0.03 * fig_h / np.maximum(mag, 1e-6))
    full = np.zeros((H_, W_, 2), np.float32)
    full[y0:y1, x0:x1] = f
    wgt = np.zeros((H_, W_), np.float32)
    wgt[y0:y1, x0:x1] = 1.0
    wgt = cv2.GaussianBlur(wgt, (0, 0), 0.01 * fig_h) * (wmask & imask)
    return full, np.clip(wgt * 1.5, 0, 1)


# Colour groups for matching. Part labels were the first idea, but the parser's hair label is
# unreliable (2% of Juno's vertices, most of her bob filed as clothing), so texels are grouped by
# their own colour instead: k-means in Lab on the bake, so black hair, red jacket, grey trousers
# and skin each get their own transform.
base_t = base.reshape(-1, 3)[ti]
lab_b = srgb_to_lab(base_t)
rng = np.random.default_rng(0)
sub = lab_b[rng.choice(len(lab_b), min(60000, len(lab_b)), replace=False)]
centres = sub[rng.choice(len(sub), 8, replace=False)]
for _ in range(20):
    lab_ = np.argmin(((sub[:, None, :] - centres[None]) ** 2).sum(2), 1)
    centres = np.array([sub[lab_ == k].mean(0) if (lab_ == k).any() else centres[k] for k in range(len(centres))])
group = np.empty(len(lab_b), np.int8)
for i0 in range(0, len(lab_b), 1_000_000):
    group[i0:i0 + 1_000_000] = np.argmin(((lab_b[i0:i0 + 1_000_000, None, :] - centres[None]) ** 2).sum(2), 1)
del lab_b
print("[texproj] colour groups (L,a,b): " + ", ".join(f"({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}) {(group == k).mean():.0%}"
      for k, c in enumerate(centres)), flush=True)


def match(src, dst, sel):
    """Lab mean/std transform taking src's colours onto dst's, measured over sel (per group)."""
    out = src.copy()
    for gi in np.unique(group):
        m = sel & (group == gi)
        g = group == gi
        if m.sum() < 400:
            continue
        ls, ld = srgb_to_lab(src[m]), srgb_to_lab(dst[m])
        ms, ss = ls.mean(0), ls.std(0) + 1e-6
        md, sd = ld.mean(0), ld.std(0) + 1e-6
        gain = np.clip(sd / ss, 0.8, 1.25)
        out[g] = lab_to_srgb((srgb_to_lab(src[g]) - ms) * gain + md)
    return out


acc = np.zeros((len(ti), 3), np.float64)
wsum = np.zeros(len(ti), np.float64)
base_t = base.reshape(-1, 3)[ti]
palette_set = False
hand = np.zeros(len(ti), bool)
if a.hands and os.path.exists(a.hands):
    for sd in json.load(open(a.hands))["sides"].values():
        C, x, L = np.array(sd["cut_point"]), np.array(sd["x"]), float(sd["length"])
        hand |= (((P - C) @ x) > -0.02 * L) & (np.linalg.norm(P - C, axis=1) < 1.35 * L)
    if a.hands_solid and os.path.exists(a.hands_solid) and hand.any():
        # The region round the wrist is only where to look: a hip or a bag inside it is the body's own
        # surface, which lies on the handless solid, while the modelled hand stands outside it.
        Hs = np.load(a.hands_solid)
        hv, ho = float(Hs["voxel"]), Hs["origin_ijk"]
        hi_ = np.nonzero(hand)[0]
        dh = ndimage.map_coordinates(Hs["sdf"], (P[hi_] / hv - ho).T, order=1, mode="constant", cval=1.0)
        body_ = dh < 1.5 * hv
        hand[hi_[body_]] = False
        print(f"[texproj] hands: {int(body_.sum()):,} texels by a wrist that lie on the body itself are "
              f"not the modelled hand's", flush=True)
vmeta = {v["tag"]: v for v in meta["views"]}
# The generated hands, as cut_hands.py removed them: every source image still shows them, and where the
# modelled hand does not stand in the same place the mesh behind the old one - a thigh, a bag - read the
# old hand's skin (Aoi's trousers from her side view, Wren's satchel from the front).
old_hand_pts = None
if a.old_hands:
    f_full, f_cut = a.old_hands.split(",")
    if os.path.exists(f_full) and os.path.exists(f_cut):
        Sf, Sc = np.load(f_full), np.load(f_cut)
        gone = (Sf["sdf"] < 0) & (Sc["sdf"] >= 0)
        gone[1::2] = False                                            # every other voxel is plenty
        gone[:, 1::2] = False
        idx_ = np.argwhere(gone)
        if len(idx_):
            old_hand_pts = (idx_ + Sf["origin_ijk"]) * float(Sf["voxel"])
report = []
face_skin = None
face_img, face_box = None, None
face_unaligned = False
if a.face_detail and a.face_box and os.path.exists(a.face_detail) and os.path.exists(a.face_box):
    fb = json.load(open(a.face_box))
    face_box = [float(x) for x in fb["box"]]
    face_img = np.asarray(Image.open(a.face_detail).convert("RGB"), np.float32) / 255.0
    # its colours onto the reference's own, region for region (a diffusion pass shifts exposure)
    ref_crop = os.path.join(os.path.dirname(a.face_box), "face_src.png")
    if os.path.exists(ref_crop):
        rc = np.asarray(Image.open(ref_crop).convert("RGB").resize(face_img.shape[1::-1]), np.float32) / 255.0
        lf, lr = srgb_to_lab(face_img.reshape(-1, 3)), srgb_to_lab(rc.reshape(-1, 3))
        g_ = np.clip(lr.std(0) / (lf.std(0) + 1e-6), 0.8, 1.25)
        face_img = lab_to_srgb((lf - lf.mean(0)) * g_ + lr.mean(0)).reshape(face_img.shape).astype(np.float32)
for spec in a.view:
    parts_ = spec.split(":")
    tag, path = parts_[0], parts_[1]
    mask_path = parts_[2] if len(parts_) > 2 and parts_[2] else None
    vw = float(parts_[3]) if len(parts_) > 3 else 1.0
    tag = f"{int(tag):03d}" if tag.isdigit() else tag
    if tag not in vmeta:
        report.append(f"{tag}: no such view rendered - skipped")
        continue
    v = vmeta[tag]
    img = np.asarray(Image.open(path).convert("RGB"))
    rend = np.asarray(Image.open(os.path.join(a.dir, f"view_{tag}_albedo.png")).convert("RGBA"))
    rpos = np.load(os.path.join(a.dir, f"view_{tag}_position.npy"))
    rmask = rend[..., 3] > 127
    if mask_path and os.path.exists(mask_path):
        imask = np.asarray(Image.open(mask_path).convert("L").resize((img.shape[1], img.shape[0]))) > 127
        imask = ndimage.binary_fill_holes(imask)
    else:
        imask = image_mask(img)
    (s, tx, ty), iou = similarity(rmask, imask)
    if iou < a.min_iou:
        # A view that does not line up with the mesh paints the wrong surface; skipping it
        # leaves the bake, which is blurry but right. This is a gate, not a warning: Vex's
        # painted street background once passed as her silhouette (IoU 0.01).
        report.append(f"{tag}: silhouette IoU {iou:.3f} < {a.min_iou} - view NOT used")
        continue
    M = np.array([[s, 0, tx], [0, s, ty]], np.float32)
    H_, W_ = img.shape[:2]
    warped = cv2.warpAffine(rend[..., :3], M, (W_, H_), flags=cv2.INTER_LINEAR)
    wmask = cv2.warpAffine(rmask.astype(np.uint8), M, (W_, H_), flags=cv2.INTER_NEAREST) > 0
    both = wmask & imask
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(gray(warped, wmask), gray(img, imask), None)
    # What is left after the similarity is smooth - perspective, and the generator's slightly
    # different proportions - but raw optical flow between a blurry render and a sharp photo is
    # noisy at the scale of features (it dragged Juno's lips down her chin). Keep only the smooth
    # part: a normalised Gaussian over the region both silhouettes cover, then a cap.
    inner = ndimage.binary_erosion(both, iterations=5)
    sig = 0.012 * H_
    wgt = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), sig)
    flow = np.dstack([cv2.GaussianBlur(flow[..., c] * inner, (0, 0), sig) / np.maximum(wgt, 1e-3)
                      for c in range(2)])
    mag = np.linalg.norm(flow, axis=2, keepdims=True)
    lim = 0.015 * H_
    flow = flow * np.minimum(1.0, lim / np.maximum(mag, 1e-6))
    if tag in ("000", "face") and not a.no_face_flow:
        ff, fw = face_flow(warped, img, wmask, imask)
        # A face flow is a small correction or it is wrong. Where the generator drew the face much as
        # the reference did it moves features 1-2% of the figure's height (juno 1.0%, mara 1.2%,
        # rowan 1.7%); on the cartoon and anime characters the generator's face disagrees with the
        # reference by more than a local warp can mend (pip 3.4%, aoi 4.7%), and applying it painted
        # pip a second pair of eyes and tore his face at the nose. Left out past 2.5%.
        if ff is not None and face_flow.p95 > a.face_flow_max:
            report.append(f"face flow left out: it would move features {face_flow.p95:.1%} of the figure's "
                          f"height (over {a.face_flow_max:.1%}: the generated face and the reference disagree too much)")
            ff = None
            face_unaligned = True
        if ff is not None:
            flow = flow * (1 - fw[..., None]) + ff * fw[..., None]
    # texel -> render pixel -> image pixel
    rp = to_render_px(P, v)
    ri = np.floor(rp).astype(int)
    # Visible if the render shows this same surface point at this pixel or a neighbour. A single
    # pixel lookup flickers on surfaces the camera sees at a slant (hair seen from the front):
    # neighbouring texels round to different pixels and alternately pass and fail, which left a
    # grid of dots. The nearest of the 3x3 pixels, with a soft tolerance, does not.
    best = np.full(len(P), np.inf, np.float32)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu = np.clip(ri[:, 0] + du, 0, v["res"] - 1)
            vv = np.clip(ri[:, 1] + dv, 0, v["res"] - 1)
            sp = rpos[vv, uu]
            dd = np.linalg.norm(np.nan_to_num(sp, nan=1e3) - P, axis=1)
            best = np.minimum(best, dd)
    px_mesh = v["ortho_scale"] / v["res"] / scale     # one pixel, in mesh units
    # three pixels, but never under 2 mm: the texel positions are float16 (about half a millimetre
    # of error), and three pixels of the close-up face camera is less than that - every texel
    # failed the test and the face view painted nothing
    tol = max(3.0 * px_mesh, 0.002 * (2.0 / scale) / 1.75)
    vis = np.clip(1.0 - (best - tol) / tol, 0.0, 1.0)
    if a.edge_guard > 0 and spec != a.view[0]:
        # A texel close to the outline of something in front of it takes little from this view: the
        # image's arm may stand a few pixels off the mesh's, and the texels just past the mesh's outline
        # read the image's arm (Aoi's hands streaked her trousers from the side view). An outline is a
        # depth jump between neighbouring pixels; a surface merely seen at a slant has none. Not in the
        # reference: the mesh was generated from it and lines up best, and what lies behind an outline
        # there is what the projection is for - half of Aoi's face is behind a lock of her hair.
        fcam, _, _ = camera(v)
        dep = np.nan_to_num(((rpos.astype(np.float32) - ctr) * scale) @ fcam, nan=1e3)   # depth, per pixel
        jump = 0.03 * 2.0 / 1.75                                                           # 3 cm
        pad = np.pad(dep, 1, mode="edge")
        far_nb = np.maximum.reduce([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        outline = (far_nb - dep > jump) & (dep < 1e2)                  # the near side of a depth jump
        kk_ = max(2, int(round(a.edge_guard * v["res"])))
        occ = ndimage.minimum_filter(np.where(outline, dep, 1e3), size=2 * kk_ + 1)
        dist = ndimage.distance_transform_edt(~outline)
        uu0 = np.clip(ri[:, 0], 0, v["res"] - 1)
        vv0 = np.clip(ri[:, 1], 0, v["res"] - 1)
        behind = occ[vv0, uu0] < ((P - ctr) * scale) @ fcam - jump    # that outline is in front of this texel
        g = np.clip(dist[vv0, uu0] / kk_, 0.0, 1.0)
        g = np.where(behind, g * g * (3 - 2 * g), 1.0)
        report.append(f"{tag}: {int(((g < 0.5) & (vis > 0.5)).sum()):,} visible texels by the outline of a "
                      f"nearer surface left mostly to the other views")
        vis = vis * g
    ip = rp * s + np.array([tx, ty])
    ii = np.clip(np.round(ip).astype(int), [0, 0], [W_ - 1, H_ - 1])
    ip = ip + flow[ii[:, 1], ii[:, 0]]
    f, _, _ = camera(v)
    facing = -(N @ f)
    w = np.clip((facing - 0.2) / 0.8, 0, 1) ** 1.5
    edge = ndimage.distance_transform_edt(imask)
    ii = np.clip(np.round(ip).astype(int), [0, 0], [W_ - 1, H_ - 1])
    w *= np.clip(edge[ii[:, 1], ii[:, 0]] / 6.0, 0, 1) * vis * imask[ii[:, 1], ii[:, 0]]
    w[hand] = 0
    if tag == "000" and face_box is not None:
        # the face's own texels, for the hands' skin tone: the middle of the face region, facing the camera
        fx0, fy0, fx1, fy1 = face_box
        face_skin = ((np.abs(ip[:, 0] - (fx0 + fx1) / 2) < 0.3 * (fx1 - fx0))
                     & (np.abs(ip[:, 1] - (fy0 + fy1) / 2) < 0.3 * (fy1 - fy0)) & (vis > 0.5) & (facing > 0.5))
    if tag == "000" and face_box is not None and face_unaligned and a.keep_base_face:
        # A realistic face the reference cannot be laid on. Knight's generated face sat 2.5% of his
        # height from the picture's: projected unaligned, the picture's fringe fell across his eye and
        # every feature came out twice; forced into line by the flow, his eyes smeared into bands.
        # The generator's own face is softer but it is where its geometry is - so the front view
        # leaves the face and the fringe above it alone (in the face frame 0.23-0.77 across, 0.26-0.86
        # down, fading out beyond): stopped at the hairline, the picture's fringe still fell across his eye.
        fx0, fy0, fx1, fy1 = face_box
        u = (ip[:, 0] - (fx0 + fx1) / 2) / (0.27 * (fx1 - fx0))
        q_ = (ip[:, 1] - (fy0 + 0.56 * (fy1 - fy0))) / (0.30 * (fy1 - fy0))
        keep_face = np.clip((1.15 - np.sqrt(u * u + q_ * q_)) / 0.30, 0, 1) * (facing > 0.0)
        w *= 1 - keep_face
        report.append(f"face: the generated face kept - {int((keep_face > 0.5).sum()):,} texels the front view leaves to it")
    if old_hand_pts is not None:
        # where this image shows the old hands: their points through the same camera, alignment and flow
        hp = to_render_px(old_hand_pts, v) * s + np.array([tx, ty])
        hi_ = np.clip(np.round(hp).astype(int), [0, 0], [W_ - 1, H_ - 1])
        hp = hp + flow[hi_[:, 1], hi_[:, 0]]
        hi_ = np.round(hp).astype(int)
        ok_ = (hi_[:, 0] >= 0) & (hi_[:, 0] < W_) & (hi_[:, 1] >= 0) & (hi_[:, 1] < H_)
        om = np.zeros((H_, W_), bool)
        om[hi_[ok_, 1], hi_[ok_, 0]] = True
        om = ndimage.binary_dilation(ndimage.binary_closing(om, iterations=2), iterations=max(2, int(0.006 * H_)))
        in_old = om[ii[:, 1], ii[:, 0]]
        w[in_old] = 0
        if tag == "000" or in_old.any():
            report.append(f"{tag}: {int((in_old & (vis > 0.5)).sum()):,} texels behind the old hands left to the other views")
    w *= vw
    col = np.stack([ndimage.map_coordinates(img[..., c].astype(np.float32), [ip[:, 1], ip[:, 0]],
                                            order=1, mode="nearest") for c in range(3)], 1) / 255.0
    if tag == "000" and face_img is not None and not (face_unaligned and a.keep_base_face):
        # The face region, from the detailed image instead: the same pixels, four times finer. Read
        # through this view's own alignment - a separate close-up camera aligned on its own landed a
        # few pixels off it and doubled the eyes where the two were blended.
        x0, y0, x1, y1 = face_box
        fs = face_img.shape[0] / float(x1 - x0)
        fx, fy = (ip[:, 0] - x0) * fs, (ip[:, 1] - y0) * fs
        inside_ = (fx >= 0) & (fy >= 0) & (fx < face_img.shape[1] - 1) & (fy < face_img.shape[0] - 1)
        if inside_.any():
            d_edge = np.minimum.reduce([fx, fy, face_img.shape[1] - 1 - fx, face_img.shape[0] - 1 - fy])
            wf_ = np.clip(d_edge / (0.1 * face_img.shape[0]), 0, 1)
            wf_ = (wf_ * wf_ * (3 - 2 * wf_)) * inside_
            cf_ = np.stack([ndimage.map_coordinates(face_img[..., c], [fy, fx], order=1, mode="nearest")
                            for c in range(3)], 1)
            col = col * (1 - wf_[:, None]) + cf_ * wf_[:, None]
            report.append(f"face: {int((wf_ > 0.5).sum()):,} texels of the front view read from the "
                          f"detailed face ({fs:.1f}x the reference's pixels)")
    # Colour. The first view given is the reference - the image the character was designed in -
    # so it sets the palette: the bake is moved onto its colour statistics, part by part (skin,
    # clothing, hair: one global transform turned Juno's black hair brown), over the texels both
    # see well. Later views are made from renders and repainted; they follow the corrected bake.
    conf = w > 0.6 * vw
    shift = float("nan")
    if conf.sum() > 500:
        shift = float(np.linalg.norm(srgb_to_lab(col[conf]).mean(0) - srgb_to_lab(base_t[conf]).mean(0)))
        if not palette_set:
            base_t = match(base_t, col, conf)
            palette_set = True
        else:
            col = match(col, base_t, conf)
    acc += w[:, None] * col
    wsum += w
    report.append(f"{tag}: silhouette IoU {iou:.3f}, scale {s:.3f}, flow p95 {np.percentile(np.linalg.norm(flow[both], axis=1), 95):.1f}px, "
                  f"{(w > 0.5 * vw).mean():.1%} of texels confident, colour shift dE {shift:.1f}")
    if a.debug:
        os.makedirs(a.debug, exist_ok=True)
        ov = img.copy()
        ov[wmask & ~imask] = (255, 0, 0)
        ov[imask & ~wmask] = (0, 0, 255)
        Image.fromarray(ov).save(os.path.join(a.debug, f"align_{tag}.png"))
for line in report:
    print(f"[texproj] {line}", flush=True)

out_t = (acc + a.base_weight * base_t) / (wsum + a.base_weight)[:, None]
opened = np.zeros(len(ti), bool)
fill = base_t.copy()                                   # what stands in for the bake where the bake is wrong
bw = np.full(len(ti), a.base_weight)                   # and how much it counts against the views
if a.solid and os.path.exists(a.solid):
    # The surfaces free_arms.py opened between an arm and what it was glued to - the inside of a sleeve,
    # the side of a vest under it, the back of a sleeve a puffy vest had swallowed - were inside the
    # generated solid: the source images show whatever covered them, and the bake is the solid's inside
    # (Pip's armpits came out grey). Each takes the colour of the old surface nearest to it along the
    # mesh - the cut parted the arm from the torso, so the sleeve's opened side reaches the sleeve round
    # the cut's edge before the vest - and the views are not used on it: whatever they show there is what
    # covered it. Nearest through the air, a texel facing the same way can be the other garment across
    # the gap: the back of a sleeve and the back of the vest beside it both face backwards.
    from scipy.spatial import cKDTree
    Sd = np.load(a.solid)
    vv, oo = float(Sd["voxel"]), Sd["origin_ijk"]
    d0 = ndimage.map_coordinates(Sd["sdf"], (P / vv - oo).T, order=1, mode="constant", cval=1.0)
    opened = (d0 < -2.5 * vv) & ~hand
    left_ = opened.copy()
    if opened.any() and a.mesh and os.path.exists(a.mesh):
        import trimesh
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        mm = trimesh.load(a.mesh, force="mesh", process=False)
        Vg = np.asarray(mm.vertices, np.float64)
        Vm = np.c_[Vg[:, 0], -Vg[:, 2], Vg[:, 1]]                     # glTF is y-up; the texel maps are z-up
        _, weld = np.unique(np.round(Vm, 5), axis=0, return_inverse=True)
        weld = weld.ravel()
        nw = int(weld.max()) + 1
        Vw = np.zeros((nw, 3)); Vw[weld] = Vm                        # one vertex per position: seams do not cut
        Fw = weld[np.asarray(mm.faces)]
        E = np.unique(np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1), axis=0)
        E = E[E[:, 0] != E[:, 1]]
        el = np.linalg.norm(Vw[E[:, 0]] - Vw[E[:, 1]], axis=1) + 1e-9
        G = coo_matrix((el, (E[:, 0], E[:, 1])), shape=(nw, nw)).tocsr()
        dv = ndimage.map_coordinates(Sd["sdf"], (Vw / vv - oo).T, order=1, mode="constant", cval=1.0)
        src_v = np.nonzero(np.abs(dv) < 1.5 * vv)[0]
        src_t = np.nonzero(~opened & ~hand & (np.abs(d0) < 1.5 * vv))[0]
        if len(src_v) and len(src_t):
            _, _, origin = dijkstra(G, directed=False, indices=src_v, min_only=True, return_predecessors=True)
            # each old-surface vertex: the colour of the old-surface texels about it
            _, nb_t = cKDTree(P[src_t]).query(Vw[src_v], k=8)
            vcol = np.zeros((nw, 3)); vcol[src_v] = out_t[src_t[nb_t]].mean(1)
            # each opened texel: the vertices of its own surface about it (facing its way), through them the
            # old-surface vertex each was reached from
            vn = np.zeros((nw, 3))
            fn = np.cross(Vw[Fw[:, 1]] - Vw[Fw[:, 0]], Vw[Fw[:, 2]] - Vw[Fw[:, 0]])
            for k_ in range(3):
                np.add.at(vn, Fw[:, k_], fn)
            vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
            oi = np.nonzero(opened)[0]
            dd_, nb_v = cKDTree(Vw).query(P[oi], k=6)
            wv = np.clip((vn[nb_v] * N[oi][:, None, :]).sum(-1), 0.0, 1.0) / (dd_ + 1e-4)
            ok_ = origin[nb_v] >= 0
            wv = wv * ok_
            reach = wv.sum(1) > 0
            cv = (vcol[np.where(ok_, origin[nb_v], 0)] * wv[..., None]).sum(1) / np.maximum(wv.sum(1), 1e-12)[:, None]
            fill[oi[reach]] = cv[reach]
            left_ = np.zeros(len(ti), bool); left_[oi[~reach]] = True
            print(f"[texproj] {int(opened.sum()):,} texels opened by the arm cut coloured from the old surface "
                  f"nearest along the mesh ({float(reach.mean()):.1%} reached)", flush=True)
    if left_.any():
        src = np.nonzero(~opened & ~hand & (np.abs(d0) < 1.5 * vv))[0][::3]
        dist_, nb = cKDTree(P[src]).query(P[left_], k=24)
        nb = src[nb]
        same = (N[nb] * N[left_][:, None, :]).sum(-1) > 0.35
        pick = np.where(same.any(1), same.argmax(1), 0)
        fill[left_] = out_t[nb[np.arange(len(nb)), pick]]
        print(f"[texproj] {int(left_.sum()):,} texels opened by the arm cut coloured from the nearest old surface "
              f"facing their way ({float(same.any(1).mean()):.0%} found one)", flush=True)
    bw[opened] = 1e3
out_t = (acc + bw[:, None] * fill) / (wsum + bw)[:, None]
if hand.any():
    # The modelled hands take the character's skin tone, read from texels that are skin-coloured
    # anywhere on the body outside the hands (face, neck, forearms): warm, mid-light, moderately
    # saturated. The old hand region itself mixes in cuff and shadow (brick red on Juno), and "the
    # top of the figure" is mostly hair on a character like Vex (it came out teal).
    Lh = srgb_to_lab(out_t)
    skin = (~hand & ~opened & (Lh[:, 0] > 40) & (Lh[:, 0] < 90) & (Lh[:, 1] > 6) & (Lh[:, 1] < 32)
            & (Lh[:, 2] > 8) & (Lh[:, 2] < 38) & (Lh[:, 2] > 0.6 * Lh[:, 1]))
    on_face = skin & face_skin if face_skin is not None else np.zeros_like(skin)
    if on_face.sum() > 500:
        # the face first: over the whole figure, orange hair, khaki shorts and a vest's shading all pass
        # for skin, and Pip's hands came out brown (137, 95, 71) against his peach face
        skin = on_face
    face_any = (face_skin & ~hand & ~opened) if face_skin is not None else np.zeros_like(skin)
    from_face = on_face.sum() > 500
    if skin.sum() <= 500 and face_any.sum() > 500:
        # skin that is not skin-coloured - a grey alien, a green orc: the face's own colour, whatever
        # its hue. The warm range found nothing on Gray, and the fallback peach gave him a human hand
        skin, from_face = face_any, True
    if skin.sum() > 500:
        # the largest cluster of skin-like colour, not an average over lips and tan clothing
        lab_s = Lh[skin]
        med = np.median(lab_s, 0)
        near = np.linalg.norm(lab_s - med, axis=1) < 12
        tone = np.median(out_t[skin][near], 0) if near.sum() > 500 else np.median(out_t[skin], 0)
        src_n = int(near.sum())
    else:
        tone = np.array([0.80, 0.63, 0.53])
        src_n = 0
    # a little of the bake's own light and dark variation, so the hand is not a flat plastic
    lum = srgb_to_lab(base_t[hand])[:, 0]
    var = np.clip((lum - np.median(lum)) / 60.0, -0.12, 0.12)[:, None]
    out_t[hand] = np.clip(tone * (1.0 + var), 0, 1)
    print(f"[texproj] modelled hands: {int(hand.sum()):,} texels in the body's skin tone "
          f"{tuple(int(c * 255) for c in tone)} (from {src_n:,} skin texels"
          f"{' of the face' if from_face else ''})", flush=True)
out = base.reshape(-1, 3).copy()
out[ti] = out_t
out = out.reshape(R, R, 3)
# bleed into the gutter so mip levels do not pull in the old colours at island edges
cov_img = cover.copy()
grown = out.copy()
for _ in range(8):
    m = ndimage.binary_dilation(cov_img) & ~cov_img
    if not m.any():
        break
    for c in range(3):
        s_ = ndimage.uniform_filter(grown[..., c] * cov_img, 3)
        n_ = ndimage.uniform_filter(cov_img.astype(np.float32), 3)
        grown[..., c] = np.where(m, s_ / np.maximum(n_, 1e-6), grown[..., c])
    cov_img |= m
Image.fromarray((np.clip(grown, 0, 1) * 255 + 0.5).astype(np.uint8)).save(a.out)
seen_any = (wsum > 0.5).mean()
print(f"[texproj] {seen_any:.1%} of surface texels now come mainly from a source image -> {a.out}", flush=True)
