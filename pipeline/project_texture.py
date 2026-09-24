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
ap.add_argument("--mesh", default=None, help="the mesh labels.json indexes (for per-part colour matching)")
ap.add_argument("--labels", default=None, help="labels.json - colour is matched per part when given")
ap.add_argument("--base-weight", type=float, default=0.08)
ap.add_argument("--face-detail", default=None,
                help="a detailed image of the reference's face region (pipeline/face_detail.py): the "
                     "front view samples it wherever it lands inside that region")
ap.add_argument("--face-box", default=None, help="face_src.json - where that region is in the reference")
ap.add_argument("--no-face-flow", action="store_true",
                help="leave out the finer flow on the head (for faces that are not photographic)")
ap.add_argument("--debug", default=None, help="write per-view alignment overlays here")
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
vmeta = {v["tag"]: v for v in meta["views"]}
report = []
face_img, face_box = None, None
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
        if ff is not None and face_flow.p95 > 0.025:
            report.append(f"face flow left out: it would move features {face_flow.p95:.1%} of the figure's "
                          f"height (over 2.5%: the generated face and the reference disagree too much)")
            ff = None
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
    w *= vw
    col = np.stack([ndimage.map_coordinates(img[..., c].astype(np.float32), [ip[:, 1], ip[:, 0]],
                                            order=1, mode="nearest") for c in range(3)], 1) / 255.0
    if tag == "000" and face_img is not None:
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
if hand.any():
    # The modelled hands take the character's skin tone, read from texels that are skin-coloured
    # anywhere on the body outside the hands (face, neck, forearms): warm, mid-light, moderately
    # saturated. The old hand region itself mixes in cuff and shadow (brick red on Juno), and "the
    # top of the figure" is mostly hair on a character like Vex (it came out teal).
    Lh = srgb_to_lab(out_t)
    skin = (~hand & (Lh[:, 0] > 40) & (Lh[:, 0] < 90) & (Lh[:, 1] > 6) & (Lh[:, 1] < 32)
            & (Lh[:, 2] > 8) & (Lh[:, 2] < 38) & (Lh[:, 2] > 0.6 * Lh[:, 1]))
    if skin.sum() > 2000:
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
          f"{tuple(int(c * 255) for c in tone)} (from {src_n:,} skin texels)", flush=True)
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
