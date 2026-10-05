"""Fit Hunyuan3D-Paint's side and back views onto our own cameras, for the texture stage to project (E134).

    python pipeline/hy_views_to_mv.py --work work/<name>

Both renderers are orthographic and look at the same mesh from the same directions, so a view of one maps
onto the other's by a scale and an offset (and perhaps a mirror, and perhaps an azimuth named the other way
round). Those are fitted on the silhouettes - Hunyuan's normal render against our own render of the same
azimuth (texproj/view_<az>_position.npy, from blender/uv_maps.py) - and kept only if the silhouettes then
overlap well. Writes work/<name>/mv/{side,back,side2}_hy.png in our frame (black outside the figure), the
names the texture stage looks for, and mv/hy_fit.json with each fit's IoU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def bbox(m):
    ys, xs = np.nonzero(m)
    return xs.min(), xs.max(), ys.min(), ys.max()


def fit(hy_mask, our_mask):
    """Scale and offset taking hy_mask onto our_mask, by their bounding boxes; returns (s, tx, ty)."""
    hx0, hx1, hy0, hy1 = bbox(hy_mask)
    ox0, ox1, oy0, oy1 = bbox(our_mask)
    s = (oy1 - oy0) / max(hy1 - hy0, 1)                      # height: never clipped by an arm at the side
    tx = (ox0 + ox1) / 2 - s * (hx0 + hx1) / 2
    ty = (oy0 + oy1) / 2 - s * (hy0 + hy1) / 2
    return s, tx, ty


def warp(img, s, tx, ty, size, resample=Image.BICUBIC):
    # PIL's affine maps output pixels back to input ones: x_in = (x_out - tx) / s
    return img.transform(size, Image.AFFINE, (1 / s, 0, -tx / s, 0, 1 / s, -ty / s), resample=resample)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def srgb_to_lab(c):
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / [0.95047, 1.0, 1.08883]
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * [0.95047, 1.0, 1.08883]
    M = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                                [0.0193339, 0.1191920, 0.9503041]]))
    lin = np.clip(xyz @ M.T, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def garments(lab, k=8, seed=0):
    """Colour clusters of a figure's pixels: chroma first, lightness lightly - shading moves lightness most."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([lab[:, 1:], 0.5 * lab[:, :1]], 1)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(25):
        lab_ = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([X[lab_ == i].mean(0) if (lab_ == i).any() else C[i] for i in range(k)])
    return C


def assign(lab, C):
    X = np.concatenate([lab[:, 1:], 0.5 * lab[:, :1]], 1)
    return np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)


def match_colours(view_rgb, view_mask, front_lab, C, front_cl, max_chroma=12.0):
    """Each garment of a painted view shifted onto the front's colour for it (Lab mean and spread).

    Only where the front has that garment: a view's colour is matched to the front's nearest garment colour
    if their chroma (a*, b*) lies within max_chroma, and otherwise left as painted - Cadet's back view
    painted a beige pack the front does not show, and matching it to the nearest front colour turned it
    into skin."""
    lab = srgb_to_lab(view_rgb[view_mask])
    cl = assign(lab, C)
    out = lab.copy()
    for i in range(len(C)):
        v, f = cl == i, front_cl == i
        if v.sum() < 50 or f.sum() < 50:
            continue
        mv, sv = lab[v].mean(0), lab[v].std(0) + 1e-3
        mf, sf = front_lab[f].mean(0), front_lab[f].std(0) + 1e-3
        if np.hypot(*(mv[1:] - mf[1:])) > max_chroma:
            continue
        out[v] = (lab[v] - mv) * np.clip(sf / sv, 0.7, 1.4) + mf
    res = view_rgb.copy()
    res[view_mask] = lab_to_srgb(out)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--min-iou", type=float, default=0.85)
    ap.add_argument("--front", default=None, help="match each garment's colour to this front picture (and --front-mask)")
    ap.add_argument("--front-mask", default=None)
    a = ap.parse_args()
    w = Path(a.work)
    front = None
    if a.front:
        fr = np.asarray(Image.open(a.front).convert("RGB"), np.float32) / 255
        fm = np.asarray(Image.open(a.front_mask).convert("L").resize(fr.shape[1::-1]), np.float32) > 127
        front_lab = srgb_to_lab(fr[fm])
        C = garments(front_lab[np.random.default_rng(0).choice(len(front_lab), min(40_000, len(front_lab)), replace=False)])
        front = (front_lab, C, assign(front_lab, C))
    hv = json.loads((w / "hy_views" / "views.json").read_text())
    mv = w / "mv"
    mv.mkdir(exist_ok=True)
    report = {}
    for our_az, stem in (("090", "side"), ("180", "back"), ("270", "side2")):
        pos = w / "texproj" / f"view_{our_az}_position.npy"
        if not pos.exists():
            continue
        ours = np.isfinite(np.load(pos)).all(-1)
        best = None
        for v in hv["views"]:
            # a view and the opposite one mirrored have the same silhouette, so the azimuth is not left to the
            # fit: the two renderers name azimuths alike (fitted on the sides, 090 <- 90 unmirrored, IoU 0.99)
            if abs(v["elev"]) > 1 or int(round(v["azim"])) % 360 != int(our_az):
                continue
            nrm = Image.open(w / "hy_views" / f"normal_{v['tag']}.png").convert("RGB")
            for mirror in (False, True):
                n_ = nrm.transpose(Image.FLIP_LEFT_RIGHT) if mirror else nrm
                hm = (np.asarray(n_).astype(int).sum(-1) < 3 * 250)      # the figure: not the white background
                s, tx, ty = fit(hm, ours)
                hw = np.asarray(warp(Image.fromarray(hm.astype(np.uint8) * 255), s, tx, ty,
                                     ours.shape[::-1], Image.NEAREST)) > 127
                score = iou(hw, ours)
                if best is None or score > best[0]:
                    best = (score, v, mirror, s, tx, ty)
        score, v, mirror, s, tx, ty = best
        report[stem] = {"our_azimuth": our_az, "hunyuan_azimuth": v["azim"], "mirror": mirror, "iou": round(float(score), 3),
                        "scale": round(float(s), 4)}
        if score < a.min_iou:
            print(f"[hy_mv] {stem}: best fit IoU {score:.2f} < {a.min_iou} - not used", flush=True)
            continue
        alb = Image.open(w / "hy_views" / f"albedo_{v['tag']}.png").convert("RGB")
        if mirror:
            alb = alb.transpose(Image.FLIP_LEFT_RIGHT)
        out = np.asarray(warp(alb, s, tx, ty, ours.shape[::-1])).copy()
        if front is not None:
            out = (match_colours(out.astype(np.float32) / 255, ours, *front) * 255).astype(np.uint8)
        out[~ours] = 0
        Image.fromarray(out).save(mv / f"{stem}_hy.png")
        print(f"[hy_mv] {stem} (our {our_az}) <- Hunyuan azimuth {v['azim']:.0f}{' mirrored' if mirror else ''}, "
              f"silhouette IoU {score:.3f}", flush=True)
    (mv / "hy_fit.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
