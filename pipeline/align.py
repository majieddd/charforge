"""Silhouette alignment shared by the texture stages: where a source image's figure is, and the
scale and shift that lays a render of the mesh over it."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage, optimize


def image_mask(img):
    """Foreground of a source image: its background is a plain field (grey or black)."""
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    bg = np.median(border, axis=0)
    d = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    m = d > 0.07 * 255
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndimage.binary_fill_holes(m)


def similarity(src_mask, dst_mask):
    """Scale and shift taking src_mask's frame onto dst_mask's, by silhouette overlap.
    Returns ((s, tx, ty), iou): dst pixel = s * src pixel + (tx, ty)."""
    ys, xs = np.nonzero(src_mask)
    yd, xd = np.nonzero(dst_mask)
    s0 = (yd.max() - yd.min()) / max(1, ys.max() - ys.min())
    tx0 = (xd.min() + xd.max()) / 2 - s0 * (xs.min() + xs.max()) / 2
    ty0 = (yd.min() + yd.max()) / 2 - s0 * (ys.min() + ys.max()) / 2
    k = 0.5                                          # optimise at half resolution
    src_s = cv2.resize(src_mask.astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    dst_s = cv2.resize(dst_mask.astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_AREA) > 0

    def iou(p):
        s, tx, ty = p
        M = np.array([[s, 0, tx * k], [0, s, ty * k]], np.float32)
        w = cv2.warpAffine(src_s, M, (dst_s.shape[1], dst_s.shape[0]), flags=cv2.INTER_LINEAR) > 0
        return -(w & dst_s).sum() / max(1, (w | dst_s).sum())

    r = optimize.minimize(iou, [s0, tx0, ty0], method="Nelder-Mead",
                          options={"xatol": 0.05, "fatol": 1e-5, "initial_simplex":
                                   [[s0, tx0, ty0], [s0 * 1.02, tx0, ty0], [s0, tx0 + 4, ty0], [s0, tx0, ty0 + 4]]})
    return r.x, -r.fun
