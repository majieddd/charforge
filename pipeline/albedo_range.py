"""Keep a non-metal albedo inside the range real materials have (E132's PBR check).

    python pipeline/albedo_range.py --albedo albedo.png [--orm baked_orm.png] --out albedo.png \
        [--floor 30] [--ceiling 240]

A physically based renderer multiplies light by the albedo: a black jacket painted at sRGB 5-15 comes out a hole
with no folds in it, and a white one at 250 glows. The PBR guides put non-metals between sRGB 30-50 (charcoal,
fresh asphalt) and 240 (fresh snow). The lightness (CIELAB L*) of every non-metal texel is eased into that range
by a curve - texels well inside it are untouched, and hue and chroma stay as painted. Metals (ORM blue > 0.5) are
left alone.

The floor. The old softplus shoulder lifted the darkest texels but also flattened them: its slope at black was
0.2, so the shadows of hair, shoes and dark cloth (14.7% of Mara's non-metals) lost most of their contrast, and the
albedo's fine detail fell 16% (tools/texture_sharpness.py). --curve power (the default) lifts black to about the same floor
(L* 11.3 against the old 13.0) with a slope of --slope0 (0.4) there, rising smoothly to 1 at L* 20 and the identity above. A power-law slope never
dips, so no two tones swap; on Mara's albedo before the floor it keeps 0.89 of the detail where the old floor kept 0.85 (tools/texture_sharpness.py). --curve soft is the old shoulder, kept for comparison. The ceiling is unchanged.
"""
import argparse

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


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


def l_of_srgb_grey(v):
    """CIELAB L* of the sRGB grey v (0..255)."""
    return float(srgb_to_lab(np.array([[v / 255.0] * 3]))[0, 0])


def floor_power(L, lo, slope0=0.4, L0=20.0):
    """L* lifted so that black reaches lo (sRGB 30) with slope slope0 there, rising smoothly to 1 at L0 and the
    identity above it: L = lo + L0 (s0 t + (1 - s0) t^(p+1) / (p+1)), t = L / L0, with p set so that L(L0) = L0.
    The slope never falls below slope0, so no two tones swap and the shadows keep a fraction of their folds."""
    q = lo / (L0 * (1 - slope0))
    if q > 0.95:                                     # a floor this high needs a longer ramp, or p blows up and the
        L0 = lo / (0.95 * (1 - slope0))              # darkest texels go negative (black); the default (q 0.94) is untouched
        q = 0.95
    p = q / (1 - q)
    t = np.clip(L / L0, 0.0, 1.0)
    return np.where(L < L0, lo + L0 * (slope0 * t + (1 - slope0) * t ** (p + 1) / (p + 1)), L)


def soft_range(L, lo, hi, knee=8.0, curve="power", slope0=0.4):
    """L* eased into [lo, hi]: identity well inside. The floor is floor_power (default) or the old softplus shoulder;
    the ceiling is a softplus shoulder over `knee` L* units, as before."""
    def softplus(x):
        return knee * np.log1p(np.exp(np.clip(x / knee, -40, 40)))
    if curve == "power":
        L1 = floor_power(L, lo, slope0)              # floor: the shadows keep a share of their contrast
    else:
        L1 = lo + softplus(L - lo)                   # floor, as it was (flattens black: slope 0.18)
    return hi - softplus(hi - L1)                    # ceiling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--albedo", required=True)
    ap.add_argument("--orm", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor", type=float, default=30)
    ap.add_argument("--ceiling", type=float, default=240)
    ap.add_argument("--curve", choices=("power", "soft"), default="power",
                    help="the floor: power keeps the shadows' contrast (default); soft is the old shoulder")
    ap.add_argument("--slope0", type=float, default=0.4, help="the floor's slope at black (power)")
    a = ap.parse_args()
    img = np.asarray(Image.open(a.albedo).convert("RGB"), np.float32) / 255
    metal = np.zeros(img.shape[:2], bool)
    if a.orm:
        orm = np.asarray(Image.open(a.orm).convert("RGB").resize(img.shape[1::-1]), np.float32) / 255
        metal = orm[..., 2] > 0.5
    out = img.copy()
    lo, hi = l_of_srgb_grey(a.floor), l_of_srgb_grey(a.ceiling)
    step = 512
    for r0 in range(0, img.shape[0], step):          # in strips: a 4K atlas in Lab is 400 MB at once
        sl = slice(r0, r0 + step)
        lab = srgb_to_lab(img[sl])
        L = lab[..., 0]
        L2 = soft_range(L, lo, hi, curve=a.curve, slope0=a.slope0)
        lab2 = lab.copy()                            # hue and chroma as painted; only lightness moves
        lab2[..., 0] = L2
        res = lab_to_srgb(lab2)
        keep = metal[sl]
        res[keep] = img[sl][keep]
        out[sl] = res
    lum = lambda x: (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]) * 255
    nm = ~metal
    print(f"[albedo] non-metal below sRGB {a.floor:.0f}: {100 * (lum(img)[nm] < a.floor).mean():.1f}% -> "
          f"{100 * (lum(out)[nm] < a.floor).mean():.1f}%; above {a.ceiling:.0f}: {100 * (lum(img)[nm] > a.ceiling).mean():.1f}% -> "
          f"{100 * (lum(out)[nm] > a.ceiling).mean():.1f}%", flush=True)
    Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save(a.out)


if __name__ == "__main__":
    main()
