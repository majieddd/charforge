"""Keep the bake's colour field at low frequency, and the views' own detail at high frequency, per texel.

project_texture.py's mix, per texel, is M = (acc + bw * fill) / (wsum + bw): the views' mean colour with the bake (the
TRELLIS colour, or the colour of the old surface where the arm cut opened the body) at bw. The bake is one colour field
across the front and the back, so the share it takes pulls the two sides together; that is what keeps a garment's
front and back colour within about 2 dE (tools/model_quality.py's front_back_de). But the same share is taken from
each view's detail: at bw = 0.08 a confidently seen texel kept 92% of its own detail, and the rest of the blur came
from averaging with the bake.

Here, for a texel the views see strongly (strong = 1), the low frequencies stay the mix's (so the colour field, and
the front-to-back colour, are as before) and the high frequencies are the views' own:

    G  = low-pass of the mix at the bake's full weight over the surface (sigma texels, normalised)
    Gv = low-pass of the views' mean (acc / wsum, normalised over the views; not used for the colour field)
    out = G + strong * (V - Gv) + (1 - strong) * (M - G)        V = acc / wsum, M = the mix

A weakly seen texel (strong = 0) keeps the mix exactly. strong is the same smooth ramp on the summed weight that
sets the bake's share (project_texture.py --base-ramp), zero on the modelled and opened texels.
"""
import numpy as np
from scipy import ndimage


def detail_from_views(ti, R, mix_low, mix, acc, wsum, strong, sigma=3.0):
    """ti: the surface texels (indices into the R x R atlas); mix_low (n, 3): the mix at the bake's full weight (the
    colour field, whose low frequencies are kept); mix (n, 3): the mix as it is now; acc (n, 3), wsum (n,): the views'
    weighted colour sum and weight; strong (n,) in [0, 1]. Returns (n, 3)."""
    n = len(ti)

    def grid(x):
        g = np.zeros((R * R,) + x.shape[1:], np.float32)
        g[ti] = x
        return g.reshape((R, R) + x.shape[1:])

    def smooth(g):
        return ndimage.gaussian_filter(g, (sigma, sigma) + (0,) * (g.ndim - 2))

    surf = grid(np.ones(n, np.float32))
    ds = np.maximum(smooth(surf), 1e-6)
    G = (smooth(grid(mix_low.astype(np.float32)) * surf[..., None]) / ds[..., None]).reshape(R * R, 3)[ti]
    gw = smooth(grid(wsum.astype(np.float32))).reshape(-1)[ti]
    gacc = smooth(grid(acc.astype(np.float32))).reshape(R * R, 3)[ti]
    Gv = np.where((gw > 1e-6)[:, None], gacc / np.maximum(gw, 1e-6)[:, None], G)   # the views' own low band
    V = acc / np.maximum(wsum, 1e-9)[:, None]
    s = np.clip(strong, 0.0, 1.0)[:, None]
    return (G + s * (V - Gv) + (1 - s) * (mix - G)).astype(np.float32)
