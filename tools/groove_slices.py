"""Slices through a groove cluster: which voxels are outside, shell, solid, plus the flagged retopo vertices.

    ../.venv/bin/python tools/groove_slices.py --char work/rowan --centre 0.0889 0.019 0.1908 --span 0.10 \
        --solid solid_free.npz --flags flags.npy --out slice.png

--flags is an .npy of flagged vertex positions in the Blender frame (tools/groove_ao_report.py's rule). The
planes are X-Z at the centre's Y and Y-Z at its X. Colours: white = outside air (count >= 3 of 26 directions),
dark = solid in --solid, light blue = shell in air (carved from the solid), light grey = not outside and not
in --solid (air carved after solidify).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--char", required=True)
ap.add_argument("--centre", nargs=3, type=float, required=True, help="Blender frame metres")
ap.add_argument("--span", type=float, default=0.10)
ap.add_argument("--solid", default="solid.npz")
ap.add_argument("--flags", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

d = np.load(f"{a.char}/sdf.npz")
U = np.abs(d["sdf"])
v = float(d["voxel"])
o = d["origin_ijk"].astype(np.int64)
shell = U < 0.75 * v
del U


def escapes(blocked, dvec):
    ax = int(np.argmax(np.abs(dvec)))
    step = int(np.sign(dvec[ax]))
    others = [i for i in range(3) if i != ax]
    E = np.empty(blocked.shape, bool)
    n = blocked.shape[ax]
    order = range(n - 1, -1, -1) if step > 0 else range(n)
    src, dst = [slice(None), slice(None)], [slice(None), slice(None)]
    for k, oa in enumerate(others):
        s_ = int(dvec[oa])
        if s_ > 0:
            src[k], dst[k] = slice(1, None), slice(None, -1)
        elif s_ < 0:
            src[k], dst[k] = slice(None, -1), slice(1, None)
    src, dst = tuple(src), tuple(dst)
    prev = None
    for i in order:
        sl = [slice(None)] * 3
        sl[ax] = i
        sl = tuple(sl)
        free = ~blocked[sl]
        if prev is None:
            cur = free
        else:
            nxt = np.ones_like(prev)
            nxt[dst] = prev[src]
            cur = free & nxt
        E[sl] = cur
        prev = cur
    return E


dirs = [np.array((x, y, z)) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]
count = np.zeros(shell.shape, np.uint8)
for dv in dirs:
    count += escapes(shell, dv)
outside = (count >= 3) & ~shell
del count
S = np.load(f"{a.char}/{a.solid}")["sdf"]
F_all = S < 0
c = np.round(np.array(a.centre) / v).astype(np.int64) - o
j, i = c[1], c[0]
FP = np.load(a.flags) if a.flags else None
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
panels = [("xz", (slice(None), j, slice(None))), ("yz", (i, slice(None), slice(None)))]
for ax_, (name, sl) in zip(axes, panels):
    A = shell[sl]
    E = outside[sl]
    F = F_all[sl]
    if name == "xz":
        hx = (np.arange(A.shape[0]) + o[0]) * v
        hz = (np.arange(A.shape[1]) + o[2]) * v
        ctr = (a.centre[0], a.centre[2])
        lab = ("X (m)", "Z (m)")
    else:
        hx = (np.arange(A.shape[0]) + o[1]) * v
        hz = (np.arange(A.shape[1]) + o[2]) * v
        ctr = (a.centre[1], a.centre[2])
        lab = ("Y (m)", "Z (m)")
    img = np.full(A.shape + (3,), (0.85, 0.85, 0.85))
    img[E] = (1, 1, 1)
    img[A & ~F] = (0.5, 0.8, 1.0)
    img[F] = (0.1, 0.1, 0.1)
    img[A & F] = (0.45, 0.45, 0.45)
    img = np.transpose(img, (1, 0, 2))[::-1]
    ax_.imshow(img, extent=[hx[0], hx[-1], hz[0], hz[-1]], origin="upper", interpolation="nearest")
    if FP is not None:
        if name == "xz":
            m = np.abs(FP[:, 1] - a.centre[1]) < 2 * v
            ax_.scatter(FP[m][:, 0], FP[m][:, 2], s=6, c="orange", marker="s", label="flagged retopo vertices")
        else:
            m = np.abs(FP[:, 0] - a.centre[0]) < 2 * v
            ax_.scatter(FP[m][:, 1], FP[m][:, 2], s=6, c="orange", marker="s", label="flagged retopo vertices")
    ax_.set_xlim(ctr[0] - a.span, ctr[0] + a.span)
    ax_.set_ylim(ctr[1] - a.span, ctr[1] + a.span)
    ax_.set_aspect("equal")
    ax_.set_xlabel(lab[0])
    ax_.set_ylabel(lab[1])
    ax_.set_title(f"{name.upper()} slice of {a.solid}")
    ax_.legend(loc="lower left", fontsize=7)
plt.tight_layout()
plt.savefig(a.out, dpi=100)
print(f"[groove_slices] -> {a.out}")
