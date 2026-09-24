"""Turn a generated mesh's volume into one clean solid: flakes closed, shards gone, no cavities.

Input is the distance volume blender/sdf_io.py writes. What comes out of the generator is not a
surface a modeller would recognise:

* hair arrives as dozens of thin overlapping flakes with gaps between them (Vex: 11k hair
  vertices spread over flakes a few millimetres apart, scalp visible through every gap);
* the shell is double-walled - 59% of Vex's vertices can never be seen from outside - with
  shirts modelled under jackets and air between them;
* loose shards float off sleeves, collars and hair.

There is no reliable "inside" to start from. OpenVDB's level set is only signed when the mesh
is closed, and these leak: on Juno 99.98% of the volume came back positive, the middle of the
torso included, so every earlier remesh of these characters was a thickened shell with a second
surface hidden inside it. So inside is defined the way a game asset needs it: a voxel is outside
if it can see out - if a straight line from it leaves the character along at least a few of 26
directions (axes, edge diagonals, corner diagonals). A leak lets an interior voxel escape along
one or two of them; a real concavity - an armpit, the gap between the legs, between the fingers
- is seen along many. Everything that cannot see out is solid: double walls, the shirt under the
jacket, the air between them. The thin shell around every triangle stays solid as well, so a
single-sided sheet survives.

Then three repairs that do not care how the triangles were connected. Hair gets a morphological
closing (dilate, then erode by the same radius): gaps narrower than twice the radius fill and
the envelope stays where it was - only where hair is the nearest part, so bangs do not close
over the eyes. Everywhere else a much smaller closing seals cracks and keeps folds. Pieces are
kept only if large, so a shard goes and a separate satchel stays.

Run: python solidify.py --sdf sdf.npz --mesh mesh.glb --parts parts.json --out solid.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import trimesh
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--sdf", required=True)
ap.add_argument("--mesh", required=True, help="the mesh the parts labels index (glTF, Y up)")
ap.add_argument("--parts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--hair-close", type=float, default=0.006,
                help="closing radius in the hair, as a fraction of height (~1 cm on a person)")
ap.add_argument("--base-close", type=float, default=0.0012,
                help="closing radius everywhere else (~2 mm) - seals cracks, keeps folds")
ap.add_argument("--min-piece", type=float, default=0.002,
                help="drop pieces smaller than this share of the character's volume")
ap.add_argument("--min-escapes", type=int, default=3,
                help="a voxel is outside if it sees out along at least this many of 26 directions")
ap.add_argument("--smooth", type=float, default=0.8, help="Gaussian sigma on the distance, voxels")
a = ap.parse_args()

d = np.load(a.sdf)
v = float(d["voxel"])
o = d["origin_ijk"].astype(np.int64)
H = float(d["height"])
U = np.abs(d["sdf"])                                # unsigned in practice - see above
shape = U.shape
shell = U < 0.75 * v                                # every surface, thickened to ~1.5 voxels
print(f"[solidify] volume {shape}, voxel {v*1000:.2f} (height {H:.3f}); "
      f"{(d['sdf'] < 0).mean():.3%} of voxels came back signed inside", flush=True)
del d


def escapes(blocked, dvec):
    """True where a ray from the voxel along integer direction dvec leaves the grid unobstructed."""
    ax = int(np.argmax(np.abs(dvec)))               # march along an axis the direction moves on
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
            # the next voxel along the ray is one slice on, shifted by the other components;
            # whatever is shifted in from beyond the grid has escaped
            nxt = np.ones_like(prev)
            nxt[dst] = prev[src]
            cur = free & nxt
        E[sl] = cur
        prev = cur
    return E


dirs = [np.array((x, y, z)) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
        if (x, y, z) != (0, 0, 0)]
count = np.zeros(shape, np.uint8)
for dv in dirs:
    count += escapes(shell, dv)
exterior = (count >= a.min_escapes) & ~shell
leak = int(((count > 0) & (count < a.min_escapes) & ~shell).sum())
del count
occ = ~exterior
print(f"[solidify] outside (seen along >= {a.min_escapes} of 26 directions): {exterior.mean():.2%} of "
      f"the box; {leak:,} voxels seen along fewer, taken as leaks; solid {occ.mean():.2%}", flush=True)


def ball(r):
    r = max(1, int(round(r)))
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (g ** 2).sum(0) <= r * r + 0.5


def to_ijk(P):
    return np.round(P / v).astype(np.int64) - o


# ---- the hair region, from the part labels ---------------------------------------------------
# Closing is only safe where both sides of a gap are hair. Bangs lie over the brow, so "near a
# hair vertex" reaches the eyes, and the first version closed the eye sockets under a veil. So
# every voxel takes the label of the nearest labelled vertex, and the hair closing acts only
# where hair is nearest AND no other part is within a margin.
mesh = trimesh.load(a.mesh, force="mesh", process=False)
parts = json.load(open(a.parts))
lab = np.asarray(parts["labels"])
if len(lab) != len(mesh.vertices):
    raise SystemExit(f"[solidify] {len(lab)} labels for {len(mesh.vertices)} vertices")
Vgl = np.asarray(mesh.vertices)
Vbl = np.stack([Vgl[:, 0], -Vgl[:, 2], Vgl[:, 1]], axis=1)       # glTF Y-up -> Blender Z-up
hair_id = parts["group_ids"]["hair"]
ijk = to_ijk(Vbl)
ok = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
seed_hair = np.zeros(shape, bool)
seed_other = np.zeros(shape, bool)
seed_hair[tuple(ijk[ok & (lab == hair_id)].T)] = True
seed_other[tuple(ijk[ok & (lab != hair_id)].T)] = True
seed_hair &= ~seed_other                            # a voxel holding both is not hair's to close
rc_h = a.hair_close * H / v
rc_b = a.base_close * H / v
if seed_hair.any():
    d_h = ndimage.distance_transform_edt(~seed_hair)
    d_o = ndimage.distance_transform_edt(~seed_other)
    margin = max(2.0, 0.6 * rc_h)
    hair_zone = (d_h < d_o) & (d_o > margin) & (d_h < 2.5 * rc_h)
    del d_h, d_o
else:
    hair_zone = np.zeros(shape, bool)
print(f"[solidify] hair: {int((lab == hair_id).sum()):,} vertices, closing zone {hair_zone.mean():.3%} "
      f"of the box, radius {rc_h:.1f} voxels", flush=True)

# ---- closing -------------------------------------------------------------------------------------
pad = int(np.ceil(max(rc_h, rc_b))) + 2


def close(m, r):
    mp = np.pad(m, pad)
    st = ball(r)
    mp = ndimage.binary_dilation(mp, structure=st)
    mp = ndimage.binary_erosion(mp, structure=st, border_value=0)
    return mp[pad:-pad, pad:-pad, pad:-pad]


closed = occ.copy()
if hair_zone.any():
    zi = np.argwhere(hair_zone)                     # the big closing only inside the hair's box
    lo_, hi_ = zi.min(0), zi.max(0) + 1
    sl = tuple(slice(l, h) for l, h in zip(lo_, hi_))
    closed[sl] = np.where(hair_zone[sl], close(occ[sl], rc_h) | occ[sl], occ[sl])
closed = close(closed, rc_b) | closed
added = int(closed.sum() - occ.sum())

# ---- loose pieces --------------------------------------------------------------------------------
lbl, n = ndimage.label(closed, structure=np.ones((3, 3, 3)))
sizes = np.bincount(lbl.ravel())
sizes[0] = 0
keep = sizes >= a.min_piece * sizes.sum()
keep[0] = False
dropped = int(n - keep.sum())
dropped_vox = int(sizes[~keep].sum())
solid = keep[lbl]
del lbl
print(f"[solidify] closing added {added:,} voxels; {dropped:,} loose pieces dropped "
      f"({dropped_vox:,} voxels), {int(keep.sum())} kept", flush=True)

# ---- back to a distance field -------------------------------------------------------------------
# The source field is unsigned, so the signed one comes from the solid itself. Re-distancing a
# binary volume quantises the surface to voxels; sdf_io.py then puts each vertex back on the
# source surface wherever that surface is within reach, which restores the sub-voxel detail.
din = ndimage.distance_transform_edt(solid)
dout = ndimage.distance_transform_edt(~solid)
dist = np.where(solid, -(din - 0.5), dout - 0.5).astype(np.float32)
del din, dout
if a.smooth > 0:
    dist = ndimage.gaussian_filter(dist, a.smooth)
dist *= v
band = 4 * v
dist = np.clip(dist, -band * 3, band * 3)
np.savez_compressed(a.out, sdf=dist, origin_ijk=o, voxel=v, background=float(band * 3), height=H)
print(f"[solidify] -> {a.out}", flush=True)
