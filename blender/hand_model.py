"""A procedural hand: palm, four fingers and a thumb as a smooth signed distance field.

Generators fuse fingers into a paddle, and no amount of reweighting animates a paddle. So the
generated hand is replaced by one built the way a modeller blocks a hand out: capsules for the
bones, blended so the palm, knuckles and webbing come out of the blend rather than being
sculpted. Everything is a proportion of the hand's length, measured on the character, so the
same code makes a large realistic hand and a small stylised one.

The hand is modelled in the rest convention the animation library uses (Mixamo's T-pose):
fingers straight and slightly spread, thumb forward and down at about 40 degrees, palm facing
the floor once the arm is raised. Straight fingers are what the finger clips' rotations are
measured from; a curled rest would add the curl to every pose.

Local frame, right hand: x along the hand from the wrist, y toward the thumb, z out of the back
of the hand. A left hand is the mirror image, produced by mapping this frame with a left-handed
basis (see place()).

This module is imported by blender/hands.py; it has no Blender dependency except in mesh().
"""
from __future__ import annotations

import numpy as np

# name, knuckle position (x, y) along/across the palm, spread (deg), phalanx lengths, base radius
# - all as fractions of the hand length (wrist crease to middle fingertip). Proportions from
# anthropometric tables (finger length ratios ~0.88 : 1 : 0.93 : 0.74, phalanges ~ 5 : 3 : 2).
FINGERS = [
    ("index",  (0.455,  0.128),  3.0, (0.210, 0.125, 0.095), 0.056),
    ("middle", (0.470,  0.043),  0.0, (0.230, 0.140, 0.100), 0.058),
    ("ring",   (0.460, -0.043), -3.0, (0.215, 0.130, 0.095), 0.054),
    ("pinky",  (0.425, -0.124), -7.0, (0.165, 0.100, 0.085), 0.047),
]
THUMB_BASE = (0.090, 0.100, -0.030)                  # carpometacarpal joint
THUMB = (0.200, 0.150, 0.125)                        # metacarpal, proximal, distal
THUMB_RADIUS = 0.062


def smin(a, b, k):
    """Polynomial smooth minimum - a blend of radius k between two distance fields."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def capsule(p, a, b, ra, rb):
    """Round cone from a (radius ra) to b (radius rb)."""
    pa, ba = p - a, b - a
    L2 = float(ba @ ba)
    t = np.clip((pa @ ba) / max(L2, 1e-12), 0.0, 1.0)
    r = ra + (rb - ra) * t
    return np.linalg.norm(pa - t[:, None] * ba, axis=1) - r


def ellipsoid(p, c, r):
    """Approximate distance to an axis-aligned ellipsoid (radii r)."""
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / r, axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])


def rot_y(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def skeleton(width_scale=1.0, thickness_scale=1.0):
    """Joint chains in the unit local frame: {finger: [base, j1, j2, tip]} plus radii."""
    chains, radii = {}, {}
    for name, (kx, ky), spread, lens, r0 in FINGERS:
        base = np.array([kx, ky * width_scale, 0.0])
        d = rot_z(spread) @ np.array([1.0, 0.0, 0.0])
        pts = [base]
        for L in lens:
            pts.append(pts[-1] + d * L)
        chains[name] = pts
        radii[name] = [r0 * s for s in (1.0, 0.93, 0.86, 0.78)]
    # thumb: out toward the thumb side and down toward the palm, then its own slight bend
    tb = np.array([THUMB_BASE[0], THUMB_BASE[1] * width_scale, THUMB_BASE[2] * thickness_scale])
    d = rot_z(30.0) @ rot_y(22.0) @ np.array([1.0, 0.0, 0.0])
    pts = [tb]
    for i, L in enumerate(THUMB):
        di = rot_z(-6.0 * i) @ d
        pts.append(pts[-1] + di * L)
    chains["thumb"] = pts
    radii["thumb"] = [THUMB_RADIUS * s for s in (1.15, 0.92, 0.86, 0.78)]
    return chains, radii


def sdf(p, wrist_r=(0.15, 0.10), width_scale=1.0, thickness_scale=1.0, stub=0.30):
    """Signed distance of the hand at points p (N x 3, unit hand length, local frame).

    wrist_r: the wrist's half-width (y) and half-thickness (z); the stub reaches back `stub`
    into the forearm so the hand can be blended or joined to the arm."""
    chains, radii = skeleton(width_scale, thickness_scale)
    t = thickness_scale
    # wrist and forearm stub: an elliptical cylinder, tapering into the palm
    q = p.copy()
    q[:, 1] /= wrist_r[0] / wrist_r[1]
    d = capsule(q, np.array([-stub, 0, 0]), np.array([0.10, 0, 0]), wrist_r[1], wrist_r[1] * 1.02)
    d *= min(1.0, wrist_r[1] / wrist_r[0])
    # the palm: metacarpals from the carpus to each knuckle, blended wide and flat
    for name, (kx, ky), _, _, r0 in FINGERS:
        a = np.array([0.08, ky * width_scale * 0.55, 0.0])
        b = chains[name][0]
        q = p.copy()
        q[:, 2] /= 0.72 * t                          # flatten the palm front-to-back
        dm = capsule(q, a, b, r0 * 1.25, r0 * 1.12) * 0.72 * t
        d = smin(d, dm, 0.055)
    # palm pads: thenar (thumb side) and hypothenar (little finger side) on the palm face
    d = smin(d, ellipsoid(p, np.array([0.19, 0.08 * width_scale, -0.045 * t]),
                          np.array([0.13, 0.075, 0.045 * t])), 0.03)
    d = smin(d, ellipsoid(p, np.array([0.26, -0.10 * width_scale, -0.035 * t]),
                          np.array([0.14, 0.05, 0.04 * t])), 0.03)
    # fingers: three phalanges each, knuckles slightly proud
    for name, pts in chains.items():
        rs = radii[name]
        k_join = 0.07 if name == "thumb" else 0.024
        for i in range(3):
            q = p.copy()
            q[:, 2] /= 0.92 * t                      # fingers are a little flatter than round
            dc = capsule(q, pts[i], pts[i + 1], rs[i], rs[i + 1]) * 0.92 * t
            d = smin(d, dc, k_join if i == 0 else 0.012)
        for i in (1, 2):                             # knuckles: barely proud, well blended
            d = smin(d, np.linalg.norm(p - pts[i], axis=1) - rs[i] * 1.015, 0.016)
    return d, chains


def place(frame_x, frame_y, frame_z, origin, length, mirror):
    """World transform for the unit local frame. mirror=True for a left hand: the basis is
    left-handed, which reflects the right-hand model into a left hand."""
    B = np.stack([frame_x, frame_y, frame_z], axis=1) * length
    return B, np.asarray(origin, float), mirror
