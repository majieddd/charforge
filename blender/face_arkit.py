"""ARKit-named face shapes for the face rig: the names engines and face-capture apps drive.

face_rig.py calls build() after its own six shapes, from the same landmarks, in the same units: the
eye distance (ied) for the face and brows, the eye's height for the lids, the mouth's width for the
mouth. build() returns {ARKit name: displacement (n, 3), world metres}; face_rig.py adds each as a
shape key. Left and right are the subject's: the subject's left is +X (the landmarks' "left" eye and
mouth corner sit at +X), forward is -Y, up is +Z.

Every field is smooth: smoothstep falloffs and squared-clip radii only, and no sign() or hard cut
between two regions that move apart, so a shape stretches and folds a region instead of tearing the
mesh along a line. The front mask is soft for the same reason. Eye and brow shapes need the eyes the
landmark finder trusted (eyes_ok), mouth and jaw shapes the mouth it trusted (mouth_ok).

Not built: mouthClose. The rest mouth is already closed, so its only motion would undo jawOpen's
lip motion - which, driven alone, pushes the lower lip through the upper. Shapes that need a
lip-to-lip contact are left to the engine's combination of jawOpen and the lips' own geometry.

jawOpen is exact linear skinning of the jaw bone: the jaw-weighted vertices rotated about the bone's
head and local X by the jaw's open angle (face_rig.py's jaw_open_deg), so the morph and the bone agree.
Driving both doubles the opening; a game drives one or the other.
"""
import numpy as np


def ss(t):
    """Smoothstep: 0 at or below 0, 1 at or above 1."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _vec(n, x=None, y=None, z=None):
    """An (n, 3) displacement with the given per-vertex components (None is zero)."""
    D = np.zeros((n, 3))
    for k, c in enumerate((x, y, z)):
        if c is not None:
            D[:, k] = c
    return D


def _corner(V, ff, C, r):
    """A soft disc around C in the x-z plane (the face's plane), radius r, times the front mask."""
    d = np.linalg.norm((V - C)[:, [0, 2]], axis=1)
    return np.clip(1.0 - (d / r) ** 2, 0.0, 1.0) ** 2 * ff


def _eye(V, ff, ied, e):
    """Per-vertex eye coordinates: u across the eye (half-width units), v up it (half-height units),
    and the mask that is 1 on the eye's surround and fades out with distance and depth."""
    E = np.asarray(e["centre"], float)
    w = float(e["width_m"])
    h = max(float(e["height_m"]), 0.28 * w)
    u = (V[:, 0] - E[0]) / (0.5 * w)
    v = (V[:, 2] - E[2]) / (0.5 * h)
    fu = np.clip(1.0 - (np.abs(u) / 1.15) ** 2, 0.0, 1.0) ** 0.7
    dep = ss((0.45 * ied - np.abs(V[:, 1] - E[1])) / (0.2 * ied))
    return E, h, v, fu * dep * ff


def _upper(v):
    """The upper lid and the skin above it: 0 below the eye's middle, 1 at its top edge, fading by the brow."""
    return ss(v / 1.2) * ss((3.4 - v) / 2.4)


def _lower(v):
    """The lower lid's margin, a band a little below the eye."""
    return ss((-0.2 - v) / 0.9) * ss((v + 1.9) / 0.9)


def _band(V, E, h, ied, reach=0.55, width=0.4):
    """The band a brow lies in, above an eye: 1 at its middle, falling to 0 at reach (units of 0.75 ied in height)."""
    t = (V[:, 2] - (E[2] + 0.45 * h)) / (0.75 * ied)
    return ss((reach - np.abs(t - 0.45)) / width)


def _rot(P, axis, t):
    """Rodrigues' rotation of the rows of P about a unit axis by t radians."""
    c, s = np.cos(t), np.sin(t)
    return P * c + np.cross(axis, P) * s + np.outer(P @ axis, axis) * (1.0 - c)


SIDES = (("left", "Left", 1.0), ("right", "Right", -1.0))


def build(V, N, Whead, F, ied, wj, hinge, axis, deg, eyes_ok, mouth_ok, blink=None, pucker=None):
    """The ARKit shapes the landmarks support, in the order they are added.

    V, N      world vertex positions and normals (n, 3); Whead the head bone's weight (n,)
    F         face_landmarks.py's dict (eyes, mouth, nose, chin)
    wj        the jaw bone's weight per vertex (n,), zero without a jaw
    hinge     the jaw bone's head (world), axis its local X (world, unit), deg its open angle
    blink     {"left": D, "right": D} - face_rig.py's blink shapes, aliased to eyeBlink*
    pucker    face_rig.py's pucker shape, aliased to mouthPucker
    """
    n = len(V)
    ff = ss((Whead - 0.2) / 0.3) * ss((0.4 - N[:, 1]) / 0.4)      # the head's front, softly
    out = {}
    if eyes_ok:
        eyes = {s: _eye(V, ff, ied, F["eyes"][s]) for s, _, _ in SIDES}
        if blink is not None:
            out["eyeBlinkLeft"] = blink["left"]
            out["eyeBlinkRight"] = blink["right"]
        for s, tag, sg in SIDES:
            E, h, v, m = eyes[s]
            out[f"eyeWide{tag}"] = _vec(n, z=(0.45 * h * _upper(v) - 0.15 * h * _lower(v)) * m)
        for s, tag, sg in SIDES:
            E, h, v, m = eyes[s]
            out[f"eyeSquint{tag}"] = _vec(n, z=(-0.30 * h * _upper(v) + 0.18 * h * _lower(v)) * m)
        for s, tag, sg in SIDES:
            E = np.asarray(F["eyes"][s]["centre"], float)
            h = max(float(F["eyes"][s]["height_m"]), 0.28 * float(F["eyes"][s]["width_m"]))
            Bd = _band(V, E, h, ied) * ff * ss((0.5 * ied - np.abs(V[:, 1] - E[1])) / (0.2 * ied))
            Xp = ss((0.6 * ied - np.abs(V[:, 0] - E[0])) / (0.25 * ied))         # this side's brow
            out[f"browDown{tag}"] = _vec(n, z=-0.06 * ied * Bd * Xp)   # no inward push: on Aoi it folded the inner brow (real flips 13 -> 2 on the left)
        EL, ER = np.asarray(F["eyes"]["left"]["centre"], float), np.asarray(F["eyes"]["right"]["centre"], float)
        hm = 0.5 * (max(float(F["eyes"]["left"]["height_m"]), 0.28 * float(F["eyes"]["left"]["width_m"]))
                    + max(float(F["eyes"]["right"]["height_m"]), 0.28 * float(F["eyes"]["right"]["width_m"])))
        Em = 0.5 * (EL + ER)
        Xi = ss((0.3 * ied - np.abs(V[:, 0] - 0.5 * (EL[0] + ER[0]))) / (0.2 * ied))  # between the brows
        Bw = _band(V, Em, hm, ied, reach=0.8, width=0.5) * ff * ss((0.5 * ied - np.abs(V[:, 1] - Em[1])) / (0.2 * ied))
        out["browInnerUp"] = _vec(n, z=0.09 * ied * Bw * Xi)
        for s, tag, sg in SIDES:
            E = np.asarray(F["eyes"][s]["centre"], float)
            h = max(float(F["eyes"][s]["height_m"]), 0.28 * float(F["eyes"][s]["width_m"]))
            Bd = _band(V, E, h, ied) * ff * ss((0.5 * ied - np.abs(V[:, 1] - E[1])) / (0.2 * ied))
            Xo = ss((0.35 * ied - np.abs(V[:, 0] - (E[0] + sg * 0.5 * ied))) / (0.2 * ied))   # the outer brow
            out[f"browOuterUp{tag}"] = _vec(n, z=0.07 * ied * Bd * Xo)
    if mouth_ok:
        mc = np.asarray(F["mouth"]["centre"], float)
        C = {"left": np.asarray(F["mouth"]["left"], float), "right": np.asarray(F["mouth"]["right"], float)}
        mw = float(np.linalg.norm(C["left"] - C["right"]))
        for s, tag, sg in SIDES:
            f = _corner(V, ff, C[s], 0.62 * mw)
            out[f"mouthSmile{tag}"] = _vec(n, x=sg * 0.09 * mw * f, z=0.11 * mw * f)
        for s, tag, sg in SIDES:
            f = _corner(V, ff, C[s], 0.6 * mw)
            out[f"mouthFrown{tag}"] = _vec(n, x=sg * 0.04 * mw * f, z=-0.10 * mw * f)
        fM = _corner(V, ff, mc, 0.75 * mw)        # wide enough to take in the corners, which sit at 0.5 mw
        out["mouthFunnel"] = _vec(n, x=-0.35 * (V[:, 0] - mc[0]) * fM, y=-0.10 * mw * fM)
        for s, tag, sg in SIDES:
            f = _corner(V, ff, C[s], 0.65 * mw)
            out[f"mouthStretch{tag}"] = _vec(n, x=sg * 0.12 * mw * f, z=-0.01 * mw * f)
        if pucker is not None:
            out["mouthPucker"] = pucker
        D_cheek = np.zeros((n, 3))
        for s, tag, sg in SIDES:
            f = _corner(V, ff, C[s] + np.array([sg * 0.15 * mw, 0.0, 0.35 * mw]), 0.6 * mw)
            D_cheek += _vec(n, x=sg * 0.03 * mw * f, y=-0.12 * mw * f)
        out["cheekPuff"] = D_cheek
        nose = np.asarray(F["nose"], float)
        for s, tag, sg in SIDES:
            fn = _corner(V, ff, nose + np.array([sg * 0.2 * ied, 0.0, 0.0]), 0.35 * ied)
            fl = _corner(V, ff, mc + np.array([sg * 0.25 * mw, 0.0, 0.12 * mw]), 0.35 * mw)
            out[f"noseSneer{tag}"] = _vec(n, x=sg * 0.04 * mw * fn, z=0.05 * mw * fn + 0.06 * mw * fl)
        if axis is not None and np.linalg.norm(axis) > 0:
            ax = np.asarray(axis, float) / np.linalg.norm(axis)
            hg = np.asarray(hinge, float)
            P = V - hg
            t = np.deg2rad(deg)
            chin_p = np.asarray(F["chin"], float) - hg
            if (_rot(chin_p[None], ax, t) - chin_p[None])[0, 2] > 0:      # the chin drops when the jaw opens
                t = -t
            out["jawOpen"] = (_rot(P, ax, t) - P) * np.asarray(wj, float)[:, None]
    return out
