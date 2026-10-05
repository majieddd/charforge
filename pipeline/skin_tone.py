"""One skin, one tone: lighting taken out of the skin's broad light and dark, its fine detail kept (E138).

    python pipeline/skin_tone.py --albedo albedo.png --dir texproj --out albedo.png [--strength 0.9] [--radius 0.004]
        [--hands hands_spec.json] [--labels labels.json --retopo retopo.glb]

The texture's skin comes from three places - the reference projected onto the front, with the picture's light and
shadow in it; the generator's own colours on the sides and back; and one flat tone on the modelled hands - so the
same skin is three colours: a ring at every bare wrist (Mara's forearm L* 50 against her hand's 69), and on the face
the picture's shadows land on geometry that is not shaped like the picture's (a smile line across Boyscout's cheek,
dark patches under Mara's eyes). A professionally painted skin is one tone with its detail on it.

Skin texels are found by colour - near the character's skin tone (read from the face, as the hands' is) in hue and
chroma, at a lightness within a band of it, so that shading counts and brows, lashes, nostrils, lips and the whites
of the eyes do not - and, with --labels, only on the parts the parser called skin (face, arms, hands, legs, torso:
brown boots and khaki are skin-coloured too). Their local colour is measured in 3D, not in the atlas (one cheek's
islands lie apart there): a normalised Gaussian of skin colour over --radius on a voxel grid. Each skin texel has
that local colour replaced by the skin tone, by --strength, and keeps its own difference from it. At the default
7 mm the shading of the picture (1-2 cm patches) goes and freckles and pores (2-3 mm) stay. Skipped on a grey or
unsaturated skin tone, whose hue cannot tell skin from cloth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--albedo", required=True)
ap.add_argument("--dir", required=True, help="uv_maps.py output: uv_position.npy")
ap.add_argument("--out", required=True)
ap.add_argument("--strength", type=float, default=0.9, help="how much of the local variation goes (0-1)")
ap.add_argument("--radius", type=float, default=0.004, help="the local scale, as a fraction of the height (7 mm at 1.75 m)")
ap.add_argument("--labels", default=None, help="labels.json (transfer_labels.py): skin only on the parts it calls skin")
ap.add_argument("--retopo", default=None, help="retopo.glb, whose vertices labels.json numbers")
ap.add_argument("--hands", default=None, help="hands_spec.json: the modelled hands' tone is the target's anchor")
ap.add_argument("--tone", default=None, help="r,g,b (0-255): the skin tone, if not read from the texture")
ap.add_argument("--keep", default=None, help="a mask over the atlas (0-255) of what to leave as it is: "
                                             "project_texture.py's face_keep.png, the face as the picture painted it")
a = ap.parse_args()


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


img = np.asarray(Image.open(a.albedo).convert("RGB"), np.float32) / 255
pos = np.load(os.path.join(a.dir, "uv_position.npy"))
if pos.shape[:2] != img.shape[:2]:
    raise SystemExit(f"[skin] uv maps are {pos.shape[0]}^2 but the albedo is {img.shape[0]}^2")
cover = np.isfinite(pos[..., 0])
ti = np.nonzero(cover.ravel())[0]
P = pos.reshape(-1, 3)[ti].astype(np.float64)
C = img.reshape(-1, 3)[ti].astype(np.float64)
lab = srgb_to_lab(C)
H = float(np.percentile(P[:, 2], 99.5) - np.percentile(P[:, 2], 0.5))

# the skin tone: given, or the modelled hands' (the texture stage paints them in the skin tone read from the face)
tone_lab = None
if a.tone:
    tone_lab = srgb_to_lab(np.array([float(x) for x in a.tone.split(",")]) / 255)
elif a.hands and os.path.exists(a.hands):
    hand = np.zeros(len(P), bool)
    for sd in json.load(open(a.hands))["sides"].values():
        c0, x, L = np.array(sd["cut_point"]), np.array(sd["x"]), float(sd["length"])
        t = (P - c0) @ x
        hand |= (t > 0.25 * L) & (t < 0.8 * L) & (np.linalg.norm(P - c0, axis=1) < 1.2 * L)
    if hand.sum() > 500:
        tone_lab = np.median(lab[hand], 0)
if tone_lab is None:
    warm = (lab[:, 0] > 40) & (lab[:, 0] < 90) & (lab[:, 1] > 6) & (lab[:, 1] < 32) & (lab[:, 2] > 8) & (lab[:, 2] < 38)
    if warm.sum() < 500:
        raise SystemExit("[skin] no skin tone found")
    tone_lab = np.median(lab[warm], 0)

# skin by hue and chroma near the tone's, lightness within a wide band of it (shading), but not the dark of brows,
# lashes and nostrils nor the white of an eye
if np.hypot(tone_lab[1], tone_lab[2]) < 8:
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(a.out)
    print(f"[skin] tone L*a*b* ({tone_lab[0]:.0f}, {tone_lab[1]:.0f}, {tone_lab[2]:.0f}) is grey: left as it is", flush=True)
    sys.exit(0)
hue = np.arctan2(lab[:, 2], lab[:, 1])
chroma = np.hypot(lab[:, 1], lab[:, 2])
t_hue = np.arctan2(tone_lab[2], tone_lab[1])
t_chroma = np.hypot(tone_lab[1], tone_lab[2])
dh = np.abs(np.angle(np.exp(1j * (hue - t_hue))))
skin = ((dh < np.radians(14)) & (chroma > 0.55 * t_chroma) & (chroma < 1.6 * t_chroma)
        & (lab[:, 0] > tone_lab[0] - 30) & (lab[:, 0] < tone_lab[0] + 18))
if a.labels and a.retopo and os.path.exists(a.labels) and os.path.exists(a.retopo):
    import trimesh
    from scipy.spatial import cKDTree
    lb = json.load(open(a.labels))
    mesh = trimesh.load(a.retopo, force="mesh", process=False)
    V = np.asarray(mesh.vertices, float)
    if len(V) == lb["vertices"]:
        V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)          # glTF's Y up, as Blender imported it for uv_maps.py
        names = {int(k): v for k, v in lb["class_names"].items()}
        body = np.array([g == 0 or names.get(c) in ("face", "arms", "hands", "legs", "torso")
                         for g, c in zip(lb["labels"], lb["classes"])])
        cand = np.nonzero(skin)[0]
        d_, j_ = cKDTree(V).query(P[cand])
        off = np.median(d_)
        skin[cand[~body[j_]]] = False
        print(f"[skin] {int((~body[j_]).sum()):,} skin-coloured texels on clothing, hair or accessories left out "
              f"(texel to labelled vertex: median {off:.4f})", flush=True)
    else:
        print(f"[skin] labels.json numbers {lb['vertices']} vertices, retopo.glb has {len(V)}: labels not used", flush=True)
# a texel is weighed by how sure it is skin: full inside the bands, fading at their edges
wt = (np.clip(1 - (dh - np.radians(9)) / np.radians(5), 0, 1)
      * np.clip((lab[:, 0] - (tone_lab[0] - 30)) / 8, 0, 1) * skin)

# broad colour of the skin, in 3D: normalised Gaussian on a voxel grid
vox = a.radius * H / 2.5
lo = P.min(0) - 3 * vox
idx = np.floor((P - lo) / vox).astype(int)
shape = idx.max(0) + 4
flat = np.ravel_multi_index(idx.T, shape)
sig = a.radius * H / vox
den = ndimage.gaussian_filter(np.bincount(flat, wt, int(np.prod(shape))).reshape(shape), sig)
broad = np.zeros((len(P), 3))
for k in range(3):
    num = ndimage.gaussian_filter(np.bincount(flat, wt * lab[:, k], int(np.prod(shape))).reshape(shape), sig)
    field = num / np.maximum(den, 1e-9)
    broad[:, k] = ndimage.map_coordinates(field, ((P - lo) / vox - 0.5).T, order=1, mode="nearest")
have = ndimage.map_coordinates(den, ((P - lo) / vox - 0.5).T, order=1, mode="nearest") > 1e-3
# the local skin colour, replaced by the tone; the texel's own difference from it kept
s = a.strength * wt * have
if a.keep and os.path.exists(a.keep):
    kp_ = np.asarray(Image.open(a.keep).convert("L").resize(img.shape[1::-1]), np.float32).reshape(-1)[ti] / 255.0
    s = s * (1 - kp_)
    print(f"[skin] {int((kp_ > 0.5).sum()):,} texels of the face left as the picture painted them", flush=True)
new = lab + s[:, None] * (tone_lab[None] - broad)
out_c = C.copy()
sel = s > 0.01
out_c[sel] = lab_to_srgb(new[sel])
out = img.reshape(-1, 3).copy()
out[ti] = out_c
Image.fromarray((np.clip(out.reshape(img.shape), 0, 1) * 255 + 0.5).astype(np.uint8)).save(a.out)
moved = np.linalg.norm((new - lab)[sel], axis=1)
print(f"[skin] tone L*a*b* ({tone_lab[0]:.0f}, {tone_lab[1]:.0f}, {tone_lab[2]:.0f}); {int(sel.sum()):,} skin texels "
      f"({sel.mean():.1%} of the surface) brought toward it, by a median {np.median(moved):.1f} and p95 "
      f"{np.quantile(moved, 0.95):.1f} L*a*b* units -> {a.out}", flush=True)
