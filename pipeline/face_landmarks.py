"""Find the eyes, nose, mouth and chin on a character's face, in 3D.

    python face_landmarks.py --dir face --out face.json

Input is blender/face_render.py's output. A body pose model (ViTPose, already used for the
skeleton) marks the nose, eyes and ears on the whole-body render - it is trained on people and
holds up on stylised ones, which a face-landmark model trained on photographs does not. Its eye
marks are refined on the close-up head render, which shows the texture itself: an eye is the
darkest compact blob near the mark, and its width and height are that blob's. The mouth is the
reddest line below the nose, its corners where the redness fades. Every point is lifted to 3D
through the head render's position map, so the rig works on the mesh directly.

Writes face.json and face_qa.png (the head render with every landmark drawn, for a person or a
model to check).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--repo", default="usyd-community/vitpose-base-simple")
ap.add_argument("--style", default="realistic", help="realistic faces have smaller eyes, set further apart")
a = ap.parse_args()

meta = json.load(open(os.path.join(a.dir, "face_render.json")))
cb, ch = meta["cameras"]["body_front"], meta["cameras"]["head_front"]
body = Image.open(os.path.join(a.dir, "body_front.png")).convert("RGBA")
head = Image.open(os.path.join(a.dir, "head_front.png")).convert("RGBA")
pos = np.load(os.path.join(a.dir, "head_front_pos.npy"))
R = ch["res"]


def px_to_xz(cam, u, v):
    half = cam["ortho_scale"] / 2
    return cam["centre"][0] + (u / cam["res"] * 2 - 1) * half, cam["centre"][2] + (1 - v / cam["res"] * 2) * half


def xz_to_px(cam, x, z):
    half = cam["ortho_scale"] / 2
    return ((x - cam["centre"][0]) / half + 1) / 2 * cam["res"], (1 - (z - cam["centre"][2]) / half) / 2 * cam["res"]


# ---- pose model on the whole body -------------------------------------------------------------
import torch  # noqa: E402
from transformers import AutoProcessor, VitPoseForPoseEstimation  # noqa: E402

bg = Image.new("RGBA", body.size, (255, 255, 255, 255))
rgb = Image.alpha_composite(bg, body).convert("RGB")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained(a.repo)
model = VitPoseForPoseEstimation.from_pretrained(a.repo).to(dev).eval()
boxes = [[[0.0, 0.0, float(body.size[0]), float(body.size[1])]]]
inp = proc(rgb, boxes=boxes, return_tensors="pt").to(dev)
with torch.no_grad():
    out = model(**inp)
kp = proc.post_process_pose_estimation(out, boxes=boxes)[0][0]
K = kp["keypoints"].cpu().numpy()
S = kp["scores"].cpu().numpy()
NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
pose = {n: (px_to_xz(cb, *K[i]), float(S[i])) for i, n in enumerate(NAMES)}

# ---- the head render: lightness and redness --------------------------------------------------
hr = np.asarray(head).astype(np.float32) / 255.0
alpha = hr[..., 3] > 0.5
rgbh = hr[..., :3]


def lab(rgb_):
    c = np.where(rgb_ <= 0.04045, rgb_ / 12.92, ((rgb_ + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


L_ = lab(rgbh)
Lum, A_ = L_[..., 0], L_[..., 1]
eyes_px = {s: xz_to_px(ch, *pose[f"{s}_eye"][0]) for s in ("left", "right")}
ied_px = abs(eyes_px["left"][0] - eyes_px["right"][0])
if ied_px < 5:
    raise SystemExit("[face] the pose model did not separate the eyes; no face rig")
# The face's midline and scale come from the ears when the model is sure of them. On aoi (anime)
# its eye and nose marks sat 1-2 cm to one side of her face and 40% too close together, while its
# ears straddled the face within a few pixels: a model trained on photographs finds a drawn
# head's outline far better than its drawn features. Eye spacing is at least a share of the head's
# width - a drawn face's eyes are larger and further apart than a photographed one's.
ears_px = {s: xz_to_px(ch, *pose[f"{s}_ear"][0]) for s in ("left", "right")}
ears_ok = min(pose["left_ear"][1], pose["right_ear"][1]) > 0.5 and abs(ears_px["left"][0] - ears_px["right"][0]) > 1.2 * ied_px
cx_ref = (ears_px["left"][0] + ears_px["right"][0]) / 2 if ears_ok else (eyes_px["left"][0] + eyes_px["right"][0]) / 2
if ears_ok:
    head_w = abs(ears_px["left"][0] - ears_px["right"][0])
    ied_px = max(ied_px, (0.42 if a.style == "realistic" else 0.55) * head_w)
ey_ref = (eyes_px["left"][1] + eyes_px["right"][1]) / 2
sgn = 1.0 if eyes_px["left"][0] > eyes_px["right"][0] else -1.0        # the character's left is image right
eyes_px = {"left": (cx_ref + sgn * ied_px / 2, ey_ref), "right": (cx_ref - sgn * ied_px / 2, ey_ref)}

# the skin level around the eyes, to judge "dark" against
cx_mid = cx_ref
cy_eye = ey_ref
skin_box = (slice(int(cy_eye + 0.25 * ied_px), int(cy_eye + 0.7 * ied_px)),
            slice(int(cx_mid - 0.35 * ied_px), int(cx_mid + 0.35 * ied_px)))
skin_L = float(np.median(Lum[skin_box][alpha[skin_box]])) if alpha[skin_box].any() else 70.0
skin_ab = np.median(L_[skin_box][alpha[skin_box]][:, 1:], 0) if alpha[skin_box].any() else np.array([10.0, 15.0])
# the hair's colour: the crown of the head, in the head render's top rows
ys_a = np.nonzero(alpha.any(1))[0]
crown = alpha & (np.arange(R)[:, None] < ys_a.min() + 0.08 * R) if len(ys_a) else alpha
hair_lab = np.median(L_[crown], 0) if crown.any() else np.array([30.0, 5.0, 10.0])

eyes = {}
for s, (ex, ey) in eyes_px.items():
    # Eyes and brows are both dark, and on a realistic face the brow is often the larger, darker
    # blob - the first version put both eye ellipses on the eyebrows. The eye is the one BELOW:
    # of the dark blobs around the pose model's mark (not much smaller than the largest), the
    # lowest. The window stops short of the nostrils.
    rx = int(0.42 * ied_px)
    # a drawn face's eyes can sit well below where the pose model marks them - it marked pip's
    # heavy cartoon brows - so the window reaches further down; the lowest blob is still the eye
    y0, y1 = int(ey - 0.45 * ied_px), int(ey + (0.38 if a.style == "realistic" else 0.8) * ied_px)
    x0, x1 = int(ex - rx), int(ex + rx)
    win = Lum[y0:y1, x0:x1]
    # dark (lashes, a realistic iris) or strongly coloured against the skin (a drawn iris) - and
    # not the hair's colour: aoi's silver-blue fringe runs into her eyes and made one blob of both
    cdist = np.linalg.norm(L_[y0:y1, x0:x1, 1:] - skin_ab, axis=-1)
    # (drawn faces only: dark hair and a photographed eye's lashes are the same colour)
    # (loosely: the hair in shadow at the side of the head is well off the lit crown's colour)
    not_hair = (np.linalg.norm(L_[y0:y1, x0:x1] - hair_lab, axis=-1) > 30) if a.style != "realistic" \
        else np.ones_like(win, bool)
    dark = ((win < skin_L - 18) | ((cdist > 28) & (win < skin_L - 4))) & alpha[y0:y1, x0:x1] & not_hair
    dark = ndimage.binary_opening(dark, iterations=1)
    lab_, nl = ndimage.label(dark)
    blobs = []
    for i in range(1, nl + 1):
        yy, xx = np.nonzero(lab_ == i)
        if len(yy) < 6:
            continue
        blobs.append((len(yy), yy.mean() + y0, xx.mean() + x0, yy, xx))
    if blobs:
        big = max(b[0] for b in blobs)
        cands_ = [b for b in blobs if b[0] > 0.15 * big and abs(b[2] - ex) < rx]
        # the lowest on a photographed face (its brow can be the bigger dark blob); the largest on a
        # drawn one, whose eyes are its biggest features and whose other dark marks - a mouth line,
        # a chin shadow - can sit lower in the window
        n_, cy, cx, yy, xx = max(cands_, key=lambda b: b[1] if a.style == "realistic" else b[0])
        if a.style != "realistic" and np.ptp(yy) > 1.3 * np.ptp(xx):
            # taller than wide: a drawn brow joined to the eye by its lashes (pip) - the eye is the
            # lower part
            low = yy > yy.min() + 0.45 * np.ptp(yy)
            yy, xx = yy[low], xx[low]
            cy, cx = yy.mean() + y0, xx.mean() + x0
        w_px = max(np.ptp(xx) + 1, 0.30 * ied_px)
        h_px = max(np.ptp(yy) + 1, 0.12 * ied_px)
    else:
        cx, cy, w_px, h_px = ex, ey, 0.4 * ied_px, 0.18 * ied_px
    touches = blobs and (yy.min() == 0 or xx.min() == 0 or yy.max() == y1 - y0 - 1 or xx.max() == x1 - x0 - 1)
    aspect = w_px / max(h_px, 1)
    # a photographed eye is wide; a drawn one can be as tall as it is wide
    min_aspect = 1.3 if a.style == "realistic" else 0.7
    # a blob running off its window is usually eye, brow and fringe merged - on a photographed face;
    # a drawn eye is simply big (aoi's and vex's filled 0.6 of the eye spacing), and the hair's colour
    # is already left out of the mask
    edge_ok = not touches or a.style != "realistic"
    good = bool(blobs) and edge_ok and min_aspect <= aspect <= 5.0 and 0.2 * ied_px <= w_px <= 0.75 * ied_px
    eyes[s] = {"px": [float(cx), float(cy)], "w_px": float(min(w_px, 0.8 * ied_px)),
               "h_px": float(min(h_px, 0.5 * ied_px)), "good": good}

# A fringe over one eye merges hair, brow and eye into one blob. Faces in a front render are
# near-symmetric, so an implausible eye (wrong size or shape, or running off its window) is
# replaced by the plausible one mirrored across the face's midline - the midpoint of the
# pose model's two eye marks, which does not depend on either refinement.
mid0 = (eyes_px["left"][0] + eyes_px["right"][0]) / 2
# Two eyes sit level. When both passed but one is far below or above the other, the one further
# from the expected eye line took something else - a collar, a chin shadow (aoi's left eye, once) -
# and is replaced by the other's mirror below.
if eyes["left"]["good"] and eyes["right"]["good"] and \
        abs(eyes["left"]["px"][1] - eyes["right"]["px"][1]) > 0.35 * ied_px:
    worse = max(("left", "right"), key=lambda s_: abs(eyes[s_]["px"][1] - ey_ref))
    eyes[worse]["good"] = False
ok_ = [s for s in ("left", "right") if eyes[s]["good"]]
if len(ok_) == 1:
    g, b_ = ok_[0], ("right" if ok_[0] == "left" else "left")
    eyes[b_] = dict(eyes[g])
    eyes[b_]["px"] = [2 * mid0 - eyes[g]["px"][0], eyes[g]["px"][1]]
    eyes[b_]["mirrored"] = True
    print(f"[face] {b_} eye hidden or merged (hair over it?) - mirrored from the {g} eye", flush=True)
elif not ok_:
    # No clean blob (a drawn eye is lashes, iris and highlight, and the fringe runs into it). The
    # estimate stands at an ordinary eye's size when it came from the head's outline - on aoi it
    # landed on both eyes - and there are no blinks when it did not (face.json "eyes_ok": false):
    # a blink on a guessed eye drags the brow and cheek with it.
    print("[face] neither eye is a clean blob - positions estimated, no blinks", flush=True)
    for s_, (ex_, ey_) in eyes_px.items():
        eyes[s_] = {"px": [float(ex_), float(ey_)], "w_px": 0.45 * ied_px, "h_px": 0.2 * ied_px, "good": False}
    # (estimated eyes stay estimates: no blinks on them - the pose model's eye height on a drawn
    # face moved 1 cm with nothing but a change of texture)
cx_mid = cx_ref if ears_ok else (eyes["left"]["px"][0] + eyes["right"]["px"][0]) / 2

# ---- the lower face, from the bottom up ----------------------------------------------------------
# The chin first: down the midline, the surface steps back from the chin to the neck by several
# centimetres - more than the step under a nose (~2 cm), so a 3 cm step is the chin. Then the nose,
# the most forward point of a narrow strip down the midline between the eyes and the chin; then
# the mouth between the nose and the chin. The order matters on a drawn face: its cheeks stand
# further forward than its small nose, the "most forward point below the eyes" was a cheek level
# with the eyes, and the mouth searched below it was drawn on the lower eyelids.
ey_mean = (eyes["left"]["px"][1] + eyes["right"]["px"][1]) / 2
col_ = int(round(cx_mid))
chin_px = None
prev_y = None
for y in range(int(ey_mean + 0.5 * ied_px), R):
    p = pos[y, col_]
    if not np.isfinite(p[0]):
        break
    # the step back to the neck: 3 cm on a sculpted face (a nose steps ~2), 2 cm on a drawn one,
    # whose small chin stands barely in front of its neck and whose nose barely stands out at all
    if prev_y is not None and p[1] - prev_y > (0.018 if a.style == "realistic" else 0.011) * meta["height"]:
        chin_px = [float(col_), float(y - 1)]
        break
    prev_y = p[1]
if chin_px is None:
    chin_px = [float(col_), float(ey_mean + 1.3 * ied_px)]
span = chin_px[1] - ey_mean
nb0, nb1 = int(ey_mean + 0.25 * span), int(ey_mean + 0.7 * span)
nw = max(2, int(0.1 * ied_px))
band = pos[nb0:nb1, col_ - nw:col_ + nw, 1]
if np.isfinite(band).any():
    iy, ix = np.unravel_index(np.nanargmin(band), band.shape)
    nx, ny = ix + col_ - nw, iy + nb0
else:
    nx, ny = col_, int(ey_mean + 0.45 * span)

# ---- mouth: the reddest (or darkest, on a drawn face) line between the nose and the chin -----------
ys = np.arange(int(ny + 0.08 * span), int(chin_px[1] - 0.15 * span))       # clear of the chin's shadow
xs = np.arange(int(cx_mid - 0.3 * ied_px), int(cx_mid + 0.3 * ied_px))
ys = ys[(ys > 0) & (ys < R)]
redness = (A_ - np.median(A_[alpha])) + 0.35 * (skin_L - Lum)
prof = np.array([np.nanmean(np.where(alpha[y, xs], redness[y, xs], np.nan)) for y in ys]) if len(ys) else np.array([])
my = int(ys[int(np.nanargmax(prof))]) if len(prof) and np.isfinite(prof).any() else int(ny + 0.4 * (chin_px[1] - ny))
row = redness[my, int(cx_mid - 0.6 * ied_px):int(cx_mid + 0.6 * ied_px)]
thr = np.nanmax(row) * (0.45 if a.style == "realistic" else 0.3)     # a drawn mouth is a faint thin line
hot = np.nonzero(row > thr)[0]
if len(hot):
    # the run of "mouth" pixels through the middle, not every red pixel in the row (a cheek blush
    # or a hair shadow at the side is not a mouth corner)
    lab_r, _ = ndimage.label(row > thr)
    mid_i = int(round(cx_mid - int(cx_mid - 0.6 * ied_px)))
    k_ = lab_r[mid_i] if 0 <= mid_i < len(row) and lab_r[mid_i] else lab_r[hot[np.argmin(np.abs(hot - mid_i))]]
    run = np.nonzero(lab_r == k_)[0]
    ml = run.min() + int(cx_mid - 0.6 * ied_px)
    mr = run.max() + int(cx_mid - 0.6 * ied_px)
else:
    ml, mr = cx_mid - 0.25 * ied_px, cx_mid + 0.25 * ied_px
mouth_px = {"centre": [float((ml + mr) / 2), float(my)], "left": [float(mr), float(my)],
            "right": [float(ml), float(my)]}                 # the character's left is image right
# The seam between the lips - where the face rig cuts the mouth open - is the darkest row across
# the middle of the mouth, near the reddest one (which lies inside a lip, not between them). On a
# drawn mouth it is the mouth line itself.
cxm = int(round((ml + mr) / 2))
half = max(2, int(0.1 * ied_px))
ys2 = np.arange(max(my - int(0.18 * ied_px), 0), min(my + int(0.18 * ied_px), R))
lum_prof = np.array([np.nanmean(np.where(alpha[y, cxm - half:cxm + half], Lum[y, cxm - half:cxm + half], np.nan))
                     for y in ys2])
sy = int(ys2[int(np.nanargmin(lum_prof))]) if np.isfinite(lum_prof).any() else my
mouth_px["seam"] = [float(cxm), float(sy)]

# ---- is this a face? ---------------------------------------------------------------------------
# Order and size checks. A face rig on wrong landmarks is worse than none: aoi's first mouth cut
# ran across her lower eyelids. When these fail the rig keeps the blinks it can trust and makes no
# mouth cut and no jaw (face.json "mouth_ok": false).
mw_px = mr - ml
checks = {"nose below the eyes": ny > ey_mean + 0.15 * ied_px,
          "mouth below the nose": my > ny + 0.05 * span,
          "chin below the mouth": chin_px[1] > my + 0.05 * span,
          "mouth width plausible": (0.25 if a.style == "realistic" else 0.12) * ied_px <= mw_px <= 1.5 * ied_px,
          "mouth near the midline": abs((ml + mr) / 2 - cx_mid) < 0.25 * ied_px}
if a.style != "realistic" and not checks["mouth width plausible"] and mw_px < 0.12 * ied_px \
        and all(v for k, v in checks.items() if k != "mouth width plausible"):
    # a drawn mouth is a short faint line and its darkest part is all that reads: where it is holds
    # (every other check passed), and it gets a small mouth's width around that centre
    c_ = (ml + mr) / 2
    ml, mr = c_ - 0.15 * ied_px, c_ + 0.15 * ied_px
    mouth_px.update({"left": [float(mr), float(my)], "right": [float(ml), float(my)]})
    checks["mouth width plausible"] = True
    print(f"[face] a drawn mouth: {mw_px * ch['ortho_scale'] / R * 100:.1f} cm read, "
          f"{0.3 * ied_px * ch['ortho_scale'] / R * 100:.1f} cm used", flush=True)
    mw_px = mr - ml
if a.style != "realistic" and not ok_:
    # on a drawn face the mouth is found below eyes and nose; with no eye found, nothing above it is
    # known either (aoi's mouth landed on her chin that way)
    checks["eyes found (a drawn face)"] = False
mouth_ok = all(checks.values())
if not mouth_ok:
    print("[face] mouth not trusted (" + ", ".join(k for k, v in checks.items() if not v) + " failed) - "
          "no mouth cut or jaw", flush=True)


def to3d(pt):
    u, v = int(round(pt[0])), int(round(pt[1]))
    for rad in range(0, 6):
        win = pos[max(v - rad, 0):v + rad + 1, max(u - rad, 0):u + rad + 1].reshape(-1, 3)
        ok = np.isfinite(win[:, 0])
        if ok.any():
            return win[ok][np.argmin(win[ok][:, 1])].tolist()   # the front-most surface point
    return None


pxs = ch["ortho_scale"] / R
face = {"eyes": {}, "ied_m": ied_px * pxs, "scores": {n: pose[n][1] for n in NAMES}, "mouth_ok": bool(mouth_ok),
        "eyes_ok": bool(ok_)}
for s, e in eyes.items():
    face["eyes"][s] = {"centre": to3d(e["px"]), "width_m": e["w_px"] * pxs, "height_m": e["h_px"] * pxs}
face["mouth"] = {k: to3d(v) for k, v in mouth_px.items()}
face["mouth"]["width_m"] = (mouth_px["left"][0] - mouth_px["right"][0]) * pxs
face["nose"] = to3d([nx, ny])
face["chin"] = to3d(chin_px)
face["ears"] = {s: [float(v) for v in pose[f"{s}_ear"][0]] for s in ("left", "right")}
bad = [k for k in ("nose", "chin") if face[k] is None] + [s for s in face["eyes"] if face["eyes"][s]["centre"] is None]
if bad or face["mouth"]["centre"] is None:
    raise SystemExit(f"[face] landmarks off the surface: {bad}")
json.dump(face, open(a.out, "w"), indent=1, default=float)

qa = head.convert("RGB")
d = ImageDraw.Draw(qa)
for s, e in eyes.items():
    x, y = e["px"]
    d.ellipse([x - e["w_px"] / 2, y - e["h_px"] / 2, x + e["w_px"] / 2, y + e["h_px"] / 2], outline=(0, 255, 120), width=2)
for k, (x, y) in mouth_px.items():
    d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 60, 60), width=2)
d.line([mouth_px["right"][0], my, mouth_px["left"][0], my], fill=(255, 60, 60), width=1)
for (x, y), c in (((nx, ny), (60, 160, 255)), (tuple(chin_px), (255, 200, 0))):
    d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=c, width=2)
qa.save(os.path.join(os.path.dirname(a.out), "face_qa.png"))
print(f"[face] eyes {face['eyes']['left']['width_m'] * 100:.1f} x {face['eyes']['left']['height_m'] * 100:.1f} cm, "
      f"{face['ied_m'] * 100:.1f} cm apart; mouth {face['mouth']['width_m'] * 100:.1f} cm wide; "
      f"pose scores nose {pose['nose'][1]:.2f} eyes {pose['left_eye'][1]:.2f}/{pose['right_eye'][1]:.2f}", flush=True)
