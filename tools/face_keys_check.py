"""Do the face rig's shapes move the face's landmarks the way their names say? (DWPose, 68 points.)

    vendor/trellis2mlx/.venv/bin/python tools/face_keys_check.py --dir <unlit renders> --out <check.json> \
        [--sheet <contact sheet.png>]

<dir> is blender/face_keys_render.py --mode unlit's folder: Basis.png (every key at 0) and <key>.png (that
key alone at 1.0), from the front. The camera looks along +Y, so image right is +X - the subject's LEFT
(the landmarks' "left" eye, brow and mouth corner). Each key is compared with Basis point by point:
a check passes when the mean displacement of its points is in the named direction and at least
--min-px. Keys whose names have no check are listed but not judged. DWPose reads the flat albedo the
face landmark finder reads (face_render.py's convention), so the points are what the detector sees.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dwpose import DWPose, read_face  # noqa: E402

# iBUG 68 indices: 17-21 the subject's right brow (image left), 22-26 the left; 36-41 right eye, 42-47 left;
# 48 the right mouth corner (image left), 54 the left; 31 the right nose wing, 35 the left
L_TOP, L_BOT, R_TOP, R_BOT = [43, 44], [46, 47], [37, 38], [40, 41]
L_BROW, R_BROW = list(range(22, 27)), list(range(17, 22))
L_BROW_IN, R_BROW_IN, L_BROW_OUT, R_BROW_OUT = [22], [21], [25, 26], [17, 18]   # the outer third: two points a side
L_CORNER, R_CORNER = [54], [48]
LOWER_LIP, CHIN = [57, 58, 59], [8]
L_WING, R_WING = [35], [31]
L_CHEEK, R_CHEEK = [13], [3]
BROWS = L_BROW + R_BROW
# name -> [(points, axis 0 = x / 1 = y, sign)]; y grows downward in the image
CHECKS = {
    # the lids: the upper lid is what moves (the rig leaves the lower margin near still, so DWPose sees no lower move)
    "eyeBlinkLeft": [(L_TOP, 1, +1)],
    "eyeBlinkRight": [(R_TOP, 1, +1)],
    "blink_L": [(L_TOP, 1, +1)],
    "blink_R": [(R_TOP, 1, +1)],
    "eyeWideLeft": [(L_TOP, 1, -1)],
    "eyeWideRight": [(R_TOP, 1, -1)],
    "eyeSquintLeft": [(L_TOP, 1, +1)],
    "eyeSquintRight": [(R_TOP, 1, +1)],
    "browDownLeft": [(L_BROW, 1, +1)],
    "browDownRight": [(R_BROW, 1, +1)],
    "browInnerUp": [(L_BROW_IN + R_BROW_IN, 1, -1)],
    "browOuterUpLeft": [(L_BROW_OUT, 1, -1)],
    "browOuterUpRight": [(R_BROW_OUT, 1, -1)],
    "brows_up": [(BROWS, 1, -1)],
    "jawOpen": [(LOWER_LIP, 1, +1), (CHIN, 1, +1)],
    "mouthSmileLeft": [(L_CORNER, 1, -1), (L_CORNER, 0, +1)],
    "mouthSmileRight": [(R_CORNER, 1, -1), (R_CORNER, 0, -1)],
    "smile": [(L_CORNER + R_CORNER, 1, -1), (L_CORNER, 0, +1), (R_CORNER, 0, -1)],
    "mouthFrownLeft": [(L_CORNER, 1, +1)],
    "mouthFrownRight": [(R_CORNER, 1, +1)],
    "mouthStretchLeft": [(L_CORNER, 0, +1)],
    "mouthStretchRight": [(R_CORNER, 0, -1)],
    "mouthFunnel": [(L_CORNER, 0, -1), (R_CORNER, 0, +1)],
    "mouthPucker": [(L_CORNER, 0, -1), (R_CORNER, 0, +1)],
    "pucker": [(L_CORNER, 0, -1), (R_CORNER, 0, +1)],
    # the nostril flare (x) is 2 mm and DWPose does not follow the wing: only the lip lift is judged
    "noseSneerLeft": [(L_WING, 1, -1)],
    "noseSneerRight": [(R_WING, 1, -1)],
}
# cheekPuff is not judged here: it pushes the cheeks forward, which the front view cannot see (3/4 by eye)


def load(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im), np.ones((im.height, im.width), dtype=bool)


ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sheet", default="")
ap.add_argument("--min-px", type=float, default=1.5, help="a mean displacement below this counts as no motion")
a = ap.parse_args()
d = Path(a.dir)
dw = DWPose()
rgb0, m0 = load(d / "Basis.png")
P0, s0 = read_face(dw, rgb0, m0)
res = {"basis_conf_median": round(float(np.nanmedian(s0)), 3), "keys": {}}
print(f"[check] basis: {int(np.isfinite(P0).all(1).sum())}/68 points, conf median {res['basis_conf_median']:.2f}", flush=True)
keys = sorted(p.stem for p in d.glob("*.png") if p.stem != "Basis")
for k in keys:
    rgb, m = load(d / f"{k}.png")
    P, s = read_face(dw, rgb, m)
    ok = np.isfinite(P).all(1) & np.isfinite(P0).all(1)
    disp = np.where(ok[:, None], P - P0, np.nan)
    row = {"conf_median": round(float(np.nanmedian(s)), 3), "points_ok": int(ok.sum())}
    checks = CHECKS.get(k)
    if checks is None:
        row["judged"] = False
    else:
        row["judged"] = True
        parts = []
        passed = True
        for pts, ax, sg in checks:
            idx = [i for i in pts if ok[i]]
            if not idx:
                passed = False
                parts.append({"points": pts, "axis": "xy"[ax], "sign": sg, "mean_px": None, "pass": False})
                continue
            mu = float(np.mean(disp[idx, ax]))
            good = np.sign(mu) == sg and abs(mu) >= a.min_px
            passed = passed and good
            parts.append({"points": pts, "axis": "xy"[ax], "sign": sg, "mean_px": round(mu, 2), "pass": bool(good)})
        row["checks"] = parts
        row["pass"] = bool(passed)
    res["keys"][k] = row
    tag = ("PASS" if row.get("pass") else "FAIL") if row["judged"] else "----"
    detail = " ".join(f"{c['axis']}{c['sign']:+d}:{c['mean_px']}" for c in row.get("checks", []))
    print(f"[check] {k:20s} conf {row['conf_median']:.2f} pts {row['points_ok']:2d}  {tag}  {detail}", flush=True)
judged = [k for k, r in res["keys"].items() if r["judged"]]
res["summary"] = {"judged": len(judged), "pass": sum(res["keys"][k]["pass"] for k in judged),
                  "failed": [k for k in judged if not res["keys"][k]["pass"]]}
json.dump(res, open(a.out, "w"), indent=1)
print(f"[check] {res['summary']['pass']}/{res['summary']['judged']} judged keys pass; failed: {res['summary']['failed']}",
      flush=True)
if a.sheet:
    names = ["Basis"] + keys
    S, cols = 220, 6
    rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (S + 4), rows * (S + 18)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, nm in enumerate(names):
        im = Image.open(d / f"{nm}.png").convert("RGB")
        w, h = im.size
        c = im.crop((w // 4, h // 6, w - w // 4, h - h // 6)).resize((S, S))
        x, y = (i % cols) * (S + 4), (i // cols) * (S + 18)
        sheet.paste(c, (x, y + 16))
        dr.text((x + 3, y + 2), nm, fill=(0, 0, 0))
    sheet.save(a.sheet)
    print(f"[check] sheet -> {a.sheet}", flush=True)
