"""Are both modelled hands on the shipped mesh, whole, and alone? A check of every character's hands.

    python tools/hands_check.py [names...] [--strict]   # default: every character in out/; --strict fails on a missing hand

hands.json (blender/hands.py) records each modelled finger's joints in the mesh's own frame. A hand is on the mesh if
the low-poly (work/<n>/retopo.glb, the same frame) has surface at each fingertip and at each finger's base knuckle,
within a few millimetres (FOUND, as a share of the hand's length). The exact boolean that joins a hand to the arm has
failed silently: after one hand fell back to a separate shell, the other's union returned the body unchanged and the
check then in place (the volume must not shrink) passed it - Rivet and Bo shipped with no right hand.

An extra hand is surface where no hand should be: vertices beyond the wrist, along the hand's axis, farther from every
modelled finger joint than the fingers are thick (a generated hand left beside the modelled one, or a second copy).
Writes work/<n>/qa/hands_check.json and prints one line a character.
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
FOUND = 0.08            # a joint counts as on the mesh within 8% of the hand's length (~8 mm)
EXTRA = 0.22            # surface beyond the wrist and this far from every joint is not the modelled hand


def check(n):
    hj = ROOT / "work" / n / "hands.json"
    rt = ROOT / "work" / n / "retopo.glb"
    if not (hj.exists() and rt.exists()):
        return None
    H = json.load(open(hj))
    m = trimesh.load(rt, force="mesh", process=False)
    V = np.asarray(m.vertices, float)
    V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)               # glTF's Y up, as Blender imported it for hands.py
    tree = cKDTree(V)
    out = {}
    for side, s in H["sides"].items():
        L = float(s["length"])
        joints = np.array([p for f in s["fingers"].values() for p in f])
        tips = np.array([f[-1] for f in s["fingers"].values()])
        bases = np.array([f[0] for f in s["fingers"].values()])
        d_tip, _ = tree.query(tips)
        d_base, _ = tree.query(bases)
        found_tips = int((d_tip < FOUND * L).sum())
        found_bases = int((d_base < FOUND * L).sum())
        # beyond the wrist along the hand's axis, within reach of a hand, but away from every finger joint
        W, x = np.array(s["wrist"]), np.array(s["frame"]["x"])
        along = (V - W) @ x
        near = (along > 0.15 * L) & (along < 1.6 * L) & (np.linalg.norm(V - W, axis=1) < 1.6 * L)
        far = cKDTree(joints).query(V[near])[0] > EXTRA * L
        stray = int(far.sum())
        # stray surface is only reported: a puffy cuff (Pip) or hair hanging past the wrist (Aoi) count too
        verdict = ("missing" if found_tips <= 1 else "partial" if found_tips < 4 else "ok")
        out[side] = {"tips_found": found_tips, "bases_found": found_bases, "tip_gap_mm_max": round(float(d_tip.max()) / L * 100, 1),
                     "stray_vertices": stray, "verdict": verdict}
    json.dump(out, open(ROOT / "work" / n / "qa" / "hands_check.json", "w"), indent=1)
    return out


if __name__ != "__main__":
    raise SystemExit
args = [x for x in sys.argv[1:] if not x.startswith("--")]
strict = "--strict" in sys.argv
names = args or sorted(p.name for p in (ROOT / "out").iterdir() if (ROOT / "out" / p.name / f"{p.name}.json").exists())
for n in names:
    r = check(n)
    if r is None:
        print(f"{n:9s} no hands.json or retopo.glb")
        continue
    print(f"[hands_check] {n:9s} " + "   ".join(f"{side}: {v['verdict']:8s} tips {v['tips_found']}/5 stray {v['stray_vertices']}"
                                                for side, v in sorted(r.items())), flush=True)
    if strict and any(v["verdict"] != "ok" for v in r.values()):
        raise SystemExit(f"[hands_check] {n}: a modelled hand is missing from the low-poly - see work/{n}/qa/hands_check.json")
