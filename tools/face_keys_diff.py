"""Compare two builds' face shape keys without relying on vertex order (the mouth cut and density change it).

    /Applications/Blender.app/Contents/MacOS/Blender -b -noaudio --python blender/face_keys_hash.py -- \
        --blend <build A rig_f.blend> --out a.json --npz a.npz      (and the same for build B)
    ../.venv/bin/python tools/face_keys_diff.py a.npz b.npz [--json out.json]

Vertices are grouped by rest position (the cut's two lips share one): a group present in both builds has its
displacements compared in sorted order, so the duplicated seam copies pair up. A key is identical when every
compared group agrees to 1e-9 m and no position is in one build only. Reports the largest displacement
difference (mm) and the rows over --tol_mm.
Runs on the CPU only.
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--json", default="")
ap.add_argument("--tol_mm", type=float, default=0.01)
args = ap.parse_args()
A, B = np.load(args.a), np.load(args.b)
na, nb = list(A["names"]), list(B["names"])
ka = {n: A[f"key_{i}"] for i, n in enumerate(na)}
kb = {n: B[f"key_{i}"] for i, n in enumerate(nb)}
out = {"vertices_a": int(len(ka["Basis"])), "vertices_b": int(len(kb["Basis"])), "keys": {}}


def groups_of(base):
    """Vertex indices grouped by rest position (1 micrometre): the cut's seam copies share a position."""
    groups = {}
    for i, k in enumerate(map(tuple, np.round(base * 1e6).astype(np.int64))):
        groups.setdefault(k, []).append(i)
    return groups


base_a, base_b = ka["Basis"], kb["Basis"]
GROUPS_A, GROUPS_B = groups_of(base_a), groups_of(base_b)
COMMON = [(GROUPS_A[k], GROUPS_B[k]) for k in GROUPS_A if k in GROUPS_B]
ONLY_A = sum(len(v) for k, v in GROUPS_A.items() if k not in GROUPS_B)
ONLY_B = sum(len(v) for k, v in GROUPS_B.items() if k not in GROUPS_A)
out["unmatched_a"], out["unmatched_b"] = int(ONLY_A), int(ONLY_B)
out["method"] = "vertices grouped by rest position; coincident copies compared in sorted order"
print(f"[diff] vertices {out['vertices_a']:,} (A) and {out['vertices_b']:,} (B); {ONLY_A:,} of A and {ONLY_B:,} of B "
      f"sit at a position the other lacks (not compared)", flush=True)


def compare(n):
    Pa, Pb = ka[n], kb[n]
    da_all, db_all = Pa - base_a, Pb - base_b
    diffs = []
    bad_groups = 0
    for ia, ib in COMMON:
        if len(ia) != len(ib):
            bad_groups += 1
            continue
        xa = da_all[ia]
        xb = db_all[ib]
        oa = np.lexsort(np.round(xa * 1e9).T[::-1])
        ob = np.lexsort(np.round(xb * 1e9).T[::-1])
        diffs.append(np.linalg.norm(xa[oa] - xb[ob], axis=1))
    d = np.concatenate(diffs) * 1000.0 if diffs else np.zeros(0)
    if d.size == 0:
        return {"identical": False, "rows": 0, "note": "no common positions"}
    return {"max_diff_mm": round(float(d.max()), 4), "rows_over_tol": int((d > args.tol_mm).sum()),
            "rows": int(d.size), "groups_with_other_count": bad_groups,
            "identical": bool(d.max() < 1e-9 and bad_groups == 0 and ONLY_A == 0 and ONLY_B == 0)}


for n in na:
    out["keys"][n] = compare(n) if n in kb else "only in A"
for n in nb:
    if n not in ka:
        out["keys"][n] = "only in B"
same = [n for n, r in out["keys"].items() if isinstance(r, dict) and r.get("identical")]
diff = [n for n, r in out["keys"].items() if isinstance(r, dict) and not r.get("identical")]
print(f"[diff] identical keys {len(same)}: {', '.join(same) if same else '-'}", flush=True)
print(f"[diff] differing keys {len(diff)}:", flush=True)
for n in diff:
    r = out["keys"][n]
    print(f"[diff]   {n:18s} max {r.get('max_diff_mm')} mm, rows over {args.tol_mm} mm: {r.get('rows_over_tol')} of {r.get('rows')}",
          flush=True)
others = [f"{n} ({r})" for n, r in out["keys"].items() if not isinstance(r, dict)]
if others:
    print("[diff] only in one build:", "; ".join(others), flush=True)
if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
