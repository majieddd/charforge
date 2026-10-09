"""Where a face rig's shape key folds: the rest position and jaw weight of its flipped triangles.

    /Applications/Blender.app/Contents/MacOS/Blender -b -noaudio --python blender/face_fold_locate.py -- \
        --blend <rig_f.blend> --face <face.json> --key jawOpen [--out folds.json]

For the named key at 1.0 (the others at 0), finds the triangles of more than 1 mm2 at rest whose normal
turns more than 90 degrees (the folds face_keys_audit.py counts as flips_1mm2) and reports, in centimetres
and in the face's own frame where the eyes, mouth and chin are known:
  the rest centroid of each fold (mean, min and max), the mouth's and chin's positions for comparison,
  the jaw and head weight of the fold's vertices, and how far the fold's vertices move under the key.
The per-triangle list goes to --out. Runs on the CPU only.
"""
import argparse
import json
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--face", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--out", default="")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH" and o.find_armature() == rig)
me = mesh.data
F = json.load(open(a.face))
kbs = me.shape_keys.key_blocks
if a.key not in kbs:
    sys.exit(f"no shape key {a.key!r}: {[k.name for k in kbs][:40]}")
n = len(me.vertices)
base = np.empty(n * 3)
kbs[0].data.foreach_get("co", base)
base = base.reshape(-1, 3)
P = np.empty(n * 3)
kbs[a.key].data.foreach_get("co", P)
P = P.reshape(-1, 3)
me.calc_loop_triangles()
nt = len(me.loop_triangles)
T = np.empty(nt * 3, dtype=np.int64)
me.loop_triangles.foreach_get("vertices", T)
T = T.reshape(-1, 3)
Mw = np.array(mesh.matrix_world)
g = {vg.name: vg.index for vg in mesh.vertex_groups}
W = {nm: np.zeros(n) for nm in ("jaw", "head", "neck")}
for v in me.vertices:
    for gg in v.groups:
        for nm, gi in (("jaw", g.get("jaw")), ("head", g.get("mixamorig:Head", g.get("head"))),
                       ("neck", g.get("mixamorig:Neck", g.get("neck")))):
            if gi is not None and gg.group == gi:
                W[nm][v.index] = gg.weight


def tri_n(X):
    return np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])


N0, N1 = tri_n(base), tri_n(P)
A0 = 0.5 * np.linalg.norm(N0, axis=1)
c0 = np.linalg.norm(N0, axis=1)
cos = (N0 * N1).sum(1) / (c0 * np.linalg.norm(N1, axis=1) + 1e-30)
bad = np.nonzero((cos < 0) & (A0 > 1e-6))[0]
world = lambda X: X @ Mw[:3, :3].T + Mw[:3, 3]
cen = world(base[T[bad]].mean(axis=1))
disp = np.linalg.norm(P - base, axis=1)
mouth = np.array(F["mouth"]["centre"])
chin = np.array(F["chin"])
ied = float(F["ied_m"])
print(f"[fold] {a.key}: {len(bad)} folded triangles (>1 mm2 at rest), ied {ied * 100:.1f} cm", flush=True)
rows = []
if len(bad):
    print(f"[fold] centroid mean (cm) x {cen[:, 0].mean() * 100:.2f} y {cen[:, 1].mean() * 100:.2f} "
          f"z {cen[:, 2].mean() * 100:.2f}; min z {cen[:, 2].min() * 100:.2f} max z {cen[:, 2].max() * 100:.2f}", flush=True)
    print(f"[fold] mouth centre (cm) {np.round(mouth * 100, 2).tolist()}, chin {np.round(chin * 100, 2).tolist()}",
          flush=True)
    mats = np.array([lt.material_index for lt in me.loop_triangles])[bad]
    names = [m.name if m else "-" for m in me.materials]
    cnt = {}
    for mi in mats:
        cnt[names[mi] if mi < len(names) else "?"] = cnt.get(names[mi] if mi < len(names) else "?", 0) + 1
    print(f"[fold] by material: {cnt}", flush=True)
    jw = W["jaw"][T[bad]]
    spread = jw.max(1) - jw.min(1)
    print(f"[fold] jaw weight spread inside the fold's triangle: median {np.median(spread):.2f}, "
          f"over 0.3 in {int((spread > 0.3).sum())} of {len(bad)}; uniform (spread < 0.05) in {int((spread < 0.05).sum())}",
          flush=True)
    vv = T[bad].ravel()
    print(f"[fold] vertices in folds: {len(np.unique(vv))}; jaw weight mean {W['jaw'][vv].mean():.2f}, "
          f"head {W['head'][vv].mean():.2f}, neck {W['neck'][vv].mean():.2f}; "
          f"median key displacement {np.median(disp[vv]) * 1000:.2f} mm", flush=True)
    for i, t in enumerate(bad):
        c = cen[i]
        rows.append({"tri": int(t), "centre_cm": np.round(c * 100, 3).tolist(),
                     "area_mm2": round(float(A0[t]) * 1e6, 3), "cos": round(float(cos[t]), 3),
                     "material": names[mats[i]] if mats[i] < len(names) else "?",
                     "jaw_w": round(float(W["jaw"][T[t]].max()), 3),
                     "jaw_min": round(float(W["jaw"][T[t]].min()), 3),
                     "head_w": round(float(W["head"][T[t]].max()), 3),
                     "disp_mm": round(float(disp[T[t]].max()) * 1000, 2)})
if a.out:
    json.dump({"key": a.key, "folds": rows, "mouth_cm": np.round(mouth * 100, 2).tolist(),
               "chin_cm": np.round(chin * 100, 2).tolist()}, open(a.out, "w"), indent=1)
    print(f"[fold] -> {a.out}", flush=True)
