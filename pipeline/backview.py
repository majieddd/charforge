"""Measure how much real detail a generated character has on the side the reference never showed.

A single-image 3D model invents everything it cannot see. On this character the reference is a
front view, so the back of the jacket is a guess, and the question multi-view conditioning has
to answer is whether that guess is actually impoverished. This module measures it instead of
assuming it.

Three measures, geometry and texture, front versus back:

  curvature   dihedral angle over edges whose faces point away from the camera, reported as a
              mean, a 95th percentile and a creased-edge fraction. Real garment structure (a
              yoke seam, a centre vent, a collar roll) is sparse and sharp, so the tail
              statistics discriminate where the mean does not - see curvature_by_side.
  relief      std-dev of surface height along the view axis within small cells, i.e. how much
              the surface actually moves in and out, normalised by character size.
  texel       gradient energy of the albedo texels that land on back-facing triangles.

Each is a back/front ratio, which isolates "is the unseen half poorer than the seen half" from
how detailed the character is overall. What this actually found, measured on the pass-1 meshes:
TRELLIS.2 does *not* produce a blank back. char01 carries 10.10% creased edges on the back
against 12.16% on the front (ratio 0.83) and char02 reaches 0.92; the shortfall is real but
modest, and it is far smaller than the prior assumption. Hunyuan3D-2.1 is the weaker case at
0.68, on a surface that is less detailed everywhere (6.5% creased edges on its *front*, half
of TRELLIS's). So the headroom multi-view conditioning can recover here is roughly 8-17% of
back-side crease density, not the rescue of a featureless shell.

A second lesson, learned the hard way and now built in as `integrity`: these ratios are only
readable next to a damage measure. Re-conditioning TRELLIS on renders of its own first-pass mesh
scored +16% back-side crease density and looked like a win in every column here - and the mesh
was visibly shredded, with 4.6x the open edges. Crease density cannot tell a seam from a hole.
Always read `open_edge_fraction` first; the comparison printer now says so out loud.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _load(path):
    m = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return m


def _facing(mesh, axis=np.array([0.0, 0.0, 1.0])):
    """Per-face cosine to the view axis; +1 faces the camera, -1 faces away."""
    n = mesh.face_normals
    return n @ (axis / np.linalg.norm(axis))


def curvature_by_side(mesh, axis, thresh=0.35, crease_deg=20.0):
    """Dihedral-angle statistics over edges interior to the front / back face sets.

    The *mean* dihedral is a poor detail measure on a marching-cubes surface: isotropic
    tessellation noise contributes to it on every edge, so a blank but bumpy shell scores as
    highly as a tailored one. Real garment structure is sparse and large - a yoke seam is a few
    hundred edges bent 30 degrees, not a uniform wobble - so the discriminating statistics are
    the tail: the 95th percentile and the fraction of edges creased past `crease_deg`.
    """
    face_cos = _facing(mesh, axis)
    adj = mesh.face_adjacency                      # (E, 2) face pairs sharing an edge
    ang = np.degrees(np.abs(mesh.face_adjacency_angles))
    a, b = face_cos[adj[:, 0]], face_cos[adj[:, 1]]
    front = (a > thresh) & (b > thresh)
    back = (a < -thresh) & (b < -thresh)
    out = {}
    for name, m in (("front", front), ("back", back)):
        if m.sum() <= 50:
            out[name] = out[name + "_p95"] = out[name + "_crease"] = float("nan")
            continue
        v = ang[m]
        out[name] = float(v.mean())
        out[name + "_p95"] = float(np.percentile(v, 95))
        out[name + "_crease"] = float((v > crease_deg).mean())
        out[name + "_edges"] = int(m.sum())
    return out


def relief_by_side(mesh, axis, thresh=0.35, grid=48):
    """Depth roughness: how far the surface departs from a locally smooth version of itself.

    The mesh is binned into a grid across the two axes perpendicular to the view; within each
    cell the spread of the view-axis coordinate is the local relief. Summed over cells and
    normalised by the character's size, this is scale-free and insensitive to silhouette.
    """
    axis = axis / np.linalg.norm(axis)
    V = np.asarray(mesh.vertices, float)
    vn = np.asarray(mesh.vertex_normals, float)
    facing = vn @ axis
    ext = float(np.linalg.norm(V.max(0) - V.min(0)))
    basis = np.eye(3)[np.argsort(np.abs(axis))[:2]]        # two axes most perpendicular
    uv = V @ basis.T
    depth = V @ axis
    out = {}
    for name, sel in (("front", facing > thresh), ("back", facing < -thresh)):
        if sel.sum() < 200:
            out[name] = float("nan")
            continue
        u, v, d = uv[sel, 0], uv[sel, 1], depth[sel]
        iu = np.clip(((u - u.min()) / max(np.ptp(u), 1e-9) * (grid - 1)).astype(int), 0, grid - 1)
        iv = np.clip(((v - v.min()) / max(np.ptp(v), 1e-9) * (grid - 1)).astype(int), 0, grid - 1)
        cell = iu * grid + iv
        order = np.argsort(cell)
        cell, d = cell[order], d[order]
        uniq, start = np.unique(cell, return_index=True)
        spreads = []
        for s, e in zip(start, list(start[1:]) + [len(d)]):
            if e - s >= 4:
                spreads.append(d[s:e].std())
        out[name] = float(np.mean(spreads) / ext * 1000) if spreads else float("nan")
        out[name + "_cells"] = int(len(spreads))
    return out


def texel_by_side(mesh, axis, thresh=0.35):
    """Gradient energy of the albedo texels covered by front- vs back-facing triangles."""
    mat = getattr(mesh.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    uv = getattr(mesh.visual, "uv", None)
    if img is None or uv is None:
        return {"front": float("nan"), "back": float("nan"), "note": "no texture"}
    tex = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(tex)
    energy = np.hypot(gx, gy)
    H, W = energy.shape
    uv = np.asarray(uv, float)
    face_cos = _facing(mesh, axis)
    out = {}
    for name, sel in (("front", face_cos > thresh), ("back", face_cos < -thresh)):
        faces = mesh.faces[sel]
        if len(faces) < 50:
            out[name] = float("nan")
            continue
        # sample each covered triangle at its barycentre and corners - cheap, unbiased enough
        pts = np.concatenate([uv[faces].mean(1), uv[faces].reshape(-1, 2)])
        px = np.clip((pts[:, 0] * (W - 1)).astype(int), 0, W - 1)
        py = np.clip(((1 - pts[:, 1]) * (H - 1)).astype(int), 0, H - 1)
        out[name] = float(energy[py, px].mean() * 1000)
        out[name + "_faces"] = int(sel.sum())
    return out


def integrity(mesh, weld_tol=1e-5):
    """Tear and hole statistics - the guard that stops crease density being read as quality.

    Crease density cannot tell structure from damage: a torn surface has plenty of edges bent
    past 20 degrees. Measured on the multi-view experiment, feeding the model renders of its own
    first-pass mesh raised back-side crease density 16% while actually shredding the surface,
    and the tail statistics did not catch it because tearing produces genuine high dihedral.

    An open (boundary) edge belongs to exactly one face, so on a closed generated surface it is
    a hole. Counting those separates "more detail" from "more damage", and any reading of the
    crease numbers is only meaningful when this has not moved.
    """
    V = np.asarray(mesh.vertices, float)
    n = len(V)
    scale = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    keys = np.round(V / (weld_tol * scale)).astype(np.int64)
    rep, remap = {}, np.empty(n, dtype=np.int64)
    for i in range(n):
        remap[i] = rep.setdefault((keys[i, 0], keys[i, 1], keys[i, 2]), len(rep))
    F = remap[mesh.faces]
    e = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).reshape(-1, 2), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    ne = max(len(uniq), 1)
    return {"welded_vertices": int(len(rep)),
            "open_edge_fraction": round(float((cnt == 1).sum()) / ne, 4),
            "nonmanifold_edge_fraction": round(float((cnt > 2).sum()) / ne, 4)}


def ratio(d, key="front", bkey="back"):
    f, b = d.get(key, float("nan")), d.get(bkey, float("nan"))
    return float(b / f) if f and np.isfinite(f) and np.isfinite(b) and f > 1e-9 else float("nan")


def measure(path, axis=(0.0, 0.0, 1.0)):
    mesh = _load(path)
    axis = np.array(axis, float)
    cur = curvature_by_side(mesh, axis)
    rel = relief_by_side(mesh, axis)
    tex = texel_by_side(mesh, axis)
    integ = integrity(mesh)
    return {
        "mesh": str(path),
        "integrity": integ,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "curvature_deg": cur, "curvature_back_over_front": round(ratio(cur), 4),
        "crease_back_over_front": round(ratio(cur, "front_crease", "back_crease"), 4),
        "p95_back_over_front": round(ratio(cur, "front_p95", "back_p95"), 4),
        "relief_per_mille": rel, "relief_back_over_front": round(ratio(rel), 4),
        "texel_energy": tex, "texel_back_over_front": round(ratio(tex), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--axis", nargs=3, type=float, default=[0.0, 0.0, 1.0],
                    help="camera/view axis the reference image was taken along")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    labels = a.labels or [Path(m).stem for m in a.meshes]
    rows = []
    for lab, m in zip(labels, a.meshes):
        r = measure(m, a.axis)
        r["label"] = lab
        rows.append(r)
        c = r["curvature_deg"]
        print(f"[backview] {lab}", flush=True)
        print(f"    mean dihedral   front {c['front']:6.2f}deg   back {c['back']:6.2f}deg   "
              f"back/front {r['curvature_back_over_front']:.3f}", flush=True)
        print(f"    p95 dihedral    front {c['front_p95']:6.2f}deg   back {c['back_p95']:6.2f}deg   "
              f"back/front {r['p95_back_over_front']:.3f}", flush=True)
        print(f"    creased edges   front {c['front_crease']:6.2%}   back {c['back_crease']:6.2%}   "
              f"back/front {r['crease_back_over_front']:.3f}", flush=True)
        print(f"    relief          back/front {r['relief_back_over_front']:.3f}   "
              f"texel back/front {r['texel_back_over_front']:.3f}", flush=True)
        ig = r["integrity"]
        print(f"    integrity       open edges {ig['open_edge_fraction']:.2%}   "
              f"non-manifold {ig['nonmanifold_edge_fraction']:.2%}", flush=True)
    if len(rows) > 1:
        base = rows[0]
        print(f"[backview] --- change vs {base['label']} (positive = more back-side detail) ---",
              flush=True)
        for r in rows[1:]:
            def chg(k):
                b, v = base[k], r[k]
                return (v / b - 1.0) if b and np.isfinite(b) and np.isfinite(v) and b > 1e-9 else float("nan")
            open_b = base["integrity"]["open_edge_fraction"]
            open_r = r["integrity"]["open_edge_fraction"]
            print(f"    {r['label']:<26} crease {chg('crease_back_over_front'):+7.1%}   "
                  f"p95 {chg('p95_back_over_front'):+7.1%}   "
                  f"relief {chg('relief_back_over_front'):+7.1%}   "
                  f"texel {chg('texel_back_over_front'):+7.1%}", flush=True)
            # a crease gain that arrives with a jump in open edges is damage, not detail
            if open_b > 1e-9 and open_r > open_b * 1.5:
                print(f"    {'':<26} ^ DISCOUNT THIS: open edges {open_b:.2%} -> {open_r:.2%} "
                      f"({open_r/open_b:.1f}x). The surface is torn; the extra creases are holes.",
                      flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=2)
        print(f"[backview] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
