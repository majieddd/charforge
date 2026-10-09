#!/usr/bin/env python3
"""How much of the dense surface a low-poly and its baked normal map keep (E168).

    ../.venv/bin/python tools/detail_retention.py --low work/<c>/e168/<v>/retopo.glb --cage work/<c>/solid_hands.glb \
        --generator work/<c>/mesh.glb --joints work/<c>/joints_refined.json \
        --labels work/<c>/labels.json --labels-ref work/<c>/retopo.glb --out work/<c>/qa/detail_<v>.json

Reads the shipped low (its own normals, UVs and the normal map embedded as its normal texture) and measures it against
two dense surfaces: the clean solid the normal map was baked from (--cage) and the generator's mesh (--generator, TRELLIS's
own surface). Everything is sampled uniformly by area with a fixed seed (SEED).

  distance      two-way closest-point distance: low samples to the cage, and cage samples to the low; p50, p95, max;
                in mm at 1.75 m (distance / height * 1750). The nearest point is exact among the 24-64 triangles whose
                centroids are nearest.
  normal        angle between the shading normal and the cage's normal at the closest point. "mapped" uses the low's
                normal with its baked normal map applied (tangent frame from the UVs, the map read at the sample's UV);
                "geometric" uses the low's own smooth normal. The gap is what the map carries.
  regions       head (retopo.py's head test on the height and the neck and head joints), hands (within 10% of the height
                of a wrist-to-hand segment), front and back (the surface's normal faces the front camera, Blender -Y, or
                not). With --labels: body, clothing, hair, accessory, read from the labels of the nearest vertex of
                --labels-ref (the mesh labels.json was made on: the default retopo.glb).
  shape         tilt_deg (tools/model_quality.py's measure: each triangle corner's smooth normal against its face
                normal; p50 and p99), skinny triangles (smallest angle under 10 degrees), median edge in mm at 1.75 m,
                open and non-manifold edges per 10k triangles (on the welded surface), head share of the triangles.
Writes JSON and prints one summary line. Uses only numpy, scipy and PIL (no Blender, no GPU).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import remesh_io as R  # noqa: E402

SEED = 168
GROUP_NAMES = ["body", "clothing", "hair", "accessory"]     # labels.json's group_ids (their order)


def pct(x, qs=(50, 95)):
    """p50, p95 and max of x (distances in mm, or angles in degrees)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    out = {f"p{q}": round(float(np.percentile(x, q)), 3) for q in qs}
    out["p99"] = round(float(np.percentile(x, 99)), 3)     # the max is one sample of 200k: a head outlier moves it
    out["max"] = round(float(x.max()), 3)
    out["n"] = int(x.size)
    return out


def sample_area(V, F, n, rng):
    """n points uniform by area on the triangles: (points, face index, barycentric weights)."""
    _, area = R.face_normals_areas(V, F)
    fi = rng.choice(len(F), n, p=area / area.sum())
    s, t = np.sqrt(rng.random(n)), rng.random(n)
    bary = np.stack([1 - s, s * (1 - t), s * t], 1)
    P = np.einsum("nk,nkj->nj", bary, V[F[fi]])
    return P, fi, bary


def tangent_frames(V, F, UV):
    """Per-vertex tangent T and bitangent B from the UV gradients (B along Blender's V, i.e. -dP/dv' in glTF)."""
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    du1 = UV[F[:, 1], 0] - UV[F[:, 0], 0]
    dv1 = UV[F[:, 1], 1] - UV[F[:, 0], 1]
    du2 = UV[F[:, 2], 0] - UV[F[:, 0], 0]
    dv2 = UV[F[:, 2], 1] - UV[F[:, 0], 1]
    det = du1 * dv2 - du2 * dv1
    ok = np.abs(det) > 1e-30
    det = np.where(ok, det, 1.0)
    dPdu = (dv2[:, None] * e1 - dv1[:, None] * e2) / det[:, None]
    dPdv = (-du2[:, None] * e1 + du1[:, None] * e2) / det[:, None]
    w = np.linalg.norm(np.cross(e1, e2), axis=1) / 2 * ok
    T = np.zeros_like(V)
    B = np.zeros_like(V)
    for k in range(3):
        np.add.at(T, F[:, k], w[:, None] * dPdu)
        np.add.at(B, F[:, k], w[:, None] * (-dPdv))
    return T, B


def decode_map(tex, uv):
    H, W = tex.shape[:2]
    u = uv[:, 0] - np.floor(uv[:, 0])
    v = uv[:, 1] - np.floor(uv[:, 1])
    x = np.clip((u * W).astype(np.int64), 0, W - 1)
    y = np.clip((v * H).astype(np.int64), 0, H - 1)
    n = tex[y, x].astype(np.float64) / 255.0 * 2 - 1
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


def angle_deg_vec(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, 1) / np.maximum(na * nb, 1e-300), -1, 1)))


def head_and_hands(P_b, J, H):
    """retopo.py's head test (above the neck, within 0.12 H of the head joint) and the hand test (within 10% of H of a
    wrist-to-hand segment), on Blender-frame points."""
    neck, head = J["neck"], J["head"]
    head_m = (P_b[:, 2] > neck[2]) & (np.linalg.norm(P_b[:, :2] - head[:2], axis=1) < 0.12 * H)
    d = np.full(len(P_b), np.inf)
    for w_name, h_name in (("left_wrist", "left_hand"), ("right_wrist", "right_hand")):
        if w_name in J and h_name in J:
            p0, p1 = J[w_name], J[h_name]
            ab = p1 - p0
            t = np.clip(((P_b - p0) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)[:, None]
            d = np.minimum(d, np.linalg.norm(P_b - (p0 + t * ab), axis=1))
    return head_m, d < 0.10 * H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", required=True, help="the low: a retopo.glb (normals, UVs and the normal map)")
    ap.add_argument("--cage", required=True, help="the clean solid the normal map was baked from (solid_hands.glb)")
    ap.add_argument("--generator", default=None, help="the generator's mesh.glb: a second dense reference")
    ap.add_argument("--joints", required=True)
    ap.add_argument("--labels", default=None, help="labels.json: body / clothing / hair / accessory per vertex")
    ap.add_argument("--labels-ref", default=None, help="the mesh the labels index (the default retopo.glb)")
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    low = R.read_mesh(a.low, want_tex=True)
    cage = R.read_mesh(a.cage)
    Vl, Fl, Nl, UVl, tex = low["V"], low["F"], low["N"], low["UV"], low["tex"]
    Vc, Fc = cage["V"], cage["F"]
    # the cage's pieces retopo keeps. retopo drops debris from the decimated low (under 0.25% of its faces); the
    # decimation keeps about len(Fl)/len(Fc) of a piece's faces, so the same rule on the cage is 0.25% of the cage's
    # faces. Keeping only the largest piece cut a real 3% shell of Vex's low off the cage (its samples read 171 mm
    # away); a rule on the low's count kept pieces retopo had already dropped (Pip's max read 26 mm)
    keep_c = R.piece_mask(Vc, Fc, min_faces=0.0025 * len(Fc))
    Fc = Fc[keep_c]
    if a.generator:
        gen = R.read_mesh(a.generator)
        lo, hi = R.bbox_blender(gen["V"])
    else:
        lo, hi = R.bbox_blender(Vc)
    Jw, H, _ = R.joints_world(R.load_joints(a.joints), lo, hi)
    mm = lambda d: d / H * 1750.0          # noqa: E731 - distance in mm at 1.75 m
    fro = np.array([0.0, -1.0, 0.0])        # the front camera is at Blender -Y

    # labels: the group of a point is the label of the nearest vertex of the reference mesh
    lab_tree = lab_ids = None
    if a.labels:
        ref = R.read_mesh(a.labels_ref or a.low)
        lab = json.load(open(a.labels))
        lab_ids = np.asarray(lab["labels"], dtype=np.int64)
        if len(lab_ids) != len(ref["V"]):
            raise SystemExit(f"labels ({len(lab_ids)}) do not index the reference mesh ({len(ref['V'])} vertices)")
        lab_tree = cKDTree(ref["V"])

    def group(P):
        if lab_tree is None:
            return None
        _, idx = lab_tree.query(P)
        return lab_ids[idx]

    # low samples: position, smooth normal, UV, mapped normal
    Pl, fl, bl = sample_area(Vl, Fl, a.samples, rng)
    Nl_v = Nl if Nl is not None else R.vertex_normals(Vl, Fl)
    N_geo = unit(np.einsum("nk,nkj->nj", bl, Nl_v[Fl[fl]]))
    Pl_b = R.to_blender(Pl)
    head_l, hand_l = head_and_hands(Pl_b, Jw, H)
    front_l = (R.to_blender(N_geo) @ fro) >= 0
    grp_l = group(Pl)
    # the cage's normal at the low samples' closest points
    cage_n = R.vertex_normals(Vc, Fc)
    cage_index = R.TriangleIndex(Vc, Fc, k=24)
    _, fc, bc, dc = cage_index.query(Pl)
    Nc_at = unit(np.einsum("nk,nkj->nj", bc, cage_n[Fc[fc]]))
    geo_angle = angle_deg(N_geo, Nc_at)
    map_angle = None
    out = {"tag": a.tag, "low_path": a.low, "cage_path": a.cage, "generator_path": a.generator,
           "labels_path": a.labels, "height_units": round(H, 5), "triangles": int(len(Fl)), "vertices": int(len(Vl))}
    if tex is not None and UVl is not None:
        T, B = tangent_frames(Vl, Fl, UVl)
        Tn = unit(np.einsum("nk,nkj->nj", bl, T[Fl[fl]]))
        Bn = unit(np.einsum("nk,nkj->nj", bl, B[Fl[fl]]))
        UVs = np.einsum("nk,nkj->nj", bl, UVl[Fl[fl]])
        nt = decode_map(tex, UVs)
        Tn = unit(Tn - np.sum(Tn * N_geo, 1)[:, None] * N_geo)
        Bn = unit(Bn - np.sum(Bn * N_geo, 1)[:, None] * N_geo - np.sum(Bn * Tn, 1)[:, None] * Tn)
        Ns = unit(nt[:, :1] * Tn + nt[:, 1:2] * Bn + nt[:, 2:3] * N_geo)
        map_angle = angle_deg(Ns, Nc_at)
        # the other sign of the bitangent: the convention check (the right one should win)
        Ns_flip = unit(nt[:, :1] * Tn - nt[:, 1:2] * Bn + nt[:, 2:3] * N_geo)
        out["bitangent_check_median_deg"] = {"blender_V": round(float(np.median(map_angle)), 3),
                                             "opposite": round(float(np.median(angle_deg(Ns_flip, Nc_at))), 3)}
        out["normal_map_size"] = list(tex.shape[:2])
    else:
        out["normal_map_size"] = None

    # cage samples: distance to the low, and the low's normal at their closest point
    Pc, fcs, bcs = sample_area(Vc, Fc, a.samples, rng)
    Pc_b = R.to_blender(Pc)
    head_c, hand_c = head_and_hands(Pc_b, Jw, H)
    Nc_s = unit(np.einsum("nk,nkj->nj", bcs, cage_n[Fc[fcs]]))
    front_c = (R.to_blender(Nc_s) @ fro) >= 0
    grp_c = group(Pc)
    low_index = R.TriangleIndex(Vl, Fl, k=64)
    _, _, _, dcl = low_index.query(Pc)

    def region_stats(d, nang, masks):
        res = {}
        for name, m in masks.items():
            if not m.any():
                continue
            res[name] = {"distance_mm_at_1750": pct(mm(d[m]), (50, 95))}
            if nang is not None:
                res[name]["normal_deg"] = pct(nang[m], (50, 95))
        return res

    def masks_for(head, hand, front, grp, n):
        m = {"all": np.ones(n, bool), "head": head, "hands": hand, "front": front, "back": ~front}
        if grp is not None:
            for gid, gname in enumerate(GROUP_NAMES):
                m[gname] = grp == gid
        return m

    ml = masks_for(head_l, hand_l, front_l, grp_l, len(dc))
    mc = masks_for(head_c, hand_c, front_c, grp_c, len(dcl))
    out["vs_cage"] = {
        "low_to_cage": region_stats(dc, map_angle, ml),
        "cage_to_low": region_stats(dcl, None, mc),
        "two_way_p50_mm_at_1750": round(float(mm(np.percentile(np.concatenate([dc, dcl]), 50))), 3),
        "two_way_p95_mm_at_1750": round(float(mm(np.percentile(np.concatenate([dc, dcl]), 95))), 3),
        "two_way_p99_mm_at_1750": round(float(mm(np.percentile(np.concatenate([dc, dcl]), 99))), 3),
        "two_way_max_mm_at_1750": round(float(mm(np.concatenate([dc, dcl]).max())), 3),
    }
    out["normal_vs_cage_deg"] = {
        "mapped": pct(map_angle, (50, 95)) if map_angle is not None else None,
        "geometric": pct(geo_angle, (50, 95)),
    }
    for name, m in (("head", head_l), ("hands", hand_l), ("front", front_l), ("back", ~front_l)):
        out["normal_vs_cage_deg"][f"geometric_{name}"] = pct(geo_angle[m], (50, 95)) if m.any() else None
        if map_angle is not None:
            out["normal_vs_cage_deg"][f"mapped_{name}"] = pct(map_angle[m], (50, 95)) if m.any() else None
    if grp_l is not None:
        for gid, gname in enumerate(GROUP_NAMES):
            m = grp_l == gid
            if m.any():
                out["normal_vs_cage_deg"][f"geometric_{gname}"] = pct(geo_angle[m], (50, 95))
                if map_angle is not None:
                    out["normal_vs_cage_deg"][f"mapped_{gname}"] = pct(map_angle[m], (50, 95))

    # the generator's surface, if given: low to it and it to the low
    if a.generator:
        gi = R.TriangleIndex(gen["V"], gen["F"], k=24)
        _, _, _, dg = gi.query(Pl)
        Pg, _, _ = sample_area(gen["V"], gen["F"], a.samples // 2, rng)
        _, _, _, dgl = low_index.query(Pg)
        Pg_b = R.to_blender(Pg)
        out["generator"] = {"low_to_generator_mm_at_1750": pct(mm(dg)),
                            "generator_to_low_mm_at_1750": pct(mm(dgl)),
                            "generator_to_low_head_mm_at_1750": pct(mm(dgl[head_and_hands(Pg_b, Jw, H)[0]])),
                            "two_way_p95_mm_at_1750": round(float(mm(np.percentile(np.concatenate([dg, dgl]), 95))), 3)}

    # shape of the low itself (tools/model_quality.py's measures, on the welded surface)
    n_f = R.face_normals_areas(Vl, Fl)[0]
    VNl = Nl_v[Fl]
    tilt = np.degrees(np.arccos(np.clip((VNl * n_f[:, None, :]).sum(-1), -1, 1))).ravel()
    e = np.linalg.norm(np.stack([Vl[Fl[:, 1]] - Vl[Fl[:, 0]], Vl[Fl[:, 2]] - Vl[Fl[:, 1]],
                                 Vl[Fl[:, 0]] - Vl[Fl[:, 2]]], 1), axis=2)
    ang = []
    for k in range(3):
        p, q, r = Vl[Fl[:, k]], Vl[Fl[:, (k + 1) % 3]], Vl[Fl[:, (k + 2) % 3]]
        ang.append(angle_deg_vec(q - p, r - p))
    min_angle = np.min(np.stack(ang, 1), 1)
    Vw, Fw = R.weld(Vl, Fl)                              # topology on the welded surface (UV seams are not edges)
    Ew = np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(Ew, axis=0, return_counts=True)
    per10k = 1e4 / len(Fl)
    cen_l = Vl[Fl].mean(1)
    head_tri = head_and_hands(R.to_blender(cen_l), Jw, H)[0]
    out["shape"] = {"tilt_deg": {"p50": round(float(np.median(tilt)), 2), "p99": round(float(np.quantile(tilt, 0.99)), 2)},
                    "skinny_tri": round(float((min_angle < 10).mean()), 4),
                    "edge_mm_at_1750_p50": round(float(mm(np.median(e))), 2),
                    "open_edges_per10k": round(float((cnt == 1).sum() * per10k), 2),
                    "non_manifold_per10k": round(float((cnt >= 3).sum() * per10k), 2),
                    "head_share": round(float(head_tri.mean()), 4)}
    if lab_tree is not None:
        grp_f = group(cen_l)
        out["shape"]["tilt_deg_by_group"] = {}
        for gid, gname in enumerate(GROUP_NAMES):
            m = np.repeat(grp_f == gid, 3)
            if m.any():
                out["shape"]["tilt_deg_by_group"][gname] = {"faces": int((grp_f == gid).sum()),
                                                           "p50": round(float(np.median(tilt[m])), 2),
                                                           "p99": round(float(np.quantile(tilt[m], 0.99)), 2)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    s = out["vs_cage"]["low_to_cage"]["all"]
    print(f"[detail] {a.tag or os.path.basename(os.path.dirname(os.path.abspath(a.low)))}: "
          f"{out['triangles']:,} tris, low->cage p50/p95 {s['distance_mm_at_1750']['p50']}/"
          f"{s['distance_mm_at_1750']['p95']} mm, cage->low p95 "
          f"{out['vs_cage']['cage_to_low']['all']['distance_mm_at_1750']['p95']} mm, normal mapped p95 "
          f"{out['normal_vs_cage_deg']['mapped']['p95'] if map_angle is not None else None} deg, geometric p95 "
          f"{out['normal_vs_cage_deg']['geometric']['p95']} deg, tilt p50 {out['shape']['tilt_deg']['p50']}",
          flush=True)


if __name__ == "__main__":
    main()
