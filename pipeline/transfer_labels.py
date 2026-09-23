"""Carry part labels from the generated mesh onto the retopologised one, and fix the face.

The segmentation stage is expensive - eight orbit renders, a SegFormer pass per view, and an
exact-visibility back-projection - and it has already run on the generated mesh. The
retopologised mesh describes the *same surface*, so its labels can be looked up rather than
recomputed: for each retopo vertex, take the label of the nearest original vertex, then let the
surface itself clean up the result by neighbour voting.

The second job is the one that matters more. The segmentation labels ~13% of the hair group as
hair when its albedo is plainly skin - pieces of cheek, jaw and forehead. Downstream that
geometry gets split into a separate object with different binding, and under animation the
rigid fragments stop matching the skinned face and tear through it. Reclassifying here, before
anything is split, means those vertices never leave the skin in the first place.

Skin and hair are separated in Lab rather than RGB, because the discriminating direction is
a* (red-green) together with lightness, which Lab makes a single threshold instead of three
coupled ones. Both class centres are seeded from the mesh's own extremes, so nothing is
hard-coded to this character's colouring.

    python transfer_labels.py --retopo retopo.glb --source trellis_mesh.glb \
        --parts parts.json --out retopo_labels.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def to_lab(rgb):
    r = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    t = (r @ M.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], 1)


def vertex_colours(mesh):
    mat = getattr(mesh.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) if mat else None
    uv = getattr(mesh.visual, "uv", None)
    if img is None or uv is None:
        return None
    tex = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    H, W, _ = tex.shape
    uv = np.asarray(uv, float)
    px = np.clip((uv[:, 0] * (W - 1)).astype(int), 0, W - 1)
    py = np.clip(((1 - uv[:, 1]) * (H - 1)).astype(int), 0, H - 1)
    return tex[py, px]


def adjacency(mesh, weld_tol=1e-5):
    """Welded vertex adjacency - glTF splits vertices at every UV seam."""
    V = np.asarray(mesh.vertices, float)
    n = len(V)
    scale = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    keys = np.round(V / (weld_tol * scale)).astype(np.int64)
    adj = [set() for _ in range(n)]
    for i, j in mesh.edges_unique:
        adj[i].add(j)
        adj[j].add(i)
    bucket = defaultdict(list)
    for i in range(n):
        bucket[tuple(keys[i])].append(i)
    for members in bucket.values():
        if len(members) < 2:
            continue
        union = set()
        for m in members:
            union |= adj[m]
        for m in members:
            adj[m] = union
    return adj


def vote(values, adj, rounds):
    """Majority vote over neighbours - removes speckle without moving real boundaries."""
    cur = values.copy()
    for _ in range(rounds):
        nxt = cur.copy()
        for i, nb in enumerate(adj):
            if not nb:
                continue
            c = defaultdict(int)
            for j in nb:
                c[cur[j]] += 1
            best = max(c.items(), key=lambda kv: kv[1])
            if best[1] * 2 > len(nb):
                nxt[i] = best[0]
        cur = nxt
    return cur


def main(a):
    src = trimesh.load(a.source, force="mesh", process=False)
    dst = trimesh.load(a.retopo, force="mesh", process=False)
    P = json.load(open(a.parts))
    labels = np.asarray(P["labels"])
    groups = P["group_ids"]
    gid = dict(groups) if isinstance(groups, dict) else {n: i for i, n in enumerate(groups)}
    inv = {v: k for k, v in gid.items()}

    sv = np.asarray(src.vertices, float)
    dv = np.asarray(dst.vertices, float)
    if len(labels) != len(sv):
        raise SystemExit(f"labels {len(labels)} != source vertices {len(sv)}")

    # nearest source vertex, then vote over the retopo surface to clean the lookup
    _, idx = cKDTree(sv).query(dv, k=1)
    lab = labels[idx]
    adj = adjacency(dst)
    lab = vote(lab, adj, a.vote_rounds)
    before = {inv.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(lab, return_counts=True))}
    print(f"[labels] transferred: {before}", flush=True)

    moved = 0
    col = vertex_colours(dst)
    if col is not None and "hair" in gid and "body" in gid:
        lab_col = to_lab(np.clip(col, 0, 1))
        hm = lab == gid["hair"]
        if hm.sum() > 50:
            hl = lab_col[hm]
            hair_seed = hl[hl[:, 0] <= np.percentile(hl[:, 0], 10)].mean(0)
            # The skin reference is the character's own skin - body vertices with a skin-like
            # chroma - not the brightest, reddest part of the hair group. That older seed assumed
            # dark hair: on Vex's pink hair the bright pink WAS the reddest part, and 89% of the
            # hair was moved into the face. A hair vertex now moves only if it is close to real
            # skin in absolute terms, and nearer to it than to the hair.
            bm = lab == gid["body"]
            bl = lab_col[bm]
            bl = bl[(bl[:, 1] > 4) & (bl[:, 2] > 4) & (bl[:, 0] > 25) & (bl[:, 0] < 90)]
            if len(bl) >= 50:
                skin_seed = np.median(bl, axis=0)
            else:                                   # no usable body colour: the old in-group seed
                cand = hl[(hl[:, 1] >= np.percentile(hl[:, 1], 90)) &
                          (hl[:, 0] >= np.percentile(hl[:, 0], 50))]
                if len(cand) < 20:
                    cand = hl[hl[:, 0] >= np.percentile(hl[:, 0], 90)]
                skin_seed = cand.mean(0)
            where = np.where(hm)[0]
            d_skin = np.linalg.norm(lab_col[where] - skin_seed, axis=1)
            is_skin = (d_skin < np.linalg.norm(lab_col[where] - hair_seed, axis=1)) & (d_skin < 15.0)
            if is_skin.mean() > 0.5:
                # moving most of a group is not rescuing fragments: the hair is skin-coloured
                print(f"[labels] {is_skin.mean():.0%} of the hair is close to skin tone - taking "
                      "the hair colour to be close to skin and leaving it labelled hair", flush=True)
                is_skin[:] = False
            lab[where[is_skin]] = gid["body"]
            moved = int(is_skin.sum())
            # re-vote so the reclassified patch has clean edges
            lab = vote(lab, adj, 2)
            print(f"[labels] hair seed L*a*b* {np.round(hair_seed,1)}  "
                  f"skin seed {np.round(skin_seed,1)}", flush=True)
            print(f"[labels] reclassified {moved:,} face-coloured vertices out of hair "
                  f"({100*moved/max(int(hm.sum()),1):.1f}% of the hair group)", flush=True)

    after = {inv.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(lab, return_counts=True))}
    print(f"[labels] final: {after}", flush=True)
    out = {"retopo": a.retopo, "source": a.source, "group_ids": gid,
           "vertices": int(len(dv)), "reclassified_from_hair": moved,
           "counts": after, "labels": [int(x) for x in lab]}
    json.dump(out, open(a.out, "w"))
    print(f"[labels] -> {a.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retopo", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vote-rounds", type=int, default=3)
    main(ap.parse_args())
