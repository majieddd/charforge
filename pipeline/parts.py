"""Semantic part segmentation of a character mesh by multi-view label back-projection.

Why this exists: a generated character arrives as one fused mesh. Skinned naively, a jacket
hem welds to the legs, a bag strap stretches with the spine, and long hair smears across the
shoulders. Splitting the mesh into semantic parts is what lets the rigging stage bind each
part the way a game pipeline would: hair rigid to the head, accessories to a single bone,
clothing weighted from the body beneath it.

Method: render N orbit views (blender/render_views.py), run a human-parsing segmentation
network on each view, read each visible vertex's label from its pixel, accumulate votes
across views, then smooth the result over the mesh graph so parts are connected regions.

Input : views dir (view_XX.png, projection.json, meta.json)
Output: parts.json  {labels: [per-vertex label id], label_names: {...}, stats: {...}}
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PARSER_REPO = "fashn-ai/fashn-human-parser"

# 18 parser classes -> the part groups that matter for rigging
GROUPS = {
    "body": ["face", "arms", "hands", "legs", "feet", "torso"],
    "hair": ["hair"],
    "clothing": ["top", "dress", "skirt", "pants", "belt", "scarf"],
    "accessory": ["bag", "hat", "glasses", "jewelry"],
}
CLASS_TO_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}
GROUP_IDS = {"body": 0, "clothing": 1, "hair": 2, "accessory": 3}

# A thin accessory lying on a garment - a bag strap across a sleeve - is seen as the garment
# from most angles and as the accessory from few, so plain majority voting erases it. Accessory
# and hair classes are specific and rarely guessed by accident, so their votes count for more.
CLASS_VOTE_WEIGHT = {"accessory": 1.8, "hair": 1.8, "clothing": 1.0, "body": 1.0}


def parse_views(views_dir: Path, device="mps", batch=4):
    """Per-view pixel label maps from the human parser."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc = SegformerImageProcessor.from_pretrained(PARSER_REPO)
    model = SegformerForSemanticSegmentation.from_pretrained(PARSER_REPO).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    files = sorted(views_dir.glob("view_*.png"))
    maps = {}
    with torch.no_grad():
        for f in files:
            im = Image.open(f).convert("RGBA")
            # composite on mid-grey: the parser was trained on photos, not alpha cut-outs
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(bg, im).convert("RGB")
            inp = proc(images=rgb, return_tensors="pt").to(device)
            logits = model(**inp).logits
            up = torch.nn.functional.interpolate(logits, size=rgb.size[::-1], mode="bilinear", align_corners=False)
            maps[f.stem] = up.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return maps, id2label


def mesh_adjacency(mesh_path: str, n_vertices: int, weld_tol: float = 1e-5):
    """Vertex adjacency over the *welded* surface.

    glTF duplicates vertices at every UV and normal seam, so the raw index graph of a generated
    mesh falls apart into thousands of disconnected patches - on this character the largest
    connected body region was 731 vertices out of 52,808. Any reasoning about regions (flood
    fill, smoothing, island removal, part separation) is meaningless on that graph. Linking
    co-located vertices restores the real surface topology without changing the geometry.
    """
    import trimesh
    m = trimesh.load(mesh_path, force="mesh", process=False)
    V = np.asarray(m.vertices, dtype=np.float64)
    n = max(n_vertices, len(V))
    adj = [[] for _ in range(n)]
    for a, b in m.edges_unique:
        adj[a].append(b)
        adj[b].append(a)

    scale = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    keys = np.round(V / (weld_tol * scale)).astype(np.int64)
    seam = {}
    for i in range(len(V)):
        seam.setdefault((keys[i, 0], keys[i, 1], keys[i, 2]), []).append(i)
    welded = 0
    for members in seam.values():
        if len(members) < 2:
            continue
        head = members[0]
        for other in members[1:]:
            adj[head].append(other)
            adj[other].append(head)
            welded += 1
    print(f"[parts] welded {welded:,} seam duplicates across {len(seam):,} positions", flush=True)
    return adj, m


def backproject(views_dir: Path, maps, id2label):
    meta = json.load(open(views_dir / "meta.json"))
    proj = json.load(open(views_dir / "projection.json"))
    n = meta["n_vertices"]
    n_classes = len(id2label)
    votes = np.zeros((n, n_classes), dtype=np.float32)
    for v in meta["views"]:
        i = v["index"]
        lab = maps[f"view_{i:02d}"]
        idx = np.asarray(proj[f"idx_{i:02d}"], dtype=np.int64)
        px = np.asarray(proj[f"px_{i:02d}"], dtype=np.int64)
        py = np.asarray(proj[f"py_{i:02d}"], dtype=np.int64)
        if len(idx) == 0:
            continue
        px = np.clip(px, 0, lab.shape[1] - 1)
        py = np.clip(py, 0, lab.shape[0] - 1)
        cls = lab[py, px]
        # front-facing views are more reliable for the parser than steep back views
        np.add.at(votes, (idx, cls), 1.0)
    return votes, meta


def flood(labels, adj, max_iters=60):
    """Propagate labels into unlabelled regions along the mesh graph."""
    lab = labels.copy()
    for _ in range(max_iters):
        todo = np.where(lab < 0)[0]
        if len(todo) == 0:
            break
        changed = 0
        new = lab.copy()
        for v in todo:
            c = collections.Counter(lab[n] for n in adj[v] if lab[n] >= 0)
            if c:
                new[v] = c.most_common(1)[0][0]
                changed += 1
        lab = new
        if not changed:
            break
    return lab


def smooth(labels, adj, iters=6):
    """Majority filter over the mesh graph: removes speckle, makes parts connected."""
    lab = labels.copy()
    for _ in range(iters):
        new = lab.copy()
        for v, nb in enumerate(adj):
            if not nb:
                continue
            c = collections.Counter(lab[n] for n in nb)
            c[lab[v]] += 1.2           # slight self-bias so stable regions do not flip
            new[v] = c.most_common(1)[0][0]
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def despeckle(labels, adj, min_frac=0.0015, min_abs=120):
    """Reassign small islands to the label surrounding them.

    Multi-view voting produces speckle: a few hundred stray "body" vertices scattered through a
    jacket, or accessory fragments inside hair. Those islands are invisible in a render but they
    wreck the split, because each becomes its own object bound to its own bone. A part should be
    a contiguous region, so anything below a size floor is absorbed by its neighbours.
    """
    n = len(labels)
    floor = max(min_abs, int(min_frac * n))
    lab = labels.copy()
    for _ in range(3):
        seen = np.zeros(n, dtype=bool)
        reassigned = 0
        for start in range(n):
            if seen[start]:
                continue
            target = lab[start]
            stack, comp = [start], []
            seen[start] = True
            while stack:
                v = stack.pop()
                comp.append(v)
                for nb in adj[v]:
                    if not seen[nb] and lab[nb] == target:
                        seen[nb] = True
                        stack.append(nb)
            if len(comp) >= floor:
                continue
            border = collections.Counter()
            cs = set(comp)
            for v in comp:
                for nb in adj[v]:
                    if nb not in cs:
                        border[lab[nb]] += 1
            if border:
                new = border.most_common(1)[0][0]
                if new != target:
                    for v in comp:
                        lab[v] = new
                    reassigned += len(comp)
        if not reassigned:
            break
    return lab


def run(views_dir, out=None, device="mps"):
    views_dir = Path(views_dir)
    maps, id2label = parse_views(views_dir, device=device)
    votes, meta = backproject(views_dir, maps, id2label)

    # class votes -> group votes
    n = votes.shape[0]
    gvotes = np.zeros((n, len(GROUP_IDS)), dtype=np.float32)
    for cid, cname in id2label.items():
        g = CLASS_TO_GROUP.get(cname)
        if g is None:      # background
            continue
        gvotes[:, GROUP_IDS[g]] += votes[:, cid] * CLASS_VOTE_WEIGHT.get(g, 1.0)

    visible = votes.sum(1) > 0
    seen = gvotes.sum(1) > 0
    labels = np.where(seen, gvotes.argmax(1), -1).astype(np.int32)

    adj, m = mesh_adjacency(meta["mesh"], n)
    # Unseen vertices (interior geometry, crevices, under the arms) must inherit from
    # labelled neighbours rather than defaulting to "body" - a default would silently
    # swallow the inside of the hair and jacket.
    labels = flood(labels, adj)
    labels[labels < 0] = GROUP_IDS["body"]
    labels = smooth(labels, adj, iters=8)
    before_islands = None
    labels = despeckle(labels, adj)

    # per-class detail (kept for the report / UI)
    cls_labels = np.where(votes.sum(1) > 0, votes.argmax(1), 0).astype(np.int32)

    stats = {g: int((labels == i).sum()) for g, i in GROUP_IDS.items()}
    cls_stats = {id2label[c]: int((cls_labels == c).sum()) for c in sorted(set(cls_labels.tolist()))}
    payload = {"mesh": meta["mesh"], "n_vertices": int(n),
               "never_visible": int((~visible).sum()), "visible_but_background": int((visible & ~seen).sum()),
               "group_ids": GROUP_IDS, "labels": labels.tolist(), "class_labels": cls_labels.tolist(),
               "id2label": id2label, "stats": stats, "class_stats": cls_stats}
    out = out or (views_dir.parent / "parts.json")
    json.dump(payload, open(out, "w"))
    print(f"[parts] {stats}  never_visible={int((~visible).sum())} "
          f"bg_only={int((visible & ~seen).sum())} -> {out}")
    print(f"[parts] parser classes: {cls_stats}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    run(a.views, a.out, a.device)
