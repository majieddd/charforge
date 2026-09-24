"""Give the parts that hang their own bones, so they swing: braids, ponytails, bags.

    blender -b --python springs.py -- --blend rig.blend --labels labels.json --out rig.blend \
        --json springs.json

A braid bound rigidly to the head reads as a carving; a satchel bound to the spine reads as a
piece of the jacket. Games give such parts short bone chains driven by a spring simulation at
runtime (Unity's Dynamic Bone and Magica Cloth, Unreal's AnimDynamics, VRM's spring bones) -
the animation clips never key them, the simulation does. This stage finds the parts and builds
the chains; the playground simulates them, and springs.json (copied into the manifest) carries
the same chains and settings for any other engine.

What hangs, found from the skin weights and labels rather than guessed:

  hair  vertices bound mostly to the head that sit below the neck - a braid over the shoulder, a
        ponytail down the back. A bob that stops at the jaw has none, and gets no chain.
  bag   accessory vertices below the waist - the body of a satchel; its strap stays on the torso.

Each connected piece longer than a few centimetres becomes a chain: its principal axis is cut
into segments, the joints go on the piece's centre line, and its vertices are weighted along the
chain - the first segment blending into the bone it hangs from, so the root does not tear.
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--labels", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--json", required=True)
ap.add_argument("--min-length", type=float, default=0.035, help="shortest chain, fraction of height")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH" and o.find_armature() == rig)
me = mesh.data
n = len(me.vertices)
V = np.empty(n * 3)
me.vertices.foreach_get("co", V)
V = V.reshape(-1, 3)
M = np.array(mesh.matrix_world)
V = V @ M[:3, :3].T + M[:3, 3]
H = float(V[:, 2].max() - V[:, 2].min())
names = [g.name for g in mesh.vertex_groups]
W = np.zeros((n, len(names)))
for v in me.vertices:
    for g in v.groups:
        W[v.index, g.group] = g.weight
gi = {nm: i for i, nm in enumerate(names)}
head = lambda b: np.array((rig.matrix_world @ rig.data.bones[b].head_local)[:])

# the mesh graph, welded across UV seams
_, weld = np.unique(np.round(V / (1e-6 * H)).astype(np.int64), axis=0, return_inverse=True)
weld = weld.ravel()
E = np.empty(len(me.edges) * 2, np.int64)
me.edges.foreach_get("vertices", E)
E = weld[E.reshape(-1, 2)]
nw = int(weld.max()) + 1
nbr = [[] for _ in range(nw)]
for i, j in E:
    if i != j:
        nbr[i].append(j)
        nbr[j].append(i)

lab = None
gids = {}
if a.labels and os.path.exists(a.labels):
    Lj = json.load(open(a.labels))
    if len(Lj["labels"]) == n:
        lab = np.asarray(Lj["labels"])
        gids = Lj["group_ids"]

neck_z = head("neck")[2]
waist_z = head("spine1")[2]
cands = []
# hair: head-bound below the neck (the hair label, when the parser found it, counts too)
hw = W[:, gi["head"]] if "head" in gi else np.zeros(n)
is_hair = (hw > 0.5) | ((lab == gids["hair"]) if lab is not None and "hair" in gids else False)
cands.append(("hair", "head", is_hair & (V[:, 2] < neck_z - 0.01 * H)))
if lab is not None and "accessory" in gids:
    cands.append(("bag", "spine1", (lab == gids["accessory"]) & (V[:, 2] < waist_z)))


def pieces(mask):
    """Connected pieces of the masked vertices, on the welded graph."""
    mw = np.zeros(nw, bool)
    mw[weld[mask]] = True
    seen = np.zeros(nw, bool)
    out = []
    for s0 in np.nonzero(mw)[0]:
        if seen[s0]:
            continue
        comp, st = [], [s0]
        seen[s0] = True
        while st:
            u = st.pop()
            comp.append(u)
            for w_ in nbr[u]:
                if mw[w_] and not seen[w_]:
                    seen[w_] = True
                    st.append(w_)
        out.append(np.array(comp))
    return out


chains = []
rejected = 0
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
eb = rig.data.edit_bones
new_groups = []
for kind, parent, mask in cands:
    for comp in pieces(mask):
        vids = np.nonzero(np.isin(weld, comp))[0]
        X = V[vids]
        if len(vids) < 40:
            continue
        c = X.mean(0)
        u, s_, vt = np.linalg.svd(X - c, full_matrices=False)
        ax = vt[0]
        if ax[2] > 0:                                    # point the axis down, root at the top
            ax = -ax
        t = (X - c) @ ax
        length = float(t.max() - t.min())
        if length < a.min_length * H:
            continue
        # Hanging free, or fused to the body along its length? A ponytail meets the rest of the mesh
        # only where it leaves the head - its border is at the root. A braid generated lying in a
        # hood is one surface with it all the way down (Wren's), and a chain on it would tear that
        # surface on the first swing; it stays rigid. Hair must also hang: a curtain wider than it
        # is long (the back of a bob) is not a tail.
        inside = np.zeros(nw, bool)
        inside[comp] = True
        border_nodes = np.array([u for u in comp if any(not inside[w_] for w_ in nbr[u])], np.int64)
        f_along = (t - t.min()) / max(length, 1e-9)
        f_border = f_along[np.isin(weld[vids], border_nodes)] if len(border_nodes) else np.zeros(0)
        fused = float((f_border > 0.35).mean()) if len(f_border) else 0.0
        if fused > 0.2:
            rejected += 1
            print(f"[springs] {kind} piece ({length / H * 175:.0f} cm at 1.75 m) is joined to the body along "
                  f"its length ({fused:.0%} of its border below the root) - left rigid", flush=True)
            continue
        if kind == "hair" and (s_[0] < 1.6 * s_[1] or abs(ax[2]) < 0.5):
            print(f"[springs] hair piece wider than it hangs (axis {ax.round(2).tolist()}) - left rigid", flush=True)
            continue
        nseg = int(np.clip(round(length / (0.06 * H)), 2, 5))
        edges_t = np.linspace(t.min(), t.max(), nseg + 1)
        joints = []
        for k in range(nseg + 1):
            lo_ = edges_t[max(k - 1, 0)] if k else edges_t[0]
            hi_ = edges_t[min(k + 1, nseg)]
            sel = (t >= lo_) & (t <= hi_)
            p = X[sel].mean(0) if sel.any() else c + ax * edges_t[k]
            # keep the joint on the axis position for its slice, centred across it
            p = p + ax * (edges_t[k] - (p - c) @ ax)
            joints.append(p)
        idx = len(chains) + 1
        bnames = []
        prev = eb[parent]
        for k in range(nseg):
            b = eb.new(f"spring_{kind}{idx}_{k + 1}")
            b.head, b.tail = Vector(joints[k]), Vector(joints[k + 1])
            b.parent = prev
            b.use_connect = k > 0
            prev = b
            bnames.append(b.name)
        # thickness: the piece's radius about its axis, for collisions
        radial = np.linalg.norm((X - c) - np.outer(t, ax), axis=1)
        chains.append({"kind": kind, "parent": parent, "bones": bnames,
                       "joints": [p.tolist() for p in joints], "vertices": vids.tolist(),
                       "t": t.tolist(), "edges_t": edges_t.tolist(),
                       "radius_h": float(np.percentile(radial, 60)) / H, "length_h": length / H})
bpy.ops.object.mode_set(mode="OBJECT")

# ---- weights along each chain ------------------------------------------------------------------
for ch in chains:
    vids = np.array(ch.pop("vertices"))
    t = np.array(ch.pop("t"))
    et = np.array(ch.pop("edges_t"))
    nseg = len(ch["bones"])
    for b in ch["bones"]:
        mesh.vertex_groups.new(name=b)
    f = np.clip((t - et[0]) / (et[-1] - et[0]), 0, 1) * nseg        # 0 .. nseg along the chain
    Wn = np.zeros((len(vids), nseg + 1))                           # column 0 = the parent bone
    for k in range(nseg):
        # bone k spans f in [k, k+1]; it owns its middle, blending into its neighbours
        centre = k + 0.5
        Wn[:, k + 1] = np.clip(1.0 - np.abs(f - centre), 0, 1)
    root = np.clip(1.0 - f / 0.6, 0, 1)                              # the first part stays on the parent
    Wn[:, 0] = root
    Wn /= np.maximum(Wn.sum(1, keepdims=True), 1e-9)
    # the parent's share keeps whatever the vertex had before (head, spine...), scaled down
    old = W[vids] * Wn[:, :1]
    for j, vi in enumerate(vids):
        for g in mesh.vertex_groups:
            if g.name.startswith("spring_"):
                continue
            w0 = old[j, gi[g.name]] if g.name in gi else 0.0
            if w0 > 1e-4:
                g.add([int(vi)], float(w0), "REPLACE")
            elif g.name in gi and W[vi, gi[g.name]] > 0:
                g.remove([int(vi)])
        for k, b in enumerate(ch["bones"]):
            w = Wn[j, k + 1]
            if w > 1e-4:
                mesh.vertex_groups[b].add([int(vi)], float(w), "REPLACE")
    # runtime settings: hair is light and lively, a bag heavier and slower
    if ch["kind"] == "hair":
        ch.update({"stiffness": 0.55, "drag": 0.35, "gravity": 0.6})
    else:
        ch.update({"stiffness": 0.8, "drag": 0.55, "gravity": 1.0})
    ch["vertices"] = int(len(vids))
    print(f"[springs] {ch['kind']}: {nseg}-bone chain from {ch['parent']}, {ch['length_h'] * 175:.0f} cm at "
          f"1.75 m, {len(vids):,} vertices", flush=True)

if not chains:
    print("[springs] no chains: " + (f"{rejected} hanging part(s) are joined to the body and stay rigid"
          if rejected else "nothing hangs below the neck or the waist"), flush=True)
# colliders: spheres a chain may not pass through, sized from the body around each joint
colliders = []
for b, r in (("head", 0.075), ("neck", 0.05), ("spine3", 0.11), ("spine2", 0.11),
             ("left_collar", 0.06), ("right_collar", 0.06), ("pelvis", 0.11)):
    if b in rig.data.bones:
        colliders.append({"bone": b, "radius_h": r / 1.75})           # a fraction of height
for ch in chains:
    ch.pop("joints", None)                            # the armature carries them from here on
json.dump({"chains": chains, "colliders": colliders,
           "note": "radius_h and length_h are fractions of the character's height"},
          open(a.json, "w"), indent=1)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
print(f"[springs] {len(chains)} chain(s) -> {a.out}, {a.json}", flush=True)
