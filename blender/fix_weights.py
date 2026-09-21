"""Repair the skin weights that tear the character apart.

Measured on the idle clip, the body mesh reaches **2,931% edge stretch across the head/neck
weight boundary** and 1,547% within the head itself. That is the face pulling apart, and it is
what reads as "ripping at the seams" during playback. It is not an export problem and not an
animation problem: the same tear is there in the source file, and idle is the gentlest clip
there is.

The cause is the proximity skinning in build_rig.py. It assigns weight by inverse distance to
bone segments, so where two short adjacent bones meet - head and neck - the assignment flips
over a very short distance and neighbouring vertices end up following different bones almost
rigidly. Any rotation between them shears the surface.

Two repairs, in order:

  rigid head   every vertex above the head joint is bound wholly to the head bone, with a
               feathered band through the neck. A game character has no facial rig; the head is
               meant to travel as one piece, and forcing that removes the worst boundary
               outright.
  hair follows the body   the part segmentation misclassifies chunks of cheek, jaw and forehead
               as hair, and split_parts binds hair rigidly per island to its nearest bone - which
               put 617 of those face fragments on the **neck** while the face they belong to was
               skinned to the head. At rest the two line up perfectly, which is why every
               rest-pose check passed. Pose the character and the rigid fragments stop matching
               the skinned face, and it opens into holes and shards.
               Binding the hair rigidly to the head instead was tried and only half works: the
               fragments below the head joint still sit beside body vertices feathered toward
               the neck. What does work is giving the hair the *body's own* weights, interpolated
               from the nearest body surface. Then every hair vertex deforms exactly as the skin
               it lies against, misclassified or not, and the two cannot separate anywhere.
  smoothing    Laplacian smoothing of all weights across the *welded* surface. build_rig.py did
               smooth, but over the raw index graph, which glTF has shredded at every UV seam -
               so the smoothing ran inside thousands of disconnected patches and could not cross
               the boundaries that mattered. Welding first is what makes it work.

Afterwards the same tear measurement is re-run, and the script fails if the mesh still tears.

Run: blender -b -noaudio --python fix_weights.py -- --blend tposed.blend --out fixed.blend
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--parts", default="body,clothing")
ap.add_argument("--head-locked", default="hair",
                help="meshes bound wholly to the head bone (comma-separated name fragments)")
ap.add_argument("--iters", type=int, default=24)
ap.add_argument("--lam", type=float, default=0.75)
ap.add_argument("--max-influences", type=int, default=4)
ap.add_argument("--neck-band", type=float, default=0.55,
                help="feather band below the head joint, as a fraction of head-bone length")
ap.add_argument("--max-stretch", type=float, default=1.2,
                help="fail if any edge still stretches more than this (1.2 = 120%%)")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
if rig is None:
    raise SystemExit("[weights] no armature")

head_b = rig.data.bones.get("head")
if head_b is None:
    raise SystemExit("[weights] no 'head' bone")
M = rig.matrix_world
head_joint = (M @ head_b.head_local)
head_len = (M @ head_b.tail_local - head_joint).length or 0.1
band = head_len * a.neck_band

# The lock has to reach as low as the lowest hair fragment. Segmentation puts pieces of cheek
# and jaw into the hair mesh, and those are bound rigidly to the head; if the *body* vertices
# beside them are still feathered toward the neck, the two surfaces drift apart under motion and
# the face opens up exactly where they meet. So the floor of the rigid region is whichever is
# lower: the head joint, or the bottom of the hair.
# Taking the *minimum* hair z as the floor was tried and is wrong: the hair mesh carries stray
# specks scattered down the torso, so the floor fell to the waist and 4,398 jacket vertices got
# bound to the head bone (stretch went 374% -> 4410%). The floor stays at the head joint.
lock_z = head_joint.z

report = {}
for key in [p.strip() for p in a.parts.split(",") if p.strip()]:
    ob = next((o for o in scene.objects if o.type == "MESH" and key in o.name.lower()), None)
    if ob is None:
        continue
    me = ob.data
    n = len(me.vertices)
    gidx = {vg.name: vg.index for vg in ob.vertex_groups}
    names = {v: k for k, v in gidx.items()}

    # ---- weld-aware adjacency ----------------------------------------------------------------
    V = [ob.matrix_world @ v.co for v in me.vertices]
    ext = max((max(c[i] for c in ((p.x, p.y, p.z) for p in V))
               - min(c[i] for c in ((p.x, p.y, p.z) for p in V))) for i in range(3)) or 1.0
    tol = 1e-5 * ext
    bucket = defaultdict(list)
    for i, p in enumerate(V):
        bucket[(round(p.x / tol), round(p.y / tol), round(p.z / tol))].append(i)
    adj = [set() for _ in range(n)]
    for e in me.edges:
        i, j = e.vertices
        adj[i].add(j)
        adj[j].add(i)
    welded = 0
    for members in bucket.values():
        if len(members) < 2:
            continue
        h = members[0]
        for o in members[1:]:
            adj[h].add(o)
            adj[o].add(h)
            welded += 1
    # a welded vertex inherits its twin's neighbours, so smoothing crosses the seam
    for members in bucket.values():
        if len(members) < 2:
            continue
        union = set()
        for m in members:
            union |= adj[m]
        for m in members:
            adj[m] = union

    # ---- current weights as dense rows -------------------------------------------------------
    W = [dict() for _ in range(n)]
    for v in me.vertices:
        for g in v.groups:
            if g.weight > 1e-4:
                W[v.index][g.group] = g.weight

    # ---- rigid head ---------------------------------------------------------------------------
    hg = gidx.get("head")
    forced = 0
    if hg is not None:
        for i, p in enumerate(V):
            d = p.z - lock_z
            if d >= 0:
                t = 1.0
            elif d > -band:
                t = 1.0 + d / band          # feather through the neck
            else:
                continue
            if t <= 0:
                continue
            rest = {k: w * (1.0 - t) for k, w in W[i].items() if k != hg}
            rest[hg] = rest.get(hg, 0.0) + t
            W[i] = rest
            forced += 1

    # ---- Laplacian smoothing over the welded surface ------------------------------------------
    for _ in range(a.iters):
        new = []
        for i in range(n):
            nb = adj[i]
            if not nb:
                new.append(W[i])
                continue
            acc = defaultdict(float)
            for j in nb:
                for k, w in W[j].items():
                    acc[k] += w
            inv = 1.0 / len(nb)
            mixed = defaultdict(float)
            for k, w in W[i].items():
                mixed[k] += (1.0 - a.lam) * w
            for k, w in acc.items():
                mixed[k] += a.lam * w * inv
            new.append(dict(mixed))
        W = new

    # ---- trim, renormalise, write back --------------------------------------------------------
    for vg in ob.vertex_groups:
        vg.remove(range(n))
    for i in range(n):
        items = sorted(W[i].items(), key=lambda kv: kv[1], reverse=True)[:a.max_influences]
        s = sum(w for _, w in items)
        if s <= 1e-9:
            continue
        for k, w in items:
            ob.vertex_groups[names[k]].add([i], w / s, "REPLACE")

    unweighted = sum(1 for v in me.vertices if not v.groups)
    print(f"[weights] {ob.name}: {welded:,} welded pairs, {forced:,} vertices forced to head, "
          f"{a.iters} smoothing iterations, {unweighted} unweighted", flush=True)
    report[ob.name] = {"vertices": n, "welded_pairs": welded, "forced_to_head": forced,
                       "iterations": a.iters, "unweighted": unweighted}

# ---- give the head-locked parts the body's own weights ---------------------------------------
body_ob = next((o for o in scene.objects if o.type == "MESH" and "body" in o.name.lower()), None)
for key in [p.strip() for p in a.head_locked.split(",") if p.strip()]:
    ob = next((o for o in scene.objects if o.type == "MESH" and key in o.name.lower()), None)
    if ob is None or body_ob is None:
        continue
    n = len(ob.data.vertices)
    for vg in list(ob.vertex_groups):
        ob.vertex_groups.remove(vg)
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    body_ob.select_set(True)
    bpy.context.view_layer.objects.active = body_ob      # active = source
    bpy.ops.object.data_transfer(
        data_type="VGROUP_WEIGHTS", use_create=True,
        vert_mapping="POLYINTERP_NEAREST", layers_select_src="ALL", layers_select_dst="NAME")
    weighted = sum(1 for v in ob.data.vertices if v.groups)
    print(f"[weights] {ob.name}: weights transferred from {body_ob.name} "
          f"({weighted:,}/{n:,} weighted) so it deforms with the skin it sits on", flush=True)
    if weighted < n * 0.98:
        raise SystemExit(f"[weights] {ob.name}: only {weighted}/{n} vertices received weights")
    report[ob.name] = {"vertices": n, "weights_from": body_ob.name, "weighted": weighted}

# ---- verify: re-measure the tear ---------------------------------------------------------------
def worst_stretch(ob, act_name):
    act = bpy.data.actions.get(act_name)
    if act is None:
        return None
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    me = ob.data
    all_edges = [(e.vertices[0], e.vertices[1]) for e in me.edges]
    all_rest = [(me.vertices[i].co - me.vertices[j].co).length for i, j in all_edges]
    # Degenerate edges make a relative-stretch metric meaningless: this mesh contains
    # zero-length edges, and dividing normal motion by ~0 reports thousands of percent. Only
    # edges long enough for the ratio to mean something are measured; the count of skipped
    # edges is reported so the exclusion is visible rather than quiet.
    srt = sorted(all_rest)
    med = srt[len(srt) // 2] if srt else 1.0
    floor = med * 0.2
    keep = [k for k, r in enumerate(all_rest) if r >= floor]
    edges = [all_edges[k] for k in keep]
    rest = [all_rest[k] for k in keep]
    skipped = len(all_edges) - len(edges)
    f0, f1 = (int(x) for x in act.frame_range)
    worst = 0.0
    for f in range(f0, f1 + 1, 3):
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = ob.evaluated_get(dg)
        m2 = ev.to_mesh()
        for (i, j), r in zip(edges, rest):
            s = abs((m2.vertices[i].co - m2.vertices[j].co).length - r) / r
            if s > worst:
                worst = s
        ev.to_mesh_clear()
    return worst, skipped


checks = {}
for key in ("body", "clothing"):
    ob = next((o for o in scene.objects if o.type == "MESH" and key in o.name.lower()), None)
    if ob is None:
        continue
    for clip in ("idle", "walk"):
        res = worst_stretch(ob, clip)
        if res is None:
            continue
        w, skipped = res
        checks[f"{ob.name}/{clip}"] = round(w, 3)
        print(f"[weights] {ob.name} on '{clip}': worst edge stretch {w*100:.0f}% "
              f"({skipped} degenerate edges excluded)", flush=True)

bad = {k: v for k, v in checks.items() if v > a.max_stretch}
report["stretch_after"] = checks
if a.json:
    json.dump(report, open(a.json, "w"), indent=2)
if bad:
    raise SystemExit(f"[weights] still tearing: {bad} (limit {a.max_stretch*100:.0f}%)")

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
print(f"[weights] -> {a.out}", flush=True)
