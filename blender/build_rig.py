"""Build an armature for a character mesh and skin it with per-part rules.

The point of the part labels: a single automatic-weight bind welds everything to whatever bone
is nearest, which is why AI-generated characters so often show hair smearing across the
shoulders and straps stretching with the spine. Here:

  body, clothing -> bone-heat automatic weights (smooth deformation across joints)
  hair           -> rigid to the head bone (a hair mesh should travel with the skull, not shear)
  accessory      -> rigid to the single nearest bone (straps, bags, glasses keep their shape)

Then influences are limited to 4 per vertex and normalised, which is what game engines expect.

Run: blender -b -noaudio --python build_rig.py -- --mesh m.glb --joints joints.json
     [--parts parts.json] --out rigged.glb
"""
import argparse
import json
import os
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--joints", required=True)
ap.add_argument("--parts", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--blend", default=None)
a = ap.parse_args(argv)

J = json.load(open(a.joints))
JOINTS = {k: mathutils.Vector(v) for k, v in J["joints"].items()}
PARENTS = J["parents"]
PARTS = json.load(open(a.parts)) if a.parts and os.path.exists(a.parts) else None

# bone -> (head joint, tail joint)
CHAIN = [
    ("pelvis", "pelvis", "spine1"), ("spine1", "spine1", "spine2"), ("spine2", "spine2", "spine3"),
    ("spine3", "spine3", "neck"), ("neck", "neck", "head"), ("head", "head", "head_top"),
    ("left_collar", "left_collar", "left_shoulder"), ("left_shoulder", "left_shoulder", "left_elbow"),
    ("left_elbow", "left_elbow", "left_wrist"), ("left_wrist", "left_wrist", "left_hand"),
    ("right_collar", "right_collar", "right_shoulder"), ("right_shoulder", "right_shoulder", "right_elbow"),
    ("right_elbow", "right_elbow", "right_wrist"), ("right_wrist", "right_wrist", "right_hand"),
    ("left_hip", "left_hip", "left_knee"), ("left_knee", "left_knee", "left_ankle"),
    ("left_ankle", "left_ankle", "left_foot"),
    ("right_hip", "right_hip", "right_knee"), ("right_knee", "right_knee", "right_ankle"),
    ("right_ankle", "right_ankle", "right_foot"),
]
BONE_PARENT = {
    "pelvis": None, "spine1": "pelvis", "spine2": "spine1", "spine3": "spine2", "neck": "spine3", "head": "neck",
    "left_collar": "spine3", "left_shoulder": "left_collar", "left_elbow": "left_shoulder", "left_wrist": "left_elbow",
    "right_collar": "spine3", "right_shoulder": "right_collar", "right_elbow": "right_shoulder", "right_wrist": "right_elbow",
    "left_hip": "pelvis", "left_knee": "left_hip", "left_ankle": "left_knee",
    "right_hip": "pelvis", "right_knee": "right_hip", "right_ankle": "right_knee",
}


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    {".glb": bpy.ops.import_scene.gltf, ".gltf": bpy.ops.import_scene.gltf}.get(ext, None)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise SystemExit("unsupported mesh")
    ms = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for m in ms:
        m.select_set(True)
    bpy.context.view_layer.objects.active = ms[0]
    if len(ms) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def normalize_like_render(obj):
    """Match render_views.py's normalisation so joints.json lines up with the mesh."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    vs = [v.co for v in obj.data.vertices]
    mn = mathutils.Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
    mx = mathutils.Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
    ctr = (mn + mx) / 2
    s = 2.0 / max(1e-6, mx.z - mn.z)
    for v in obj.data.vertices:
        v.co = (v.co - ctr) * s
    obj.data.update()


def build_armature():
    arm = bpy.data.armatures.new("rig")
    rig = bpy.data.objects.new("rig", arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    made = {}
    for name, h, t in CHAIN:
        b = arm.edit_bones.new(name)
        b.head = JOINTS[h]
        b.tail = JOINTS[t]
        if (b.tail - b.head).length < 1e-4:
            b.tail = b.head + mathutils.Vector((0, 0, 0.02))
        made[name] = b
    for name, _, _ in CHAIN:
        p = BONE_PARENT.get(name)
        if p and p in made:
            made[name].parent = made[p]
            made[name].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def part_vertex_sets(obj):
    """vertex index sets per semantic group, from parts.json (empty when not supplied)."""
    if not PARTS:
        return {}
    labels = PARTS["labels"]
    gid = PARTS["group_ids"]
    inv = {v: k for k, v in gid.items()}
    sets = {g: set() for g in gid}
    n = len(obj.data.vertices)
    for i, l in enumerate(labels[:n]):
        sets[inv[int(l)]].add(i)
    return sets


def unweighted_fraction(obj, rig):
    bones = {b.name for b in rig.data.bones}
    gi = {g.index: g.name for g in obj.vertex_groups}
    n = len(obj.data.vertices)
    if n == 0:
        return 1.0
    bad = 0
    for v in obj.data.vertices:
        if not any(gi.get(g.group) in bones and g.weight > 1e-4 for g in v.groups):
            bad += 1
    return bad / n


def proximity_skin(obj, rig, power=4.0, max_infl=4):
    """Distance-to-bone skinning restricted to the skeleton's local neighbourhood.

    Plain inverse-distance weighting bleeds across the body - a hand resting near the hip
    picks up hip weight and the arm drags the pelvis. Restricting each vertex to its nearest
    bone plus that bone's parent/children keeps deformation local while still blending
    smoothly across a joint, and unlike bone heat it cannot fail on messy geometry.
    """
    import numpy as np

    bones = list(rig.data.bones)
    names = [b.name for b in bones]
    idx_of = {n: i for i, n in enumerate(names)}
    H = np.array([list(b.head_local) for b in bones], dtype=np.float64)
    T = np.array([list(b.tail_local) for b in bones], dtype=np.float64)

    # neighbourhood: self + parent + children (one hop on the bone graph)
    neigh = [{i} for i in range(len(bones))]
    for i, b in enumerate(bones):
        if b.parent is not None and b.parent.name in idx_of:
            p = idx_of[b.parent.name]
            neigh[i].add(p)
            neigh[p].add(i)

    V = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)

    seg = T - H
    L2 = np.einsum("ij,ij->i", seg, seg)
    L2[L2 < 1e-12] = 1e-12
    # distance from every vertex to every bone segment
    d = np.empty((len(V), len(bones)), dtype=np.float64)
    for j in range(len(bones)):
        w = V - H[j]
        t = np.clip((w @ seg[j]) / L2[j], 0.0, 1.0)
        proj = H[j] + t[:, None] * seg[j]
        d[:, j] = np.linalg.norm(V - proj, axis=1)

    primary = d.argmin(1)
    eps = 1e-4
    inv = 1.0 / np.power(d + eps, power)

    # dense weight matrix, masked to each vertex's local bone neighbourhood
    W = np.zeros((len(V), len(bones)), dtype=np.float32)
    for vi in range(len(V)):
        cand = sorted(neigh[primary[vi]])
        W[vi, cand] = inv[vi, cand]
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)

    # Laplacian smoothing across mesh edges. Without it, weights step abruptly at the
    # boundary between two bones' regions and the surface tears when that joint rotates -
    # which is exactly where the measured stretch was concentrated (the hip/waist band).
    E = np.empty(len(obj.data.edges) * 2, dtype=np.int32)
    obj.data.edges.foreach_get("vertices", E)
    E = E.reshape(-1, 2)
    deg = np.bincount(E.ravel(), minlength=len(V)).astype(np.float32)
    lam, iters = 0.65, 12
    for _ in range(iters):
        S = np.zeros_like(W)
        np.add.at(S, E[:, 0], W[E[:, 1]])
        np.add.at(S, E[:, 1], W[E[:, 0]])
        S /= np.maximum(deg, 1)[:, None]
        W = (1.0 - lam) * W + lam * S
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)

    # sparsify to the engine budget
    order = np.argsort(-W, axis=1)[:, :max_infl]
    Wt = np.zeros_like(W)
    rows = np.arange(len(V))[:, None]
    Wt[rows, order] = W[rows, order]
    Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-12)

    weights = {n: {} for n in names}          # bone -> {quantised weight: [vertex ids]}
    Q = np.round(Wt, 2)
    for bi, bname in enumerate(names):
        col = Q[:, bi]
        nz = np.where(col >= 0.01)[0]
        if len(nz) == 0:
            continue
        for q in np.unique(col[nz]):
            ids = nz[col[nz] == q]
            weights[bname][float(q)] = ids.tolist()

    # write in bulk: one add() per (bone, quantised weight) instead of per vertex
    groups = {g.name: g for g in obj.vertex_groups}
    for n in names:
        if n not in groups:
            groups[n] = obj.vertex_groups.new(name=n)
    for n, buckets in weights.items():
        for q, ids in buckets.items():
            groups[n].add(ids, q, "REPLACE")
    print(f"[rig] proximity skin: {len(V)} vertices, mean influences "
          f"{sum(len(i) for b in weights.values() for i in b.values())/max(1,len(V)):.2f}")


def dominant_bones(obj, rig):
    """vertex index -> the bone holding its largest weight (after the smooth skin pass)."""
    bones = {b.name for b in rig.data.bones}
    gi = {g.index: g.name for g in obj.vertex_groups}
    out = {}
    for v in obj.data.vertices:
        best, bw = None, 0.0
        for g in v.groups:
            n = gi.get(g.group)
            if n in bones and g.weight > bw:
                best, bw = n, g.weight
        if best:
            out[v.index] = best
    return out


def nearest_bone(co, rig):
    best, bd = None, 1e18
    for b in rig.data.bones:
        h, t = b.head_local, b.tail_local
        seg = t - h
        L2 = seg.length_squared or 1e-9
        u = max(0.0, min(1.0, (co - h).dot(seg) / L2))
        d = (co - (h + seg * u)).length
        if d < bd:
            best, bd = b.name, d
    return best


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = import_mesh(a.mesh)
    normalize_like_render(obj)
    rig = build_armature()

    sets = part_vertex_sets(obj)
    hair = sets.get("hair", set())
    acc = sets.get("accessory", set())

    # persist part membership on the mesh itself: vertex order changes on every glTF
    # round-trip, so an external index list cannot be trusted downstream.
    if sets:
        me = obj.data
        for nm in ("part_id", "_PARTID"):
            if nm in me.attributes:
                me.attributes.remove(me.attributes[nm])
        attr = me.attributes.new(name="_PARTID", type="INT", domain="POINT")
        gid = PARTS["group_ids"]
        vals = [gid["body"]] * len(me.vertices)
        for gname, idxs in sets.items():
            for i in idxs:
                if i < len(vals):
                    vals[i] = gid[gname]
        attr.data.foreach_set("value", vals)

    # Try bone heat first; it gives the smoothest joints when it works. On generated meshes
    # it usually does not: TRELLIS/Hunyuan output has interior shells and non-manifold edges,
    # and Blender reports the failure as a *warning*, leaving most vertices unweighted. So
    # check the result and fall back to our own solver rather than trusting the operator.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    method = "bone_heat"
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as e:
        print(f"[rig] bone-heat raised ({e})")
    if unweighted_fraction(obj, rig) > 0.02:
        print(f"[rig] bone-heat left {unweighted_fraction(obj,rig)*100:.1f}% of vertices unweighted "
              f"- using proximity skinning instead")
        obj.parent = rig
        obj.parent_type = "OBJECT"
        if not any(m.type == "ARMATURE" for m in obj.modifiers):
            md = obj.modifiers.new("Armature", "ARMATURE")
            md.object = rig
        proximity_skin(obj, rig)
        method = "proximity_hierarchy"

    # per-part overrides
    bpy.context.view_layer.objects.active = obj
    groups = {g.name: g for g in obj.vertex_groups}

    def rigid_bind(indices, bone):
        if not indices or bone is None:
            return
        idx = list(indices)
        for g in obj.vertex_groups:
            g.remove(idx)
        if bone not in groups:
            groups[bone] = obj.vertex_groups.new(name=bone)
        groups[bone].add(idx, 1.0, "REPLACE")

    stats = {"hair_rigid": 0, "accessory_rigid": 0, "accessory_bones": {}}
    if hair:
        rigid_bind(hair, "head")
        stats["hair_rigid"] = len(hair)

    if acc:
        # cluster accessories by connectivity so a bag and glasses can bind to different bones
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        seen, clusters = set(), []
        for i in acc:
            if i in seen:
                continue
            stack, comp = [i], []
            seen.add(i)
            while stack:
                v = stack.pop()
                comp.append(v)
                for e in bm.verts[v].link_edges:
                    o = e.other_vert(bm.verts[v]).index
                    if o in acc and o not in seen:
                        seen.add(o)
                        stack.append(o)
            clusters.append(comp)
        bm.free()
        # Bind each accessory island to the bone of whatever it rests on, not to the nearest
        # bone by distance. A satchel strap crossing the chest passes close to the upper arm,
        # and distance alone binds it to the elbow - so the strap follows the arm when it lifts.
        # The body underneath is the correct answer: strap on chest -> spine, bag at hip -> hip.
        support = [i for i in range(len(obj.data.vertices)) if i not in acc and i not in hair]
        dom = dominant_bones(obj, rig)
        kd = None
        if support:
            kd = mathutils.kdtree.KDTree(len(support))
            for n, i in enumerate(support):
                kd.insert(obj.data.vertices[i].co, n)
            kd.balance()
        # Rigid-bind only islands that are genuinely separate props. A 2D human parser labels
        # parts of a hood or a textured panel as "bag", and rigid-binding a patch that is
        # continuous with smoothly-skinned cloth tears the surface along the boundary. A real
        # prop touches the body over a small contact patch; a mislabelled patch is almost all
        # boundary. That ratio separates them.
        nverts = len(obj.data.vertices)
        adj = {}
        for e in obj.data.edges:
            x, y = e.vertices
            adj.setdefault(x, []).append(y)
            adj.setdefault(y, []).append(x)
        # A part can only be attached rigidly if it is actually a separate object. Generated
        # meshes are usually one fused shell: the "bag strap" is a painted-on surface patch of
        # the jacket, not a prop resting on it. Rigid-binding such a patch tears it away from
        # the cloth it is continuous with - which is exactly what raising the arm did. So:
        # rigid-bind only accessory islands that occupy their own connected component of the
        # whole mesh, and leave surface patches to deform with the surface they belong to.
        seen_all, comp_of = set(), {}
        for start in range(nverts):
            if start in seen_all:
                continue
            cid = len(set(comp_of.values()))
            stack = [start]
            seen_all.add(start)
            while stack:
                v = stack.pop()
                comp_of[v] = cid
                for n in adj.get(v, ()):
                    if n not in seen_all:
                        seen_all.add(n)
                        stack.append(n)
        comp_is_pure = {}
        for v, cid in comp_of.items():
            if v not in acc:
                comp_is_pure[cid] = False
            else:
                comp_is_pure.setdefault(cid, True)

        kept = []
        for comp in clusters:
            if comp_is_pure.get(comp_of.get(comp[0], -1), False):
                kept.append(comp)
            else:
                stats.setdefault("accessory_left_smooth", 0)
                stats["accessory_left_smooth"] += len(comp)
        clusters = kept

        for comp in clusters:
            c = mathutils.Vector((0, 0, 0))
            for i in comp:
                c += obj.data.vertices[i].co
            c /= len(comp)
            b = None
            if kd is not None:
                votes = {}
                for i in comp[:: max(1, len(comp) // 200)]:
                    for (_, n, _) in kd.find_n(obj.data.vertices[i].co, 3):
                        nb = dom.get(support[n])
                        if nb:
                            votes[nb] = votes.get(nb, 0) + 1
                if votes:
                    b = max(votes, key=votes.get)
            b = b or nearest_bone(c, rig)
            # A worn accessory belongs to the torso or head. A satchel strap lies across the
            # sleeve, so proximity and support votes both point at the arm - and then raising
            # the arm rips the strap off the body. Snap any limb answer back to the torso.
            WEARABLE = {"pelvis", "spine1", "spine2", "spine3", "neck", "head",
                        "left_collar", "right_collar"}
            if b not in WEARABLE:
                best, bd = "spine2", 1e18
                for bone in rig.data.bones:
                    if bone.name not in WEARABLE:
                        continue
                    h, t = bone.head_local, bone.tail_local
                    seg = t - h
                    L2 = seg.length_squared or 1e-9
                    u = max(0.0, min(1.0, (c - h).dot(seg) / L2))
                    dd = (c - (h + seg * u)).length
                    if dd < bd:
                        best, bd = bone.name, dd
                b = best
            rigid_bind(comp, b)
            stats["accessory_rigid"] += len(comp)
            stats["accessory_bones"][b] = stats["accessory_bones"].get(b, 0) + len(comp)

    # game-engine hygiene: <=4 influences, normalised
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.vertex_group_limit_total(limit=4)
    bpy.ops.object.vertex_group_normalize_all(lock_active=False)

    blend_path = a.blend or (os.path.splitext(a.out)[0] + ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=True,
                              export_attributes=True)

    info = {"method": method, "blend": blend_path, "bones": len(rig.data.bones), "vertices": len(obj.data.vertices),
            "parts": {k: len(v) for k, v in sets.items()}, **stats}
    print("[rig] " + json.dumps(info))
    json.dump(info, open(os.path.splitext(a.out)[0] + "_riginfo.json", "w"), indent=2)


main()
