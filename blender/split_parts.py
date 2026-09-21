"""Split a rigged character into separate meshes per semantic part, and bind each its own way.

Why this matters: while body, clothing, hair and accessories share one surface, every binding
choice is a compromise. A strap that lies on a sleeve is joined by real edges to that sleeve, so
rigid-binding it tears the surface (measured at 757% edge stretch on the satchel character) and
skinning it drags it off the body. Separating the parts removes those edges, after which each
part can be bound correctly:

  body      skinned normally
  clothing  skinned, and now eligible for cloth simulation (its own object, own collision)
  hair      rigid to the head
  accessory rigid to the bone of whatever it hangs from - now safe, because there is nothing
            left to tear

Run: blender -b -noaudio --python split_parts.py -- --blend rigged.blend --out split.blend
     [--glb split.glb]
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--glb", default=None)
ap.add_argument("--min-verts", type=int, default=64, help="parts smaller than this stay with the body")
a = ap.parse_args(argv)

GROUP_NAMES = {0: "body", 1: "clothing", 2: "hair", 3: "accessory"}
WEARABLE = {"pelvis", "spine1", "spine2", "spine3", "neck", "head", "left_collar", "right_collar"}


def part_ids(mesh):
    me = mesh.data
    name = next((x for x in ("_PARTID", "part_id") if x in me.attributes), None)
    if name is None:
        return None
    vals = np.zeros(len(me.vertices), dtype=np.int32)
    me.attributes[name].data.foreach_get("value", vals)
    return vals


def dominant_bone(obj, rig, idx):
    """The bone holding the most weight across a set of vertices."""
    bones = {b.name for b in rig.data.bones}
    gi = {g.index: g.name for g in obj.vertex_groups}
    votes = {}
    for i in idx[:: max(1, len(idx) // 3000)]:
        best, bw = None, 0.0
        for g in obj.data.vertices[int(i)].groups:
            n = gi.get(g.group)
            if n in bones and g.weight > bw:
                best, bw = n, g.weight
        if best:
            votes[best] = votes.get(best, 0) + 1
    return max(votes, key=votes.get) if votes else None


def connected_components(obj, weld_tol=1e-5):
    """Islands over the welded surface - glTF splits vertices at UV seams, so the raw index
    graph reports thousands of fragments where the surface is actually continuous."""
    adj = {}
    for e in obj.data.edges:
        x, y = e.vertices
        adj.setdefault(x, []).append(y)
        adj.setdefault(y, []).append(x)
    V = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    scale = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    keys = np.round(V / (weld_tol * scale)).astype(np.int64)
    seam = {}
    for i in range(len(V)):
        seam.setdefault((keys[i, 0], keys[i, 1], keys[i, 2]), []).append(i)
    for members in seam.values():
        for other in members[1:]:
            adj.setdefault(members[0], []).append(other)
            adj.setdefault(other, []).append(members[0])
    seen, comps = set(), []
    for start in range(len(obj.data.vertices)):
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            v = stack.pop()
            comp.append(v)
            for n in adj.get(v, ()):
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def nearest_wearable_bone(obj, rig, comp):
    """Which torso/head bone an island hangs from, by centroid distance."""
    import mathutils
    c = mathutils.Vector((0, 0, 0))
    step = max(1, len(comp) // 500)
    n = 0
    for i in comp[::step]:
        c += obj.data.vertices[i].co
        n += 1
    c /= max(1, n)
    c = rig.matrix_world.inverted() @ (obj.matrix_world @ c)
    best, bd = "spine2", 1e18
    for b in rig.data.bones:
        if b.name not in WEARABLE:
            continue
        h, t = b.head_local, b.tail_local
        seg = t - h
        L2 = seg.length_squared or 1e-9
        u = max(0.0, min(1.0, (c - h).dot(seg) / L2))
        d = (c - (h + seg * u)).length
        if d < bd:
            best, bd = b.name, d
    return best


def main():
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
    rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    pid = part_ids(mesh)
    if pid is None:
        raise SystemExit("no _PARTID attribute on the mesh - run build_rig.py with --parts first")

    counts = {GROUP_NAMES[k]: int((pid == k).sum()) for k in GROUP_NAMES}
    print(f"[split] parts: {counts}", flush=True)

    # remember, per part, the bone a rigid part should follow (computed before separating)
    rigid_target = {}
    for k, name in GROUP_NAMES.items():
        if name in ("hair", "accessory"):
            idx = np.where(pid == k)[0]
            if len(idx) >= a.min_verts:
                b = dominant_bone(mesh, rig, idx)
                if b and b not in WEARABLE and name == "accessory":
                    b = "spine2"
                rigid_target[name] = b or ("head" if name == "hair" else "spine2")

    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode="OBJECT")
    made = {}
    # separate largest-id first so indices stay valid on the shrinking source mesh
    for k in sorted(GROUP_NAMES, reverse=True):
        name = GROUP_NAMES[k]
        if k == 0:
            continue
        cur = part_ids(mesh)
        idx = np.where(cur == k)[0]
        if len(idx) < a.min_verts:
            print(f"[split] {name}: {len(idx)} vertices, too small to separate", flush=True)
            continue
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        for i in idx:
            mesh.data.vertices[int(i)].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        before = {o.name for o in bpy.context.scene.objects}
        bpy.ops.mesh.separate(type="SELECTED")
        bpy.ops.object.mode_set(mode="OBJECT")
        new = [o for o in bpy.context.scene.objects if o.name not in before]
        if not new:
            print(f"[split] {name}: separation produced nothing", flush=True)
            continue
        part = new[0]
        part.name = f"char_{name}"
        made[name] = part
        print(f"[split] {name}: {len(part.data.vertices)} vertices -> {part.name}", flush=True)

    mesh.name = "char_body"
    made["body"] = mesh

    # rebind the rigid parts now that they are their own objects: no shared edges left,
    # so a rigid bind can no longer tear a neighbouring surface
    info = {"parts": {}, "rigid_target": rigid_target}
    for name, obj in made.items():
        obj.modifiers.clear()
        md = obj.modifiers.new("Armature", "ARMATURE")
        md.object = rig
        obj.parent = rig
        if name in rigid_target:
            # Bind per connected island, not per object: "accessory" covers a hat on the head
            # and a satchel at the hip, and one bone for both would ride the wrong body part.
            for g in list(obj.vertex_groups):
                obj.vertex_groups.remove(g)
            comps = connected_components(obj)
            binds = {}
            for comp in comps:
                bone = nearest_wearable_bone(obj, rig, comp)
                vg = obj.vertex_groups.get(bone) or obj.vertex_groups.new(name=bone)
                vg.add(comp, 1.0, "REPLACE")
                binds[bone] = binds.get(bone, 0) + len(comp)
            info["parts"][name] = {"vertices": len(obj.data.vertices),
                                   "bind": "rigid per island", "islands": len(comps),
                                   "bones": binds}
        else:
            info["parts"][name] = {"vertices": len(obj.data.vertices), "bind": "skinned"}

    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    if a.glb:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.export_scene.gltf(filepath=a.glb, export_format="GLB", use_selection=True,
                                  export_animations=True, export_animation_mode="ACTIONS",
                                  export_attributes=True)
    json.dump(info, open(os.path.splitext(a.out)[0] + "_split.json", "w"), indent=2)
    print("[split] " + json.dumps(info), flush=True)


main()
