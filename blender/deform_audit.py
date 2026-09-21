"""Localise skinning artefacts to the joint that causes them.

verify_deform.py answers "does this part tear?" with one number per part. That was enough to
catch a jacket hem welded to a thigh, but it is useless for the thing left now - a garment that
reads as subtly wrong while every whole-part figure looks acceptable. A p99 of 40% spread over
37,582 clothing vertices says nothing about *where*, and "where" is the only thing that leads to
a fix.

So every measurement here is bucketed by the bone that dominates the vertex, and reported per
bone, sorted worst-first. Four things get measured, on **every frame** of every clip:

  edge stretch    |current - rest| / rest, per edge, bucketed by the dominant bone of its two
                  endpoints. Degenerate rest edges are excluded (dividing motion by ~0 reports
                  thousands of percent and drowns the real signal).
  area ratio      per face, current area / rest area. Linear blend skinning collapses volume on
                  the inside of a bending joint - the classic candy-wrapper - and that shows up
                  as area ratio well under 1 at elbows and knees while edge stretch stays
                  unremarkable, because the edges rotate rather than lengthen.
  normal swing    the angle each face normal turns relative to the *rigid* motion of its dominant
                  bone. A face that swings far more than its bone did is being sheared by
                  disagreeing weights; this is what reads as shimmering and seam-crawling in
                  motion, and no static pose shows it.
  penetration     clothing vertices that end up behind the body surface. A jacket that sinks
                  into the torso does not look like a bug, it looks like the garment is
                  dissolving - which is exactly the reported symptom.

Nothing here is a pass/fail gate. It is a ranked list of joints to go and fix.

Run: blender -b -noaudio --python deform_audit.py -- --blend p4_final.blend --out audit.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import bpy
import numpy as np
from mathutils import kdtree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--step", type=int, default=1, help="frame stride (1 = every frame)")
ap.add_argument("--pen-step", type=int, default=4, help="frame stride for the penetration test")
ap.add_argument("--pen-samples", type=int, default=4000)
ap.add_argument("--clips", default="")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
meshes = [o for o in scene.objects if o.type == "MESH"]
if rig is None or not meshes:
    raise SystemExit("[audit] need an armature and at least one mesh")


def set_action(name):
    act = bpy.data.actions.get(name)
    if act is None:
        return None
    rig.animation_data.action = act
    # Blender 5 slotted actions: an action does nothing until a slot is assigned
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        for s in act.slots:
            if getattr(s, "target_id_type", "OBJECT") == "OBJECT":
                rig.animation_data.action_slot = s
                break
        else:
            rig.animation_data.action_slot = act.slots[0]
    return act


def verts_of(ob):
    me = ob.data
    v = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", v)
    return v.reshape(-1, 3)


def eval_mesh(ob, dg):
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    v = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", v)
    nrm = np.empty(len(me.polygons) * 3, np.float64)
    me.polygons.foreach_get("normal", nrm)
    area = np.empty(len(me.polygons), np.float64)
    me.polygons.foreach_get("area", area)
    out = (v.reshape(-1, 3), nrm.reshape(-1, 3), area)
    ev.to_mesh_clear()
    return out


# ---- static per-mesh tables ---------------------------------------------------------------
tables = {}
for ob in meshes:
    me = ob.data
    n = len(me.vertices)
    names = [vg.name for vg in ob.vertex_groups]

    # dominant bone per vertex
    dom = np.full(n, -1, np.int32)
    for v in me.vertices:
        best, bw = -1, 0.0
        for g in v.groups:
            if g.weight > bw:
                best, bw = g.group, g.weight
        dom[v.index] = best

    edges = np.array([(e.vertices[0], e.vertices[1]) for e in me.edges], np.int64)
    rest_v = verts_of(ob)
    rest_len = np.linalg.norm(rest_v[edges[:, 0]] - rest_v[edges[:, 1]], axis=1)
    floor = np.median(rest_len) * 0.2
    keep = rest_len >= floor
    print(f"[audit] {ob.name}: {len(edges):,} edges, {int((~keep).sum()):,} degenerate excluded",
          flush=True)

    faces = [list(p.vertices) for p in me.polygons]
    fdom = np.array([np.bincount(dom[f][dom[f] >= 0], minlength=len(names)).argmax()
                     if (dom[f] >= 0).any() else -1 for f in faces], np.int32)
    rest_area = np.empty(len(me.polygons), np.float64)
    me.polygons.foreach_get("area", rest_area)
    rest_nrm = np.empty(len(me.polygons) * 3, np.float64)
    me.polygons.foreach_get("normal", rest_nrm)

    tables[ob.name] = dict(
        names=names, dom=dom, edges=edges[keep], rest_len=rest_len[keep],
        edom=dom[edges[keep][:, 0]], fdom=fdom,
        rest_area=np.maximum(rest_area, 1e-12), rest_nrm=rest_nrm.reshape(-1, 3),
        n=n)

body = next((o for o in meshes if "skin" in o.name.lower() or "body" in o.name.lower()), None)
cloth = next((o for o in meshes if "cloth" in o.name.lower()), None)

clips = [c.strip() for c in a.clips.split(",") if c.strip()] or [x.name for x in bpy.data.actions]
report = {"blend": a.blend, "clips": {}}

for clip in clips:
    act = set_action(clip)
    if act is None:
        continue
    f0, f1 = (int(x) for x in act.frame_range)
    frames = list(range(f0, f1 + 1, a.step))

    acc = {ob.name: dict(stretch=defaultdict(list), area=defaultdict(list),
                         swing=defaultdict(list)) for ob in meshes}
    pen_depth, pen_count, pen_total = [], 0, 0

    for fi, f in enumerate(frames):
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        bone_rot = {}
        for pb in rig.pose.bones:
            # rotation the bone actually underwent, rest -> posed, in world space
            R = (rig.matrix_world @ pb.matrix).to_3x3().normalized()
            R0 = (rig.matrix_world @ pb.bone.matrix_local).to_3x3().normalized()
            bone_rot[pb.name] = np.array((R @ R0.inverted()).normalized())

        ev = {}
        for ob in meshes:
            T = tables[ob.name]
            V, N, A = eval_mesh(ob, dg)
            ev[ob.name] = (V, N)

            cur = np.linalg.norm(V[T["edges"][:, 0]] - V[T["edges"][:, 1]], axis=1)
            s = np.abs(cur - T["rest_len"]) / T["rest_len"]
            for b in np.unique(T["edom"]):
                if b < 0:
                    continue
                acc[ob.name]["stretch"][int(b)].append(s[T["edom"] == b])

            ar = A / T["rest_area"]
            fd = T["fdom"]
            for b in np.unique(fd):
                if b < 0:
                    continue
                m = fd == b
                acc[ob.name]["area"][int(b)].append(ar[m])
                # how far the normal turned beyond its bone's own rotation
                Rb = bone_rot.get(T["names"][b])
                if Rb is not None:
                    expect = T["rest_nrm"][m] @ Rb.T
                    d = np.clip((expect * N[m]).sum(1), -1.0, 1.0)
                    acc[ob.name]["swing"][int(b)].append(np.degrees(np.arccos(d)))

        if body is not None and cloth is not None and fi % a.pen_step == 0:
            BV, BN = ev[body.name]
            tree = kdtree.KDTree(len(BV))
            for i, p in enumerate(BV):
                tree.insert(p, i)
            tree.balance()
            CV, _ = ev[cloth.name]
            idx = np.linspace(0, len(CV) - 1, min(a.pen_samples, len(CV))).astype(int)
            # body vertex normals, evaluated
            bev = body.evaluated_get(dg)
            bm = bev.to_mesh()
            vn = np.empty(len(bm.vertices) * 3, np.float64)
            bm.vertices.foreach_get("normal", vn)
            vn = vn.reshape(-1, 3)
            bev.to_mesh_clear()
            for i in idx:
                p = CV[i]
                _, j, _ = tree.find(p)
                d = float(np.dot(p - BV[j], vn[j]))
                pen_total += 1
                if d < -1e-4:
                    pen_count += 1
                    pen_depth.append(-d)

    def pct(chunks, q):
        if not chunks:
            return None
        v = np.concatenate(chunks)
        return float(np.percentile(v, q))

    cr = {}
    for ob in meshes:
        T = tables[ob.name]
        rows = []
        for b, chunks in acc[ob.name]["stretch"].items():
            name = T["names"][b]
            ar = acc[ob.name]["area"].get(b, [])
            sw = acc[ob.name]["swing"].get(b, [])
            rows.append({
                "bone": name,
                "edges": int(sum(len(c) for c in chunks) / max(len(frames), 1)),
                "stretch_p99": round(pct(chunks, 99) or 0, 4),
                "stretch_max": round(float(max(c.max() for c in chunks)), 4),
                "area_p01": round(pct(ar, 1) or 0, 4),
                "area_p99": round(pct(ar, 99) or 0, 4),
                "swing_p99": round(pct(sw, 99) or 0, 2),
            })
        rows.sort(key=lambda r: -r["stretch_p99"])
        cr[ob.name] = rows

    entry = {"frames": len(frames), "per_bone": cr}
    if pen_total:
        entry["penetration"] = {
            "sampled": pen_total, "inside": pen_count,
            "fraction": round(pen_count / pen_total, 4),
            "mean_depth_cm": round(float(np.mean(pen_depth)) * 100, 3) if pen_depth else 0.0,
            "p99_depth_cm": round(float(np.percentile(pen_depth, 99)) * 100, 3) if pen_depth else 0.0,
        }
    report["clips"][clip] = entry

    print(f"\n[audit] === {clip}  ({len(frames)} frames) ===", flush=True)
    for mname, rows in cr.items():
        print(f"[audit] {mname}: worst bones by p99 edge stretch", flush=True)
        for r in rows[:6]:
            print(f"[audit]    {r['bone']:<18s} stretch p99 {r['stretch_p99']*100:6.1f}%  "
                  f"max {r['stretch_max']*100:7.1f}%   area p01 {r['area_p01']:.2f} "
                  f"p99 {r['area_p99']:.2f}   swing p99 {r['swing_p99']:5.1f}deg", flush=True)
    if "penetration" in entry:
        p = entry["penetration"]
        print(f"[audit] clothing inside body: {p['fraction']*100:.2f}% of samples, "
              f"mean {p['mean_depth_cm']:.2f}cm, p99 {p['p99_depth_cm']:.2f}cm", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(report, open(a.out, "w"), indent=2)
print(f"\n[audit] -> {a.out}", flush=True)
