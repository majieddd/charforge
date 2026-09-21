"""Measure whether a rigged character actually deforms correctly, per part.

"Looks fine in the viewport" is not a check. This samples every animation clip and measures:

  edge stretch     per part, how far edges depart from their rest length. Skinning artefacts
                   (a jacket hem welded to a thigh, hair shearing off the skull) show up here
                   as large p99 stretch on that part while the body stays low.
  rigid residual   for hair and accessories, the distance between where each vertex actually
                   lands and where it would land if it were rigidly attached to its bone.
                   Near zero means the part travels with the head/strap bone instead of
                   being dragged by the spine - exactly the failure the pipeline exists to avoid.
  weight hygiene   influences per vertex (engines cap at 4), unweighted vertices, weight sums.
  ground contact   lowest vertex per frame, to catch feet sinking through the floor.

Run: blender -b -noaudio --python verify_deform.py -- --blend animated.blend --out verify.json
     [--render sheets_dir]
"""
import argparse
import json
import math
import os
import sys
from statistics import median

import bpy
import mathutils
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--render", default=None)
ap.add_argument("--samples", type=int, default=6, help="frames sampled per clip")
a = ap.parse_args(argv)

GROUP_NAMES = {0: "body", 1: "clothing", 2: "hair", 3: "accessory"}


def scene_objects():
    """All character meshes plus the armature. After split_parts.py a character is several
    objects, and each must be measured on its own - edges no longer cross between parts."""
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    rig = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if not meshes or rig is None:
        raise SystemExit("need at least one mesh and an armature")
    return meshes, rig


NAME_TO_ID = {v: k for k, v in {0: "body", 1: "clothing", 2: "hair", 3: "accessory"}.items()}


def part_ids(mesh):
    me = mesh.data
    n = len(me.vertices)
    name = next((x for x in ("_PARTID", "part_id") if x in me.attributes), None)
    if name is not None:
        vals = np.zeros(n, dtype=np.int32)
        me.attributes[name].data.foreach_get("value", vals)
        return vals
    # split objects are named char_<part>; fall back to that
    suffix = mesh.name.replace("char_", "").split(".")[0]
    return np.full(n, NAME_TO_ID.get(suffix, 0), dtype=np.int32)


def eval_positions(mesh, deps):
    ob = mesh.evaluated_get(deps)
    me = ob.to_mesh()
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    out = co.reshape(-1, 3).copy()
    ob.to_mesh_clear()
    return out


def edges_of(mesh):
    me = mesh.data
    e = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", e)
    return e.reshape(-1, 2)


def weight_stats(mesh, rig):
    bones = {b.name for b in rig.data.bones}
    gi_to_bone = {g.index: g.name for g in mesh.vertex_groups}
    infl, sums, unweighted = [], [], 0
    for v in mesh.data.vertices:
        ws = [g.weight for g in v.groups if gi_to_bone.get(g.group) in bones and g.weight > 1e-4]
        infl.append(len(ws))
        s = sum(ws)
        sums.append(s)
        if s <= 1e-4:
            unweighted += 1
    infl = np.array(infl)
    sums = np.array(sums)
    return {"max_influences": int(infl.max()) if len(infl) else 0,
            "mean_influences": float(infl.mean()) if len(infl) else 0,
            "over_4_influences": int((infl > 4).sum()),
            "unweighted_vertices": int(unweighted),
            "weight_sum_min": float(sums.min()) if len(sums) else 0,
            "weight_sum_max": float(sums.max()) if len(sums) else 0}


def main():
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    meshes, rig = scene_objects()
    gi_to_name = {}

    # per-mesh static data
    M = []
    for mesh in meshes:
        pid = part_ids(mesh)
        edges = edges_of(mesh)
        M.append({"obj": mesh, "pid": pid, "edges": edges,
                  "epart": pid[edges[:, 0]] if len(edges) else np.zeros(0, dtype=np.int32),
                  "gi": {g.index: g.name for g in mesh.vertex_groups}})

    if rig.animation_data:
        rig.animation_data.action = None
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    all_rest = []
    for m in M:
        m["rest"] = eval_positions(m["obj"], deps)
        e = m["edges"]
        m["rest_len"] = (np.linalg.norm(m["rest"][e[:, 0]] - m["rest"][e[:, 1]], axis=1)
                         if len(e) else np.zeros(0))
        all_rest.append(m["rest"])
    stacked = np.concatenate(all_rest, 0)
    height = float(stacked[:, 2].max() - stacked[:, 2].min())
    usable = 0
    for m in M:
        m["ok"] = m["rest_len"] > max(1e-9, height * 1e-3)
        usable += int(m["ok"].sum())
    print(f"[verify] {len(meshes)} mesh object(s), {usable} usable edges", flush=True)

    bpy.ops.object.mode_set(mode="OBJECT")
    rest_bone = {b.name: b.matrix_local.copy() for b in rig.data.bones}

    parts_total = {}
    for m in M:
        for k in GROUP_NAMES:
            parts_total[GROUP_NAMES[k]] = parts_total.get(GROUP_NAMES[k], 0) + int((m["pid"] == k).sum())

    wstats = {"max_influences": 0, "over_4_influences": 0, "unweighted_vertices": 0,
              "mean_influences": 0.0, "weight_sum_min": 1e9, "weight_sum_max": 0.0}
    nv = 0
    for m in M:
        w = weight_stats(m["obj"], rig)
        wstats["max_influences"] = max(wstats["max_influences"], w["max_influences"])
        wstats["over_4_influences"] += w["over_4_influences"]
        wstats["unweighted_vertices"] += w["unweighted_vertices"]
        n = len(m["obj"].data.vertices)
        wstats["mean_influences"] += w["mean_influences"] * n
        wstats["weight_sum_min"] = min(wstats["weight_sum_min"], w["weight_sum_min"])
        wstats["weight_sum_max"] = max(wstats["weight_sum_max"], w["weight_sum_max"])
        nv += n
    wstats["mean_influences"] = round(wstats["mean_influences"] / max(1, nv), 3)

    results = {"blend": a.blend, "height": height, "vertices": nv,
               "meshes": [m["obj"].name for m in M],
               "edges": int(sum(len(m["edges"]) for m in M)),
               "parts": parts_total, "weights": wstats, "clips": {}}

    # rigid groups: per mesh, per part, vertices grouped by the bone they actually follow
    def bone_groups(m, mask, cap=6000):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return {}
        if len(idx) > cap:
            idx = idx[:: max(1, len(idx) // cap)]
        out = {}
        for i in idx:
            best, bw = None, 0.0
            for g in m["obj"].data.vertices[int(i)].groups:
                if g.weight > bw:
                    best, bw = m["gi"].get(g.group), g.weight
            if best:
                out.setdefault(best, []).append(int(i))
        return {b: np.array(v) for b, v in out.items() if len(v) >= 10}

    rigid = []          # (mesh_index, part_name, {bone: idx})
    for mi, m in enumerate(M):
        for k, name in GROUP_NAMES.items():
            if name in ("hair", "accessory") and (m["pid"] == k).sum() > 0:
                gset = bone_groups(m, m["pid"] == k)
                if gset:
                    rigid.append((mi, name, gset))

    for act in bpy.data.actions:
        rig.animation_data.action = act
        if hasattr(rig.animation_data, "action_slot") and len(act.slots):
            rig.animation_data.action_slot = act.slots[0]
        for pb in rig.pose.bones:
            pb.rotation_euler = (0.0, 0.0, 0.0)
            pb.location = (0.0, 0.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
        fr = [int(x) for x in act.frame_range]
        frames = np.linspace(fr[0], max(fr[0] + 1, fr[1]), a.samples).astype(int)
        per_part = {GROUP_NAMES[k]: [] for k in GROUP_NAMES}
        rigid_res = {"hair": [], "accessory": []}
        lowest = []
        for f in frames:
            bpy.context.scene.frame_set(int(f))
            bpy.context.view_layer.update()
            d = bpy.context.evaluated_depsgraph_get()
            cur = [eval_positions(m["obj"], d) for m in M]
            frame_stretch = {GROUP_NAMES[k]: [] for k in GROUP_NAMES}
            for mi, m in enumerate(M):
                e, ok = m["edges"], m["ok"]
                if not len(e) or not ok.any():
                    continue
                c = cur[mi]
                L = np.linalg.norm(c[e[:, 0]] - c[e[:, 1]], axis=1)
                stretch = np.abs(L[ok] / m["rest_len"][ok] - 1.0)
                ep = m["epart"][ok]
                for k, name in GROUP_NAMES.items():
                    sel = ep == k
                    if sel.sum():
                        frame_stretch[name].append(stretch[sel])
                lowest.append(float(c[:, 2].min()))
            for name, chunks in frame_stretch.items():
                if chunks:
                    per_part[name].append(float(np.percentile(np.concatenate(chunks), 99)))
            for mi, pname, gset in rigid:
                worst = 0.0
                for bname, sel in gset.items():
                    pb = rig.pose.bones.get(bname)
                    if pb is None:
                        continue
                    Mx = (rig.matrix_world @ pb.matrix @ rest_bone[bname].inverted()
                          @ rig.matrix_world.inverted())
                    pred = np.array([(Mx @ mathutils.Vector(M[mi]["rest"][i])) for i in sel])
                    res = np.linalg.norm(pred - cur[mi][sel], axis=1) / max(height, 1e-6)
                    worst = max(worst, float(np.percentile(res, 95)))
                rigid_res[pname].append(worst)
        results["clips"][act.name] = {
            "frames_sampled": [int(x) for x in frames],
            "edge_stretch_p99_by_part": {k: (round(max(v), 4) if v else None) for k, v in per_part.items()},
            "rigid_residual_p95_by_part": {k: (round(max(v), 5) if v else None) for k, v in rigid_res.items()},
            "lowest_point": round(min(lowest), 4) if lowest else None,
        }
        c = results["clips"][act.name]
        print(f"[verify] {act.name:8s} stretch p99 " +
              " ".join(f"{k}={v}" for k, v in c["edge_stretch_p99_by_part"].items() if v is not None) +
              "  rigid " + " ".join(f"{k}={v}" for k, v in c["rigid_residual_p95_by_part"].items()
                                     if v is not None), flush=True)

    worst_body = max((c["edge_stretch_p99_by_part"].get("body") or 0) for c in results["clips"].values()) if results["clips"] else 0
    worst_hair = max((c["rigid_residual_p95_by_part"].get("hair") or 0) for c in results["clips"].values()) if results["clips"] else 0
    worst_acc_s = max((c["edge_stretch_p99_by_part"].get("accessory") or 0) for c in results["clips"].values()) if results["clips"] else 0
    results["verdict"] = {
        "body_stretch_ok": bool(worst_body < 0.35),
        "hair_rigid_ok": bool(worst_hair < 0.01),
        "accessory_stretch_ok": bool(worst_acc_s < 0.35),
        "influences_ok": bool(wstats["over_4_influences"] == 0),
        "all_weighted": bool(wstats["unweighted_vertices"] == 0),
        "worst_body_stretch_p99": round(worst_body, 4),
        "worst_hair_rigid_residual_p95": round(worst_hair, 5),
        "worst_accessory_stretch_p99": round(worst_acc_s, 4),
    }
    json.dump(results, open(a.out, "w"), indent=2)
    print("[verify] " + json.dumps(results["verdict"]))


main()
