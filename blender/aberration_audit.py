"""The aberrations a person sees in a moving character, counted: every clip, every sampled frame.

    blender -b --python aberration_audit.py -- --blend final.blend --out aberrations.json [--step 2]

Experiment E102. A person watching the character notices a crushed armpit, a sleeve that stretches
like gum, a patch of surface that shears against its own bone, an arm passing through the torso, a
shoe through the floor, a vertex that pops for one frame. Each of these is measured per frame of the
posed mesh against the rig's rest pose, and attributed to the body region of the bone that dominates
the surface there:

  crushed      faces whose area falls below half of rest (linear skinning's candy-wrapper, armpits)
  stretched    faces whose area grows past twice rest (weights pulling a surface between two bones)
  sheared      faces whose normal turns more than 60 degrees further than their own bone did
               (disagreeing weights; reads as shimmer and crawling seams)
  intersecting pairs of triangles that intersect in the posed mesh although their centroids were more
               than 3 cm apart at rest. Blender's BVHTree.overlap() runs a triangle-triangle test
               after the box test (checked: boxes that overlap without touching are not returned),
               so these are real intersections - but an arm resting on the torso intersects too
  penetration  how deep one body region has gone inside another, in cm (also without the head region, which
               carries the hair - *_body - since hair should swing aside): from the vertices round each
               intersection, the signed distance to the other region's surface (negative = inside),
               counting only vertices that were more than 3 cm from that region at rest. Contact
               is a few millimetres; an arm through a body is centimetres
  floor        vertices more than 1 cm below the fixed floor plane (not contact-phase foot slip)
  popping      vertices whose local shape (offset from their neighbours' mean) has a large,
               mostly out-and-back second difference over three sampled frames

On sparse BVH frames, crushed, stretched and sheared flags are also split by a single normal-ray
exposure test. This is a geometric exposure heuristic, not visibility from a player camera.

Skinning is dual quaternion by default, as the playground, Unity and Unreal show it (--skin lbs
for linear, which is what glTF viewers without the switch show).

Shares are of all faces (or vertices), so characters and clips compare directly. Nothing is a gate:
the report ranks clips and regions, and names the worst frames to render and look at.
"""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--clips", default="", help="comma list; default every action")
ap.add_argument("--step", type=int, default=1, help="frame stride for the surface measures (above 1, a one-frame pop on a skipped frame is missed; every frame costs ~20% more than every second)")
ap.add_argument("--bvh-step", "--through-step", dest="bvh_step", type=int, default=6,
                help="exact frame stride for the intersection, penetration and exposure checks (legacy alias: --through-step)")
ap.add_argument("--apart", type=float, default=0.03,
                help="minimum rest separation (m) for an intersection or penetration to count")
ap.add_argument("--deep", type=float, default=2.0, help="penetration (cm) that counts as deep")
ap.add_argument("--render", default="", help="a folder: render each clip's worst frame of each kind, the flagged faces red")
ap.add_argument("--res", type=int, default=520)
ap.add_argument("--skin", default="dqs", choices=("dqs", "lbs"))
ap.add_argument("--pop", type=float, default=0.5,
                help="minimum center-sample local-shape excursion in mean-rest-edge lengths")
ap.add_argument("--series", action="store_true", help="also write every sampled frame's values (for the controls)")
ap.add_argument("--force", action="store_true", help="allow replacing an existing JSON/render output")
a = ap.parse_args(argv)
if a.step < 1 or a.bvh_step < 1:
    raise SystemExit("[aberr] --step and --bvh-step must both be positive frame counts")
if os.path.exists(a.out) and not a.force:
    raise SystemExit(f"[aberr] refusing to overwrite existing output: {a.out} (pass --force to replace it)")
if a.render and os.path.isdir(a.render) and os.listdir(a.render) and not a.force:
    raise SystemExit(f"[aberr] refusing to overwrite render files in: {a.render} (pass --force to replace them)")

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
meshes = [o for o in sc.objects if o.type == "MESH" and o.visible_get()
          and any(m.type == "ARMATURE" for m in o.modifiers)]
if not meshes:
    raise SystemExit("[aberr] no skinned mesh")
for o in meshes:
    for m in o.modifiers:
        if m.type == "ARMATURE":
            m.use_deform_preserve_volume = a.skin == "dqs"
fps = sc.render.fps / sc.render.fps_base
POP_X = a.pop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_contact import Contact, region_of  # noqa: E402  (regions, rest pose, intersections, penetration)


def set_action(name):
    act = bpy.data.actions.get(name)
    if act is None:
        return None
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    return act


# ---- rest pose: areas, normals, dominant bones, regions (mesh_contact.Contact) ----------------------
C = Contact(rig, meshes, a.apart)
V0, F, A0, N0 = C.V0, C.F, C.A0, C.N0
bones, bidx, vdom, fdom = C.bones, C.bidx, C.vdom, C.fdom
region_names, ridx, breg, freg, vreg = C.region_names, C.ridx, C.breg, C.freg, C.vreg
live, cent0 = C.live, C.cent0
nbr_idx, nbr_start, deg, nbr_ok, edge_len = C.nbr_idx, C.nbr_start, C.deg, C.nbr_ok, C.edge_len
nF, nV = int(live.sum()), len(V0)
print(f"[aberr] {os.path.basename(os.path.dirname(a.blend))}: {nF:,} faces, {nV:,} vertices, {len(bones)} bones, "
      f"regions {', '.join(region_names)}", flush=True)


def evaluated(dg, F_rest=None):
    """Posed vertices with the rest triangles (and their areas and normals); fails if the vertex count changed."""
    V, A, N = C.posed(dg)
    return V, F, A, N


def require_rest_topology(V_now, frame_label):
    """Kept for the call sites: Contact.posed() already refuses a changed vertex count."""


intersections, penetration = C.intersections, C.penetration
extent = float(np.linalg.norm(V0.max(0) - V0.min(0)))
ray_eps = max(1e-6, extent * 1e-4)
ray_reach = max(1e-5, extent * 0.30)


def normal_ray_exposed(tree, V, F, N, flags):
    """Flag triangles whose outward-normal ray escapes the posed combined mesh.

    The result is aligned to the audit's concatenated loop-triangle order. It is an exposure
    heuristic only: one ray and the mesh normals do not establish camera/pixel visibility.
    """
    exposed = np.zeros(len(F), dtype=bool)
    centers = V[F].mean(axis=1)
    for i in np.flatnonzero(flags):
        origin = Vector(centers[i] + N[i] * ray_eps)
        exposed[i] = tree.ray_cast(origin, Vector(N[i]), ray_reach)[0] is None
    return exposed


def bone_transforms():
    """Each bone's rest-to-posed transform (4x4, world)."""
    T = np.zeros((len(bones), 4, 4))
    for b, i in bidx.items():
        pb = rig.pose.bones[b]
        T[i] = np.array((rig.matrix_world @ pb.matrix) @ (rig.matrix_world @ pb.bone.matrix_local).inverted())
    return T


def bone_rotations():
    R = np.zeros((len(bones), 3, 3))
    for b, i in bidx.items():
        pb = rig.pose.bones[b]
        M = (rig.matrix_world @ pb.matrix).to_3x3().normalized()
        M0 = (rig.matrix_world @ pb.bone.matrix_local).to_3x3().normalized()
        R[i] = np.array(M @ M0.inverted())
    return R


clips = [c.strip() for c in a.clips.split(",") if c.strip()] or sorted(x.name for x in bpy.data.actions)
report = {"blend": os.path.basename(a.blend), "faces": nF, "vertices": nV, "regions": region_names,
          "thresholds": {"crushed_area": 0.5, "stretched_area": 2.0, "sheared_deg": 60,
                         "rest_separation_m": a.apart, "deep_cm": a.deep,
                         "floor_m": -0.01, "popping_center_excursion_edges": a.pop},
          "sampling": {"surface_step_frames": a.step, "bvh_step_frames": a.bvh_step,
                       "fps": round(float(fps), 6),
                       "popping_center_spacing_frames": a.step,
                       "popping_center_spacing_seconds": round(a.step / fps, 6),
                       "popping_window_seconds": round(2 * a.step / fps, 6),
                       "popping_index": "||Lmid-(Lprev+Lnext)/2|| / mean_rest_edge_length; center-sample deviation from the chord, flagged above threshold and >1.5x endpoint displacement",
                       "normal_ray_exposure": "no hit along triangle normal within 0.30 x rest-mesh diagonal",
                       "normal_ray_origin_offset_m": round(ray_eps, 7),
                       "normal_ray_max_distance_m": round(ray_reach, 6)}, "clips": {}}

for clip in clips:
    act = set_action(clip)
    if act is None:
        continue
    f0, f1 = (int(x) for x in act.frame_range)
    surface_frames = list(range(f0, f1 + 1, a.step))
    bvh_frames = list(range(f0, f1 + 1, a.bvh_step))
    surface_set, bvh_set = set(surface_frames), set(bvh_frames)
    sample_frames = sorted(surface_set | bvh_set)
    per = []                                            # surface-metric sample frames
    sparse = []                                         # BVH / normal-ray sample frames
    allrows = []
    reg_hits = {k: Counter() for k in ("crushed", "stretched", "sheared")}
    inter_pairs = Counter()                             # region pair -> frames it intersected
    pen_max = defaultdict(float)                        # region pair -> deepest penetration (m)
    exposure_totals = {k: {"flagged": 0, "exposed": 0} for k in ("crushed", "stretched", "sheared")}
    prev = []                                           # last two posed vertex arrays, for acceleration
    prev_frames = []
    pop_worst = (0.0, None)
    pop_regions = Counter()
    for fr in sample_frames:
        sc.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        V, _, A, N = evaluated(dg, F)
        require_rest_topology(V, f"clip {clip} frame {fr}")
        ratio = A / np.maximum(A0, 1e-12)
        crushed = live & (ratio < 0.5)
        stretched = live & (ratio > 2.0)
        R = bone_rotations()
        expect = np.einsum("fij,fj->fi", R[np.maximum(fdom, 0)], N0)
        swing = np.degrees(np.arccos(np.clip((expect * N).sum(1), -1, 1)))
        sheared = live & (fdom >= 0) & (swing > 60)
        row = {"frame": fr, "t": round((fr - f0) / fps, 3), "popping": 0}
        allrows.append(row)
        if fr in surface_set:
            for key, m in (("crushed", crushed), ("stretched", stretched), ("sheared", sheared)):
                reg_hits[key].update(dict(zip(*np.unique(freg[m], return_counts=True))))
            zmin = float(V[:, 2].min())
            floor_n = int((V[:, 2] < -0.01).sum())
            row.update({"crushed": int(crushed.sum()), "stretched": int(stretched.sum()),
                        "sheared": int(sheared.sum()), "floor": floor_n, "zmin_cm": round(zmin * 100, 2)})
            # Use the explicit surface schedule; this is a local-shape excursion heuristic, not jerk.
            lap = V - np.add.reduceat(V[nbr_idx], nbr_start) / deg[:, None] if nbr_ok else None
            if lap is not None:
                prev.append(lap)
                prev_frames.append(fr)
                if len(prev) == 3:
                    # Measure center-sample local-shape excursion from the chord through its
                    # neighbors. Thus --pop=.5 means a half-mean-rest-edge excursion.
                    excursion = np.linalg.norm(prev[1] - (prev[0] + prev[2]) * 0.5, axis=1)
                    endpoint_motion = np.linalg.norm(prev[2] - prev[0], axis=1)
                    jump = excursion / edge_len
                    # Require a mostly out-and-back event, not a simple one-way translation.
                    m = (jump > POP_X) & (excursion > 1.5 * endpoint_motion)
                    center_row = per[-1]
                    center_row["popping"] = int(m.sum())
                    if m.any():
                        pop_regions.update(dict(zip(*np.unique(np.where(vdom[m] >= 0, breg[np.maximum(vdom[m], 0)],
                                                                          ridx["other"]), return_counts=True))))
                        if float(jump[m].max()) > pop_worst[0]:
                            pop_worst = (float(jump[m].max()), prev_frames[1])
                    prev.pop(0)
                    prev_frames.pop(0)
            per.append(row)
        if fr in bvh_set:
            tree, pairs, by_region = intersections(V)
            for (ri, rj), (ti, tj) in by_region.items():
                inter_pairs[(region_names[ri], region_names[rj])] += 1
            row["intersecting_pairs"] = len(pairs)
            depth, _ = penetration(V, by_region)
            deep_now = deep_body = 0.0
            for (ri, rj), d in depth.items():
                key = (region_names[ri], region_names[rj])
                pen_max[key] = max(pen_max[key], d)
                deep_now = max(deep_now, d)
                if "head" not in key:           # the head region carries the hair, which should swing aside
                    deep_body = max(deep_body, d)
            row["penetration_cm"] = round(deep_now * 100, 2)
            row["penetration_body_cm"] = round(deep_body * 100, 2)
            for key, flags in (("crushed", crushed), ("stretched", stretched), ("sheared", sheared)):
                exposed = normal_ray_exposed(tree, V, F, N, flags)
                row[f"{key}_normal_ray_exposed"] = int(exposed.sum())
                exposure_totals[key]["flagged"] += int(flags.sum())
                exposure_totals[key]["exposed"] += int(exposed.sum())
            sparse.append(row)
    if not per:
        continue

    def worst(rows, key, value_key=None):
        value_key = value_key or key
        r = max(rows, key=lambda r: r.get(value_key, 0))
        return {"frame": r["frame"], "t": r["t"], "count": r.get(value_key, 0)}
    n = len(per)
    popping_windows = max(n - 2, 0)
    popping_event_windows = sum(1 for r in per if r.get("popping", 0) > 0)
    candidate_counts = [r["intersecting_pairs"] for r in sparse]
    depths = [r["penetration_cm"] for r in sparse]
    depths_body = [r["penetration_body_cm"] for r in sparse]
    exposure = {}
    for key, counts in exposure_totals.items():
        flagged_samples = counts["flagged"]
        exposed_samples = counts["exposed"]
        exposure[key] = {
            "sample_frames": [r["frame"] for r in sparse],
            "flagged_triangle_samples": flagged_samples,
            "normal_ray_exposed_triangle_samples": exposed_samples,
            "normal_ray_exposed_pct_of_flagged": round(100 * exposed_samples / flagged_samples, 3) if flagged_samples else None,
            "normal_ray_exposed_pct_of_all_live_triangle_samples": round(100 * exposed_samples / (len(sparse) * nF), 3) if sparse else None,
            "worst_sample": worst(sparse, key, f"{key}_normal_ray_exposed") if sparse else None,
        }
    entry = {
        "frames": n,
        "surface_sample_frames": [r["frame"] for r in per],
        "bvh_sample_frames": [r["frame"] for r in sparse],
        "crushed_pct": round(100 * np.mean([r["crushed"] for r in per]) / nF, 3),
        "stretched_pct": round(100 * np.mean([r["stretched"] for r in per]) / nF, 3),
        "sheared_pct": round(100 * np.mean([r["sheared"] for r in per]) / nF, 3),
        "intersecting_pairs_mean": round(float(np.mean(candidate_counts)), 1) if candidate_counts else None,
        "intersecting_pairs_max": int(max(candidate_counts)) if candidate_counts else None,
        "penetration_max_cm": round(max(depths), 2) if depths else None,
        "penetration_median_cm": round(float(np.median(depths)), 2) if depths else None,
        "deep_frames_pct": round(100 * np.mean([d > a.deep for d in depths]), 1) if depths else None,
        "penetration_body_max_cm": round(max(depths_body), 2) if depths_body else None,
        "deep_body_frames_pct": round(100 * np.mean([d > a.deep for d in depths_body]), 1) if depths_body else None,
        "penetration_regions_cm": {f"{k[0]}×{k[1]}": round(v * 100, 2)
                                   for k, v in sorted(pen_max.items(), key=lambda kv: -kv[1])[:6]},
        "floor_frames_pct": round(100 * np.mean([r["floor"] > 0 for r in per]), 1),
        "floor_vertices_pct": round(100 * np.mean([r["floor"] for r in per]) / nV, 3),
        "zmin_cm": min(r["zmin_cm"] for r in per),
        "popping_eligible_windows": popping_windows,
        "popping_event_windows": popping_event_windows,
        "popping_event_window_pct": round(100 * popping_event_windows / popping_windows, 2) if popping_windows else None,
        "popping_worst_center": {"edges": round(pop_worst[0], 2), "frame": pop_worst[1]},
        "normal_ray_exposure": exposure,
        "worst": {**{k: worst(per, k) for k in ("crushed", "stretched", "sheared", "floor", "popping")},
                  "intersecting_pairs": worst(sparse, "intersecting_pairs") if sparse else None,
                  "penetration": worst(sparse, "penetration_cm") if sparse else None},
        "regions": {k: {region_names[i]: round(100 * c / n / nF, 3) for i, c in reg_hits[k].most_common(5)} for k in reg_hits},
        "intersecting_regions_frames": {f"{p[0]}×{p[1]}": c for p, c in inter_pairs.most_common(6)},
        "popping_regions": {region_names[i]: int(c) for i, c in pop_regions.most_common(4)},
    }
    if a.series:
        entry["series"] = allrows
    report["clips"][clip] = entry
    print(f"[aberr] {clip:16s} crushed {entry['crushed_pct']:5.2f}%  stretched {entry['stretched_pct']:5.2f}%  "
          f"sheared {entry['sheared_pct']:5.2f}%  intersecting {entry['intersecting_pairs_mean']} "
          f"(max {entry['intersecting_pairs_max']})  penetration max {entry['penetration_max_cm']} cm, "
          f"deep in {entry['deep_frames_pct']}% of frames  "
          f"floor {entry['floor_frames_pct']}% of frames (min {entry['zmin_cm']} cm)  "
          f"popping {entry['popping_event_windows']}/{entry['popping_eligible_windows']} sample windows  "
          f"normal-ray exposed C/S/H {exposure['crushed']['normal_ray_exposed_pct_of_flagged']}/"
          f"{exposure['stretched']['normal_ray_exposed_pct_of_flagged']}/"
          f"{exposure['sheared']['normal_ray_exposed_pct_of_flagged']}% of flagged triangle samples",
          flush=True)

def flags_at(fr, kind):
    """The flagged faces and normal-ray-exposed subset at one frame (for the diagnostic render)."""
    sc.frame_set(fr)
    dg = bpy.context.evaluated_depsgraph_get()
    V, _, A, N = evaluated(dg, F)
    require_rest_topology(V, f"diagnostic frame {fr}")
    ratio = A / np.maximum(A0, 1e-12)
    if kind == "crushed":
        m = live & (ratio < 0.5)
    elif kind == "stretched":
        m = live & (ratio > 2.0)
    elif kind == "sheared":
        R = bone_rotations()
        expect = np.einsum("fij,fj->fi", R[np.maximum(fdom, 0)], N0)
        m = live & (fdom >= 0) & (np.degrees(np.arccos(np.clip((expect * N).sum(1), -1, 1))) > 60)
    elif kind == "intersecting_pairs":
        _, pairs, _ = intersections(V)
        m = np.zeros(len(F), bool)
        for i, j in pairs:
            m[i] = m[j] = True
    elif kind == "penetration":
        _, _, by_region = intersections(V)
        _, vdepth = penetration(V, by_region)
        # red where deeper than --deep, amber where the surfaces only just cross
        return V, (vdepth[F] > 0).any(axis=1), (vdepth[F] > a.deep / 100).any(axis=1)
    elif kind == "popping":
        action = rig.animation_data.action
        f0, f1 = (int(x) for x in action.frame_range)
        if fr - a.step < f0 or fr + a.step > f1:
            raise SystemExit(f"[aberr] popping render frame {fr} has no full three-sample window")
        center_V = V
        laps = []
        for sample_fr in (fr - a.step, fr, fr + a.step):
            sc.frame_set(sample_fr)
            dg = bpy.context.evaluated_depsgraph_get()
            V_sample, _, _, _ = evaluated(dg, F)
            require_rest_topology(V_sample, f"popping diagnostic frame {sample_fr}")
            laps.append(V_sample - np.add.reduceat(V_sample[nbr_idx], nbr_start) / deg[:, None])
        excursion = np.linalg.norm(laps[1] - (laps[0] + laps[2]) * 0.5, axis=1)
        endpoint_motion = np.linalg.norm(laps[2] - laps[0], axis=1)
        vertex_flags = (excursion / edge_len > POP_X) & (excursion > 1.5 * endpoint_motion)
        m = vertex_flags[F].any(axis=1)
        return center_V, m, np.zeros(len(F), dtype=bool)
    else:
        m = live & (V[F].min(1)[:, 2] < -0.01)
    exposed = np.zeros(len(F), dtype=bool)
    if kind in ("crushed", "stretched", "sheared") and m.any():
        tree = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)
        exposed = normal_ray_exposed(tree, V, F, N, m)
    return V, m, exposed


def render_flags(V, m, exposed, path, title):
    """Show blocked flags amber and normal-ray-exposed flags red from two sides."""
    import math
    from mathutils import Vector
    me = bpy.data.meshes.new("aberr")
    me.from_pydata(V.tolist(), [], F.tolist())
    grey = bpy.data.materials.new("aberr_grey"); grey.diffuse_color = (0.72, 0.73, 0.76, 1)
    amber = bpy.data.materials.new("aberr_blocked"); amber.diffuse_color = (1.0, 0.55, 0.08, 1)
    red = bpy.data.materials.new("aberr_exposed"); red.diffuse_color = (0.95, 0.12, 0.10, 1)
    for mat, col in ((grey, (0.72, 0.73, 0.76, 1)), (amber, (1.0, 0.55, 0.08, 1)),
                     (red, (0.95, 0.12, 0.10, 1))):
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = col
    me.materials.append(grey); me.materials.append(amber); me.materials.append(red)
    idx = np.zeros(len(F), np.int32); idx[m] = 1; idx[exposed] = 2
    me.polygons.foreach_set("material_index", idx)
    ob = bpy.data.objects.new("aberr", me); sc.collection.objects.link(ob)
    hidden = [o for o in sc.objects if o.type == "MESH" and o is not ob]
    for o in hidden:
        o.hide_render = True
    lo, hi = V.min(0), V.max(0); ctr = (lo + hi) / 2; size = float(max(hi - lo))
    cam = bpy.data.objects.get("aberr_cam") or bpy.data.objects.new("aberr_cam", bpy.data.cameras.new("aberr_cam"))
    if cam.name not in sc.collection.objects:
        sc.collection.objects.link(cam)
    sc.camera = cam; cam.data.type = "ORTHO"; cam.data.ortho_scale = size * 1.08
    sc.render.resolution_x = sc.render.resolution_y = a.res
    eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
    if not bpy.data.objects.get("aberr_sun"):
        for nm, rot, e in (("aberr_sun", (50, 0, 35), 3.0), ("aberr_fill", (60, 0, 215), 1.5)):
            lt = bpy.data.objects.new(nm, bpy.data.lights.new(nm, "SUN")); sc.collection.objects.link(lt)
            lt.data.energy = e; lt.rotation_euler = tuple(math.radians(x) for x in rot)
    outs = []
    for az in (30, 210):
        r = math.radians(az)
        cam.location = Vector((ctr[0] + math.sin(r) * size * 3, ctr[1] - math.cos(r) * size * 3, ctr[2]))
        cam.rotation_euler = (Vector(ctr.tolist()) - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = f"{path}_{az}.png"
        bpy.ops.render.render(write_still=True)
        outs.append(sc.render.filepath)
    bpy.data.objects.remove(ob, do_unlink=True)
    for o in hidden:
        o.hide_render = False
    return outs


if a.render:
    os.makedirs(a.render, exist_ok=True)
    for clip, e in report["clips"].items():
        set_action(clip)
        for kind in ("crushed", "stretched", "sheared", "intersecting_pairs", "penetration", "floor", "popping"):
            w = e["worst"][kind]
            if not w or not w["count"]:
                continue
            V, m, exposed = flags_at(w["frame"], kind)
            render_flags(V, m, exposed, os.path.join(a.render, f"{clip}_{kind}_f{w['frame']}"), f"{clip} {kind}")
        for kind in ("crushed", "stretched", "sheared"):
            w = e["normal_ray_exposure"][kind]["worst_sample"]
            if not w or not w["count"]:
                continue
            V, m, exposed = flags_at(w["frame"], kind)
            render_flags(V, m, exposed,
                         os.path.join(a.render, f"{clip}_{kind}_exposure_worst_f{w['frame']}"),
                         f"{clip} {kind} exposure worst")
    print(f"[aberr] worst frames rendered -> {a.render}", flush=True)

# One line per character: equal-weight clip means and the clips/regions to inspect first.
if not report["clips"]:
    raise SystemExit("[aberr] no requested actions were found in the blend")
cl = report["clips"].values()
report["summary"] = {k: round(float(np.mean([c[k] for c in cl if c[k] is not None])), 3)
                     for k in ("crushed_pct", "stretched_pct", "sheared_pct",
                               "intersecting_pairs_mean", "penetration_max_cm", "deep_frames_pct",
                               "penetration_body_max_cm", "deep_body_frames_pct",
                               "floor_frames_pct")}
report["summary"]["popping_event_windows"] = sum(c["popping_event_windows"] for c in cl)
pop_pct = [c["popping_event_window_pct"] for c in cl if c["popping_event_window_pct"] is not None]
report["summary"]["popping_event_window_pct_equal_clip_mean"] = round(float(np.mean(pop_pct)), 2) if pop_pct else None
report["summary"]["normal_ray_exposed_pct_of_flagged_equal_clip_mean"] = {
    k: round(float(np.mean([c["normal_ray_exposure"][k]["normal_ray_exposed_pct_of_flagged"]
                             for c in cl if c["normal_ray_exposure"][k]["normal_ray_exposed_pct_of_flagged"] is not None])), 3)
    if any(c["normal_ray_exposure"][k]["normal_ray_exposed_pct_of_flagged"] is not None for c in report["clips"].values()) else None
    for k in ("crushed", "stretched", "sheared")}
report["summary"]["aggregation"] = "equal-weight arithmetic mean across clips; no frame-weighted inference"
report["skin"] = a.skin
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
out_dir = os.path.dirname(os.path.abspath(a.out))
fd, temp_out = tempfile.mkstemp(prefix=f".{os.path.basename(a.out)}.", suffix=".tmp", dir=out_dir)
try:
    with os.fdopen(fd, "w") as f:
        json.dump(report, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    if a.force:
        os.replace(temp_out, a.out)
    else:
        # A hard link provides an atomic no-clobber publish if another process created the
        # requested output after the preflight check.
        os.link(temp_out, a.out)
        os.unlink(temp_out)
finally:
    if os.path.exists(temp_out):
        os.unlink(temp_out)
print(f"[aberr] summary {report['summary']} -> {a.out}", flush=True)
