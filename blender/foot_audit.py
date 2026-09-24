"""Foot contact, measured the way a player sees it: on the deformed shoe, moving at clip speed.

A foot "slides" when the part of the shoe touching the floor moves across it. That is the only
definition that survives contact with a screen, so this measures exactly that - the evaluated
mesh, not a joint (a generated rig's foot joints sit inside the shoe, and the instep joint rises
with the heel while the sole stays planted). Each in-place clip is played with the character
carried forward at the ground speed its manifest states, the way a controller moves it:

  slip       horizontal speed of the lowest shoe vertex while it touches the floor (< 4 mm), as a
             share of ground speed - the same vertex tracked from one key to the next, so a foot
             rolling heel to toe does not count as sliding
  on floor   share of keys with that foot down: about 60% for a walk, 35% for a jog or run
  through    how far the shoe goes below the floor at worst, and how often by more than 1 cm

Run: blender -b -noaudio --python foot_audit.py -- --blend final.blend --report retarget.json
         [--clips walk run] [--json out.json]
"""
import argparse
import json
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--report", required=True, help="retarget.py's JSON report (ground speeds)")
ap.add_argument("--clips", nargs="*", default=None)
ap.add_argument("--json", default=None)
a = ap.parse_args(argv)

rep = json.load(open(a.report))
bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
fps = sc.render.fps / sc.render.fps_base
meshes = [o for o in sc.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)]


def foot_groups(side):
    names = {f"{side}_ankle", f"{side}_foot", f"mixamorig:{side.title()}Foot",
             f"mixamorig:{side.title()}ToeBase"}
    out = []
    for o in meshes:
        gi = {g.index for g in o.vertex_groups if g.name in names}
        idx = [v.index for v in o.data.vertices if sum(g.weight for g in v.groups if g.group in gi) > 0.5]
        out.append((o, np.array(idx, dtype=int)))
    return out


FEET = {"left": foot_groups("left"), "right": foot_groups("right")}


def feet_now():
    dg = bpy.context.evaluated_depsgraph_get()
    out = {}
    for side, lst in FEET.items():
        pts = []
        for o, idx in lst:
            if len(idx) == 0:
                continue
            ev = o.evaluated_get(dg)
            m = ev.to_mesh()
            X = np.empty(len(m.vertices) * 3)
            m.vertices.foreach_get("co", X)
            X = X.reshape(-1, 3)[idx]
            mw = np.array(o.matrix_world)
            pts.append(X @ mw[:3, :3].T + mw[:3, 3])
            ev.to_mesh_clear()
        out[side] = np.vstack(pts)
    return out


results = {}
for name in (a.clips or list(rep)):
    act = bpy.data.actions.get(name)
    if act is None:
        continue
    rig.animation_data_create()
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    f0, f1 = (int(x) for x in act.frame_range)
    v = (rep.get(name) or {}).get("stride_speed_mps") or 0.0
    # carried the way the clip travels: back-pedals backward, strafes sideways (travel_deg is
    # counter-clockwise from forward, forward is -Y). Carrying every clip forward read a
    # back-pedal as 200% slip and a strafe as 141%.
    th = np.radians((rep.get(name) or {}).get("travel_deg") or 0.0)
    carry = np.array([np.sin(th), -np.cos(th), 0.0]) * v
    F = []
    for f in range(f0, f1 + 1):
        sc.frame_set(f)
        F.append({s: p + carry * (f - f0) / fps for s, p in feet_now().items()})
    ref = v if v >= 1.0 else 1.0                     # stationary clips: m/s per 1 m/s
    row = {"ground_speed_mps": v}
    for side in ("left", "right"):
        slip, down, depth = [], 0, []
        for i in range(1, len(F) - 1):
            z = F[i][side][:, 2]
            depth.append(float(z.min()))
            if z.min() < 0.004:
                down += 1
                k = int(np.argmin(z))
                d = (F[i + 1][side][k, :2] - F[i - 1][side][k, :2]) * fps / 2
                slip.append(float(np.linalg.norm(d)))
        depth = np.array(depth)
        row[side] = {"slip": round(float(np.mean(slip)) / ref, 4) if slip else None,
                     "on_floor": round(down / max(len(F) - 2, 1), 3),
                     "deepest_cm": round(float(depth.min()) * 100, 2),
                     "keys_over_1cm_through": round(float(np.mean(depth < -0.01)), 3)}
    results[name] = row
    L, R = row["left"], row["right"]
    fmt = lambda x: "  n/a" if x is None else f"{x*100:5.1f}%"
    print(f"[feet] {name:6s} {v:4.2f} m/s | slip L {fmt(L['slip'])} R {fmt(R['slip'])} | on floor "
          f"L {L['on_floor']*100:3.0f}% R {R['on_floor']*100:3.0f}% | deepest "
          f"{min(L['deepest_cm'], R['deepest_cm']):+.1f} cm", flush=True)
if a.json:
    json.dump(results, open(a.json, "w"), indent=2)
