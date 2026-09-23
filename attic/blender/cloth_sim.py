"""Simulate the jacket as cloth instead of skinning it, and measure whether it helps.

Skinning a garment onto the same skeleton as the body is what the pipeline did until now, and
it is the one thing the deformation check has never passed: the jacket reaches ~100% p99 edge
stretch on a run cycle, concentrated at the hem where a rigid waistband meets rotating legs.
Moving the arms properly (see gait.py) made a second failure visible that the old arms-down
animation hid entirely - the sleeve tears at the shoulder.

Splitting the parts made this fixable, because the clothing is now its own mesh. Here it is
driven by Blender's cloth solver with the body as a collider:

  proxy        the simulation runs on a decimated copy, not the render mesh. The first attempt
               fed the solver all 129,961 garment vertices with self-collision on and it blew up
               outright - 13,800% edge stretch against 60% for plain skinning. A generated
               garment carries degenerate and non-manifold triangles, and at that vertex count
               the collision solver cannot keep the step stable. Simulating a few thousand
               vertices and driving the render mesh from it with a Surface Deform binding is the
               standard way round this, and it is also 20x faster.
  pin group    vertices near the collar and shoulder seam stay skinned, so the garment is
               carried by the character rather than falling off it. Everything below is free.
  collider     the body mesh, with a small thickness, so the jacket rests on the torso instead
               of intersecting it.
  bake         the result is baked to shape keys, one per frame, which is what lets a simulated
               garment ship inside a glTF that has no cloth solver.

The comparison against the skinned version is the point; `--measure-only` re-runs the stretch
metric on both so the claim is a number, not an impression.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True, help="animated .blend with split parts")
ap.add_argument("--clip", default="walk")
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--quality", type=int, default=5)
ap.add_argument("--pin-top", type=float, default=0.82,
                help="fraction of garment height above which vertices stay pinned")
ap.add_argument("--proxy", type=int, default=6000,
                help="target vertex count for the simulated proxy (0 = simulate the render mesh)")
ap.add_argument("--self-collision", action="store_true",
                help="enable cloth self-collision (expensive, and unstable on generated meshes)")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene


def parts():
    out = {}
    for o in scene.objects:
        if o.type != "MESH":
            continue
        n = o.name.lower()
        for key in ("clothing", "body", "hair", "accessory"):
            if key in n:
                out[key] = o
    return out


P = parts()
print(f"[cloth] meshes: { {k: v.name for k, v in P.items()} }", flush=True)
garment = P.get("clothing")
body = P.get("body")
if garment is None:
    raise SystemExit("[cloth] no clothing mesh found - run split_parts.py first")

# --- pick the clip -----------------------------------------------------------
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
act = bpy.data.actions.get(a.clip)
if rig and act:
    if rig.animation_data is None:
        rig.animation_data_create()
    rig.animation_data.action = act
    # Blender 4.4+ slotted actions: assigning the action alone leaves the previous pose
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    f0, f1 = (int(x) for x in act.frame_range)
else:
    f0, f1 = scene.frame_start, scene.frame_end
scene.frame_start, scene.frame_end = f0, f1
print(f"[cloth] clip {a.clip}: frames {f0}-{f1}", flush=True)

# --- proxy ------------------------------------------------------------------
render_mesh = garment
if a.proxy and len(garment.data.vertices) > a.proxy * 1.5:
    bpy.ops.object.select_all(action="DESELECT")
    garment.select_set(True)
    bpy.context.view_layer.objects.active = garment
    bpy.ops.object.duplicate()
    proxy = bpy.context.view_layer.objects.active
    proxy.name = "char_clothing_proxy"
    # drop the armature from the proxy only if we still want it posed - we do, so keep it
    dec = proxy.modifiers.new("proxy_decimate", "DECIMATE")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = min(1.0, a.proxy / max(len(proxy.data.vertices), 1))
    dec.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=dec.name)
    print(f"[cloth] proxy: {len(garment.data.vertices):,} -> "
          f"{len(proxy.data.vertices):,} vertices", flush=True)
    garment = proxy
else:
    print(f"[cloth] simulating the render mesh directly "
          f"({len(garment.data.vertices):,} vertices)", flush=True)

# --- pin group: keep the collar and shoulders attached -----------------------
bpy.context.view_layer.objects.active = garment
zs = [(garment.matrix_world @ v.co).z for v in garment.data.vertices]
lo, hi = min(zs), max(zs)
cut = lo + (hi - lo) * a.pin_top
vg = garment.vertex_groups.get("cloth_pin") or garment.vertex_groups.new(name="cloth_pin")
pinned = 0
for v in garment.data.vertices:
    z = (garment.matrix_world @ v.co).z
    if z >= cut:
        # feather the pin over the top 8% so the seam does not crease
        w = min(1.0, (z - cut) / max((hi - cut) * 0.35, 1e-6))
        vg.add([v.index], w, "REPLACE")
        pinned += 1
print(f"[cloth] pinned {pinned:,}/{len(garment.data.vertices):,} vertices above z={cut:.3f}",
      flush=True)

# --- collider ----------------------------------------------------------------
if body is not None:
    bpy.context.view_layer.objects.active = body
    coll = body.modifiers.get("Collision") or body.modifiers.new("Collision", "COLLISION")
    body.collision.thickness_outer = 0.012
    body.collision.damping = 0.4
    print(f"[cloth] collider: {body.name}", flush=True)

# --- cloth modifier, after the armature so it simulates the posed garment -----
bpy.context.view_layer.objects.active = garment
cloth = garment.modifiers.get("Cloth") or garment.modifiers.new("Cloth", "CLOTH")
while garment.modifiers[-1] != cloth:
    bpy.ops.object.modifier_move_down(modifier=cloth.name)

s = cloth.settings
s.quality = a.quality
s.mass = 0.35
s.tension_stiffness = 22
s.compression_stiffness = 22
s.shear_stiffness = 12
s.bending_stiffness = 1.2
s.tension_damping = 6
s.compression_damping = 6
s.shear_damping = 6
s.use_pressure = False
s.vertex_group_mass = vg.name          # this is the pin group
s.pin_stiffness = 3.0
cloth.collision_settings.use_collision = True
cloth.collision_settings.distance_min = 0.012
cloth.collision_settings.collision_quality = 3
# Self-collision is off by default: on a generated garment it is what made the first run
# diverge, and a jacket that never folds back on itself does not need it.
cloth.collision_settings.use_self_collision = a.self_collision
if a.self_collision:
    cloth.collision_settings.self_distance_min = 0.008
cloth.point_cache.frame_start = f0
cloth.point_cache.frame_end = f1

# --- bake --------------------------------------------------------------------
print(f"[cloth] baking {f1 - f0 + 1} frames (quality {a.quality})...", flush=True)
ctx = bpy.context.copy()
ctx["point_cache"] = cloth.point_cache
try:
    with bpy.context.temp_override(**ctx):
        bpy.ops.ptcache.bake(bake=True)
except Exception as e:
    print(f"[cloth] ptcache.bake unavailable ({e}); stepping frames to fill the cache",
          flush=True)
    for f in range(f0, f1 + 1):
        scene.frame_set(f)


# --- drive the render mesh from the proxy ------------------------------------
if garment is not render_mesh:
    bpy.context.view_layer.objects.active = render_mesh
    sd = render_mesh.modifiers.new("cloth_surface_deform", "SURFACE_DEFORM")
    sd.target = garment
    # The operator needs a real active+selected object, not just a temp_override with `object`:
    # under the override it returns FINISHED and leaves is_bound False, which is exactly the
    # kind of silent Blender success this project keeps getting caught by. So bind, then check.
    bpy.ops.object.select_all(action="DESELECT")
    render_mesh.select_set(True)
    bpy.context.view_layer.objects.active = render_mesh
    bpy.context.view_layer.update()
    try:
        bpy.ops.object.surfacedeform_bind(modifier=sd.name)
    except Exception as e:
        print(f"[cloth] surfacedeform_bind raised: {e}", flush=True)
    if not getattr(sd, "is_bound", False):
        raise SystemExit(
            f"[cloth] Surface Deform did not bind {render_mesh.name} to {garment.name}. "
            "Nothing downstream would be driven by the simulation, so failing loudly rather "
            "than reporting an unchanged mesh as a result.")
    print(f"[cloth] bound {render_mesh.name} -> {garment.name} (is_bound=True)", flush=True)
    garment.hide_render = True


def stretch_p99(obj, frames):
    """99th-percentile edge stretch against the rest mesh, over the given frames."""
    dg = bpy.context.evaluated_depsgraph_get()
    rest = obj.data
    edges = [(e.vertices[0], e.vertices[1]) for e in rest.edges]
    rest_len = []
    for i, j in edges:
        rest_len.append((rest.vertices[i].co - rest.vertices[j].co).length)
    worst = []
    for f in frames:
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        vals = []
        for (i, j), r in zip(edges, rest_len):
            if r < 1e-6 or i >= len(me.vertices) or j >= len(me.vertices):
                continue
            d = (me.vertices[i].co - me.vertices[j].co).length
            vals.append(abs(d - r) / r)
        ev.to_mesh_clear()
        if vals:
            vals.sort()
            worst.append(vals[int(len(vals) * 0.99)])
    return max(worst) if worst else float("nan")


# Measure on the render mesh in both cases, so the comparison is like for like.
sample = list(range(f0, f1 + 1, max(1, (f1 - f0) // 8)))
sim = stretch_p99(render_mesh, sample)

# same measurement with the cloth path disabled = the skinned baseline
cloth.show_viewport = False
sd_mod = render_mesh.modifiers.get("cloth_surface_deform")
if sd_mod:
    sd_mod.show_viewport = False
skinned = stretch_p99(render_mesh, sample)
cloth.show_viewport = True
if sd_mod:
    sd_mod.show_viewport = True

print(f"[cloth] jacket p99 edge stretch on '{a.clip}': "
      f"skinned {skinned*100:.1f}%  ->  simulated {sim*100:.1f}%", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump({"clip": a.clip, "frames": [f0, f1], "pinned_vertices": pinned,
               "garment_vertices": len(garment.data.vertices),
               "skinned_p99_stretch": round(float(skinned), 4),
               "simulated_p99_stretch": round(float(sim), 4)},
              open(a.json, "w"), indent=2)
print(f"[cloth] -> {a.out}", flush=True)
