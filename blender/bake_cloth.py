"""Bake cloth simulation into glTF morph targets, for every clip.

glTF has no cloth solver. A simulated garment can only reach a browser as baked vertex
animation, which in glTF means morph targets: one target per simulated frame, its weight keyed
to 1 on its own frame and 0 on the neighbours, so playback steps through the simulation.

Three things make this less obvious than it sounds:

  * **The target stores simulated minus skinned, not the simulated shape.** glTF applies morph
    targets *before* skinning, so storing the posed result would apply the pose a second time.
  * **The garment is decimated once, before any clip is simulated.** Morph targets index
    vertices, so a per-clip decimation would make each clip's targets incompatible with the
    mesh they are attached to.
  * **Every clip's targets live on the same mesh**, so each clip's shape-key action has to pin
    the *other* clips' targets to zero across its own frame range, or a walk drags the jump's
    cloth along with it.

Cost is three floats per vertex per target, which is why the garment is decimated for the web
build and the frame count is capped; the byte count is reported rather than hidden.

Run: blender -b -noaudio --python bake_cloth.py -- --blend animated.blend \
         --clips walk,run,jump,idle,wave --out cloth_baked.blend
"""
import argparse
import json
import math
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--clips", default="walk")
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--part", default="clothing")
ap.add_argument("--web-verts", type=int, default=3200)
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--max-targets", type=int, default=13)
ap.add_argument("--quality", type=int, default=6)
ap.add_argument("--pin-top", type=float, default=0.82)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
garment = next((o for o in scene.objects if o.type == "MESH" and a.part in o.name.lower()), None)
# the clean rig calls the skinned mesh "char_skin"; the older one called it "char_body"
body = next((o for o in scene.objects if o.type == "MESH"
             and ("body" in o.name.lower() or "skin" in o.name.lower())), None)
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
if garment is None or rig is None:
    raise SystemExit("[bake] need a garment mesh and an armature")

clip_names = [c.strip() for c in a.clips.split(",") if c.strip()]
missing = [c for c in clip_names if bpy.data.actions.get(c) is None]
if missing:
    raise SystemExit(f"[bake] missing action(s) {missing}; "
                     f"have {[x.name for x in bpy.data.actions]}")

# ---- decimate once, before any simulation ---------------------------------------------------
bpy.context.view_layer.objects.active = garment
n0 = len(garment.data.vertices)
if a.web_verts and n0 > a.web_verts * 1.2:
    dec = garment.modifiers.new("web_decimate", "DECIMATE")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = a.web_verts / n0
    dec.use_collapse_triangulate = True
    while garment.modifiers[0] != dec:
        bpy.ops.object.modifier_move_up(modifier=dec.name)
    bpy.ops.object.modifier_apply(modifier=dec.name)
print(f"[bake] garment {n0:,} -> {len(garment.data.vertices):,} vertices", flush=True)

# ---- pin group, collider, cloth --------------------------------------------------------------
zs = [(garment.matrix_world @ v.co).z for v in garment.data.vertices]
lo, hi = min(zs), max(zs)
cut = lo + (hi - lo) * a.pin_top
vg = garment.vertex_groups.get("cloth_pin") or garment.vertex_groups.new(name="cloth_pin")
pinned = 0
for v in garment.data.vertices:
    z = (garment.matrix_world @ v.co).z
    if z >= cut:
        vg.add([v.index], min(1.0, (z - cut) / max((hi - cut) * 0.35, 1e-6)), "REPLACE")
        pinned += 1

if body is not None:
    bpy.context.view_layer.objects.active = body
    body.modifiers.get("Collision") or body.modifiers.new("Collision", "COLLISION")
    body.collision.thickness_outer = 0.012
    body.collision.damping = 0.4

bpy.context.view_layer.objects.active = garment
cloth = garment.modifiers.get("Cloth") or garment.modifiers.new("Cloth", "CLOTH")
while garment.modifiers[-1] != cloth:
    bpy.ops.object.modifier_move_down(modifier=cloth.name)
s = cloth.settings
s.quality = a.quality
s.mass = 0.3
s.tension_stiffness = 18
s.compression_stiffness = 18
s.shear_stiffness = 10
s.bending_stiffness = 1.0
s.vertex_group_mass = vg.name
s.pin_stiffness = 3.0
cloth.collision_settings.use_collision = True
cloth.collision_settings.distance_min = 0.012
cloth.collision_settings.use_self_collision = False

if garment.data.shape_keys is None:
    garment.shape_key_add(name="Basis", from_mix=False)
basis = garment.data.shape_keys.key_blocks["Basis"]

all_info = []
for clip_name in clip_names:
    act = bpy.data.actions[clip_name]
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    f0, f1 = (int(x) for x in act.frame_range)
    scene.frame_start, scene.frame_end = f0, f1
    cloth.point_cache.frame_start = f0
    cloth.point_cache.frame_end = f1
    cloth.show_viewport = True

    stride = max(a.stride, math.ceil((f1 - f0 + 1) / max(a.max_targets, 1)))
    print(f"[bake] '{clip_name}' frames {f0}-{f1}, stride {stride}", flush=True)
    ctx = bpy.context.copy()
    ctx["point_cache"] = cloth.point_cache
    # Free the previous clip's cache explicitly. free_bake() takes no arguments; passing one
    # makes the whole call raise, which silently dropped this through to the frame-stepping
    # fallback and risked a clip being baked against the cache of the clip before it.
    try:
        with bpy.context.temp_override(**ctx):
            bpy.ops.ptcache.free_bake()
    except Exception as e:
        print(f"[bake]   free_bake failed ({e})", flush=True)
    try:
        with bpy.context.temp_override(**ctx):
            bpy.ops.ptcache.bake(bake=True)
    except Exception as e:
        print(f"[bake]   ptcache.bake unavailable ({e}); stepping frames", flush=True)
        scene.frame_set(f0)
        for f in range(f0, f1 + 1):
            scene.frame_set(f)

    made = []
    for f in range(f0, f1 + 1, stride):
        scene.frame_set(f)

        cloth.show_viewport = True
        dg = bpy.context.evaluated_depsgraph_get()
        ev = garment.evaluated_get(dg)
        me = ev.to_mesh()
        sim = [v.co.copy() for v in me.vertices]
        ev.to_mesh_clear()

        cloth.show_viewport = False
        dg = bpy.context.evaluated_depsgraph_get()
        ev = garment.evaluated_get(dg)
        me = ev.to_mesh()
        skin = [v.co.copy() for v in me.vertices]
        ev.to_mesh_clear()
        cloth.show_viewport = True

        n = len(garment.data.vertices)
        if len(sim) != n or len(skin) != n:
            raise SystemExit(f"[bake] vertex count changed at {clip_name} f{f}: "
                             f"{n} base, {len(sim)} sim, {len(skin)} skin")

        kb = garment.shape_key_add(name=f"{clip_name}_{f:04d}", from_mix=False)
        for i in range(n):
            kb.data[i].co = basis.data[i].co + (sim[i] - skin[i])
        kb.value = 0.0
        made.append((f, kb))
    print(f"[bake]   {len(made)} morph targets", flush=True)

    # one shape-key action per clip; detach before keying the next
    kd = garment.data.shape_keys.animation_data
    if kd:
        kd.action = None
    for idx, (f, kb) in enumerate(made):
        prev_f = made[idx - 1][0] if idx > 0 else f - stride
        next_f = made[idx + 1][0] if idx + 1 < len(made) else f + stride
        kb.value = 0.0
        kb.keyframe_insert("value", frame=prev_f)
        kb.value = 1.0
        kb.keyframe_insert("value", frame=f)
        kb.value = 0.0
        kb.keyframe_insert("value", frame=next_f)
    # pin the other clips' targets to zero over this clip's range
    for other in all_info:
        for _, kb in other["keys"]:
            kb.value = 0.0
            kb.keyframe_insert("value", frame=f0)
            kb.keyframe_insert("value", frame=f1)
    kd = garment.data.shape_keys.animation_data
    if kd and kd.action:
        kd.action.use_fake_user = True
        kd.action.name = f"cloth_{clip_name}"
        kd.action = None
    all_info.append({"clip": clip_name, "targets": len(made), "stride": stride,
                     "frames": [f0, f1], "keys": made})

cloth.show_viewport = False
cloth.show_render = False
for kb in garment.data.shape_keys.key_blocks:
    kb.value = 0.0

# ---- pair each clip's skeletal and cloth actions on identically-named NLA tracks -------------
# Blender's glTF exporter in ACTIONS mode only ships the shape-key action that happens to be
# *assigned* when the export runs, so a multi-clip bake exported 57 morph targets with nothing
# driving them - the jacket loaded and never moved. Exported as NLA tracks instead, one track
# per clip on each datablock, the exporter merges the armature channels and the morph-weight
# channels of same-named tracks into a single glTF animation per clip.
rig.animation_data.action = None
keys_id = garment.data.shape_keys
if keys_id.animation_data is None:
    keys_id.animation_data_create()
keys_id.animation_data.action = None
for tr in list(rig.animation_data.nla_tracks):
    rig.animation_data.nla_tracks.remove(tr)
for tr in list(keys_id.animation_data.nla_tracks):
    keys_id.animation_data.nla_tracks.remove(tr)

for x in all_info:
    name = x["clip"]
    skel = bpy.data.actions.get(name)
    cl = bpy.data.actions.get(f"cloth_{name}")
    if skel:
        t = rig.animation_data.nla_tracks.new()
        t.name = name
        t.strips.new(name, int(skel.frame_range[0]), skel)
    if cl:
        t = keys_id.animation_data.nla_tracks.new()
        t.name = name                      # same track name is what pairs them
        t.strips.new(name, int(cl.frame_range[0]), cl)
    print(f"[bake] NLA track '{name}': skeletal={bool(skel)} cloth={bool(cl)}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
total = sum(x["targets"] for x in all_info)
info = {"clips": [{k: v for k, v in x.items() if k != "keys"} for x in all_info],
        "garment_vertices": len(garment.data.vertices), "morph_targets": total,
        "pinned": pinned,
        "approx_target_bytes": total * len(garment.data.vertices) * 12}
if a.json:
    json.dump(info, open(a.json, "w"), indent=2)
print(f"[bake] {json.dumps({k: v for k, v in info.items() if k != 'clips'})}", flush=True)
print(f"[bake] -> {a.out}", flush=True)
