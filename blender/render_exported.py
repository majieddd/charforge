"""Render every frame of the *exported* GLB, not the source .blend.

Everything checked so far has been checked in the file the pipeline authors. The browser loads
something else: a glTF, with the skinning re-expressed as joint matrices, the cloth re-expressed
as morph targets, and normals that were *not* morphed. Any of those can differ from the source,
and a defect introduced by the export is invisible to every check that runs before it.

So this re-imports the shipped GLB and renders it frame by frame - every frame, not a sample.
Eight frames out of forty is enough to see whether a walk cycle is a walk cycle; it is not
enough to see a garment tear for three frames and recover, which is what reads as "ripping at
the seams" when it plays back at speed.

Run: blender -b -noaudio --python render_exported.py -- --glb character.glb \
         --clip idle --out frames/ --res 512
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Euler, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--glb", required=True)
ap.add_argument("--clip", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--az", type=float, default=35.0)
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--max-frames", type=int, default=200)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.glb)
scene = bpy.context.scene
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
meshes = [o for o in scene.objects if o.type == "MESH"]
if rig is None:
    raise SystemExit("[render] no armature in the GLB")

print(f"[render] meshes: {[(o.name, len(o.data.vertices)) for o in meshes]}", flush=True)
print(f"[render] actions: {[x.name for x in bpy.data.actions]}", flush=True)

# glTF animations import as actions; the armature action and the shape-key action are separate
# objects again, so both have to be assigned for the clip to play with its cloth.
skel = next((x for x in bpy.data.actions if x.name == a.clip), None)
if skel is None:
    cands = [x for x in bpy.data.actions if a.clip in x.name]
    skel = cands[0] if cands else None
if skel is None:
    raise SystemExit(f"[render] no action matching {a.clip!r}; "
                     f"have {[x.name for x in bpy.data.actions]}")
if rig.animation_data is None:
    rig.animation_data_create()
rig.animation_data.action = skel
if hasattr(rig.animation_data, "action_slot") and getattr(skel, "slots", None):
    rig.animation_data.action_slot = skel.slots[0]

n_morph = 0
for ob in meshes:
    keys = ob.data.shape_keys
    if keys is None:
        continue
    n_morph = len(keys.key_blocks) - 1
    ad = keys.animation_data
    if ad is None:
        ad = keys.animation_data_create()
    # glTF puts the morph-weight channels in the same animation, which Blender imports as a
    # second action whose name matches or is suffixed. Find it by what it drives.
    def drives_keys(act):
        try:
            fcs = list(act.fcurves)
        except Exception:
            fcs = []
            for L in act.layers:
                for st in L.strips:
                    for sl in act.slots:
                        cb = st.channelbag(sl)
                        if cb:
                            fcs.extend(cb.fcurves)
        return any("key_blocks" in fc.data_path or "value" == fc.data_path for fc in fcs)

    match = [x for x in bpy.data.actions if drives_keys(x) and a.clip in x.name]
    if match:
        ad.action = match[0]
        # Slots are typed: an action can hold both an object slot (OB) and a shape-key slot
        # (KE), and assigning the wrong one raises "not suitable for this data-block type".
        if hasattr(ad, "action_slot") and getattr(match[0], "slots", None):
            slot = next((sl for sl in match[0].slots
                         if getattr(sl, "target_id_type", "") == "KEY"), None)
            if slot is not None:
                ad.action_slot = slot
        print(f"[render] {ob.name}: morph action '{match[0].name}' ({n_morph} targets)", flush=True)
    else:
        print(f"[render] {ob.name}: {n_morph} morph targets but NO morph action matched "
              f"{a.clip!r} - cloth will not play", flush=True)

f0, f1 = (int(x) for x in skel.frame_range)
f1 = min(f1, f0 + a.max_frames - 1)
scene.frame_start, scene.frame_end = f0, f1

# ---- camera framed on the character, held still so tearing is not masked by motion ----------
scene.render.engine = "BLENDER_EEVEE"
scene.render.film_transparent = True
scene.render.resolution_x = scene.render.resolution_y = a.res
cd = bpy.data.cameras.new("c")
cam = bpy.data.objects.new("c", cd)
scene.collection.objects.link(cam)
cd.type = "ORTHO"
scene.camera = cam

lo_z, hi_z = 1e9, -1e9
scene.frame_set(f0)
for ob in meshes:
    for c in ob.bound_box:
        z = (ob.matrix_world @ Vector(c)).z
        lo_z, hi_z = min(lo_z, z), max(hi_z, z)
cd.ortho_scale = (hi_z - lo_z) * 1.12
mid = (hi_z + lo_z) * 0.5
r = math.radians(a.az)
cam.location = (5 * math.sin(r), -5 * math.cos(r), mid)
cam.rotation_euler = Euler((math.radians(90), 0, r))

key = bpy.data.lights.new("key", "SUN")
ko = bpy.data.objects.new("key", key)
scene.collection.objects.link(ko)
key.energy = 4.0
ko.rotation_euler = Euler((math.radians(55), 0, math.radians(40)))
fill = bpy.data.lights.new("fill", "SUN")
fo = bpy.data.objects.new("fill", fill)
scene.collection.objects.link(fo)
fill.energy = 1.6
fo.rotation_euler = Euler((math.radians(70), 0, math.radians(-120)))

os.makedirs(a.out, exist_ok=True)
n = 0
for f in range(f0, f1 + 1, a.every):
    scene.frame_set(f)
    scene.render.filepath = os.path.join(a.out, f"f{f:04d}.png")
    bpy.ops.render.render(write_still=True)
    n += 1
    if n % 20 == 0:
        print(f"[render] {n} frames", flush=True)
print(f"[render] {n} frames of '{a.clip}' ({f0}-{f1}) -> {a.out}", flush=True)
