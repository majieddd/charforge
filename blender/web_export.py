"""Export a web-sized build of the animated character.

The full asset is a 170k-vertex mesh with 2K PBR maps and six clips - about 14 MB of GLB, which
is more than a browser page should carry. This decimates the skinned meshes and downsizes the
textures, keeping the armature, the vertex groups and every action intact, so the result drops
straight into three.js and still plays all six animations.

Decimate is applied with the armature still bound: Blender's collapse decimator carries vertex
groups through, so the skinning survives. The part-id attribute is preserved too, which is what
lets the viewer colour the character by body / clothing / hair / accessory.

Run: blender -b -noaudio --python web_export.py -- --blend animated2.blend --out web/character.glb
"""
import argparse
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ratio", type=float, default=0.22, help="decimate collapse ratio")
ap.add_argument("--tex", type=int, default=1024, help="max texture edge")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)

# Drop anything that is not part of the character. An Icosphere left over from an earlier
# stage rode along into the shipped GLB unnoticed.
KEEP = ("body", "skin", "clothing", "hair", "accessory")
for o in list(bpy.context.scene.objects):
    if o.type == "MESH" and not any(k in o.name.lower() for k in KEEP):
        print(f"  [web] dropping stray object {o.name!r} ({len(o.data.vertices)} verts)",
              flush=True)
        bpy.data.objects.remove(o, do_unlink=True)

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
before = sum(len(o.data.polygons) for o in meshes)

_rig = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
_tracks = len(_rig.animation_data.nla_tracks) if (_rig and _rig.animation_data) else 0
ANIM_MODE = "NLA_TRACKS" if _tracks else "ACTIONS"
print(f"  [web] {_tracks} NLA track(s) -> animation mode {ANIM_MODE}, "
      f"{len(bpy.data.actions)} action(s) in file", flush=True)

for o in meshes:
    if o.data.shape_keys is not None:
        # The baked-cloth garment carries one morph target per simulated frame. Decimating it
        # would invalidate every one of them, so it arrives pre-decimated from bake_cloth.py.
        print(f"  [web] {o.name}: {len(o.data.shape_keys.key_blocks)-1} morph targets, "
              f"{len(o.data.vertices):,} verts - left alone", flush=True)
        continue
    bpy.context.view_layer.objects.active = o
    m = o.modifiers.new("web_decimate", "DECIMATE")
    m.decimate_type = "COLLAPSE"
    m.ratio = a.ratio
    m.use_collapse_triangulate = True
    try:
        bpy.ops.object.modifier_apply(modifier=m.name)
    except Exception as e:
        print(f"  [web] {o.name}: decimate failed ({e}), leaving as-is", flush=True)

after = sum(len(o.data.polygons) for o in bpy.context.scene.objects if o.type == "MESH")

# Downsize textures in place. scale() is destructive, which is what we want here - this file is
# a throwaway build, never saved back over the master .blend.
for img in bpy.data.images:
    if img.size[0] > a.tex or img.size[1] > a.tex:
        w, h = img.size
        s = a.tex / max(w, h)
        img.scale(max(1, int(w * s)), max(1, int(h * s)))
        print(f"  [web] texture {img.name}: {w}x{h} -> {img.size[0]}x{img.size[1]}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=a.out,
    export_format="GLB",
    export_animations=True,
    # NLA_TRACKS when the cloth bake has paired each clip's skeletal and morph-weight actions
    # on identically-named tracks, so they export as one animation per clip; ACTIONS otherwise.
    # Picking the wrong one is silent: NLA_TRACKS on a file with no tracks exports zero
    # animations and the character arrives frozen.
    export_animation_mode=ANIM_MODE,
    export_skins=True,
    export_apply=False,
    # PNG, not JPEG. A marching-cubes character's UV atlas is thousands of tiny scattered
    # islands packed tight against each other, and JPEG's block transform bleeds colour across
    # those boundaries: black atlas gaps bled into the face islands as dark gashes, and skin
    # islands bled into the jacket as rust-coloured speckles. Lossless is not a luxury on this
    # kind of atlas, it is a correctness requirement. The source maps are 1024 PNG already, so
    # this costs a few MB and loses nothing.
    export_image_format="AUTO",
    export_morph=True,
    export_morph_normal=False,
)
size = os.path.getsize(a.out) / 1e6
print(f"[web] {before:,} -> {after:,} polygons, {size:.1f} MB -> {a.out}", flush=True)
