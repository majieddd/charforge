"""The packaged character as UniMate should read it: the same rig, mesh and clips, without morph targets.

    blender -b -noaudio --python unimate_input.py -- --glb out/<n>/<n>.glb --out work/<n>/unimate/<n>.glb

UniMate's preprocess_char.py moves and scales the mesh into its canonical frame, but a mesh's shape keys
keep their own coordinates: Cadet's face shapes (blink, smile, ...) then pulled every vertex back
towards its old place and the export tore into ribbons, even for his own wave. The jaw bone is fine.
UniMate only makes bone motion, and retarget.py carries only rotations back to the character, whose own
face rig is untouched.
"""
import argparse
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--glb", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.glb)
n = 0
for o in bpy.context.scene.objects:
    if o.type == "MESH" and o.data.shape_keys:
        n += len(o.data.shape_keys.key_blocks) - 1
        o.shape_key_clear()
bpy.ops.export_scene.gltf(filepath=a.out, export_animations=True, export_animation_mode="ACTIONS")
print(f"[unimate_input] {n} morph targets dropped -> {a.out}", flush=True)
