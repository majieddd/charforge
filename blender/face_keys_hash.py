"""Hash every shape key of a face rig, so two builds can be compared key by key.

    /Applications/Blender.app/Contents/MacOS/Blender -b -noaudio --python blender/face_keys_hash.py -- \
        --blend work/mara/rig_f.blend --out qa/face_keys/mara_hash.json

Each key block's rest coordinates (metres, rounded to 0.1 micrometre) go through sha1; "Basis" is the
neutral face. Also records the vertex, triangle and shape counts. The hash follows the vertex order, so a
rebuilt mesh with the same geometry but another order does not match: tools/face_keys_diff.py compares
keys by position instead (--npz saves the coordinates it reads). Runs on the CPU only.
"""
import argparse
import hashlib
import json
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--npz", default="", help="also save every key's rest coordinates (for tools/face_keys_diff.py)")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH" and o.find_armature() == rig)
me = mesh.data
n = len(me.vertices)
out = {"n_verts": n, "n_tris": None, "keys": {}}
me.calc_loop_triangles()
out["n_tris"] = len(me.loop_triangles)
saved = {}
for kb in me.shape_keys.key_blocks:
    P = np.empty(n * 3)
    kb.data.foreach_get("co", P)
    out["keys"][kb.name] = hashlib.sha1(np.round(P.reshape(-1, 3), 7).tobytes()).hexdigest()[:16]
    saved[kb.name] = P.reshape(-1, 3)
with open(a.out, "w") as f:
    json.dump(out, f, indent=1)
if a.npz:
    np.savez(a.npz, **{f"key_{i}": v for i, v in enumerate(saved.values())},
             names=np.array(list(saved.keys())))
print(f"[hash] {n:,} verts, {out['n_tris']:,} tris, {len(out['keys'])} keys -> {a.out}", flush=True)
