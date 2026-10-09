"""Count faces whose UVs overlap another face's (folded or doubled islands), with Blender's own test.

    blender -b --python tools/marks_uv_overlap.py -- --mesh work/rowan/retopo.glb

Prints the number of faces Blender's select_overlap flags. Earlier garment blotches on this project were
folded islands (charforge-defect-tracing): a non-zero count with the flagged faces on the defect is the cue.
"""
import sys
import argparse
import bpy
import bmesh

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
a = ap.parse_args(argv)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
ob = next(o for o in bpy.context.scene.objects if o.type == "MESH")
bpy.context.view_layer.objects.active = ob
ob.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bm = bmesh.from_edit_mesh(ob.data)
bpy.ops.mesh.select_all(action="SELECT")
bm.faces.ensure_lookup_table()
bpy.context.scene.tool_settings.use_uv_select_sync = False
bpy.ops.uv.select_all(action="SELECT")
bpy.ops.uv.select_overlap()
bm = bmesh.from_edit_mesh(ob.data)
flag = [f.index for f in bm.faces if getattr(f, "uv_select", False)]
print(f"[uv_overlap] {a.mesh}: {len(bm.faces)} faces, {len(flag)} flagged by select_overlap", flush=True)
if flag:
    cs = [bm.faces[i].calc_center_median() for i in flag[:4000]]
    xs = [c.x for c in cs]; ys = [c.y for c in cs]; zs = [c.z for c in cs]
    print(f"[uv_overlap] flagged faces span x[{min(xs):+.3f},{max(xs):+.3f}] y[{min(ys):+.3f},{max(ys):+.3f}] z[{min(zs):+.3f},{max(zs):+.3f}] (Blender frame)", flush=True)
bpy.ops.object.mode_set(mode="OBJECT")
