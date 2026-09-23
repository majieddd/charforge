"""Build a stand-in character (body + jacket shell + hair + bag strap) for pipeline testing.

This is NOT a generated asset - it exists so the rigging, segmentation, animation and
verification stages can be developed and tested before the 3D generators finish downloading.
Run: blender -b -noaudio --python make_stub_character.py -- --out stub.glb
"""
import argparse
import sys

import bmesh
import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
parts = []


def add(name, kind="uv_sphere", loc=(0, 0, 0), scale=(1, 1, 1), rot=(0, 0, 0), **kw):
    if kind == "uv_sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=loc, **kw)
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, location=loc, **kw)
    elif kind == "cube":
        bpy.ops.mesh.primitive_cube_add(location=loc, **kw)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    o.rotation_euler = rot
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    parts.append(o)
    return o


# body (A-pose): head, neck, torso, arms angled down-out, legs
add("head", loc=(0, 0, 1.62), scale=(0.115, 0.125, 0.145))
add("neck", "cylinder", loc=(0, 0, 1.47), scale=(0.05, 0.05, 0.06))
add("torso", "cube", loc=(0, 0, 1.15), scale=(0.19, 0.11, 0.30))
add("hips", "cube", loc=(0, 0, 0.87), scale=(0.16, 0.10, 0.10))
for s, sx in (("l", 1), ("r", -1)):
    add(f"upperarm_{s}", "cylinder", loc=(sx * 0.32, 0, 1.22), scale=(0.055, 0.055, 0.19), rot=(0, sx * 0.62, 0))
    add(f"forearm_{s}", "cylinder", loc=(sx * 0.50, 0, 0.94), scale=(0.045, 0.045, 0.17), rot=(0, sx * 0.62, 0))
    add(f"hand_{s}", loc=(sx * 0.60, 0, 0.78), scale=(0.05, 0.03, 0.07))
    add(f"thigh_{s}", "cylinder", loc=(sx * 0.09, 0, 0.60), scale=(0.075, 0.075, 0.22))
    add(f"shin_{s}", "cylinder", loc=(sx * 0.09, 0, 0.22), scale=(0.06, 0.06, 0.20))
    add(f"foot_{s}", "cube", loc=(sx * 0.09, -0.05, 0.03), scale=(0.055, 0.11, 0.03))

# jacket shell: slightly larger than the torso, open at the bottom
add("jacket", "cube", loc=(0, 0, 1.17), scale=(0.215, 0.135, 0.31))
for s, sx in (("l", 1), ("r", -1)):
    add(f"sleeve_{s}", "cylinder", loc=(sx * 0.33, 0, 1.20), scale=(0.07, 0.07, 0.20), rot=(0, sx * 0.62, 0))

# hair: skull cap plus a long tail down the back (the part that must NOT follow the spine)
add("hair_cap", loc=(0, 0.01, 1.66), scale=(0.128, 0.135, 0.135))
add("hair_tail", "cylinder", loc=(0, 0.115, 1.40), scale=(0.075, 0.045, 0.22))

# accessory: a bag strap across the chest (rigid, must not stretch)
add("strap", "cube", loc=(0, -0.115, 1.18), scale=(0.035, 0.02, 0.26), rot=(0, 0.55, 0))

bpy.ops.object.select_all(action="DESELECT")
for o in parts:
    o.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.object.join()
obj = bpy.context.object
obj.name = "character"

# a little shading smoothness + UVs so downstream steps have something to work with
bpy.ops.object.shade_smooth()
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.005)
bpy.ops.object.mode_set(mode="OBJECT")

mat = bpy.data.materials.new("skin")
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.62, 0.48, 0.40, 1)
obj.data.materials.append(mat)

bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=False)
print(f"[stub] {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces -> {a.out}")
