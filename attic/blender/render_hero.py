"""Hero renders for the report: textured, wireframe, and part-coloured views.

Run: blender -b -noaudio --python render_hero.py -- --blend animated.blend --out dir
"""
import argparse
import math
import os
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=900)
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)

PART_RGB = {0: (0.16, 0.34, 0.85), 1: (0.78, 0.52, 0.10), 2: (0.09, 0.62, 0.42), 3: (0.82, 0.25, 0.19)}

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
mesh = next(o for o in sc.objects if o.type == "MESH")
rig = next(o for o in sc.objects if o.type == "ARMATURE")

engines = [i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else engines[0]
sc.render.resolution_x = int(a.res * 0.72)
sc.render.resolution_y = a.res
sc.render.film_transparent = False
sc.render.image_settings.file_format = "PNG"
if sc.render.engine == "BLENDER_EEVEE_NEXT":
    sc.eevee.taa_render_samples = 48

for name, loc, energy in (("key", (3, -4, 4), 1400), ("fill", (-4, -2, 1.5), 520), ("rim", (1, 5, 3), 900)):
    lt = bpy.data.lights.new(name, type="AREA")
    lt.energy, lt.size = energy, 6
    ob = bpy.data.objects.new(name, lt)
    ob.location = loc
    ob.rotation_euler = (mathutils.Vector((0, 0, 0.15)) - mathutils.Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(ob)
w = bpy.data.worlds.new("w")
w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.055, 0.062, 0.078, 1)
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
sc.world = w

cd = bpy.data.cameras.new("cam")
cd.type = "ORTHO"
cd.ortho_scale = 2.25
cam = bpy.data.objects.new("cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam
az = math.radians(28)
cam.location = (4 * math.sin(az), -4 * math.cos(az), 0.25)
cam.rotation_euler = (mathutils.Vector((0, 0, 0.0)) - mathutils.Vector(cam.location)).to_track_quat("-Z", "Y").to_euler()

# rest pose
if rig.animation_data:
    rig.animation_data.action = None
for pb in rig.pose.bones:
    pb.rotation_euler = (0, 0, 0)
bpy.context.view_layer.update()

orig_mats = [m for m in mesh.data.materials]

def render(path):
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)

render(os.path.join(a.out, "hero_textured.png"))

# part-coloured: bake _PARTID into a colour attribute and shade it flat
me = mesh.data
name = next((x for x in ("_PARTID", "part_id") if x in me.attributes), None)
if name:
    import numpy as np
    vals = np.zeros(len(me.vertices), dtype=np.int32)
    me.attributes[name].data.foreach_get("value", vals)
    if "partcol" in me.color_attributes:
        me.color_attributes.remove(me.color_attributes["partcol"])
    ca = me.color_attributes.new(name="partcol", type="FLOAT_COLOR", domain="POINT")
    flat = np.zeros((len(me.vertices), 4), dtype=np.float32)
    for i, v in enumerate(vals):
        r, g, b = PART_RGB.get(int(v), (0.5, 0.5, 0.5))
        flat[i] = (r, g, b, 1.0)
    ca.data.foreach_set("color", flat.ravel())

    mat = bpy.data.materials.new("parts")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.85
    col = nt.nodes.new("ShaderNodeVertexColor")
    col.layer_name = "partcol"
    nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
    me.materials.clear()
    me.materials.append(mat)
    render(os.path.join(a.out, "hero_parts.png"))

# wireframe over flat shading
me.materials.clear()
wmat = bpy.data.materials.new("wire")
wmat.use_nodes = True
wmat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.13, 0.15, 0.19, 1)
me.materials.append(wmat)
mod = mesh.modifiers.new("wf", "WIREFRAME")
mod.thickness = 0.0012
mod.use_replace = False
wire_mat = bpy.data.materials.new("wirecol")
wire_mat.use_nodes = True
e = wire_mat.node_tree.nodes.new("ShaderNodeEmission")
e.inputs["Color"].default_value = (0.35, 0.62, 1.0, 1)
e.inputs["Strength"].default_value = 1.6
wire_mat.node_tree.links.new(e.outputs["Emission"],
                             wire_mat.node_tree.nodes["Material Output"].inputs["Surface"])
me.materials.append(wire_mat)
mod.material_offset = 1
render(os.path.join(a.out, "hero_wire.png"))
print("[hero] done ->", a.out)
