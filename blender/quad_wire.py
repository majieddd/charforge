"""Close-up of a game mesh's edge flow: the grey surface with its polygon edges drawn over it (lane topology).

    blender -b -noaudio --python blender/quad_wire.py -- --mesh work/<name>/retopo_polys.obj --out dir/name \
        [--target 0.5,0.9,0.5] [--span 0.35] [--az 0] [--res 900]

--target is the point to look at as fractions of the mesh's bounding box (x across, y depth, z height), --span
the view's width as a fraction of the height. Quads and triangles both show: their edges are the mesh's own
(an .obj keeps the quads; a .glb shows the triangle pairs the exporter made). Writes <out>_<az>.png.
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--target", default="0.5,0.5,0.5")
ap.add_argument("--span", type=float, default=0.35)
ap.add_argument("--az", type=float, default=0.0)
ap.add_argument("--res", type=int, default=900)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
ext = os.path.splitext(a.mesh)[1].lower()
if ext == ".obj":
    bpy.ops.wm.obj_import(filepath=a.mesh)
else:
    # a glTF renders black here (its split normals take the material's shading); go through a triangulated
    # .obj, which is what the same mesh looks like as a polygon file
    bpy.ops.import_scene.gltf(filepath=a.mesh)
    tmp_obj = os.path.splitext(a.out)[0] + "_tmp_triangulated.obj"
    os.makedirs(os.path.dirname(os.path.abspath(tmp_obj)), exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.wm.obj_export(filepath=tmp_obj, export_selected_objects=True, export_triangulated_mesh=True,
                          export_uv=False, export_normals=False, export_materials=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=tmp_obj)
    os.remove(tmp_obj)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
obj = max(meshes, key=lambda o: len(o.data.polygons))
for o in meshes:
    if o is not obj:
        bpy.data.objects.remove(o, do_unlink=True)
# the surface in matte grey, its polygon edges drawn over it by a wireframe modifier
mat = bpy.data.materials.new("matte")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (0.62, 0.62, 0.64, 1.0)
bsdf.inputs["Roughness"].default_value = 0.8
obj.data.materials.clear()
obj.data.materials.append(mat)
wf = obj.modifiers.new("edges", "WIREFRAME")
wf.thickness = 0.0009
wf.use_relative_offset = False
wf.offset = 0.0
wf.use_even_offset = True
wf.material_offset = 0
bpy.context.view_layer.update()

# world space: a glTF import carries a rotation (Y-up to Z-up) that the camera must see too
world = [obj.matrix_world @ v.co for v in obj.data.vertices]
lo = Vector((min(p.x for p in world), min(p.y for p in world), min(p.z for p in world)))
hi = Vector((max(p.x for p in world), max(p.y for p in world), max(p.z for p in world)))
print("[wire] bounds", tuple(round(x, 3) for x in lo), tuple(round(x, 3) for x in hi), flush=True)
size = hi - lo
H = max(size.x, size.y, size.z)
fx, fy, fz = (float(x) for x in a.target.split(","))
tgt = lo + Vector((fx * size.x, fy * size.y, fz * size.z))

sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items} else "BLENDER_EEVEE"
sc.render.resolution_x = sc.render.resolution_y = a.res
sc.render.film_transparent = False
sc.world = bpy.data.worlds.new("w")
sc.world.color = (0.12, 0.12, 0.13)
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = a.span * H
cam = bpy.data.objects.new("cam", cam_data)
sc.collection.objects.link(cam)
dist = 2.0 * H
th = math.radians(a.az)
cam.location = tgt + Vector((math.sin(th) * dist, -math.cos(th) * dist, 0.0))
cam.rotation_euler = (math.radians(90), 0.0, th)
sc.camera = cam
for pos, energy in (((2.0, -3.0, 3.0), 3.0), ((-3.0, -1.0, 2.0), 1.2)):
    ld = bpy.data.lights.new("sun", "SUN")
    ld.energy = energy
    lo_ = bpy.data.objects.new("sun", ld)
    lo_.location = Vector(pos) * H
    lo_.rotation_euler = (math.radians(60), 0.0, math.radians(30))
    sc.collection.objects.link(lo_)
sc.render.filepath = f"{a.out}_{int(a.az):03d}.png"
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.render.render(write_still=True)
print("[wire] wrote", sc.render.filepath, flush=True)
