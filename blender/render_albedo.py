"""A textured mesh unlit - its albedo exactly as painted - from the front and the back (and sides).

    blender -b -noaudio --python render_albedo.py -- --mesh retopo.glb --albedo albedo.png --out prefix \
        [--az 0,180] [--res 900] [--cpu]

Writes <prefix>_<az>.png. Orthographic, level, framed on the whole model, emission shading: what differs
between two pictures is the texture, not the light.
"""
import argparse
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--albedo", default=None, help="an image to paint on instead of the file's own base colour")
ap.add_argument("--out", required=True)
ap.add_argument("--az", default="0,180")
ap.add_argument("--res", type=int, default=900)
ap.add_argument("--cpu", action="store_true", help="Cycles on the CPU, leaving the GPU to a job running beside it")
a = ap.parse_args(argv)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
sc = bpy.context.scene
meshes = [o for o in sc.objects if o.type == "MESH" and not any(c.name == "glTF_not_exported" for c in o.users_collection)]
img = bpy.data.images.load(a.albedo) if a.albedo else None
for o in meshes:
    for slot in o.material_slots:
        m = slot.material
        nt = m.node_tree
        tex = img and nt.nodes.new("ShaderNodeTexImage")
        if tex:
            tex.image = img
            col = tex.outputs["Color"]
        else:
            bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
            link = bsdf.inputs["Base Color"].links
            col = link[0].from_socket if link else None
        em = nt.nodes.new("ShaderNodeEmission")
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        if col is not None:
            nt.links.new(col, em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
if a.cpu:
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 8                                    # emission only: the samples are for the edges
sc.view_settings.view_transform = "Standard"                 # the albedo's own colours, no filmic curve
sc.render.resolution_x = sc.render.resolution_y = a.res
w = bpy.data.worlds.new("w"); sc.world = w; w.color = (1, 1, 1)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = "ORTHO"
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
ctr, size = (lo + hi) / 2, max(hi - lo)
cam.data.ortho_scale = size * 1.05
for az in (float(x) for x in a.az.split(",")):
    r = math.radians(az)
    cam.location = ctr + Vector((math.sin(r), -math.cos(r), 0)) * size * 3
    cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{a.out}_{int(az):03d}.png"
    bpy.ops.render.render(write_still=True)
print("[albedo] rendered", a.out, flush=True)
