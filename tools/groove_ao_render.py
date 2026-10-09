"""Close-up of a proxy shaded by the per-vertex occlusion of tools/groove_ao.py (CPU Cycles, emission only).

    blender -b -noaudio --python tools/groove_ao_render.py -- --in proxy.glb --ao ao.npz --out prefix \
        --az 180 --z 0.19 --x 0.09 --y 0.02 --span 0.30 [--res 900]

Each vertex's occlusion is shown as a grey level (white open, black closed), so the picture is the measured
number itself, not a lit guess. Cycles on the CPU with emission only: no GPU, no GPU lock.
Camera: az 0 looks from the front (-Y), 180 from the back, as tools/marks_lit_probe.py.
"""
import argparse
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True)
ap.add_argument("--ao", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--az", type=float, default=180.0)
ap.add_argument("--z", type=float, default=None, help="centre height in Blender metres (default: mesh centre)")
ap.add_argument("--x", type=float, default=None)
ap.add_argument("--y", type=float, default=None)
ap.add_argument("--span", type=float, default=0.30, help="orthographic width in metres")
ap.add_argument("--res", type=int, default=900)
a = ap.parse_args(argv)

D = np.load(a.ao)
ao = D["ao"]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
ob = bpy.context.view_layer.objects.active
me = ob.data
if len(me.vertices) != len(ao):
    raise SystemExit(f"[groove_ao_render] {len(me.vertices)} vertices, AO has {len(ao)}: not the same mesh")
col = me.color_attributes.new(name="ao", type="FLOAT_COLOR", domain="POINT")
for i, v in enumerate(ao):
    col.data[i].color = (float(v), float(v), float(v), 1.0)
mat = bpy.data.materials.new("ao")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
vc = nt.nodes.new("ShaderNodeVertexColor")
vc.layer_name = "ao"
em = nt.nodes.new("ShaderNodeEmission")
outn = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(vc.outputs["Color"], em.inputs["Color"])
nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
me.materials.clear()
me.materials.append(mat)

sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = 16
sc.cycles.use_denoising = False
sc.render.resolution_x = sc.render.resolution_y = a.res
sc.render.image_settings.file_format = "PNG"
sc.view_settings.view_transform = "Standard"

pts = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
ctr = (lo + hi) / 2
ctr = Vector((a.x if a.x is not None else ctr.x, a.y if a.y is not None else ctr.y, a.z if a.z is not None else ctr.z))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"
cam.data.ortho_scale = a.span
r = math.radians(a.az)
cam.location = ctr + Vector((math.sin(r), -math.cos(r), 0)) * 2.0
cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.render.filepath = f"{a.out}_ao_{int(a.az):03d}.png"
bpy.ops.render.render(write_still=True)
print(f"[groove_ao_render] -> {sc.render.filepath}", flush=True)
