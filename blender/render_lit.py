"""A textured model under studio light: how it looks in a renderer, its albedo lit rather than shown flat.

    blender -b -noaudio --python render_lit.py -- --mesh retopo.glb [--albedo albedo.png] --out prefix \
        [--az 0,180] [--res 700] [--cpu] [--samples 48] [--z 0.88 --span 0.16]

The model's own material (base colour, normal map, roughness and metal) under a key light from the front-left,
a weaker fill from the right, a rim from behind and a grey sky, so that lighting left in the albedo shows as
shading drawn twice and a flat, unlit albedo as an evenly lit surface. --albedo paints another base colour on
(the texture stage's albedo.png over the bake). Orthographic, level, framed on the whole model; the lights turn
with the camera. Writes <prefix>_<az>.png. --cpu renders with Cycles on the CPU, leaving the GPU to a job
running beside it; otherwise Eevee.
"""
import argparse
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--albedo", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--az", default="0,180")
ap.add_argument("--res", type=int, default=700)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--samples", type=int, default=48)
ap.add_argument("--z", type=float, default=None, help="a close-up: its centre, as a fraction of the height from the feet")
ap.add_argument("--span", type=float, default=None, help="a close-up: its height, as a fraction of the model's")
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
sc = bpy.context.scene
for o in sc.objects:
    if o.type == "ARMATURE":
        o.data.pose_position = "REST"
meshes = [o for o in sc.objects if o.type == "MESH" and not any(c.name == "glTF_not_exported" for c in o.users_collection)]
img = bpy.data.images.load(a.albedo) if a.albedo else None
for o in meshes:
    for slot in o.material_slots:
        if img is None or slot.material is None or not slot.material.use_nodes:
            continue
        nt = slot.material.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        link = bsdf.inputs["Base Color"].links
        if link and link[0].from_node.type == "TEX_IMAGE":
            link[0].from_node.image = img
        else:
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = img
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

if a.cpu:
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
else:
    eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
sc.render.resolution_x = sc.render.resolution_y = a.res
world = bpy.data.worlds.new("w")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.32, 0.33, 0.35, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
lights = []
for name, elev, turn, energy in (("key", 40, -35, 3.2), ("fill", 20, 50, 1.0), ("rim", 35, 165, 2.2)):
    lt = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
    sc.collection.objects.link(lt)
    lt.data.energy = energy
    lt.data.angle = math.radians(8)
    lights.append((lt, elev, turn))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
ctr, size = (lo + hi) / 2, max(hi - lo)
cam.data.ortho_scale = size * 1.05
if a.z is not None:
    ctr = Vector((ctr.x, ctr.y, lo.z + a.z * (hi.z - lo.z)))
    cam.data.ortho_scale = (a.span or 0.2) * (hi.z - lo.z)
for az in (float(x) for x in a.az.split(",")):
    r = math.radians(az)
    cam.location = ctr + Vector((math.sin(r), -math.cos(r), 0)) * size * 3
    cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
    for lt, elev, turn in lights:
        # a sun shines along its local -Z: tilt it down from the zenith by (90 - elevation), then turn it
        # round to come from `turn` degrees beside the camera
        lt.rotation_euler = (math.radians(90 - elev), 0, math.radians(az + turn))
    sc.render.filepath = f"{a.out}_{int(az):03d}.png"
    bpy.ops.render.render(write_still=True)
    print(f"[lit] -> {sc.render.filepath}", flush=True)
