"""Raking-light close-ups of a low-poly GLB, for judging a level of detail by eye (E168 step 4).

A grazing sun from the side shows a normal map's detail (or its absence) as the eye sees it in a game: the
faces, the hem and collar edges, the fingers. Cycles on the CPU, so it takes no GPU lock. The camera is orthographic
and looks along +Y from the front (Blender's -Y is the front), at the head, the chest or the whole body. The base
colour is replaced by one grey, so only the shape and the normal map show.

Run: blender -b -noaudio --python blender/lod_render.py -- --glb <lod.glb> --out <png> --view face|torso|body
         [--res 384] [--samples 64]
"""
import argparse
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--glb", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--view", choices=("face", "torso", "body"), default="face")
ap.add_argument("--res", type=int, default=384)
ap.add_argument("--samples", type=int, default=64)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.glb)
# The colour is taken out: the comparison is of shape and normal detail, and the bakes made here carry the
# generator's unfiltered albedo (its black specks show as speckles on a skin). Every variant gets the same grey.
for mat in bpy.data.materials:
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if mat.use_nodes else None
    if bsdf is None:
        continue
    for link in list(bsdf.inputs["Base Color"].links):
        mat.node_tree.links.remove(link)
    bsdf.inputs["Base Color"].default_value = (0.72, 0.68, 0.64, 1.0)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
pts = [o.matrix_world @ v.co for o in meshes for v in o.data.vertices]
lo = Vector([min(p[k] for p in pts) for k in range(3)])
hi = Vector([max(p[k] for p in pts) for k in range(3)])
H = hi.z - lo.z
cx = (lo.x + hi.x) / 2
front_y = lo.y                                       # the front-most surface (the nose and the belly)
target, scale = {
    "face": (Vector((cx, front_y, hi.z - 0.10 * H)), 0.30 * H),
    "torso": (Vector((cx, front_y, lo.z + 0.62 * H)), 0.50 * H),
    "body": (Vector((cx, front_y, (lo.z + hi.z) / 2)), 1.05 * H),
}[a.view]

cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = scale
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.scene.collection.objects.link(cam)
cam.location = target + Vector((0.0, -2.0 * H, 0.0))
cam.rotation_euler = (math.radians(90), 0.0, 0.0)    # looks along +Y
bpy.context.scene.camera = cam

sun_data = bpy.data.lights.new("raking", "SUN")
sun_data.energy = 3.0
sun = bpy.data.objects.new("raking", sun_data)
bpy.context.scene.collection.objects.link(sun)
direction = Vector((-1.0, 0.25, -0.12)).normalized()  # light travels from the +X side, nearly across the surface
sun.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

world = bpy.data.worlds.new("world")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.22, 0.22, 0.22, 1.0)
bg.inputs[1].default_value = 1.0
bpy.context.scene.world = world

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = a.samples
scene.cycles.use_denoising = False
scene.render.resolution_x = scene.render.resolution_y = a.res
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = a.out
bpy.ops.render.render(write_still=True)
print(f"[lod_render] {a.view} -> {a.out}", flush=True)
