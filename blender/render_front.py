"""The textured mesh from the texture stage's front camera, at any resolution: the same frame as
blender/uv_maps.py's view_000 (its views.json normalisation and orthographic camera), so a pixel here is a pixel
there times the resolution ratio (E140: the face's landmarks on the model).

    blender -b -noaudio --python render_front.py -- --mesh retopo.glb --albedo albedo_clean.png --views texproj/views.json \
        --out front.png [--res 4096] [--lit | --grey] [--frame face]
"""
import argparse
import json
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--albedo", default=None)
ap.add_argument("--views", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=4096)
ap.add_argument("--lit", action="store_true", help="soft studio light on the material (else the colour alone)")
ap.add_argument("--grey", action="store_true", help="the shape alone: a plain grey material, lit (implies --lit)")
ap.add_argument("--frame", choices=("front", "face"), default="front",
                help="face: views.json's face_frame, the head close up (uv_maps.py --face-joints)")
a = ap.parse_args(argv)
a.lit = a.lit or a.grey
V = json.load(open(a.views))
v0 = next(v for v in V["views"] if v["tag"] == "000")
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
sc = bpy.context.scene
meshes = [o for o in sc.objects if o.type == "MESH"]
ctr, s = Vector(V["center"]), float(V["scale"])
img = bpy.data.images.load(a.albedo) if a.albedo else None
for o in meshes:
    o.data.transform(o.matrix_world)
    o.matrix_world.identity()
    for vt in o.data.vertices:
        vt.co = (vt.co - ctr) * s
    for slot in o.material_slots:
        nt = slot.material.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        lk = bsdf.inputs["Base Color"].links
        if a.grey:
            for l in list(lk):
                nt.links.remove(l)
            bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1)
            for name in ("Normal",):
                for l in list(bsdf.inputs[name].links):
                    nt.links.remove(l)
            continue
        col = None
        if img is not None:
            if lk and lk[0].from_node.type == "TEX_IMAGE":
                lk[0].from_node.image = img
                col = lk[0].from_node.outputs["Color"]
        elif lk:
            col = lk[0].from_socket
        if not a.lit and col is not None:
            em = nt.nodes.new("ShaderNodeEmission")
            nt.links.new(col, em.inputs["Color"])
            out_n = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
            nt.links.new(em.outputs["Emission"], out_n.inputs["Surface"])
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = 16 if a.lit else 4
sc.render.resolution_x = sc.render.resolution_y = a.res
sc.render.film_transparent = True
sc.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("w")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.5, 0.5, 0.52, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8 if a.lit else 0.0
if a.lit:
    for name, rot, e in (("key", (55, 0, -25), 2.2), ("fill", (70, 0, 40), 1.0)):
        lt = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        sc.collection.objects.link(lt)
        lt.data.energy = e
        lt.data.angle = 0.4
        lt.rotation_euler = tuple(math.radians(x) for x in rot)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"
if a.frame == "face":
    ff = V["face_frame"]
    cam.data.ortho_scale = float(ff["ortho_scale"])
    tgt = Vector(ff["target"])
    cam.location = tgt + Vector((0, -4.0, 0))
else:
    cam.data.ortho_scale = float(v0["ortho_scale"])
    tgt = Vector(v0.get("target", (0, 0, 0)))
    cam.location = Vector(v0["cam_location"])
cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.render.filepath = a.out
bpy.ops.render.render(write_still=True)
print(f"[front] {a.res} px -> {a.out}", flush=True)
