"""The surface alone, grey, under a low raking light: how the shape reads without its texture (E132, E135).

    blender -b -noaudio --python render_raking.py -- --mesh retopo.glb --out dir/name [--az 0,180] \
        [--z 0.5] [--span 1.05] [--no-normal-map] [--weld] [--res 700]

A light grazing the surface shows every facet, lump, sawtooth edge and baked triangle pattern that a texture
hides; it is how a modeller checks a sculpt. Every material is made the same matte grey; with the normal map
(the default) the render shows the shipped model's shading, without it the low-poly's own triangles. --weld
joins glTF's split vertices and smooth-shades, for a generator's raw output (TRELLIS's GLBs carry per-face
vertices and render faceted otherwise). The camera is orthographic, centred at --z (a fraction of the model's
height from its feet) and --span of its height across; the light comes from 70 degrees round from the camera,
12 degrees above the horizon. Writes <out>_<az>.png per azimuth (0 = the model's front).
"""
import argparse
import math
import sys

import bmesh
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--az", default="0", help="azimuths, comma-separated (0 = front)")
ap.add_argument("--z", type=float, default=0.5, help="the view's centre, as a fraction of the height")
ap.add_argument("--span", type=float, default=1.05, help="the view's height, as a fraction of the model's")
ap.add_argument("--no-normal-map", action="store_true")
ap.add_argument("--weld", action="store_true", help="join split vertices and smooth-shade (raw generator output)")
ap.add_argument("--res", type=int, default=700)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
sc = bpy.context.scene
for o in sc.objects:
    if o.type == "ARMATURE":
        o.data.pose_position = "REST"
ms = [o for o in sc.objects if o.type == "MESH" and not any(c.name == "glTF_not_exported" for c in o.users_collection)]
for o in ms:
    if a.weld:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bm.to_mesh(o.data)
        bm.free()
        if o.data.has_custom_normals:
            bpy.context.view_layer.objects.active = o
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        for p in o.data.polygons:
            p.use_smooth = True
    for slot in o.material_slots:
        if slot.material is None or not slot.material.use_nodes:
            continue
        nt = slot.material.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        for name, value in (("Base Color", (0.8, 0.8, 0.8, 1)), ("Roughness", 0.45), ("Metallic", 0.0)):
            for link in list(bsdf.inputs[name].links):
                nt.links.remove(link)
            bsdf.inputs[name].default_value = value
        if a.no_normal_map:
            for link in list(bsdf.inputs["Normal"].links):
                nt.links.remove(link)

engines = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
sc.render.resolution_x = sc.render.resolution_y = a.res
world = bpy.data.worlds.new("w")
sc.world = world
world.color = (0.02, 0.02, 0.02)
light = bpy.data.objects.new("key", bpy.data.lights.new("key", "SUN"))
sc.collection.objects.link(light)
light.data.energy = 4.0
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
pts = [o.matrix_world @ Vector(c) for o in ms for c in o.bound_box]
lo, hi = min(p.z for p in pts), max(p.z for p in pts)
z = lo + a.z * (hi - lo)
cam.data.type = "ORTHO"
cam.data.ortho_scale = a.span * (hi - lo)
for az in (float(x) for x in a.az.split(",")):
    light.rotation_euler = (math.radians(78), 0, math.radians(az + 70))
    r = math.radians(az)
    cam.location = Vector((math.sin(r) * 3, -math.cos(r) * 3, z))
    cam.rotation_euler = (Vector((0, 0, z)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{a.out}_{int(az):03d}.png"
    bpy.ops.render.render(write_still=True)
    print(f"[raking] -> {sc.render.filepath}", flush=True)
