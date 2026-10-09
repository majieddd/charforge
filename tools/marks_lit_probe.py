"""Which map darkens a lit render: render_lit.py with one input switched off at a time.

    blender -b -noaudio --python tools/marks_lit_probe.py -- --mesh retopo.glb --albedo albedo.png \
        --out prefix --az 180 --variants full,nonormal,flat [--res 1200] [--cpu] [--samples 32] \
        [--z 0.80 --span 0.30 --x 0.0]

Variants (each writes <prefix>_<variant>_<az>.png, same camera and lights as render_lit.py):
  full      the material as render_lit.py has it (base colour from --albedo, normal map, roughness, metal)
  nonormal  the normal map unlinked from the Principled BSDF: shading from the geometry alone
  flat      a constant 0.5 grey base colour: what the geometry and normal map do without the albedo
Each variant is a fresh import, so no variant sees another's edits.
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
ap.add_argument("--az", default="180")
ap.add_argument("--variants", default="full,nonormal,flat")
ap.add_argument("--res", type=int, default=1200)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--z", type=float, default=None)
ap.add_argument("--span", type=float, default=None)
ap.add_argument("--x", type=float, default=0.0, help="close-up centre along X, in model units (Blender frame)")
a = ap.parse_args(argv)


def build(variant):
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
            if slot.material is None or not slot.material.use_nodes:
                continue
            nt = slot.material.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            if variant == "nonormal":
                for lk in list(bsdf.inputs["Normal"].links):
                    nt.links.remove(lk)
            if variant == "flat":
                for lk in list(bsdf.inputs["Base Color"].links):
                    nt.links.remove(lk)
                bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
            elif img is not None:
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
        ctr = Vector((a.x, ctr.y, lo.z + a.z * (hi.z - lo.z)))
        cam.data.ortho_scale = (a.span or 0.2) * (hi.z - lo.z)
    return sc, cam, lights, ctr, size


for variant in a.variants.split(","):
    for az in (float(x) for x in a.az.split(",")):
        sc, cam, lights, ctr, size = build(variant)
        r = math.radians(az)
        cam.location = ctr + Vector((math.sin(r), -math.cos(r), 0)) * size * 3
        cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
        for lt, elev, turn in lights:
            lt.rotation_euler = (math.radians(90 - elev), 0, math.radians(az + turn))
        sc.render.filepath = f"{a.out}_{variant}_{int(az):03d}.png"
        bpy.ops.render.render(write_still=True)
        print(f"[lit_probe] {variant} -> {sc.render.filepath}", flush=True)
