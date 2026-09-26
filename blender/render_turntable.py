"""A mesh's silhouette from every side - the model half of tools/turntable_fidelity.py.

    blender -b -noaudio --python render_turntable.py -- --mesh mesh.glb --out dir [--step 5] [--res 384 576]

The mesh is imported and normalised as blender/render_views.py does it (2 units tall, centred), then
seen by a level orthographic camera every --step degrees, azimuth 0 in front (the camera on -Y
looking +Y) and rising to the character's left. Each view is a flat alpha silhouette (Workbench, no
lighting): {azimuth:03d}.png. A .blend is opened instead of imported, its armature left in the rest
pose.
"""
import argparse
import math
import os
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--step", type=float, default=5.0)
ap.add_argument("--res", type=int, nargs=2, default=(384, 576))
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)

if a.mesh.endswith(".blend"):
    bpy.ops.wm.open_mainfile(filepath=a.mesh)
    for o in bpy.context.scene.objects:
        if o.type == "ARMATURE":
            o.data.pose_position = "REST"
    for o in list(bpy.context.scene.objects):
        if o.type not in ("MESH", "ARMATURE"):
            bpy.data.objects.remove(o, do_unlink=True)
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in bpy.context.scene.objects:
        if o.type == "MESH" and o.visible_get():
            ev = o.evaluated_get(dg)
            m = ev.to_mesh()
            pts += [o.matrix_world @ v.co for v in m.vertices]
            ev.to_mesh_clear()
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=a.mesh)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    pts = [v.co.copy() for v in ob.data.vertices]
lo = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
ctr = (lo + hi) / 2
h = hi.z - lo.z

sc = bpy.context.scene
eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_WORKBENCH" if "BLENDER_WORKBENCH" in eng else sc.render.engine
sc.display.shading.light = "FLAT"
sc.display.shading.color_type = "SINGLE"
sc.render.resolution_x, sc.render.resolution_y = a.res
sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
cd = bpy.data.cameras.new("tt")
cd.type = "ORTHO"
cd.sensor_fit = "VERTICAL"
cd.ortho_scale = 1.1 * h
cam = bpy.data.objects.new("tt", cd)
sc.collection.objects.link(cam)
sc.camera = cam
n = int(round(360 / a.step))
for k in range(n):
    az = math.radians(k * a.step)
    cam.location = ctr + mathutils.Vector((6 * math.sin(az), -6 * math.cos(az), 0))
    cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = os.path.join(a.out, f"{int(round(k * a.step)):03d}.png")
    bpy.ops.render.render(write_still=True)
print(f"[turntable] {n} silhouettes of {os.path.basename(a.mesh)} -> {a.out}", flush=True)
