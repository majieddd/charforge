"""Render orbit views of a mesh plus a depth pass, for multi-view label back-projection.

Run:  blender -b -noaudio --python render_views.py -- --mesh in.glb --out dir [--views 8] [--res 768]

Per view:
  view_XX.png    - lit beauty render (input to the human parser)
  depth_XX.exr   - 32-bit camera-space depth (used to decide which vertices are visible)
Plus meta.json with the orthographic camera parameters, so vertices can be projected to
pixels in numpy and tested against the depth buffer. Depth beats an RGB vertex-id pass here:
8-bit PNG output is dithered, which corrupts a 24-bit index packed into colour channels.
"""
import argparse
import json
import math
import os
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--views", type=int, default=8)
ap.add_argument("--res", type=int, default=768)
ap.add_argument("--elev", type=float, default=8.0)
ap.add_argument("--ortho", type=float, default=2.4)
ap.add_argument("--no-normalize", action="store_true")
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise SystemExit(f"unsupported mesh: {path}")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh in file")
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def normalize(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    vs = [v.co for v in obj.data.vertices]
    mn = mathutils.Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
    mx = mathutils.Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
    ctr = (mn + mx) / 2
    h = max(1e-6, mx.z - mn.z)
    s = 2.0 / h
    for v in obj.data.vertices:
        v.co = (v.co - ctr) * s
    obj.data.update()
    return {"scale": s, "center": list(ctr), "height_before": h}


def setup(res, ortho):
    sc = bpy.context.scene
    engines = [i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else engines[0]
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.film_transparent = True
    sc.render.dither_intensity = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_layers[0].use_pass_z = True
    for name, loc, energy in (("key", (3, -4, 4), 900), ("fill", (-4, -2, 1.5), 400), ("rim", (0, 5, 3), 600)):
        lt = bpy.data.lights.new(name, type="AREA")
        lt.energy, lt.size = energy, 6
        ob = bpy.data.objects.new(name, lt)
        ob.location = loc
        ob.rotation_euler = (mathutils.Vector((0, 0, 0.9)) - mathutils.Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(ob)
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    sc.world = w
    cd = bpy.data.cameras.new("cam")
    cd.type = "ORTHO"
    cd.ortho_scale = ortho
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam

    return cam


def place(cam, az_deg, elev_deg, dist=4.0):
    az, el = math.radians(az_deg), math.radians(elev_deg)
    cam.location = (dist * math.cos(el) * math.sin(az), -dist * math.cos(el) * math.cos(az), dist * math.sin(el))
    cam.rotation_euler = (mathutils.Vector((0, 0, 0)) - mathutils.Vector(cam.location)).to_track_quat("-Z", "Y").to_euler()


def visible_projection(obj, cam, res):
    """Which vertices this camera can see, and where they land in pixels.

    Orthographic camera: every ray shares the camera's forward direction, so a vertex is
    visible when a ray cast from just outside it toward the camera hits nothing. Exact,
    and it avoids depending on the compositor API (which moved again in Blender 5.x).
    """
    from mathutils.bvhtree import BVHTree
    deps = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(obj, deps)
    mw = obj.matrix_world
    inv = cam.matrix_world.inverted()
    fwd = (cam.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))).normalized()
    to_cam = -fwd
    s = cam.data.ortho_scale
    # offset must clear the vertex's own triangles: on a 170k-vertex mesh normalised to
    # height 2.0 the mean edge is ~0.005, so 1e-4 re-hits the originating face and the
    # vertex is wrongly reported as occluded.
    import statistics
    me = obj.data
    if len(me.edges):
        sample = [me.edges[i] for i in range(0, len(me.edges), max(1, len(me.edges) // 500))]
        mean_edge = statistics.fmean(
            (me.vertices[e.vertices[0]].co - me.vertices[e.vertices[1]].co).length for e in sample)
    else:
        mean_edge = 0.01
    eps = max(1e-4, mean_edge * 0.75)
    idx, px, py = [], [], []
    for v in obj.data.vertices:
        co = mw @ v.co
        n = (mw.to_3x3() @ v.normal).normalized()
        if n.dot(to_cam) <= 0.05:          # back-facing
            continue
        hit = bvh.ray_cast(co + to_cam * eps + n * eps, to_cam, 20.0)
        if hit[0] is not None:             # something occludes it
            continue
        c = inv @ co
        x = (c.x / (s / 2)) * 0.5 + 0.5
        y = (c.y / (s / 2)) * 0.5 + 0.5
        if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0):
            continue
        idx.append(v.index)
        px.append(int(x * res))
        py.append(int((1.0 - y) * res))
    return idx, px, py


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = import_mesh(a.mesh)
    norm = None if a.no_normalize else normalize(obj)
    cam = setup(a.res, a.ortho)
    sc = bpy.context.scene
    if sc.render.engine == "BLENDER_EEVEE_NEXT":
        sc.eevee.taa_render_samples = 16

    views, proj = [], {}
    for i in range(a.views):
        az = 360.0 * i / a.views
        place(cam, az, a.elev)
        sc.render.filepath = os.path.join(a.out, f"view_{i:02d}.png")
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        idx, px, py = visible_projection(obj, cam, a.res)
        proj[f"idx_{i:02d}"] = idx
        proj[f"px_{i:02d}"] = px
        proj[f"py_{i:02d}"] = py
        views.append({"index": i, "azimuth": az, "elevation": a.elev,
                      "cam_location": list(cam.location), "ortho_scale": cam.data.ortho_scale,
                      "res": a.res, "n_visible": len(idx)})
        print(f"  view {i}: {len(idx)} visible vertices", flush=True)

    json.dump(proj, open(os.path.join(a.out, "projection.json"), "w"))
    json.dump({"mesh": os.path.abspath(a.mesh), "views": views,
               "n_vertices": len(obj.data.vertices), "n_polygons": len(obj.data.polygons),
               "normalize": norm, "res": a.res, "engine": sc.render.engine},
              open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"[render_views] {a.views} views, {len(obj.data.vertices)} verts -> {a.out}")


main()
