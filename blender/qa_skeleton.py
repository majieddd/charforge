"""QA render: the skeleton over an x-ray of the mesh, front and side, for a person or a model to check.

    blender -b --python qa_skeleton.py -- --mesh solid.glb --joints joints_refined.json \
        --frame-from sdf.npz --out qa_skeleton.png [--compare joints.json]

Joints are drawn as spheres (refined: orange; --compare: blue), traced limb centre lines as
thin tubes. The mesh is rendered as a translucent x-ray so a joint outside the body is obvious.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--joints", required=True)
ap.add_argument("--frame-from", required=True, help="sdf.npz whose points define the joints' frame")
ap.add_argument("--compare", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=900)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
body = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
P = np.load(a.frame_from)["points"]
lo, hi = P.min(0), P.max(0)
kk = 2.0 / float(hi[2] - lo[2])
cc = (lo + hi) / 2
H = float(hi[2] - lo[2])


def mat(name, rgba, alpha=1.0, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = rgba
    b.inputs["Alpha"].default_value = alpha
    if emit:
        b.inputs["Emission Color"].default_value = rgba
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1:
        m.surface_render_method = "BLENDED"
    return m


body.data.materials.clear()
body.data.materials.append(mat("xray", (0.75, 0.78, 0.85, 1), alpha=0.28))


def spheres(joints, colour, r):
    m = mat(f"j{colour}", colour + (1,), emit=2.0)
    for n, p in joints.items():
        q = np.array(p) / kk + cc
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=q.tolist(), segments=12, ring_count=8)
        o = bpy.context.active_object
        o.data.materials.append(m)


def polyline(pts, colour, r):
    cu = bpy.data.curves.new("tr", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = r
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        q = np.array(p) / kk + cc
        sp.points[i].co = (*q, 1)
    o = bpy.data.objects.new("tr", cu)
    bpy.context.scene.collection.objects.link(o)
    cu.materials.append(mat("t", colour + (1,), emit=1.5))


Jr = json.load(open(a.joints))
spheres(Jr["joints"], (1.0, 0.45, 0.05), 0.008 * H)
for k, t in Jr.get("traces", {}).items():
    polyline(t, (1.0, 0.85, 0.2), 0.002 * H)
if a.compare:
    spheres(json.load(open(a.compare))["joints"], (0.1, 0.4, 1.0), 0.006 * H)

sc = bpy.context.scene
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
w = bpy.data.worlds.new("w")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.08, 0.08, 0.09, 1)
L = bpy.data.objects.new("key", bpy.data.lights.new("key", "SUN"))
sc.collection.objects.link(L)
L.data.energy = 3
L.rotation_euler = (math.radians(40), 0, math.radians(20))
eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
ctr = Vector(((lo + hi) / 2).tolist())
tmp = []
for name, d in (("front", (0, -1, 0)), ("side", (1, 0, 0))):
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = H * 1.08
    cam.location = ctr + Vector(d) * 4
    cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.resolution_x, sc.render.resolution_y = int(a.res * 0.62), a.res
    f = a.out.replace(".png", f"_{name}.png")
    sc.render.filepath = f
    bpy.ops.render.render(write_still=True)
    tmp.append(f)
print(f"[qa] -> {', '.join(tmp)}", flush=True)
