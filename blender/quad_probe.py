"""Can Blender's QuadriFlow remesh the clean solid? (lane topology, E163)

    blender -b -noaudio --python blender/quad_probe.py -- --mesh work/<name>/solid_hands.glb \
        [--sizes 0.012,0.008,0.006,0.003] [--faces 25000]
    blender -b -noaudio --python blender/quad_probe.py -- --control ico       (controls: icosphere, accepted)
    blender -b -noaudio --python blender/quad_probe.py -- --control voxel_ico (its voxel remesh at 0.03: refused)

For the solid as it is, and for a voxel remesh of it at each size (in the solid's own units: about 0.98 for a
1.75 m character), prints the faces, the folded quads and edges that turn over 120 degrees (quad_remesh.
quadriflow_check), then runs bpy.ops.object.quadriflow_remesh and reports whether it accepted the mesh (the
face count changes) or refused it (unchanged, with QuadriFlow's manifold warning in the log).
"""
import argparse
import sys
import os

import bmesh
import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quad_remesh  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", default="")
ap.add_argument("--control", default="", help="ico: an icosphere (subdivisions 5); voxel_ico: that sphere voxel-remeshed at 0.03")
ap.add_argument("--sizes", default="", help="comma list of voxel sizes; empty: the solid as it is only")
ap.add_argument("--faces", type=int, default=25000, help="QuadriFlow's target face count")
a = ap.parse_args(argv)


def fresh_solid():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=a.mesh)
    obj = max([o for o in bpy.context.scene.objects if o.type == "MESH"], key=lambda o: len(o.data.polygons))
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


def report(obj, tag):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    chk = quad_remesh.quadriflow_check(bm)
    faces = len(bm.faces)
    bm.free()
    before = len(obj.data.polygons)
    try:
        bpy.ops.object.quadriflow_remesh(target_faces=a.faces, use_mesh_symmetry=False, use_preserve_sharp=False,
                                         use_preserve_boundary=False, smooth_normals=False)
    except RuntimeError as e:
        print(f"[probe] {tag}: quadriflow raised {e}", flush=True)
    after = len(obj.data.polygons)
    verdict = "accepted" if after != before else "refused (unchanged)"
    print(f"[probe] {tag}: faces {faces}, {chk} -> QuadriFlow {verdict} ({after} faces)", flush=True)


if a.control:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=1.0)
    obj = bpy.context.active_object
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    if a.control == "voxel_ico":
        obj.data.remesh_voxel_size = 0.03
        obj.data.remesh_voxel_adaptivity = 0.0
        bpy.ops.object.voxel_remesh()
    report(obj, f"control {a.control}")
    raise SystemExit(0)
obj = fresh_solid()
report(obj, "solid as it is")
for size in filter(None, a.sizes.split(",")):
    obj = fresh_solid()
    obj.data.remesh_voxel_size = float(size)
    obj.data.remesh_voxel_adaptivity = 0.0
    bpy.ops.object.voxel_remesh()
    report(obj, f"voxel {size}")
