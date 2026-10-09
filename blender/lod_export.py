"""Export the level-of-detail meshes of a packaged FBX as GLBs, one per LOD, for measuring (E168 step 4).

package.py writes the FBX with its LOD siblings named <base>_LOD0, _LOD1, _LOD2 (collapse decimation of the rigged
LOD0 at 0.4 and 0.15 of its triangles, no re-bake). The packaged character is in metres, soles on the floor, origin
under the pelvis. The rig frame it was made in is the generator's bounding box normalised to two units, centred
(joints_world's mapping: rig = (x - c) * 2 / H, with c and H from the generator mesh), and the solid lies in the
generator's frame. With --gen-box and --cage-box the LOD meshes are carried back into the solid's frame: metres ->
rig by the one uniform scale and shift that matches the packaged bounding box to the rig box (predicted from the
solid), then rig -> solid by the normalisation. tools/detail_retention.py then reads them beside the solid. Each
mesh is exported at rest with its armature modifier removed and its face morphs dropped (as package.py does).

Run: blender -b -noaudio --python blender/lod_export.py -- --fbx out/<c>/<c>.fbx --out-dir work/<c>/budget/fbx_lods \
         --gen-box=<lo x,y,z,hi x,y,z> --cage-box=<lo x,y,z,hi x,y,z>     (Blender frame; see tools/budget_sweep.py)
Writes lod<i>.glb per LOD and prints one line per LOD: index, triangles, bounding box.
"""
import argparse
import os
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--fbx", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--gen-box", default=None, help="the generator mesh's bounding box, Blender frame")
ap.add_argument("--cage-box", default=None, help="the solid's bounding box, Blender frame")
a = ap.parse_args(argv)
os.makedirs(a.out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=a.fbx)

lods = {}
for o in bpy.context.scene.objects:
    if o.type != "MESH" or "_LOD" not in o.name:
        continue
    lods[int(o.name.rsplit("_LOD", 1)[1])] = o
if not lods:
    raise SystemExit(f"[lod_export] no _LOD meshes in {a.fbx}")


def bbox(points):
    lo = [min(p[k] for p in points) for k in range(3)]
    hi = [max(p[k] for p in points) for k in range(3)]
    return lo, hi


if a.gen_box and a.cage_box:
    g = [float(x) for x in a.gen_box.split(",")]
    cg = [float(x) for x in a.cage_box.split(",")]
    glo, ghi, clo, chi = g[:3], g[3:], cg[:3], cg[3:]
    k = 2.0 / (ghi[2] - glo[2])                       # the rig's normalisation: 2 units tall
    c = Vector([(l + h) / 2 for l, h in zip(glo, ghi)])
    rig_lo = [(x - cc) * k for x, cc in zip(clo, c)]  # the solid's box in the rig frame
    rig_hi = [(x - cc) * k for x, cc in zip(chi, c)]
    for o in lods.values():                           # the face morphs are not measured (package.py drops them too)
        if o.data.shape_keys:
            bpy.context.view_layer.objects.active = o
            o.shape_key_clear()
    base = lods[min(lods)]                            # one mapping for every LOD: LOD0's box
    lo, hi = bbox([base.matrix_world @ v.co for v in base.data.vertices])
    a_s = sum((h - l) / (rh - rl) for l, h, rl, rh in zip(lo, hi, rig_lo, rig_hi)) / 3.0   # metres per rig unit
    cm = Vector([(l + h) / 2 for l, h in zip(lo, hi)])
    cr = Vector([(l + h) / 2 for l, h in zip(rig_lo, rig_hi)])
    b = cm - a_s * cr                                 # metres = a * rig + b
    per_axis = [round((h - l) / (rh - rl), 5) for l, h, rl, rh in zip(lo, hi, rig_lo, rig_hi)]
    print(f"[lod_export] metres per rig unit {a_s:.5f} (per axis {per_axis}), rig -> solid: x / {k:.5f} + generator "
          f"centre", flush=True)
    for o in lods.values():
        mat = o.matrix_world.copy()
        for v in o.data.vertices:
            v.co = ((mat @ v.co - b) / a_s) / k + c
        o.data.update()
        o.matrix_world = Matrix.Identity(4)

for i in sorted(lods):
    o = lods[i]
    for m in list(o.modifiers):                       # the armature would pose the mesh at the import's frame
        o.modifiers.remove(m)
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    tris = sum(len(p.vertices) - 2 for p in o.data.polygons)
    path = os.path.join(a.out_dir, f"lod{i}.glb")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True)
    lo, hi = bbox([o.matrix_world @ v.co for v in o.data.vertices])
    print(f"[lod_export] LOD{i}: {tris:,} triangles, bbox {[round(h - l, 3) for h, l in zip(hi, lo)]} -> {path}",
          flush=True)
