"""Put a rigged character into the frame every game engine assumes: metres, feet on the floor.

Until this stage existed, every character left the pipeline exactly 2.00 units tall with its
origin at the hips. Both are artefacts of the skeleton estimator working in a normalised
[-1, 1] box, and both break on import: an engine reads 1 unit as 1 metre, so the character
arrived as a two-metre giant, and it places an asset's origin on the floor, so it arrived
buried to the waist. The playground only looked right because it measured the bounding box at
load and pushed the model up - a patch over the asset rather than a property of it.

This runs straight after rigging, while the character has a skeleton and skin weights but no
animation. Doing it here rather than at export means no animation curve ever has to be
rescaled: every later stage - the T-pose, retargeting, smoothing, export - simply works in
metres from the start. Retargeting already sizes the motion from world-space leg length, so
the clips pick up the new scale on their own.

The transform is applied to the data, not the objects: mesh vertices and bone heads and tails
move together, object matrices stay identity, and the rest pose is verified to still deform to
exactly itself afterwards.

Run: blender -b -noaudio --python normalize_frame.py -- --blend rig.blend --out rig_m.blend
     [--height 1.75]
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--height", type=float, default=1.75,
                help="standing height in metres, crown to sole")
ap.add_argument("--json", default=None)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
meshes = [o for o in scene.objects if o.type == "MESH" and o.find_armature() == rig]
if rig is None or not meshes:
    raise SystemExit("[frame] need an armature and at least one mesh bound to it")
if bpy.data.actions:
    raise SystemExit("[frame] this stage must run before any animation exists; "
                     f"found {len(bpy.data.actions)} action(s)")
for o in [rig] + meshes:
    if not np.allclose(np.array(o.matrix_world), np.eye(4), atol=1e-6):
        raise SystemExit(f"[frame] {o.name} has a non-identity transform; apply it first")

# ---- measure the standing character at rest ----------------------------------------------------
V = np.vstack([np.array([v.co[:] for v in o.data.vertices]) for o in meshes])
lo, hi = V.min(0), V.max(0)
h0 = float(hi[2] - lo[2])
k = a.height / h0
# origin on the floor, centred between the feet in plan
feet = V[V[:, 2] < lo[2] + 0.05 * h0]
cx, cy = float(np.median(feet[:, 0])), float(np.median(feet[:, 1]))
origin = np.array([cx, cy, float(lo[2])])
print(f"[frame] rest height {h0:.3f} units, origin was {float(-lo[2]):.3f} above the soles; "
      f"scaling x{k:.4f} to {a.height:.2f} m and dropping the soles to z=0", flush=True)


def xf(p):
    return (np.asarray(p, dtype=np.float64) - origin) * k


# ---- meshes ----------------------------------------------------------------------------------
for o in meshes:
    me = o.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = xf(co.reshape(-1, 3))
    me.vertices.foreach_set("co", co.reshape(-1))
    me.update()

# ---- skeleton --------------------------------------------------------------------------------
bpy.ops.object.select_all(action="DESELECT")
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
for eb in rig.data.edit_bones:
    roll = eb.roll
    eb.head = Vector(xf(eb.head))
    eb.tail = Vector(xf(eb.tail))
    eb.roll = roll                                   # uniform scale + translation keeps roll
bpy.ops.object.mode_set(mode="OBJECT")

# ---- verify ----------------------------------------------------------------------------------
# At rest the armature must deform every vertex to exactly where the data says it is; if the
# bones and the mesh were moved inconsistently, this is where it shows.
for pb in rig.pose.bones:
    pb.matrix_basis.identity()
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
worst = 0.0
for o in meshes:
    ev = o.evaluated_get(dg)
    m2 = ev.to_mesh()
    P = np.empty(len(m2.vertices) * 3)
    m2.vertices.foreach_get("co", P)
    R = np.empty(len(o.data.vertices) * 3)
    o.data.vertices.foreach_get("co", R)
    worst = max(worst, float(np.abs(P - R).max()))
    ev.to_mesh_clear()
V2 = np.vstack([np.array([v.co[:] for v in o.data.vertices]) for o in meshes])
lo2, hi2 = V2.min(0), V2.max(0)
pelvis = rig.data.bones[0]
print(f"[frame] now {float(hi2[2]-lo2[2]):.3f} m tall, soles at z={float(lo2[2]):+.4f}, "
      f"root bone '{pelvis.name}' at {float(pelvis.head_local.z):.3f} m; "
      f"rest-pose deformation error {worst*1000:.3f} mm", flush=True)
if worst > 1e-4:
    raise SystemExit(f"[frame] rest pose no longer deforms to itself ({worst*1000:.2f} mm off)")
if abs(float(lo2[2])) > 1e-4 or abs(float(hi2[2] - lo2[2]) - a.height) > 1e-3:
    raise SystemExit("[frame] result is not the requested height with soles on the floor")

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
if a.json:
    json.dump({"height_m": a.height, "scale": k, "rest_height_units": h0,
               "origin_offset": origin.tolist()}, open(a.json, "w"), indent=2)
print(f"[frame] -> {a.out}", flush=True)
