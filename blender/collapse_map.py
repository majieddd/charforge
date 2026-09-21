"""Paint the deformation error onto the character, so it can be seen rather than inferred.

The audit reports that 8.8% of jacket faces fall below half their rest area during the walk.
That is a number with no address. This bakes the same quantity onto the surface as vertex
colour and renders it: cool where a face holds its rest area, hot where it has collapsed or
ballooned. Two copies are posed side by side from the same frame and the same weights - one
deformed with linear blend skinning, one with dual quaternions - so the difference between the
two algorithms is visible in place instead of as a percentage.

Run: blender -b -noaudio --python collapse_map.py -- --blend p4_final.blend --clip walk \
        --frame 13 --out heat.png
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--clip", default="walk")
ap.add_argument("--frame", type=int, default=13)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=900)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next(o for o in scene.objects if o.type == "ARMATURE")
meshes = [o for o in scene.objects if o.type == "MESH" and o.find_armature() == rig]

act = bpy.data.actions.get(a.clip)
rig.animation_data.action = act
if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
    rig.animation_data.action_slot = act.slots[0]
scene.frame_set(a.frame)


def mat_to_dq(T):
    R, t = T[:3, :3], T[:3, 3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
                      (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                      (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    w0, x0, y0, z0 = 0.0, t[0], t[1], t[2]
    w1, x1, y1, z1 = q
    d = 0.5 * np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])
    return q, d


def heat(v):
    """1.0 -> deep blue, 0.5 or 2.0 -> red. Log-symmetric so squash and stretch read alike."""
    e = np.clip(np.abs(np.log2(np.maximum(v, 1e-6))), 0, 1)
    return np.stack([e, 0.25 * (1 - e), 1.0 - e], 1)


made = []
for ob in meshes:
    me = ob.data
    n = len(me.vertices)
    names = [vg.name for vg in ob.vertex_groups]
    R = np.empty(n * 3)
    me.vertices.foreach_get("co", R)
    R = R.reshape(-1, 3)
    W = np.zeros((n, len(names)))
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)

    T = np.zeros((len(names), 4, 4))
    for bi, nm in enumerate(names):
        pb = rig.pose.bones.get(nm)
        T[bi] = np.array(pb.matrix @ pb.bone.matrix_local.inverted()) if pb else np.eye(4)

    Rh = np.concatenate([R, np.ones((n, 1))], 1)
    P_lbs = np.einsum('nb,bij,nj->ni', W, T, Rh)[:, :3]

    q = np.zeros((len(names), 4))
    dq = np.zeros((len(names), 4))
    for bi in range(len(names)):
        q[bi], dq[bi] = mat_to_dq(T[bi])
    piv = q[int(np.argmax(W.sum(0)))]
    sgn = np.where((q @ piv) < 0, -1.0, 1.0)[:, None]
    qb, db = W @ (q * sgn), W @ (dq * sgn)
    L = np.linalg.norm(qb, axis=1, keepdims=True)
    qb, db = qb / np.maximum(L, 1e-12), db / np.maximum(L, 1e-12)
    w, x, y, z = qb.T
    rot = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                   1).reshape(-1, 3, 3)
    tt = 2.0 * np.stack([-db[:, 0] * x + db[:, 1] * w - db[:, 2] * z + db[:, 3] * y,
                         -db[:, 0] * y + db[:, 1] * z + db[:, 2] * w - db[:, 3] * x,
                         -db[:, 0] * z - db[:, 1] * y + db[:, 2] * x + db[:, 3] * w], 1)
    P_dqs = np.einsum('nij,nj->ni', rot, R) + tt

    faces = [list(p.vertices) for p in me.polygons]
    rest_area = np.empty(len(faces))
    me.polygons.foreach_get("area", rest_area)

    for tag, P, dx in (("LBS", P_lbs, -0.58), ("DQS", P_dqs, 0.58)):
        cp = me.copy()
        nob = bpy.data.objects.new(f"{ob.name}_{tag}", cp)
        scene.collection.objects.link(nob)
        cp.vertices.foreach_set("co", (P + np.array([dx, 0, 0])).ravel())
        cp.update()
        for m in list(nob.modifiers):
            nob.modifiers.remove(m)
        area = np.empty(len(cp.polygons))
        cp.polygons.foreach_get("area", area)
        ratio = area / np.maximum(rest_area, 1e-12)
        col = heat(ratio)
        # per-corner colour, so a collapsed face reads as one flat patch
        lay = cp.color_attributes.new(name="heat", type="FLOAT_COLOR", domain="CORNER")
        buf = np.ones((len(cp.loops), 4))
        for fi, p in enumerate(cp.polygons):
            for li in p.loop_indices:
                buf[li, :3] = col[fi]
        lay.data.foreach_set("color", buf.ravel())
        mat = bpy.data.materials.new(f"heat_{tag}_{ob.name}")
        mat.use_nodes = True
        nt = mat.node_tree
        for nd in list(nt.nodes):
            if nd.type != "OUTPUT_MATERIAL":
                nt.nodes.remove(nd)
        out = next(nd for nd in nt.nodes if nd.type == "OUTPUT_MATERIAL")
        att = nt.nodes.new("ShaderNodeVertexColor")
        att.layer_name = "heat"
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(att.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        cp.materials.clear()
        cp.materials.append(mat)
        made.append(nob)
        print(f"[heat] {ob.name} {tag}: faces under half area "
              f"{100*(ratio<0.5).mean():.2f}%, over double {100*(ratio>2).mean():.2f}%", flush=True)

for ob in meshes:
    ob.hide_render = True

# ---- camera and render ------------------------------------------------------------------------
pts = np.vstack([np.array([v.co[:] for v in o.data.vertices]) for o in made])
lo, hi = pts.min(0), pts.max(0)
ctr = Vector(((lo + hi) / 2).tolist())
cam_d = bpy.data.cameras.new("heatcam")
cam = bpy.data.objects.new("heatcam", cam_d)
scene.collection.objects.link(cam)
scene.camera = cam
cam.location = ctr + Vector((0.0, -3.6, 0.10))
cam.rotation_euler = (math.radians(90), 0, 0)
cam_d.lens = 62

scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in \
    {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items} else "BLENDER_EEVEE"
scene.render.film_transparent = False
scene.world = scene.world or bpy.data.worlds.new("w")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1)
scene.view_settings.view_transform = "Standard"
scene.render.resolution_x = a.res * 2
scene.render.resolution_y = a.res
scene.render.filepath = os.path.abspath(a.out)
bpy.ops.render.render(write_still=True)
print(f"[heat] -> {a.out}", flush=True)
