"""Render a character at chosen moments of a clip, from chosen sides - for looking at a problem, and
for showing it fixed.

    blender -b -noaudio --python render_poses.py -- --blend final.blend --clip victory_cheer \
        --times 1.6,2.2 --az 0,60,180 --out renders/ [--res 640] [--frame upper|full] \
        [--weights left_collar,left_shoulder,spine3,spine2] [--rest]
    blender -b -noaudio --python render_poses.py -- --blend final.blend --shots idle@0.5,jump@1.1,wave@0.6 \
        --az 30 --out renders/          (several clips in one run)

Writes <out>/<clip>_<time>s_<az>.png for every time and side (az: degrees round from the front; 90
is the character's left). --frame upper frames the head and torso, where most skinning faults show.
--weights paints each vertex by the listed bones' weights (in the order red, green, blue, yellow,
magenta, cyan; every other bone grey) instead of its texture, to see which bone a fault rides on.
--rest renders the rig's rest pose instead of the clip.
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
ap.add_argument("--clip", default=None)
ap.add_argument("--times", default="0")
ap.add_argument("--az", default="0")
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=640)
ap.add_argument("--frame", choices=("upper", "full"), default="full")
ap.add_argument("--weights", default="")
ap.add_argument("--rest", action="store_true")
ap.add_argument("--shots", default="", help="clip@seconds,... - several clips in one run")
ap.add_argument("--albedo", default=None,
                help="render with this colour texture in place of the packed one: a texture fix, seen on the "
                     "posed character without a rebuild")
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
meshes = [o for o in sc.objects if o.type == "MESH" and o.visible_get()
          and any(m.type == "ARMATURE" for m in o.modifiers)]
for o in sc.objects:                                                # nothing but the character
    if o.type == "MESH" and o not in meshes:
        o.hide_render = True

if a.albedo:
    new_img = bpy.data.images.load(os.path.abspath(a.albedo))
    old_imgs = [im for im in bpy.data.images if im.filepath.endswith("albedo.png") or im.name.startswith("baked_base_color")]
    swapped = 0
    for mat in bpy.data.materials:
        if mat.use_nodes:
            for nd in mat.node_tree.nodes:
                if nd.type == "TEX_IMAGE" and nd.image in old_imgs:
                    nd.image = new_img
                    swapped += 1
    print(f"[poses] colour texture {a.albedo} in {swapped} image node(s)", flush=True)

fps = sc.render.fps / sc.render.fps_base


def use_clip(name):
    """Put a clip on the rig; the frame it starts on."""
    act = bpy.data.actions.get(name)
    if act is None:
        raise SystemExit(f"[poses] no clip {name!r} in {a.blend}; clips: {[x.name for x in bpy.data.actions]}")
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    return act.frame_range[0]


if a.rest:
    rig.data.pose_position = "REST"
shots = ([(s_.split("@")[0], float(s_.split("@")[1])) for s_ in a.shots.split(",") if s_.strip()] if a.shots
         else [(a.clip, float(t_)) for t_ in a.times.split(",")])

if a.weights:
    COL = [(0.9, 0.15, 0.1), (0.1, 0.75, 0.2), (0.15, 0.35, 0.95), (0.95, 0.8, 0.1), (0.85, 0.2, 0.85), (0.1, 0.8, 0.85)]
    bones = [b.strip() for b in a.weights.split(",") if b.strip()]
    mat = bpy.data.materials.new("weights")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = "cf_weights"
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    for o in meshes:
        me = o.data
        gi = {g.name: g.index for g in o.vertex_groups}
        col = np.full((len(me.vertices), 3), 0.55)
        for vi, v in enumerate(me.vertices):
            ws = {g.group: g.weight for g in v.groups}
            tot = sum(ws.values()) or 1.0
            c = np.zeros(3)
            rest = 1.0
            for k, b in enumerate(bones):
                w = ws.get(gi.get(b, -1), 0.0) / tot
                c += w * np.array(COL[k % len(COL)])
                rest -= w
            col[vi] = c + max(rest, 0.0) * 0.55
        attr_ = me.color_attributes.new("cf_weights", "FLOAT_COLOR", "POINT")
        attr_.data.foreach_set("color", np.c_[col, np.ones(len(col))].astype(np.float32).ravel())
        me.materials.clear()
        me.materials.append(mat)

engines = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
sc.render.resolution_x = sc.render.resolution_y = a.res
sc.render.film_transparent = False
sc.render.image_settings.file_format = "PNG"
names = {t.name for t in sc.view_settings.bl_rna.properties["view_transform"].enum_items}
sc.view_settings.view_transform = "Standard" if "Standard" in names else sc.view_settings.view_transform
world = bpy.data.worlds.new("studio")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.78, 0.78, 0.8, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
for name, rot, energy in (("key", (50, 0, 35), 3.2), ("fill", (60, 0, -60), 1.2), ("back", (60, 0, 180), 1.5)):
    lt = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
    sc.collection.objects.link(lt)
    lt.data.energy = energy
    lt.rotation_euler = tuple(math.radians(x) for x in rot)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"


def bounds():
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in meshes:
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        V = np.empty(len(m.vertices) * 3)
        m.vertices.foreach_get("co", V)
        M = np.array(o.matrix_world)
        pts.append(V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        ev.to_mesh_clear()
    return np.vstack(pts)


def head_z():
    pb = rig.pose.bones.get("head") or rig.pose.bones.get("mixamorig:Head")
    return (rig.matrix_world @ pb.head).z if pb else None


for clip, t in shots:
    f0 = use_clip(clip) if (clip and not a.rest) else sc.frame_start
    f = f0 + t * fps
    sc.frame_set(int(math.floor(f)), subframe=float(f - math.floor(f)))
    P = bounds()
    lo, hi = P.min(0), P.max(0)
    if a.frame == "upper":
        hz = head_z() or hi[2]
        top, bot = hi[2], hz - 0.55 * (hz - lo[2])
        sel = P[(P[:, 2] >= bot) & (P[:, 2] <= top)]
        lo2, hi2 = sel.min(0), sel.max(0)
        ctr = Vector(((lo2 + hi2) / 2).tolist())
        span = max(top - bot, float(np.ptp(sel[:, 0])), float(np.ptp(sel[:, 1]))) * 1.08
    else:
        ctr = Vector(((lo + hi) / 2).tolist())
        span = max(float(hi[2] - lo[2]), float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1]))) * 1.08
    cam.data.ortho_scale = span
    for az in (float(x) for x in a.az.split(",")):
        r = math.radians(az)
        cam.location = ctr + Vector((math.sin(r) * 6.0, -math.cos(r) * 6.0, 0.0))
        cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(a.out, f"{clip if (clip and not a.rest) else 'rest'}_{t:.2f}s_{az:+.0f}.png")
        bpy.ops.render.render(write_still=True)
print(f"[poses] {len(shots)} x {len(a.az.split(','))} renders -> {a.out}", flush=True)
