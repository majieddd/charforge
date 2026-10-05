"""A close-up of one joint of a rigged character, at rest or in a clip: how the skin deforms there (E137).

    blender -b -noaudio --python render_joint.py -- --blend final.blend --out prefix --bone left_shoulder \
        [--clip idle --at 0.25,0.5] [--span 0.3] [--az 35] [--elev -12] [--grey] [--albedo a.png] [--res 600] [--cpu]

The camera follows the bone's head in each pose (posed, so a raised arm stays in frame), orthographic,
--span of the character's height across, --az degrees round from the front toward the bone's side and --elev
above level (negative looks up into an armpit). --grey renders the shape alone in matte grey under a key
light from above the camera; otherwise the character's own materials. --at takes fractions of the clip's
length; without --clip the rest pose is rendered. Writes <prefix>_rest.png or <prefix>_<clip>_<at>.png.
"""
import argparse
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bone", default="left_shoulder")
ap.add_argument("--clip", default=None)
ap.add_argument("--at", default="0.5")
ap.add_argument("--span", type=float, default=0.3)
ap.add_argument("--az", type=float, default=35.0)
ap.add_argument("--elev", type=float, default=-12.0)
ap.add_argument("--grey", action="store_true")
ap.add_argument("--albedo", default=None, help="paint this base colour on instead of the file's own")
ap.add_argument("--res", type=int, default=600)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--unlit", action="store_true", help="the base colour alone, as emission - the texture without any light")
ap.add_argument("--soft", action="store_true",
                help="studio light: a wide key, a strong fill and a grey sky, no hard cast shadows - for judging a face's "
                     "texture rather than how a low sun falls on it")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
meshes = [o for o in sc.objects if o.type == "MESH" and o.visible_get() and not o.hide_render]
for o in sc.objects:
    if o.type == "LIGHT":
        o.hide_render = True
if a.albedo and not a.grey:
    img = bpy.data.images.load(a.albedo)
    for o in meshes:
        for slot in o.material_slots:
            if slot.material is None or not slot.material.use_nodes:
                continue
            bsdf = next((n for n in slot.material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            link = bsdf.inputs["Base Color"].links if bsdf else None
            if link and link[0].from_node.type == "TEX_IMAGE":
                link[0].from_node.image = img
if a.unlit:
    for o in meshes:
        for slot in o.material_slots:
            if slot.material is None or not slot.material.use_nodes:
                continue
            nt = slot.material.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            out_n = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
            if bsdf is None or out_n is None:
                continue
            em = nt.nodes.new("ShaderNodeEmission")
            lk = bsdf.inputs["Base Color"].links
            if lk:
                nt.links.new(lk[0].from_socket, em.inputs["Color"])
            else:
                em.inputs["Color"].default_value = bsdf.inputs["Base Color"].default_value
            nt.links.new(em.outputs["Emission"], out_n.inputs["Surface"])
if a.grey:
    for o in meshes:
        for slot in o.material_slots:
            if slot.material is None or not slot.material.use_nodes:
                continue
            nt = slot.material.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            for name, value in (("Base Color", (0.8, 0.8, 0.8, 1)), ("Roughness", 0.5), ("Metallic", 0.0)):
                for link in list(bsdf.inputs[name].links):
                    nt.links.remove(link)
                bsdf.inputs[name].default_value = value
if a.cpu:
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 32
    sc.cycles.use_denoising = True
else:
    eng = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
sc.render.resolution_x = sc.render.resolution_y = a.res
if a.unlit:
    sc.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("w")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.06, 0.06, 0.07, 1)
if a.soft:
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.32, 0.33, 0.35, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.7
key = bpy.data.objects.new("key", bpy.data.lights.new("key", "SUN"))
sc.collection.objects.link(key)
key.data.energy = 3.5
fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", "SUN"))
sc.collection.objects.link(fill)
fill.data.energy = 0.8
if a.soft:
    key.data.energy, fill.data.energy = 2.6, 1.3
    key.data.angle = fill.data.angle = 0.45                 # ~26 deg: soft-edged shadows
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"

# the character's height, at rest
rig.data.pose_position = "REST"
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
zs = []
for o in meshes:
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    zs += [(o.matrix_world @ v.co).z for i, v in enumerate(me.vertices) if i % 25 == 0]
    ev.to_mesh_clear()
H = max(zs) - min(zs)
cam.data.ortho_scale = a.span * H
pb = rig.pose.bones[a.bone]
side = 1.0 if (rig.matrix_world @ pb.head).x >= 0 else -1.0     # the bone's side: the camera swings toward it


def shoot(path):
    bpy.context.view_layer.update()
    target = rig.matrix_world @ pb.head
    az, el = math.radians(a.az) * side, math.radians(a.elev)
    d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    cam.location = target + d * 3 * H
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    key.rotation_euler = (math.radians(40), 0, math.radians(a.az * side - 30 * side))
    fill.rotation_euler = (math.radians(70), 0, math.radians(a.az * side + 120 * side))
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print(f"[joint] -> {path}", flush=True)


if a.clip is None:
    shoot(f"{a.out}_rest.png")
else:
    rig.data.pose_position = "POSE"
    act = bpy.data.actions[a.clip]
    if rig.animation_data is None:
        rig.animation_data_create()
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    f0, f1 = act.frame_range
    for t in (float(x) for x in a.at.split(",")):
        sc.frame_set(int(round(f0 + t * (f1 - f0))))
        shoot(f"{a.out}_{a.clip}_{int(round(t * 100)):03d}.png")
