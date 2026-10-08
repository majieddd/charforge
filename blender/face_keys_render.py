"""Render each shape key of a face rig at 1.0, alone, from the front and three-quarter.

    /Applications/Blender.app/Contents/MacOS/Blender -b -noaudio --python blender/face_keys_render.py -- \
        --blend work/mara/rig_f.blend --face work/mara/face.json --out-dir qa/face_keys/mara --mode unlit
    ... --mode lit

unlit  the base colour as emission, front only, at --front-res: the flat albedo the face landmark
       finder reads (face_render.py's convention), so DWPose sees the features as they are
lit    soft studio light, front and three-quarter (az 40, the subject's left cheek), at --side-res:
       the volume a person sees
Both write Basis.png (every key at 0) and <key>.png (unlit) or <key>_front.png / <key>_34.png (lit).
CPU only (Cycles on the CPU, two threads).
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--face", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--mode", choices=("unlit", "lit"), required=True)
ap.add_argument("--keys", default="", help="comma list; default every key")
ap.add_argument("--front-res", type=int, default=768)
ap.add_argument("--side-res", type=int, default=512)
a = ap.parse_args(argv)
os.makedirs(a.out_dir, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
rig.data.pose_position = "REST"
meshes = [o for o in sc.objects if o.type == "MESH" and o.find_armature() == rig and o.visible_get()]
face_obj = next(o for o in meshes if o.data.shape_keys)
kbs = face_obj.data.shape_keys.key_blocks
for o in list(sc.objects):
    if o.type in ("LIGHT", "CAMERA"):
        bpy.data.objects.remove(o, do_unlink=True)
F = json.load(open(a.face))
ied = float(F["ied_m"])
c = (np.array(F["eyes"]["left"]["centre"]) + np.array(F["eyes"]["right"]["centre"]) + np.array(F["mouth"]["centre"])) / 3.0

cd = bpy.data.cameras.new("cam")
cd.type = "ORTHO"
cd.ortho_scale = 4.4 * ied
cam = bpy.data.objects.new("cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam


def place(az):
    """The camera 2 m from the face at azimuth az (0 = the front, facing -Y; + swings to the subject's left),
    looking at the face: X by 90 degrees points it along +Y with +Z up, then Z by az turns it. (A track-to
    constraint does not evaluate in background mode, so the rotation is set outright.)"""
    cam.location = tuple(float(x) for x in c + 2.0 * np.array([math.sin(math.radians(az)), -math.cos(math.radians(az)), 0.0]))
    cam.rotation_euler = (math.radians(90.0), 0.0, math.radians(az))
world = bpy.data.worlds.new("w")
sc.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.32, 0.33, 0.35, 1.0)
bg.inputs["Strength"].default_value = 0.7 if a.mode == "lit" else 1.0


def sun(name, energy, rot, angle):
    l_ = bpy.data.lights.new(name, "SUN")
    l_.energy, l_.angle = energy, angle
    o = bpy.data.objects.new(name, l_)
    o.rotation_euler = rot
    sc.collection.objects.link(o)


if a.mode == "lit":
    sun("key", 2.6, (math.radians(55), 0.0, math.radians(25)), 0.45)
    sun("fill", 1.2, (math.radians(70), 0.0, math.radians(-120)), 0.6)
else:
    for mat in {s.material for o in meshes for s in o.material_slots if s.material}:
        if not mat.use_nodes:
            continue
        nt = mat.node_tree
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
    sc.view_settings.view_transform = "Standard"

sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = 24 if a.mode == "lit" else 4
sc.cycles.use_denoising = a.mode == "lit"
sc.render.threads_mode = "FIXED"
sc.render.threads = 2
sc.render.image_settings.file_format = "PNG"

names = [kb.name for kb in kbs][1:]
want = [k for k in a.keys.split(",") if k] or names
jobs = [("Basis", None)] + [(k, k) for k in want if k in names]
views = [("front", 0.0, a.front_res)] if a.mode == "unlit" else [("front", 0.0, a.side_res), ("34", 40.0, a.side_res)]
for kb in kbs:
    kb.value = 0.0
for key, on in jobs:
    if on:
        kbs[on].value = 1.0
    for tag, az, res in views:
        place(az)
        sc.render.resolution_x = sc.render.resolution_y = res
        if a.mode == "unlit":
            fn = f"{key}.png"
        else:
            fn = f"{key}_{tag}.png"
        sc.render.filepath = os.path.join(a.out_dir, fn)
        bpy.ops.render.render(write_still=True)
    if on:
        kbs[on].value = 0.0
print(f"[render] {len(jobs)} keys x {len(views)} view(s) -> {a.out_dir}", flush=True)
