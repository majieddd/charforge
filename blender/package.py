"""Turn a finished character into something a game developer can actually use.

Before this stage the pipeline's output was one .glb with a private skeleton. That fails the
first thing anyone does with a character, which is to import it into an engine and apply
animations to it:

  skeleton   bones were named by the joint at their head, SMPL-style - `left_shoulder` was the
             upper arm, `left_collar` the clavicle, `left_elbow` the forearm. Engine humanoid
             mappers match by name and read those names as something else; Unity's auto-mapper
             takes `left_shoulder` for the clavicle. They are renamed here to the Mixamo
             convention, which is what every retargeting tool ships a preset for and what the
             animation most people use is authored on. The joints map one-to-one, so this is a
             rename, not a re-rig - verified below by deforming every clip before and after.
  formats    glTF for Unreal, Godot and the web; FBX for Unity, which does not import glTF
             natively. Textures are also written out loose, so a material can be rebuilt by hand
             in any engine.
  manifest   height, clip lengths, which clips loop, and the ground speed each locomotion clip
             was captured at - the number a character controller needs so the feet do not skate.

Scale and origin are not handled here: normalize_frame.py fixes them before any animation
exists, so nothing downstream ever has to rescale an animation curve.

Run: blender -b -noaudio --python package.py -- --blend final.blend --out-dir pkg --name wren
     [--retarget retarget.json] [--prompt "..."] [--image ref.png]
"""
import argparse
import json
import math
import os
import shutil
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--retarget", default=None, help="retarget.py's JSON report (clip speeds)")
ap.add_argument("--prompt", default=None)
ap.add_argument("--image", default=None)
ap.add_argument("--springs", default=None, help="springs.json from springs.py")
ap.add_argument("--style", default="realistic",
                help="realistic | anime | stylized - recorded so a viewer or engine shades it to match")
ap.add_argument("--tex", type=int, default=0,
                help="downscale textures to this edge; 0 keeps the bake resolution (the web build is 2K)")
ap.add_argument("--no-fbx", action="store_true")
ap.add_argument("--lods", default="0.4,0.15",
                help="FBX level-of-detail ratios (triangle fraction of LOD0); empty for none")
a = ap.parse_args(argv)

LOOPING = {"idle", "walk", "run", "jog", "sprint", "crouch_walk", "strafe_left", "strafe_right",
           "walk_back", "jog_back", "crouch_idle", "fall"}

MIXAMO = {
    "pelvis": "Hips", "spine1": "Spine", "spine2": "Spine1", "spine3": "Spine2",
    "neck": "Neck", "head": "Head", "head_top": "HeadTop_End",
}
for side, S in (("left", "Left"), ("right", "Right")):
    MIXAMO.update({
        f"{side}_collar": f"{S}Shoulder", f"{side}_shoulder": f"{S}Arm",
        f"{side}_elbow": f"{S}ForeArm", f"{side}_wrist": f"{S}Hand",
        # a rig from before the modelled hands had a single stub past the wrist, named for the
        # middle finger's first joint, which is where it sat
        f"{side}_hand": f"{S}HandMiddle1",
        f"{side}_hip": f"{S}UpLeg", f"{side}_knee": f"{S}Leg",
        f"{side}_ankle": f"{S}Foot", f"{side}_foot": f"{S}ToeBase",
    })
    for F in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for i in (1, 2, 3, 4):
            MIXAMO[f"{side}_{F.lower()}{i}"] = f"{S}Hand{F}{i}"
PREFIX = "mixamorig:"

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
if rig is None:
    raise SystemExit("[pkg] no armature")

# A mesh is part of the character if this rig deforms it - not if its name matches a list (the
# name rule dropped the whole character the day the rig began producing a single mesh "char").
for o in list(scene.objects):
    if o.type == "MESH" and not any(m.type == "ARMATURE" and m.object == rig for m in o.modifiers):
        print(f"[pkg] dropping stray object {o.name!r} (not skinned to the rig)", flush=True)
        bpy.data.objects.remove(o, do_unlink=True)
meshes = [o for o in scene.objects if o.type == "MESH"]
actions = sorted(bpy.data.actions, key=lambda x: x.name)


def set_action(act):
    rig.animation_data_create()
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]


def snapshot():
    """Deformed vertex positions at the middle frame of every clip - the rename's regression test."""
    out = []
    for act in actions:
        set_action(act)
        f0, f1 = (int(x) for x in act.frame_range)
        scene.frame_set((f0 + f1) // 2)
        dg = bpy.context.evaluated_depsgraph_get()
        for o in meshes:
            ev = o.evaluated_get(dg)
            m = ev.to_mesh()
            P = np.empty(len(m.vertices) * 3)
            m.vertices.foreach_get("co", P)
            out.append(P)
            ev.to_mesh_clear()
    return np.concatenate(out) if out else np.zeros(0)


# ---- rename to the Mixamo convention, and prove nothing moved ---------------------------------
before = snapshot()
# spring-chain bones have no Mixamo equivalent; they keep their own names (a humanoid mapping
# ignores them, and the spring components look them up by name)
EXTRA = ("spring_", "jaw")            # face and spring bones: not in the Mixamo set, kept by name
unknown = [b.name for b in rig.data.bones if b.name not in MIXAMO and not b.name.startswith(EXTRA)]
if unknown:
    raise SystemExit(f"[pkg] bones with no Mixamo equivalent: {unknown}")
def action_fcurves(act):
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            for slot in act.slots:
                cb = strip.channelbag(slot)
                if cb:
                    out.extend(cb.fcurves)
    return out


rename = {old: PREFIX + new for old, new in MIXAMO.items() if old in rig.data.bones}
for b in list(rig.data.bones):
    if b.name in rename:
        b.name = rename[b.name]
# Renaming a bone renames its vertex groups, but in Blender 5's layered actions it does NOT
# rewrite the animation curves that target it: the first build of this stage shipped every clip
# still pointing at the old names, so each one would have played on nothing and left the
# character frozen in its T-pose in every engine. The drift check below is what caught it
# (961 mm). The curve paths are therefore rewritten explicitly.
moved = 0
for act in actions:
    for fc in action_fcurves(act):
        dp = fc.data_path
        if dp.startswith('pose.bones["'):
            old = dp.split('"')[1]
            if old in rename:
                fc.data_path = dp.replace(f'"{old}"', f'"{rename[old]}"', 1)
                moved += 1
        if fc.group is not None and fc.group.name in rename:
            fc.group.name = rename[fc.group.name]
print(f"[pkg] rewrote {moved:,} animation curve paths to the new bone names", flush=True)
rig.data.name = "Armature"
rig.name = "Armature"
after = snapshot()
drift = float(np.abs(after - before).max()) if len(before) else 0.0
groups_ok = all(vg.name in rig.data.bones for o in meshes for vg in o.vertex_groups)
print(f"[pkg] renamed {len(MIXAMO)} bones to {PREFIX}*; every clip deforms identically "
      f"(max drift {drift*1000:.4f} mm), vertex groups follow: {groups_ok}", flush=True)
if drift > 1e-5 or not groups_ok:
    raise SystemExit("[pkg] the rename changed the deformation - animation paths did not follow")

# ---- measurements for the manifest -------------------------------------------------------------
if rig.animation_data:
    rig.animation_data.action = None
for pb in rig.pose.bones:
    pb.matrix_basis.identity()
bpy.context.view_layer.update()
V = np.vstack([np.array([(o.matrix_world @ v.co)[:] for v in o.data.vertices]) for o in meshes])
height = float(V[:, 2].max() - V[:, 2].min())
tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes)
fps = scene.render.fps / scene.render.fps_base

speeds = {}
if a.retarget and os.path.exists(a.retarget):
    rj = json.load(open(a.retarget))
    for clip, info in rj.items():
        if isinstance(info, dict) and info.get("frames"):
            # the speed the character's own planted feet support, where it was measured; a clip
            # that does not travel (idle, wave, a jump in place) moves at zero, whatever small
            # drift its capture has
            if info.get("stride_speed_mps"):
                speeds[clip] = info["stride_speed_mps"]
            elif info.get("heading_from") == "hip line":
                speeds[clip] = 0.0
            else:
                dur = max(info["frames"] - 1, 1) / info.get("fps", 30)
                speeds[clip] = round(info.get("root_travel_m", 0.0) / dur, 3)

clips = []
lfoot, rfoot = rig.pose.bones.get(PREFIX + "LeftToeBase"), rig.pose.bones.get(PREFIX + "RightToeBase")
for act in actions:
    f0, f1 = (int(x) for x in act.frame_range)
    entry = {"name": act.name, "frames": f1 - f0 + 1, "seconds": round((f1 - f0) / fps, 3),
             "loop": act.name in LOOPING}
    if act.name in speeds:
        entry["ground_speed_mps"] = speeds[act.name]
    # Locomotion phase. To blend walk into run by speed - rather than switching at a threshold -
    # both cycles have to be at the same point in the stride, or the legs cross mid-blend. Each
    # clip starts wherever its capture happened to be cut, so record where the left heel strikes
    # (left foot furthest forward of the hips; the character faces glTF +Z = Blender -Y) as a
    # fraction of the cycle. A player then offsets each clip by its own phase and drives them
    # from one shared stride clock.
    lank = rig.pose.bones.get(PREFIX + "LeftFoot")
    hips = rig.pose.bones.get(PREFIX + "Hips")
    info0 = rj.get(act.name, {}) if a.retarget and os.path.exists(a.retarget) else {}
    trav = float(info0.get("travel_deg", 0.0) or 0.0)
    if act.name in speeds and speeds[act.name] > 0.3:
        # the way it travels, from forward, counter-clockwise from above (strafe left +90,
        # backpedal 180) - a 2D blend space places each clip by this and its speed
        entry["travel_deg"] = trav
    if act.name in LOOPING and speeds.get(act.name, 0) > 0.3 and lank and hips:
        set_action(act)
        # heel strike = the left foot furthest along the way the clip travels (forward for a
        # walk, sideways for a strafe, backwards for a backpedal), so every direction's cycle
        # is phased on the same event and they can be blended on one stride clock
        th = math.radians(trav)
        tx, ty = math.sin(th), -math.cos(th)          # Blender frame: forward is -Y
        fwd = []
        for f in range(f0, f1 + 1):
            scene.frame_set(f)
            d = lank.head - hips.head
            fwd.append(d.x * tx + d.y * ty)
        entry["left_contact_phase"] = round(float(np.argmax(fwd[:-1])) / max(len(fwd) - 1, 1), 4)
    if info0.get("heading_from") == "start" and abs(info0.get("turn_deg", 0.0)) > 20:
        # a turn in place: the body ends turned by this much (counter-clockwise from above);
        # a controller rotates the character by it when the clip ends
        entry["turn_deg"] = info0["turn_deg"]
    if info0.get("grounded") is False:
        entry["grounded"] = False
    if act.name == "land" and info0.get("flights_s"):
        first = min(info0["flights_s"], key=lambda f: f[0])
        if first[0] < 0.05:
            entry["touchdown_s"] = first[1]
    # For a jump: when both feet leave the floor and when one returns, measured by the retarget
    # on the soles of the mesh (the foot joints of a generated rig sit inside the shoe, and the
    # instep joint rises as soon as the heel does, which read as a takeoff 0.1 s early). A
    # controller launches at takeoff_s and lands at landing_s. A standing jump's anticipation is
    # long - this capture crouches for 0.87 s - so entry_s is where a game should start the clip
    # when the player presses jump: a quarter second before takeoff, already in the crouch.
    info = rj.get(act.name, {}) if a.retarget and os.path.exists(a.retarget) else {}
    if "jump" in act.name and info.get("flights_s"):
        t0, t1 = max(info["flights_s"], key=lambda f: f[1] - f[0])
        entry["takeoff_s"] = t0
        entry["landing_s"] = t1
        entry["apex_foot_height_m"] = info.get("apex_sole_m")
        entry["entry_s"] = round(max(0.0, t0 - 0.25), 3)
        entry["note"] = ("in place horizontally, but the clip keeps its vertical motion: the hips "
                         "rise along the arc. Launch the capsule at takeoff_s with the arc's own "
                         "speed and cancel the clip's lift while airborne (the playground does), "
                         "or let the clip carry the height and keep the capsule grounded - not both")
    clips.append(entry)

# ---- face ----------------------------------------------------------------------------------------
def face_manifest():
    """The jaw bone and the face morph targets, if the face stage made them."""
    keys = [m for o in meshes if o.data.shape_keys for m in o.data.shape_keys.key_blocks.keys()
            if m != "Basis"]
    if not keys and "jaw" not in rig.data.bones:
        return None
    return {"jaw_bone": "jaw" if "jaw" in rig.data.bones else None,
            "jaw_open_deg": 18, "morphs": sorted(set(keys)),
            "note": "morph targets on the LOD0 mesh; clips never key them. Blink both eyes for "
                    "~0.15 s every 2-6 s; for speech rotate the jaw about its local X toward "
                    "jaw_open_deg (a/o) with pucker for o/u, driven by the audio's loudness."}


# ---- spring chains ----------------------------------------------------------------------------------
def springs_manifest():
    """Chains for a runtime spring simulation (VRM-style: stiffness pulls each bone back toward
    its animated direction, drag damps it, gravity pulls it down, colliders keep it out of the
    body), in metres, with the Mixamo names of the bones they hang from and collide with."""
    if not (a.springs and os.path.exists(a.springs)):
        return None
    sj = json.load(open(a.springs))
    if not sj.get("chains"):
        return None
    inv = {v: k for k, v in rename.items()}
    def nm(internal):                                # internal name -> shipped name
        return rename.get(internal, internal)
    chains = []
    for ch in sj["chains"]:
        bones = [b for b in ch["bones"] if b in rig.data.bones]
        if not bones:
            continue
        lengths = [round(rig.data.bones[b].length * rig.matrix_world.to_scale()[0], 4) for b in bones]
        chains.append({"kind": ch["kind"], "parent": nm(ch["parent"]), "bones": bones,
                       "lengths_m": lengths, "radius_m": round(ch["radius_h"] * height, 4),
                       "stiffness": ch["stiffness"], "drag": ch["drag"], "gravity": ch["gravity"]})
    cols = [{"bone": nm(c["bone"]), "radius_m": round(c["radius_h"] * height, 4)}
            for c in sj.get("colliders", []) if nm(c["bone"]) in rig.data.bones]
    return {"chains": chains, "colliders": cols,
            "note": "clips key these bones at rest; a spring component drives them at runtime "
                    "(Unity: Magica Cloth / Dynamic Bone / VRM SpringBone; Unreal: AnimDynamics)"}


# ---- collision capsule ---------------------------------------------------------------------------
# What a CharacterController or a capsule component is sized from, measured on the idle pose with
# the arms down: the T-pose rest spreads the arms 1.7 m wide and says nothing about how wide the
# character stands. Radius is the 98th percentile of distance from the vertical axis through the
# origin, between 10% and 90% of height, so a stray strand of hair does not set it.
capsule = None
if bpy.data.actions.get("idle"):
    set_action(bpy.data.actions["idle"])
    scene.frame_set(int(bpy.data.actions["idle"].frame_range[0]))
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in meshes:
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        X = np.empty(len(m.vertices) * 3)
        m.vertices.foreach_get("co", X)
        mw = np.array(o.matrix_world)
        pts.append(X.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
        ev.to_mesh_clear()
    P = np.vstack(pts)
    z0, z1 = float(P[:, 2].min()), float(P[:, 2].max())
    band = P[(P[:, 2] > z0 + 0.1 * (z1 - z0)) & (P[:, 2] < z0 + 0.9 * (z1 - z0))]
    r = float(np.percentile(np.hypot(band[:, 0], band[:, 1]), 98))
    capsule = {"radius_m": round(r, 3), "height_m": round(z1 - z0, 3),
               "center_m": [0.0, round((z1 - z0) / 2, 3), 0.0],
               "note": "glTF axes (Y up); measured on the first idle frame"}
    print(f"[pkg] capsule: radius {r:.3f} m, height {z1 - z0:.3f} m", flush=True)
if rig.animation_data:
    rig.animation_data.action = None

# ---- export ------------------------------------------------------------------------------------
os.makedirs(a.out_dir, exist_ok=True)
tex_dir = os.path.join(a.out_dir, "textures")
os.makedirs(tex_dir, exist_ok=True)
for img in bpy.data.images:
    if a.tex and (img.size[0] > a.tex or img.size[1] > a.tex):
        img.scale(a.tex, a.tex)
written = []
for img in bpy.data.images:
    if img.size[0] == 0 or img.name in ("Render Result", "Viewer Node"):
        continue
    kind = ("albedo" if "base" in img.name or "color" in img.name else
            "normal" if "normal" in img.name else "orm" if "orm" in img.name else img.name)
    path = os.path.join(tex_dir, f"{a.name}_{kind}.png")
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    written.append(os.path.relpath(path, a.out_dir))

# ---- engine-specific maps ------------------------------------------------------------------------
# The glTF carries one packed ORM texture (R occlusion, G roughness, B metallic), which Unreal
# reads natively. Unity does not: its Standard and URP Lit shaders want metallic in R and
# *smoothness* - one minus roughness - in alpha, with occlusion separate, and FBX has no slot for
# a packed map at all. Unreal expects DirectX normals (green flipped); glTF, Unity and Godot use
# OpenGL. Rather than ship one map and a paragraph of instructions, ship the maps each engine
# consumes.
def _img_px(path):
    im = bpy.data.images.load(path, check_existing=False)
    px = np.empty(im.size[0] * im.size[1] * 4, np.float32)
    im.pixels.foreach_get(px)
    size = tuple(im.size)
    bpy.data.images.remove(im)
    return px.reshape(size[1], size[0], 4), size


def _save(px, size, path):
    im = bpy.data.images.new(os.path.basename(path), size[0], size[1], alpha=True,
                             float_buffer=False, is_data=True)
    im.pixels.foreach_set(px.reshape(-1))
    im.filepath_raw = path
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)
    return os.path.relpath(path, a.out_dir)


orm_path = os.path.join(tex_dir, f"{a.name}_orm.png")
nrm_path = os.path.join(tex_dir, f"{a.name}_normal.png")
if os.path.exists(orm_path):
    orm, size = _img_px(orm_path)
    ms = np.stack([orm[:, :, 2], orm[:, :, 2], orm[:, :, 2], 1.0 - orm[:, :, 1]], axis=2)
    written.append(_save(ms, size, os.path.join(tex_dir, f"{a.name}_metallic_smoothness.png")))
    occ = np.stack([orm[:, :, 0]] * 3 + [np.ones_like(orm[:, :, 0])], axis=2)
    written.append(_save(occ, size, os.path.join(tex_dir, f"{a.name}_occlusion.png")))
if os.path.exists(nrm_path):
    nrm, size = _img_px(nrm_path)
    nrm[:, :, 1] = 1.0 - nrm[:, :, 1]
    written.append(_save(nrm, size, os.path.join(tex_dir, f"{a.name}_normal_directx.png")))
print(f"[pkg] textures: {', '.join(os.path.basename(w) for w in written)}", flush=True)

# Morph targets rest at zero. glTF writes each shape key's current value as the mesh's default
# weight, and Blender 5 creates keys at 1.0 - so a face rig saved as made loads in any engine
# with every shape applied at once (eyes shut, mouth puckered). The playground drives the morphs
# each frame and hid it; an engine that does not would show it.
for o in meshes:
    if o.data.shape_keys:
        for kb in o.data.shape_keys.key_blocks:
            kb.value = 0.0
tracks = len(rig.animation_data.nla_tracks) if rig.animation_data else 0
glb = os.path.join(a.out_dir, f"{a.name}.glb")
bpy.ops.export_scene.gltf(
    filepath=glb, export_format="GLB", export_animations=True,
    export_animation_mode="NLA_TRACKS" if tracks else "ACTIONS",
    export_skins=True, export_apply=False, export_image_format="AUTO",
    export_morph=True, export_morph_normal=False,
    # Clips are keyed from Blender frame 1, which glTF writes as t = 1/30 s. A player looping
    # the clip holds the first pose through that empty span, so every cycle stalled for one
    # frame at the loop point and played ~7% slower than its measured ground speed - the
    # "run pops every half second" finding. Sliding every clip to start at t = 0 removes both.
    export_anim_slide_to_zero=True)
print(f"[pkg] glTF -> {glb} ({os.path.getsize(glb)/1e6:.1f} MB)", flush=True)

# ---- levels of detail, for the FBX ---------------------------------------------------------------
# Godot and Unreal build LODs on import; Unity does not, but it builds an LOD Group by itself from
# sibling meshes named _LOD0, _LOD1, ... So the FBX carries decimated copies - collapse decimation,
# which keeps UVs and interpolates the skin weights - and the glTF stays single-LOD. At 58k
# triangles the full mesh is hero-sized; a crowd or a distant character wants a few thousand.
lod_objs, lod_info = [], []
ratios = [float(x) for x in a.lods.split(",") if x.strip()] if not a.no_fbx else []
if ratios:
    for o in meshes:
        base = o.name
        for i, r in enumerate(ratios, start=1):
            c = o.copy()
            c.data = o.data.copy()
            c.name = f"{base}_LOD{i}"
            c.data.name = c.name
            scene.collection.objects.link(c)
            # a mesh with shape keys cannot take a modifier; the distant LODs do without the
            # face shapes (a blink at 30 m is invisible), so theirs are dropped
            if c.data.shape_keys:
                bpy.context.view_layer.objects.active = c
                c.shape_key_clear()
            d = c.modifiers.new("lod", "DECIMATE")
            d.decimate_type = "COLLAPSE"
            d.ratio = r
            # the decimator must run before the armature deforms the mesh, not after
            while c.modifiers[0].name != "lod":
                bpy.ops.object.select_all(action="DESELECT")
                bpy.context.view_layer.objects.active = c
                bpy.ops.object.modifier_move_up(modifier="lod")
            bpy.context.view_layer.objects.active = c
            bpy.ops.object.modifier_apply(modifier="lod")
            lod_objs.append(c)
        o.name = f"{base}_LOD0"
    for i, r in enumerate([1.0] + ratios):
        objs = [o for o in scene.objects if o.type == "MESH" and o.name.endswith(f"_LOD{i}")]
        t = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in objs)
        lod_info.append({"lod": i, "ratio": r, "triangles": t})
    print("[pkg] FBX LODs: " + ", ".join(f"LOD{x['lod']} {x['triangles']:,} tris" for x in lod_info), flush=True)

fbx = None
if not a.no_fbx:
    fbx = os.path.join(a.out_dir, f"{a.name}.fbx")
    bpy.ops.object.select_all(action="DESELECT")
    for o in [rig] + meshes + lod_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.fbx(
        filepath=fbx, use_selection=True, object_types={"ARMATURE", "MESH"},
        use_armature_deform_only=True, add_leaf_bones=False,
        # FBX_SCALE_ALL writes a unit-scale file, so Unity imports it at 1:1 in metres instead
        # of the notorious 100x scale on the armature root
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL",
        axis_forward="-Z", axis_up="Y", bake_space_transform=False,
        mesh_smooth_type="FACE", use_tspace=True,
        bake_anim=True, bake_anim_use_all_actions=True, bake_anim_use_nla_strips=False,
        bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0,
        path_mode="COPY", embed_textures=True)
    print(f"[pkg] FBX  -> {fbx} ({os.path.getsize(fbx)/1e6:.1f} MB)", flush=True)
    for c in lod_objs:
        bpy.data.objects.remove(c, do_unlink=True)
    for o in meshes:
        if o.name.endswith("_LOD0"):
            o.name = o.name[:-5]

if a.image and os.path.exists(a.image):
    shutil.copy(a.image, os.path.join(a.out_dir, "reference" + os.path.splitext(a.image)[1]))

manifest = {
    "name": a.name,
    "source": {"prompt": a.prompt, "image": os.path.basename(a.image) if a.image else None},
    "springs": springs_manifest(),
    "face": face_manifest(),
    "style": {"name": a.style,
              "shading": {"realistic": "PBR (metallic-roughness), as authored",
                          "anime": "cel shading: a 2-3 step toon ramp on the base colour, rim light, "
                                   "inverted-hull outline ~1.5 mm; the PBR maps still describe the "
                                   "surface for engines without a toon shader",
                          "stylized": "PBR with low specular; reads best with soft key light and a "
                                      "saturated rim"}.get(a.style, "PBR")},
    "units": "metres, Y-up in glTF / Z-up in Blender, facing +Z (glTF)",
    "origin": "on the floor, under the root (pelvis) joint - the pivot a turn in place rotates about",
    "capsule": capsule,
    "height_m": round(height, 3),
    "triangles": tris,
    "skeleton": {"convention": "Mixamo", "prefix": PREFIX, "bones": len(rig.data.bones),
                 "root": rig.data.bones[0].name if rig.data.bones else None,
                 "rest_pose": "T-pose"},
    "clips": clips,
    "files": {"gltf": os.path.basename(glb), "fbx": os.path.basename(fbx) if fbx else None,
              "textures": written},
    "lods": {"fbx": lod_info,
             "note": "the FBX meshes are named _LOD0/_LOD1/_LOD2, which Unity turns into an LOD "
                     "Group on import; the glTF carries LOD0 only - Godot and Unreal generate "
                     "their own LODs on import"} if lod_info else None,
    "materials": {
        "workflow": "metallic-roughness",
        "maps": {"albedo": "sRGB", "normal": "tangent space, OpenGL (+Y)",
                 "normal_directx": "tangent space, DirectX (-Y)",
                 "orm": "R occlusion, G roughness, B metallic (linear)",
                 "metallic_smoothness": "R metallic, A smoothness = 1 - roughness (linear)",
                 "occlusion": "greyscale ambient occlusion (linear)"},
        "per_engine": {
            "glTF / Godot / three.js": "everything is already wired inside the .glb",
            "Unreal": "import the .glb (or .fbx); Base Color = albedo, Normal = normal_directx, "
                      "ORM = orm (occlusion R, roughness G, metallic B) - Unreal's native packing",
            "Unity URP / Standard": "import the .fbx; Base Map = albedo, Normal Map = normal, "
                                    "Metallic Map = metallic_smoothness (smoothness source: "
                                    "metallic alpha), Occlusion = occlusion; set the rig to "
                                    "Humanoid - the Mixamo names auto-map"}},
    "animation_licence": "Clips are retargeted Mixamo captures; see README before commercial use.",
}
json.dump(manifest, open(os.path.join(a.out_dir, f"{a.name}.json"), "w"), indent=2)
print(f"[pkg] manifest: {height:.3f} m, {tris:,} tris, {len(rig.data.bones)} bones, "
      f"{len(clips)} clips -> {a.out_dir}", flush=True)
for c in clips:
    extra = "".join(f"  {k}={v}" for k, v in c.items() if k not in ("name", "frames", "seconds", "loop", "note"))
    print(f"[pkg]    {c['name']:<6s} {c['seconds']:5.2f}s {'loop' if c['loop'] else 'once'}{extra}",
          flush=True)
