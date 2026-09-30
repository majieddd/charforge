"""A synthetic character whose aberrations are known, to check blender/aberration_audit.py (E102).

    blender -b --python aberration_controls.py -- --out controls.blend --expect expected.json

A box torso (bone spine1) and a box arm on two bones (left_shoulder, left_elbow) blended over
8 cm at the elbow, 10 cm from the torso at rest, standing on the floor. One vertex on the forearm's
outer face is bound to its own bone (left_pinky1). Every action is keyed on every frame, so the
audit at --step 1 --bvh-step 1 sees exactly these values:

  still     nothing moves                            -> every measure zero
  push      the arm moves into the torso, held 5 frames at each of 0.5, 1, 2 and 5 cm
                                                     -> penetration 0.5 / 1 / 2 / 5 cm; rigid, so
                                                        nothing crushed, stretched, sheared or popping
  pop       the one vertex jumps 2 cm out on frame 10 only
                                                     -> popping at frame 10 and nowhere else
  bend      the forearm bends 0 -> 130 degrees evenly over 20 frames
                                                     -> no popping (a fast smooth bend is not a pop)
  squash    the torso scaled 0.6 across (x and y)    -> exactly its top and bottom crushed (x0.36)
  stretch   the torso scaled 1.6 across              -> exactly its top and bottom stretched (x2.56)
  sink      the torso moved 3 cm down                -> floor flagged every frame, lowest point -3 cm

expected.json records these, including the triangle counts, for tools/aberration_controls.py.
"""
import argparse
import json
import math
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--expect", required=True)
a = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = 30
N = 20                                                 # frames per action

TORSO = dict(ctr=(0.0, 0.0, 0.25), size=(0.30, 0.20, 0.50), cuts=12)     # x 0.15 half-width, on the floor
ARM = dict(ctr=(0.29, 0.0, 0.25), size=(0.08, 0.08, 0.30), cuts=16)      # inner face x=0.25: 10 cm gap
SHOULDER, ELBOW, WRIST = Vector((0.29, 0, 0.40)), Vector((0.29, 0, 0.25)), Vector((0.29, 0, 0.10))


def box(name, ctr, size, cuts):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts, use_grid_fill=True)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0] + ctr[0], v.co.y * size[1] + ctr[1], v.co.z * size[2] + ctr[2]))
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    return ob


torso = box("torso", **TORSO)
arm_ob = box("arm", **ARM)

# ---- skeleton ------------------------------------------------------------------------------------
arm_data = bpy.data.armatures.new("rig")
rig = bpy.data.objects.new("rig", arm_data)
sc.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
for name, head, tail, parent in (("spine1", (0, 0, 0.0), (0, 0, 0.5), None),
                                 ("left_shoulder", SHOULDER, ELBOW, None),
                                 ("left_elbow", ELBOW, WRIST, "left_shoulder"),
                                 ("left_pinky1", (0.33, 0, 0.18), (0.36, 0, 0.18), "left_elbow")):
    b = arm_data.edit_bones.new(name)
    b.head, b.tail = Vector(head), Vector(tail)
    if parent:
        b.parent = arm_data.edit_bones[parent]
bpy.ops.object.mode_set(mode="OBJECT")


def skin(ob, weights):
    for bn in ("spine1", "left_shoulder", "left_elbow", "left_pinky1"):
        ob.vertex_groups.new(name=bn)
    for v in ob.data.vertices:
        for bn, w in weights(v.co).items():
            if w > 0:
                ob.vertex_groups[bn].add([v.index], w, "REPLACE")
    m = ob.modifiers.new("Armature", "ARMATURE")
    m.object = rig


skin(torso, lambda co: {"spine1": 1.0})


def arm_weights(co):
    t = min(max((ELBOW.z + 0.04 - co.z) / 0.08, 0.0), 1.0)       # 0 above the blend, 1 below
    t = t * t * (3 - 2 * t)
    return {"left_shoulder": 1 - t, "left_elbow": t}


# the one vertex that pops: on the forearm's outer face, nearest (0.33, 0, 0.18)
pop_v = min(arm_ob.data.vertices, key=lambda v: (v.co - Vector((0.33, 0, 0.18))).length).index
skin(arm_ob, lambda co: arm_weights(co))
for bn in ("left_shoulder", "left_elbow"):
    arm_ob.vertex_groups[bn].remove([pop_v])
arm_ob.vertex_groups["left_pinky1"].add([pop_v], 1.0, "REPLACE")

# ---- actions -------------------------------------------------------------------------------------
rig.animation_data_create()
pb = rig.pose.bones


def local_delta(bone, world):
    return bone.bone.matrix_local.to_3x3().inverted() @ Vector(world)


def action(name, pose_at):
    """pose_at(frame) sets the pose bones; every bone is keyed on every frame."""
    rig.animation_data.action = None
    for f in range(1, N + 1):
        for b in pb:
            b.location, b.rotation_quaternion, b.scale = (0, 0, 0), (1, 0, 0, 0), (1, 1, 1)
        pose_at(f)
        for b in pb:
            for path in ("location", "rotation_quaternion", "scale"):
                b.keyframe_insert(path, frame=f)
    act = rig.animation_data.action
    act.name = name
    act.use_fake_user = True                              # kept in the file once no object holds it
    return act


def stage(f):
    return (f - 1) // 5                                   # 0..3, five frames each


PUSH = [0.005, 0.01, 0.02, 0.05]
GAP = ARM["ctr"][0] - ARM["size"][0] / 2 - TORSO["size"][0] / 2
action("still", lambda f: None)
action("push", lambda f: setattr(pb["left_shoulder"], "location", local_delta(pb["left_shoulder"], (-(GAP + PUSH[stage(f)]), 0, 0))))
action("pop", lambda f: setattr(pb["left_pinky1"], "location", local_delta(pb["left_pinky1"], (0.02, 0, 0))) if f == 10 else None)


def bend(f):
    ang = math.radians(130) * (f - 1) / (N - 1)
    # about the elbow, bringing the forearm forward-up (world -y side)
    pb["left_elbow"].rotation_quaternion = (pb["left_elbow"].bone.matrix_local.to_3x3().inverted()
                                           @ Matrix.Rotation(ang, 3, "X") @ pb["left_elbow"].bone.matrix_local.to_3x3()).to_quaternion()


action("bend", bend)


def across(k):
    def set_(f):
        # spine1 points up world z: its local x and z are the world's horizontal axes
        pb["spine1"].scale = (k, 1.0, k)
    return set_


action("squash", across(0.6))
action("stretch", across(1.6))
action("sink", lambda f: setattr(pb["spine1"], "location", local_delta(pb["spine1"], (0, 0, -0.03))))
rig.animation_data.action = None
sc.frame_start, sc.frame_end = 1, N

# ---- expectations --------------------------------------------------------------------------------
# triangles of the torso's top and bottom faces (normal +-z) - the ones a horizontal scale crushes or stretches
torso.data.calc_loop_triangles()
tb = sum(1 for t in torso.data.loop_triangles if abs(t.normal.z) > 0.99)
arm_ob.data.calc_loop_triangles()
n_tris = len(torso.data.loop_triangles) + len(arm_ob.data.loop_triangles)
json.dump({"frames": N, "triangles": n_tris, "torso_top_bottom_triangles": tb, "push_depths_cm": [d * 100 for d in PUSH],
           "push_stage_frames": 5, "pop_frame": 10, "pop_vertex": pop_v, "sink_cm": -3.0},
          open(a.expect, "w"), indent=1)
bpy.ops.wm.save_as_mainfile(filepath=a.out)
print(f"[controls] {n_tris} triangles, torso top+bottom {tb}, pop vertex {pop_v} -> {a.out}", flush=True)
