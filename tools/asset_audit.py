"""Import a shipped GLB into an empty Blender scene and report what a game developer would see.

Checks, in the order an engine import surfaces them:
  scale      how tall is the character in metres (engines assume 1 unit = 1 m)
  pivot      where is the origin relative to the feet (engines drop assets at origin = floor)
  facing     which way does the character face
  skeleton   bone names, hierarchy root, root-at-ground bone, bone count, finger bones
  materials  which PBR channels have textures, roughness/metallic factors
  uv         island count and texture-space utilisation
  anims      clip names, frame counts, fps, root translation, loop seam error
"""
import bpy, sys, math, json
import numpy as np
from collections import defaultdict, deque

glb = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
sc = bpy.context.scene
rig = next((o for o in sc.objects if o.type == "ARMATURE"), None)
# Blender's glTF importer adds a bone-display Icosphere in a collection it names
# "glTF_not_exported"; it is not in the file and must not count toward the character's bounds.
meshes = [o for o in sc.objects if o.type == "MESH"
          and not any(c.name == "glTF_not_exported" for c in o.users_collection)]
R = {"file": glb.split("/")[-1]}

# ---- rest-pose geometry -------------------------------------------------------------------
if rig and rig.animation_data:
    rig.animation_data.action = None
if rig:
    for pb in rig.pose.bones:
        pb.matrix_basis.identity()
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
P = []
for o in meshes:
    ev = o.evaluated_get(dg); m = ev.to_mesh()
    V = np.empty(len(m.vertices) * 3); m.vertices.foreach_get("co", V)
    mw = np.array(o.matrix_world); P.append(V.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
    ev.to_mesh_clear()
P = np.vstack(P)
lo, hi = P.min(0), P.max(0)
R["height_m"] = round(float(hi[2] - lo[2]), 3)             # Blender is Z-up after glTF import
R["width_m"] = round(float(hi[0] - lo[0]), 3)
R["depth_m"] = round(float(hi[1] - lo[1]), 3)
R["origin_to_feet_m"] = round(float(0 - lo[2]), 3)         # >0 means origin is above the floor
R["origin_xy_offset_m"] = [round(float((lo[0] + hi[0]) / 2), 3), round(float((lo[1] + hi[1]) / 2), 3)]

# facing: nose/toes point which way? compare foot-tip vs heel along Y using the lowest 3% of verts
low = P[P[:, 2] < lo[2] + 0.03 * (hi[2] - lo[2])]
head = P[P[:, 2] > hi[2] - 0.10 * (hi[2] - lo[2])]
R["feet_centroid_y"] = round(float(low[:, 1].mean()), 3)
R["toes_extend_toward"] = "-Y (Blender front, = glTF +Z)" if (low[:, 1].min() + low[:, 1].max()) / 2 < low[:, 1].mean() + 1e-6 and abs(low[:, 1].min()) > abs(low[:, 1].max()) else "+Y (Blender back, = glTF -Z)"

# ---- skeleton -----------------------------------------------------------------------------
if rig:
    bones = rig.data.bones
    R["bone_count"] = len(bones)
    roots = [b.name for b in bones if b.parent is None]
    R["root_bones"] = roots
    rb = bones[roots[0]]
    R["root_bone_head_height_m"] = round(float((rig.matrix_world @ rb.head_local).z - lo[2]), 3)
    R["bone_names"] = [b.name for b in bones]
    R["has_finger_bones"] = any(any(k in b.name.lower() for k in ("thumb", "index", "middle", "ring", "pinky", "finger")) for b in bones)
    R["has_twist_bones"] = any("twist" in b.name.lower() or "roll" in b.name.lower() for b in bones)
    std = {"mixamorig:Hips", "Hips", "pelvis", "root", "spine_01"}
    R["naming_convention"] = ("mixamo" if any(b.name.startswith("mixamorig") for b in bones) else
                              "unreal" if any(b.name in ("spine_01", "clavicle_l") for b in bones) else
                              "custom")
    # a quick check of how far the rest pose is from a T
    def ang(name):
        b = bones.get(name)
        if not b: return None
        v = (b.tail_local - b.head_local).normalized()
        return round(math.degrees(math.acos(max(-1, min(1, -v.z)))), 1)
    R["upper_arm_angle_from_down_deg"] = ang("left_shoulder") or ang("mixamorig:LeftArm")

# ---- materials ----------------------------------------------------------------------------
mats = {}
for o in meshes:
    for slot in o.material_slots:
        m = slot.material
        if not m or m.name in mats: continue
        info = {"used_by": [], "textures": {}, "roughness": None, "metallic": None}
        if m.use_nodes:
            bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf:
                for sock in ("Base Color", "Roughness", "Metallic", "Normal", "Alpha", "Emission Color"):
                    s = bsdf.inputs.get(sock)
                    if s is None: continue
                    if s.is_linked:
                        # walk back to an image
                        stack = [s.links[0].from_node]; img = None
                        while stack:
                            n = stack.pop()
                            if n.type == "TEX_IMAGE" and n.image: img = n.image; break
                            for inp in n.inputs:
                                if inp.is_linked: stack.append(inp.links[0].from_node)
                        info["textures"][sock] = f"{img.size[0]}x{img.size[1]}" if img else "linked (no image)"
                    elif sock in ("Roughness", "Metallic"):
                        info[sock.lower()] = round(float(s.default_value), 3)
        mats[m.name] = info
    for slot in o.material_slots:
        if slot.material: mats[slot.material.name]["used_by"].append(o.name)
R["materials"] = mats

# ---- UV islands ---------------------------------------------------------------------------
uv = {}
for o in meshes:
    me = o.data
    lay = me.uv_layers.active
    if lay is None: uv[o.name] = "none"; continue
    # islands = connected components of faces sharing a UV edge
    key = lambda li: (round(lay.data[li].uv[0], 6), round(lay.data[li].uv[1], 6))
    edge_faces = defaultdict(list)
    for p in me.polygons:
        L = list(p.loop_indices)
        for k in range(len(L)):
            a, b = L[k], L[(k + 1) % len(L)]
            va, vb = me.loops[a].vertex_index, me.loops[b].vertex_index
            e = (min(va, vb), max(va, vb), tuple(sorted((key(a), key(b)))))
            edge_faces[e].append(p.index)
    adj = defaultdict(set)
    for fs in edge_faces.values():
        for i in fs:
            for j in fs:
                if i != j: adj[i].add(j)
    seen = set(); islands = 0; sizes = []
    for p in me.polygons:
        if p.index in seen: continue
        islands += 1; q = deque([p.index]); seen.add(p.index); n = 0
        while q:
            u = q.popleft(); n += 1
            for v in adj[u]:
                if v not in seen: seen.add(v); q.append(v)
        sizes.append(n)
    # UV-space utilisation: summed UV triangle area / unit square
    area = 0.0
    for p in me.polygons:
        L = [lay.data[li].uv for li in p.loop_indices]
        for k in range(1, len(L) - 1):
            a, b, c = L[0], L[k], L[k + 1]
            area += abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2
    sizes.sort(reverse=True)
    uv[o.name] = {"faces": len(me.polygons), "islands": islands,
                  "islands_under_10_faces": sum(1 for s in sizes if s < 10),
                  "largest_islands": sizes[:5], "uv_coverage": round(area, 3)}
R["uv"] = uv
R["triangles"] = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes)
R["mesh_objects"] = [o.name for o in meshes]

# ---- animations ---------------------------------------------------------------------------
anims = {}
fps = sc.render.fps
for act in bpy.data.actions:
    f0, f1 = act.frame_range
    info = {"frames": int(f1 - f0 + 1), "seconds": round((f1 - f0) / fps, 2)}
    if rig:
        rig.animation_data_create(); rig.animation_data.action = act
        if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
            rig.animation_data.action_slot = act.slots[0]
        root = rig.pose.bones[R["root_bones"][0]]
        def pose_at(f):
            sc.frame_set(int(f)); bpy.context.view_layer.update()
            return {pb.name: np.array(pb.matrix) for pb in rig.pose.bones}
        a, b = pose_at(f0), pose_at(f1)
        info["root_translation_m"] = round(float(np.linalg.norm(b[root.name][:3, 3] - a[root.name][:3, 3])), 3)
        # loop seam: how far does every bone move between the last and first frame?
        seam = max(float(np.linalg.norm(b[n][:3, 3] - a[n][:3, 3])) for n in a)
        info["loop_seam_max_bone_jump_m"] = round(seam, 3)
    anims[act.name] = info
R["animations"] = anims
R["scene_fps"] = fps
print("AUDIT_JSON" + json.dumps(R))
