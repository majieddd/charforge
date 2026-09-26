"""One clip from several cameras at once - the same frame from the front, the side and above - so that
what one camera hides shows: Aoi's spell cast read as right from the front while her left elbow was
folded to 176 degrees, the forearm back through the upper arm; from the side it is plain.

    blender -b -noaudio --python render_angles.py -- --blend final.blend --clip spell_cast --out dir
        [--views front,left,top] [--res 400] [--every 2] [--yaw 0] [--start 0] [--frames N]

Every view is an orthographic camera - lengths read true, so one metre is the same number of pixels in
every panel - held still for the whole clip and framed on everything the clip reaches. The floor and the
walls behind the character carry a grid, 10 cm lines and a heavier line every metre. --yaw turns the
"front" view (a video's camera angle, relative to the body's average heading); "left", "right", "back"
and "top" are turned with it. Views: front, back, left, right (the character's own left and right side),
top (looking down, the character's front toward the bottom of the frame), and angle:<deg> for any other
azimuth.

Writes <out>/<view>/f0000.png... and <out>/angles.json: the views, the scale, and per frame each elbow's
and knee's bend (0 straight, 180 folded back on itself) and the lowest point of each shoe above the
floor - for the readout tools/make_angles.py draws under the panels.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--clip", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--views", default="front,left,top")
ap.add_argument("--res", type=int, default=400)
ap.add_argument("--every", type=int, default=0, help="render every n-th frame (default: the scene rate over 30)")
ap.add_argument("--yaw", type=float, default=0.0, help="the front view's azimuth, degrees (a video's camera)")
ap.add_argument("--start", type=int, default=0, help="skip this many rendered frames (a shorter test)")
ap.add_argument("--frames", type=int, default=0, help="render at most this many frames (0: the whole clip)")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
sc = bpy.context.scene
rig = next(o for o in sc.objects if o.type == "ARMATURE")
meshes = [o for o in sc.objects if o.type == "MESH" and o.visible_get()
          and any(m.type == "ARMATURE" for m in o.modifiers)]
for o in sc.objects:
    if o.type == "MESH" and o not in meshes:
        o.hide_render = True
act = bpy.data.actions.get(a.clip)
if act is None:
    raise SystemExit(f"[angles] no clip {a.clip!r}; clips: {sorted(x.name for x in bpy.data.actions)}")
rig.animation_data.action = act
if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
    rig.animation_data.action_slot = act.slots[0]
fps = sc.render.fps / sc.render.fps_base
every = a.every or max(1, round(fps / 30))
f0, f1 = (int(x) for x in act.frame_range)
frames = list(range(f0, f1 + 1, every))[a.start:]
if a.frames:
    frames = frames[:a.frames]


def world_points():
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for o in meshes:
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        V = np.empty(len(m.vertices) * 3)
        m.vertices.foreach_get("co", V)
        mw = np.array(o.matrix_world)
        out.append(V.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
        ev.to_mesh_clear()
    return np.vstack(out)


# the bounds of everything the clip reaches, for cameras that never move
pts = []
for fr in frames[::4]:
    sc.frame_set(fr)
    pts.append(world_points()[::7])
P = np.vstack(pts)
lo, hi = P.min(0), P.max(0)
# the ground is z = 0: the pipeline stands every character's soles on it (blender/foot_audit.py measures
# against it too); a shoe below it reads negative
floor = 0.0 if abs(float(lo[2])) < 0.25 else float(lo[2])
lo[2] = min(lo[2], floor)
ctr = Vector(((lo + hi) / 2).tolist())


def view_dirs(name):
    """(camera direction from the character, the frame's up) for a view, in world space."""
    if name == "top":
        r = math.radians(a.yaw)
        back = Vector((-math.sin(r), math.cos(r), 0.0))           # the character's back: up in the frame
        return Vector((0, 0, 1)), back
    az = {"front": 0.0, "left": 90.0, "back": 180.0, "right": -90.0}.get(name)
    if az is None:
        az = float(name.split(":", 1)[1])
    r = math.radians(a.yaw + az)
    return Vector((math.sin(r), -math.cos(r), 0.0)), Vector((0, 0, 1))


views = [v.strip() for v in a.views.split(",") if v.strip()]
# one scale for every panel: the largest extent any view needs, with a margin
corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
need = 0.0
for v in views:
    d, up = view_dirs(v)
    right = d.cross(up).normalized() * -1
    c = corners - np.array(ctr)
    need = max(need, float(np.ptp(c @ np.array(right))), float(np.ptp(c @ np.array(up))))
scale = need * 1.12
cams = {}
for v in views:
    d, up = view_dirs(v)
    cam = bpy.data.objects.new(f"cam_{v}", bpy.data.cameras.new(f"cam_{v}"))
    sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale
    cam.data.clip_end = 1000
    cam.location = ctr + d * (scale * 4)
    # the camera's own axes: it looks down its -Z (so +Z is d), with the frame's up along its Y
    z_ = d.normalized()
    y_ = (up - z_ * up.dot(z_)).normalized()
    x_ = y_.cross(z_)
    cam.rotation_euler = Matrix((x_, y_, z_)).transposed().to_euler()
    cams[v] = cam

# the grid: floor and two walls behind the character, 10 cm lines and a heavier one every metre
span = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])), float(hi[2] - lo[2])) * 1.6 + 1.0


def grid(name, size, loc, rot, step, thick, colour, strength):
    n = max(2, int(round(size / step)))
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=n, y_subdivisions=n, size=n * step, location=loc, rotation=rot)
    g = bpy.context.active_object
    g.name = name
    w = g.modifiers.new("wire", "WIREFRAME")
    w.thickness = thick
    w.use_replace = True
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = colour
    em.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs[0])
    g.data.materials.append(mat)
    return g


# a wall behind the character from each of the four side-on cameras, shown only in the view it stands behind
walls = {}
for az_ in (0.0, 90.0, 180.0, -90.0):
    rr = math.radians(a.yaw + az_)
    u = Vector((math.sin(rr), -math.cos(rr), 0.0))                  # toward that camera
    far = float(np.max((P - np.array(ctr)) @ np.array(-u))) + 0.35
    loc = ctr - u * far
    loc.z = floor + span / 2
    for step, thick, col, s_ in ((0.1, 0.0022, (0.30, 0.34, 0.40, 1), 0.7), (1.0, 0.006, (0.55, 0.60, 0.68, 1), 0.7)):
        g = grid(f"wall_{int(az_)}_{step}", span, loc, (math.radians(90), 0, rr), step, thick, col, s_)
        walls.setdefault(az_, []).append((g, u))
for step, thick, col, s_ in ((0.1, 0.0022, (0.30, 0.34, 0.40, 1), 1.0), (1.0, 0.006, (0.55, 0.60, 0.68, 1), 1.0)):
    grid(f"floor_{step}", span, Vector((ctr.x, ctr.y, floor - 0.001)), (0, 0, 0), step, thick, col, s_)


def show_walls(view):
    """Only the wall the camera looks at; none from above."""
    d = view_dirs(view)[0]
    best = max(walls, key=lambda k: walls[k][0][1].dot(d))
    for k, lst in walls.items():
        for g, u in lst:
            g.hide_render = not (view != "top" and k == best and u.dot(d) > 0.7)


engines = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
sc.render.resolution_x = sc.render.resolution_y = a.res
sc.render.film_transparent = False
if "AgX" in {t.identifier for t in sc.view_settings.bl_rna.properties["view_transform"].enum_items}:
    sc.view_settings.view_transform = "AgX"
w = bpy.data.worlds.new("angles_world")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.06, 0.07, 0.085, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
for n, rot, e in (("key", (50, 0, 35), 3.2), ("fill", (60, 0, -60), 1.3), ("rim", (60, 0, 180), 1.8),
                  ("top", (5, 0, 0), 1.0)):
    lt = bpy.data.objects.new(n, bpy.data.lights.new(n, "SUN"))
    sc.collection.objects.link(lt)
    lt.data.energy = e
    lt.rotation_euler = tuple(math.radians(x) for x in rot)

# the readout: each elbow's and knee's bend, and each shoe's lowest point above the floor
pb = rig.pose.bones
M = rig.matrix_world


def bend(upper, lower):
    if upper not in pb or lower not in pb:
        return None
    u = (M @ pb[upper].tail) - (M @ pb[upper].head)
    f = (M @ pb[lower].tail) - (M @ pb[lower].head)
    c = u.normalized().dot(f.normalized())
    return round(math.degrees(math.acos(max(-1.0, min(1.0, c)))), 1)


foot_idx = {}
for s_ in ("left", "right"):
    names = {f"{s_}_ankle", f"{s_}_foot"}
    for o in meshes:
        vg = [g.index for g in o.vertex_groups if g.name in names]
        if vg:
            ids = [v.index for v in o.data.vertices if any(g.group in vg and g.weight > 0.5 for g in v.groups)]
            foot_idx.setdefault(s_, []).append((o, np.array(ids, dtype=np.int64)))

os.makedirs(a.out, exist_ok=True)
for v in views:
    os.makedirs(os.path.join(a.out, v.replace(":", "_")), exist_ok=True)
rows = []
for k, fr in enumerate(frames):
    sc.frame_set(fr)
    dg = bpy.context.evaluated_depsgraph_get()
    feet = {}
    for s_, lst in foot_idx.items():
        zs = []
        for o, ids in lst:
            ev = o.evaluated_get(dg)
            m = ev.to_mesh()
            V = np.empty(len(m.vertices) * 3)
            m.vertices.foreach_get("co", V)
            V = V.reshape(-1, 3)[ids]
            mw = np.array(o.matrix_world)
            zs.append((V @ mw[:3, :3].T + mw[:3, 3])[:, 2].min())
            ev.to_mesh_clear()
        feet[s_] = round(float(min(zs)) - floor, 3) if zs else None
    rows.append({"frame": fr, "t": round((fr - f0) / fps, 3),
                 "elbow": {"left": bend("left_shoulder", "left_elbow"), "right": bend("right_shoulder", "right_elbow")},
                 "knee": {"left": bend("left_hip", "left_knee"), "right": bend("right_hip", "right_knee")},
                 "foot": feet})
    for v in views:
        show_walls(v)
        sc.camera = cams[v]
        sc.render.filepath = os.path.join(a.out, v.replace(":", "_"), f"f{k:04d}.png")
        bpy.ops.render.render(write_still=True)
json.dump({"clip": a.clip, "views": views, "res": a.res, "fps_out": fps / every, "metres_per_px": scale / a.res,
           "yaw": a.yaw, "frames": rows}, open(os.path.join(a.out, "angles.json"), "w"), indent=0)
print(f"[angles] {len(frames)} frames x {len(views)} views ({', '.join(views)}), {scale / a.res * 100:.2f} cm a pixel "
      f"-> {a.out}", flush=True)
