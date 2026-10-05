"""The eyes as parts (E140): where each of the generated model's eye openings is, for solidify.py to rebuild.

    blender -b -noaudio --python eye_fill.py -- render --mesh mesh.glb --out eyes/front.png --cam eyes/cam.json
    blender -b -noaudio --python eye_fill.py -- locate --mesh mesh.glb --cam eyes/cam.json --marks eyes/marks.json \
        --style realistic --out eyes/eyes.json

TRELLIS.2 models an eye as an almond opening into the head with a broken surface inside it - shards, holes and
stray flaps where the eyeball should be (Mara, Juno, Rowan, Knight, Cadet, Aoi and Kaito alike). The solid filled
the opening with streaks, collapse decimation crumpled it into a pit, and no texture laid on a pit reads as an eye.
A professional head has an eyeball behind the lids; pipeline/solidify.py --eyes puts one there, from this.

render: the model from the front in its own colours (orthographic, the whole figure, 4096 px), and the camera, so
pipeline/eye_marks.py can find the eyes on it - on the generator's own paint, which lies where its geometry has the
eyes (DWPose on a grey render of a drawn face guesses; it put Pip's mouth under his nose).

locate: each eye's outline from the marks (DWPose's six points), as an ellipse on the front plane, and the lids'
depth at 16 points on a ring just outside it, kept to hits near the cheek below the eye (a fringe in front of the
eye is not a lid); solidify.py lays a cap across the opening from them (SHAPE). The mesh is not changed: a ball added
to it was voxelised whole and its sides bulged out of the face past the corners.

Only eyes DWPose reads on the model itself are rebuilt. Two other ways to find them failed: the picture's marks
carried onto the model by its head silhouette (Aoi's landed high, and her model's openings are twice her picture's
eyes), and the openings as pits in the depth (the shards fill a realistic opening to the lids' depth, and an anime
fringe breaks the face up). An anime face is drawn, its eyes painted on, and is left as generated when unread.
"""
import argparse
import json
import math
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("render", "locate"))
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cam", required=True, help="render writes it, fill reads it")
ap.add_argument("--res", type=int, default=4096)
ap.add_argument("--marks", default=None, help="locate: pipeline/eye_marks.py's eyes")
ap.add_argument("--style", default="realistic", choices=("realistic", "stylized", "anime"))
a = ap.parse_args(argv)

# eyeball radius and how far its front sits behind the lids, as fractions of the opening's width. A human eye's
# opening is ~30 mm wide and its ball 12 mm in radius (0.4); a drawn eye is larger and flatter.
# The eye is a cap across the opening: it follows the lids' own depth round the rim, bulges forward by `bulge` of
# the opening's width at the middle - an eyeball under the lids - and sinks by `crease` just inside the rim, so the
# lids' edge reads. A ball set behind the opening (0.6 of the width, its front 0.1 behind the lids) left a slot with
# hard ledges that rendered as dark rings, goggles round Mara's eyes. A drawn eye is flatter.
SHAPE = {"realistic": (0.025, 0.02), "stylized": (0.02, 0.015), "anime": (0.01, 0.0)}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.mesh)
sc = bpy.context.scene
obj = next(o for o in sc.objects if o.type == "MESH")
me = obj.data
mw = np.array(obj.matrix_world)
nv = len(me.vertices)
co = np.empty(nv * 3)
me.vertices.foreach_get("co", co)
W = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
lo, hi = W.min(0), W.max(0)

if a.mode == "render":
    ctr = (lo + hi) / 2
    os_ = float(hi[2] - lo[2]) * 1.02
    for slot in obj.material_slots:
        nt = slot.material.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        lk = bsdf.inputs["Base Color"].links if bsdf else []
        if lk:
            em = nt.nodes.new("ShaderNodeEmission")
            nt.links.new(lk[0].from_socket, em.inputs["Color"])
            outn = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
            nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 4
    sc.render.resolution_x = sc.render.resolution_y = a.res
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "Standard"
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = os_
    cam.location = Vector((ctr[0], lo[1] - 4.0, ctr[2]))
    cam.rotation_euler = (math.radians(90), 0, 0)               # looking along +Y, Z up
    sc.render.filepath = a.out
    bpy.ops.render.render(write_still=True)
    json.dump({"cx": float(ctr[0]), "cz": float(ctr[2]), "ortho": os_, "res": a.res, "y_front": float(lo[1]),
               "height": float(hi[2] - lo[2])}, open(a.cam, "w"), indent=1)
    print(f"[eyes] front render {a.res} px -> {a.out}", flush=True)
    sys.exit(0)

C = json.load(open(a.cam))
M = json.load(open(a.marks))


def to_xz(p):
    u, v = p
    return np.array([C["cx"] + (u / C["res"] - 0.5) * C["ortho"], C["cz"] - (v / C["res"] - 0.5) * C["ortho"]])


bm = bmesh.new()
bm.from_mesh(me)
bm.transform(Matrix(mw.tolist()))
tree = BVHTree.FromBMesh(bm)
y0 = float(lo[1]) - 0.5


def hits(x, z):
    """Depths (y) of every surface a ray from the front meets at (x, z), front first."""
    out, o, d = [], Vector((x, y0, z)), Vector((0, 1, 0))
    for _ in range(16):
        loc, _n, idx, _d = tree.ray_cast(o, d)
        if loc is None:
            break
        out.append((loc.y, idx))
        o = loc + d * 1e-5
    return out


report = {"style": a.style, "eyes": []}
for side in ("left", "right"):
    e = M.get(side)
    if not e or e.get("score", 0) < M.get("min_score", 0.5):
        report["eyes"].append({"side": side, "used": False, "why": "not found"})
        continue
    P = np.array([to_xz(p) for p in e["points"]])            # 6 points: corner, upper lid x2, corner, lower lid x2
    m = P.mean(0)
    ax = P[3] - P[0]
    w = float(np.linalg.norm(ax))
    ax /= w
    nrm = np.array([-ax[1], ax[0]])
    b = float(np.mean(np.abs((P[[1, 2, 4, 5]] - m) @ nrm)))  # half the opening's height
    cheek = hits(m[0], m[1] - 0.55 * w)
    if not cheek:
        report["eyes"].append({"side": side, "used": False, "why": "no surface below the eye"})
        continue
    y_ck = cheek[0][0]
    ring, ring_at = [], np.full(16, np.nan)
    ring_ab = (0.5 * w * 1.25, b * 1.25 + 0.06 * w)                 # the ellipse the lids are sampled on
    for k_, t in enumerate(np.linspace(0, 2 * math.pi, 16, endpoint=False)):
        q = m + ax * ring_ab[0] * math.cos(t) + nrm * ring_ab[1] * math.sin(t)
        hs = [h for h, _ in hits(*q) if abs(h - y_ck) < 0.35 * w]
        if hs:
            ring.append(min(hs, key=lambda h: abs(h - y_ck)))
            ring_at[k_] = ring[-1]
    if len(ring) < 6:
        report["eyes"].append({"side": side, "used": False, "why": f"lids found at {len(ring)} of 16 points"})
        continue
    y_lid = float(np.median(ring))
    R, cen = 0.6 * w, np.array([m[0], y_lid + 0.6 * w, m[1]])     # for the report: an eyeball's size and place
    real = float(C["height"])
    # the lids' depth all round, the gaps filled from their neighbours round the ring
    okr = np.isfinite(ring_at)
    idx_ = np.arange(16)
    ring_at = np.interp(idx_, idx_[okr], ring_at[okr], period=16)
    report["eyes"].append({"side": side, "used": True, "centre_xz": [float(m[0]), float(m[1])],
                           "ring_depth": [round(float(x), 6) for x in ring_at], "ring_axes": [float(x) for x in ring_ab],
                           "bulge": SHAPE[a.style][0], "crease": SHAPE[a.style][1],
                           "axis_xz": [float(ax[0]), float(ax[1])], "ball": [float(x) for x in cen],
                           "width": round(w, 5), "half_height": round(b, 5),
                           "lid_depth": round(y_lid, 5), "cheek_depth": round(y_ck, 5), "radius": round(R, 5),
                           "ring_hits": len(ring),
                           "width_of_height_pct": round(100 * w / real, 2), "score": e.get("score")})
    print(f"[eyes] {side}: opening {w / real:.2%} of the height wide, {2 * b / w:.2f} as tall, lids found at "
          f"{len(ring)}/16 round it", flush=True)

bm.free()
json.dump(report, open(a.out, "w"), indent=1)
print(f"[eyes] {sum(e['used'] for e in report['eyes'])} eye(s) located -> {a.out}", flush=True)
