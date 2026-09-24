"""Retopologise a generated mesh to a quad cage and bake the original detail onto it.

Generated meshes are marching-cubes triangle soup: ~200k unstructured triangles with no edge
loops. Games want a quad-dominant cage at a fraction of that count, with the lost detail baked
into a normal map. That is the largest single step from "generated asset" toward "game asset",
and it is fully deterministic - QuadriFlow for the cage, Cycles for the bake.

Run: blender -b -noaudio --python retopo.py -- --mesh high.glb --out low.glb
     [--faces 30000] [--bake-res 2048]
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True, help="textured mesh - albedo source and fallback normal source")
ap.add_argument("--highres", default=None, help="pre-simplification mesh, used as the normal-bake source")
ap.add_argument("--out", required=True)
ap.add_argument("--faces", type=int, default=30000, help="target quad count")
ap.add_argument("--bake-res", type=int, default=2048)
ap.add_argument("--cage-extrusion", type=float, default=0.08)
ap.add_argument("--ray-distance", type=float, default=0.12)
ap.add_argument("--no-normal-map", action="store_true", help="bake the normal map but leave it unwired")
ap.add_argument("--uv-smooth", type=int, default=20,
                help="smoothing iterations on the twin the UVs are unwrapped from (see the table)")
ap.add_argument("--ao-distance", type=float, default=0.03,
                help="ambient occlusion reach as a fraction of character height")
ap.add_argument("--ao-samples", type=int, default=48)
ap.add_argument("--no-normal-median", dest="normal_median", action="store_false",
                help="skip the selective median on the baked normal map")
ap.add_argument("--joints", default=None,
                help="skeleton.py's joints.json - used to cut apart legs the voxel remesh fused")
ap.add_argument("--legs-dry-run", action="store_true",
                help="report fused leg faces without cutting them")
ap.add_argument("--keep-winding", action="store_true",
                help="skip the winding repair on the source mesh (for A/B comparison only)")
ap.add_argument("--cage", default=None,
                help="a clean solid to decimate into the low-poly (pipeline/solidify.py + hands.py) "
                     "instead of voxel-remeshing the generated mesh; also the normal and AO source")
ap.add_argument("--tris", type=int, default=60000, help="triangle budget when decimating --cage")
a = ap.parse_args(argv)


def import_one(path):
    """Import one file and join only what it brought in - not everything already in the scene."""
    before = {o.name for o in bpy.context.scene.objects}
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    else:
        raise SystemExit(f"unsupported mesh format: {path}")
    ms = [o for o in bpy.context.scene.objects if o.name not in before and o.type == "MESH"]
    if not ms:
        raise SystemExit(f"no mesh imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for m in ms:
        m.select_set(True)
    bpy.context.view_layer.objects.active = ms[0]
    if len(ms) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def fix_winding(ob):
    """Flip exactly the faces that are on the outside of the character but wound inward.

    A bake ray arrives from outside and lands on the outermost surface. Where that face is
    wound backwards, its normal points into the body, the baker records an inverted normal, and
    the repair step flattens it to "no detail". Measured by casting rays both ways off every
    face, area-weighted:

        character   exterior, correct   exterior, BACKWARDS   interior (never seen)
        Rowan             84.0%               1.0%                 14.2%
        Wren              36.0%              23.7%                 39.8%

    which is why 46% of Wren's normal map carried no detail while Rowan's was fine.

    The obvious repair - recalc_face_normals - was tried and made it worse (repairs 27% -> 42%,
    ambient occlusion collapsing to black). It decides "outside" from connectivity, and these
    are leaky double-walled shells with tens of thousands of non-manifold edges, so it turned
    most of the outer wall inward. Asking each face directly which of its sides can see out
    involves no heuristic: flip a face only when its back escapes and its front does not."""
    import bmesh
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    me = ob.data
    bvh = BVHTree.FromObject(ob, bpy.context.evaluated_depsgraph_get())
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    ext = float(np.linalg.norm(np.ptp(V.reshape(-1, 3), axis=0)))
    eps, reach = ext * 2e-4, ext * 1.5
    C = np.empty(len(me.polygons) * 3)
    me.polygons.foreach_get("center", C)
    N = np.empty(len(me.polygons) * 3)
    me.polygons.foreach_get("normal", N)
    C, N = C.reshape(-1, 3), N.reshape(-1, 3)
    flip = []
    for i in range(len(C)):
        c, n = Vector(C[i]), Vector(N[i])
        if bvh.ray_cast(c + n * eps, n, reach)[0] is None:
            continue                                    # front sees out: correct already
        if bvh.ray_cast(c - n * eps, -n, reach)[0] is None:
            flip.append(i)                              # only the back sees out: wound inward
    if flip:
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        bmesh.ops.reverse_faces(bm, faces=[bm.faces[i] for i in flip], flip_multires=False)
        bm.to_mesh(me)
        bm.free()
        me.update()
    return len(flip)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    high = import_one(a.mesh)
    high.name = "high"
    hi_tris = len(high.data.polygons)
    if not a.keep_winding:
        n = fix_winding(high)
        print(f"[retopo] source winding: {n:,} of {len(high.data.polygons):,} faces "
              f"({100*n/max(1,len(high.data.polygons)):.1f}%) turned to face outward", flush=True)

    # The pre-simplification mesh carries detail the exported GLB already threw away, so it is
    # the right normal-bake source. It has no texture, hence two sources: albedo from the
    # textured mesh, normals from the dense one.
    dense = None
    if a.cage:
        a.highres = a.cage
    if a.highres and os.path.exists(a.highres):
        dense = import_one(a.highres)
        dense.name = "dense"
        if not a.keep_winding and not a.cage:        # a solid from solidify.py winds correctly
            fix_winding(dense)
        bpy.ops.object.select_all(action="DESELECT")
        dense.select_set(True)
        bpy.context.view_layer.objects.active = dense
        bpy.ops.object.shade_smooth()
        print(f"[retopo] normal source: {len(dense.data.polygons):,} faces from {a.highres}", flush=True)

    if a.cage:
        # The low-poly is the clean solid, decimated. Collapse decimation keeps triangles where
        # the surface turns - face, hands, hair parting, folds - and spends few on flat fabric;
        # on Juno the head kept 14% of the budget with no weighting at all. The voxel remesh this
        # replaces spread a uniform ~1.2 cm grid over everything and blurred faces and hands.
        bpy.ops.object.select_all(action="DESELECT")
        dense.select_set(True)
        bpy.context.view_layer.objects.active = dense
        bpy.ops.object.duplicate()
        low = bpy.context.view_layer.objects.active
        low.name = "low"
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(low.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bm.to_mesh(low.data)
        bm.free()
        tris0 = sum(len(p.vertices) - 2 for p in low.data.polygons)
        md = low.modifiers.new("dec", "DECIMATE")
        md.decimate_type = "COLLAPSE"
        md.ratio = min(1.0, a.tris / max(1, tris0))
        md.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=md.name)
        # decimation can leave slivers and the odd non-manifold edge; skinning solvers and the
        # UV packer both choke on those, so repair until clean
        bm = bmesh.new()
        bm.from_mesh(low.data)
        for _ in range(3):
            bmesh.ops.dissolve_degenerate(bm, dist=1e-7, edges=bm.edges[:])
            bad = [e for e in bm.edges if not e.is_manifold and not e.is_boundary]
            if bad:
                bmesh.ops.delete(bm, geom=list({f for e in bad for f in e.link_faces}), context="FACES")
            loose = [v for v in bm.verts if not v.link_faces]
            if loose:
                bmesh.ops.delete(bm, geom=loose, context="VERTS")
            bnd = [e for e in bm.edges if e.is_boundary]
            if bnd:
                r = bmesh.ops.holes_fill(bm, edges=bnd, sides=0)
                bmesh.ops.triangulate(bm, faces=r["faces"])
        nm = sum(1 for e in bm.edges if not e.is_manifold)
        bm.to_mesh(low.data)
        bm.free()
        method = "solid_decimate"
        print(f"[retopo] cage: {tris0:,} triangles of clean solid -> {len(low.data.polygons):,} "
              f"({nm} non-manifold edges left)", flush=True)
    else:
        # duplicate -> remesh the copy, keep the original as the bake source
        bpy.ops.object.select_all(action="DESELECT")
        high.select_set(True)
        bpy.context.view_layer.objects.active = high
        bpy.ops.object.duplicate()
        low = bpy.context.view_layer.objects.active
        low.name = "low"

        bpy.context.view_layer.objects.active = low
        method = None
        try:
            bpy.ops.object.quadriflow_remesh(target_faces=a.faces, use_preserve_sharp=False,
                                              use_preserve_boundary=False, smooth_normals=True)
            if len(low.data.polygons) < hi_tris * 0.9:
                method = "quadriflow"
        except RuntimeError as e:
            print(f"[retopo] quadriflow raised ({e})")
        if method is None:
            # QuadriFlow needs a manifold surface and returns silently on generated meshes
            # (this one carries ~78k non-manifold edges). Voxel remesh does not care: it
            # rebuilds the surface from a signed distance field and outputs quads, which is
            # the right trade here because the lost detail is going into a normal map anyway.
            print(f"[retopo] quadriflow left {len(low.data.polygons)} polys; using voxel remesh")
            dims = max(low.dimensions)
            size = dims * 0.012
            # poly count scales roughly with 1/size^2, so step toward the target in both
            # directions rather than only shrinking
            for attempt in range(5):
                bpy.ops.object.select_all(action="DESELECT")
                low.select_set(True)
                bpy.context.view_layer.objects.active = low
                low.data.remesh_voxel_size = size
                low.data.remesh_voxel_adaptivity = 0.0
                bpy.ops.object.voxel_remesh()
                n = len(low.data.polygons)
                print(f"    voxel {size:.5f} -> {n} polys", flush=True)
                if 0.75 * a.faces <= n <= 1.3 * a.faces:
                    break
                size = size * (n / a.faces) ** 0.5
                if attempt == 4:
                    break
            if len(low.data.polygons) > a.faces * 1.1:
                md = low.modifiers.new("dec", "DECIMATE")
                md.decimate_type = "COLLAPSE"
                md.ratio = min(1.0, a.faces / max(1, len(low.data.polygons)))
                bpy.ops.object.modifier_apply(modifier=md.name)
            method = "voxel_remesh"

    # ---- legs the remesh fused ----------------------------------------------------------------
    # The voxel remesh rebuilds the surface from a distance field about 1 cm per voxel, so two
    # surfaces closer than that become one. Vex stands with her knees nearly touching: the remesh
    # welded her legs together at the knee, the rig split that weld between left and right, and
    # every stride stretched it into a web between her shins. The skeleton is estimated before
    # this stage, so each remeshed vertex below mid-thigh is given to the leg whose axis it is
    # nearest; a face with vertices on both legs spans the weld and is cut, and the openings are
    # closed - before UVs and baking, so the patch is textured like the leg around it. If the cut
    # would be large it is not a weld but a skirt or a coat hem, and nothing is cut.
    if a.joints and os.path.exists(a.joints):
        import bmesh
        J = json.load(open(a.joints))["joints"]
        HW = np.array([(high.matrix_world @ v.co)[:] for v in high.data.vertices])
        blo, bhi = HW.min(0), HW.max(0)
        kk = 2.0 / float(bhi[2] - blo[2])               # skeleton.py's frame: 2 units tall, centred
        cc = (blo + bhi) / 2
        jp = lambda n: np.array(J[n]) / kk + cc
        axes = {}
        for s_ in ("left", "right"):
            h_, k_, an_ = jp(f"{s_}_hip"), jp(f"{s_}_knee"), jp(f"{s_}_ankle")
            floor_ = np.array([an_[0], an_[1], blo[2]])
            axes[s_] = [(h_, k_), (k_, an_), (an_, floor_)]
        zcut = min((jp(f"{s_}_hip")[2] + jp(f"{s_}_knee")[2]) / 2 for s_ in ("left", "right"))

        def seg_d(X, p0, p1):
            ab = p1 - p0
            t = np.clip(((X - p0) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)[:, None]
            return np.linalg.norm(X - (p0 + t * ab), axis=1)

        mw = np.array(low.matrix_world)
        V = np.array([v.co[:] for v in low.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
        dL = np.min([seg_d(V, *sg) for sg in axes["left"]], axis=0)
        dR = np.min([seg_d(V, *sg) for sg in axes["right"]], axis=0)
        side = (dR < dL).astype(np.int8)
        below = V[:, 2] < zcut
        bm = bmesh.new()
        bm.from_mesh(low.data)
        bm.faces.ensure_lookup_table()
        n_below = sum(1 for f in bm.faces if all(below[v.index] for v in f.verts))
        weld = [f for f in bm.faces if all(below[v.index] for v in f.verts)
                and len({int(side[v.index]) for v in f.verts}) == 2]
        share = len(weld) / max(n_below, 1)
        if not weld:
            print("[retopo] legs: separate below mid-thigh, nothing to cut", flush=True)
        elif share > 0.02:
            print(f"[retopo] legs: {len(weld):,} faces span both legs ({share:.1%} of the legs) - "
                  "too many for a weld; taking it to be a skirt or coat and cutting nothing", flush=True)
        elif a.legs_dry_run:
            print(f"[retopo] legs: {len(weld):,} faces span both legs (dry run - not cut)", flush=True)
        else:
            # the vertices the weld touched, before it goes, so the scar can be smoothed after
            scar = {v for f in weld for v in f.verts}
            bmesh.ops.delete(bm, geom=weld, context="FACES")
            edges = [e for e in bm.edges if e.is_boundary]
            filled = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
            bmesh.ops.triangulate(bm, faces=filled["faces"])
            # Each leg keeps half of the old weld's saddle, which flares toward the other leg and
            # rides along as a fin. Smoothing the scar and three rings around it lays the fin
            # back into the leg; nothing else on the mesh moves.
            ring = {v for v in scar if v.is_valid} | {v for f in filled["faces"] for v in f.verts}
            for _ in range(3):
                ring |= {e.other_vert(v) for v in list(ring) for e in v.link_edges}
            bmesh.ops.smooth_vert(bm, verts=list(ring), factor=0.5, use_axis_x=True,
                                  use_axis_y=True, use_axis_z=True)
            for _ in range(11):
                bmesh.ops.smooth_vert(bm, verts=list(ring), factor=0.5, use_axis_x=True,
                                      use_axis_y=True, use_axis_z=True)
            bm.to_mesh(low.data)
            low.data.update()
            print(f"[retopo] legs: cut {len(weld):,} faces where the remesh had welded the legs "
                  f"together, closed {len(filled['faces']):,} openings", flush=True)
        bm.free()

    # Smooth shading before baking. On a flat-shaded cage every face carries its own tangent
    # basis, so the baked normal map encodes the facet angle rather than surface detail - which
    # is why the first attempt produced a map with a mean blue channel of 133 and rendered as
    # blown-out specular streaks.
    bpy.ops.object.select_all(action="DESELECT")
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.shade_smooth()

    # Fresh UVs for the baked maps, unwrapped on a smoothed twin and copied back.
    #
    # smart_project cuts wherever the surface normal turns sharply, and a voxel-remeshed surface
    # turns sharply everywhere at the scale of one voxel. Unwrapped directly it produced 2,746
    # islands, 2,479 of them under ten faces, filling 20% of the atlas - the texture was mostly
    # empty, and every tiny island bled into its neighbours at each mip level. Smoothing a copy
    # first removes the voxel bumps without changing topology, so the loop order still matches
    # and the UVs transfer one-to-one. Measured on Wren, texels per centimetre of surface:
    #
    #     layout                       islands  coverage   worst 5%   median
    #     shipped                        2,746     0.20       9.09     10.19
    #     repacked only                  2,746     0.32      11.57     12.98
    #     smoothed x20 + repacked          586     0.49      10.00     16.34   <- this
    #     smoothed x60 + repacked          439     0.51       8.93     16.68
    #     smoothed x150 + repacked         416     0.48       7.10     15.90
    #
    # Past 20 iterations the smoothing distorts thin features enough that the worst regions
    # drop below what shipped, so more is not better. At 20, nothing gets worse and the median
    # surface gets 60% more texels from the same file.
    import bmesh
    twin = low.copy()
    twin.data = low.data.copy()
    bpy.context.collection.objects.link(twin)
    bm = bmesh.new()
    bm.from_mesh(twin.data)
    for _ in range(a.uv_smooth):
        bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=0.5,
                              use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(twin.data)
    bm.free()
    bpy.ops.object.select_all(action="DESELECT")
    twin.select_set(True)
    bpy.context.view_layer.objects.active = twin
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.002)
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(udim_source="CLOSEST_UDIM", rotate=True, rotate_method="ANY",
                            scale=True, merge_overlap=False, margin_method="SCALED",
                            margin=0.002, shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")
    if not low.data.uv_layers:
        low.data.uv_layers.new(name="UVMap")
    src_uv, dst_uv = twin.data.uv_layers.active.data, low.data.uv_layers.active.data
    if len(src_uv) != len(dst_uv):
        raise SystemExit(f"[retopo] UV transfer mismatch: {len(src_uv)} vs {len(dst_uv)} loops")
    buf = np.empty(len(src_uv) * 2, np.float32)
    src_uv.foreach_get("uv", buf)
    dst_uv.foreach_set("uv", buf)
    bpy.data.objects.remove(twin, do_unlink=True)
    uv_cov = 0.0
    for p in low.data.polygons:
        L = [dst_uv[li].uv for li in p.loop_indices]
        for k in range(1, len(L) - 1):
            x0, x1, x2 = L[0], L[k], L[k + 1]
            uv_cov += abs((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x2[0] - x0[0]) * (x1[1] - x0[1])) / 2
    print(f"[retopo] UVs: smoothed x{a.uv_smooth} unwrap, concave repack, "
          f"atlas coverage {uv_cov:.3f}", flush=True)
    bpy.ops.object.select_all(action="DESELECT")
    low.select_set(True)
    bpy.context.view_layer.objects.active = low

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    try:
        sc.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
    except Exception as e:
        print(f"[retopo] Metal device setup skipped ({e})")
    sc.cycles.samples = 8
    sc.cycles.use_denoising = False
    sc.render.bake.use_selected_to_active = True
    sc.render.bake.cage_extrusion = a.cage_extrusion
    sc.render.bake.use_cage = False
    sc.render.bake.margin = 8

    mat = bpy.data.materials.new("baked")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    low.data.materials.clear()
    low.data.materials.append(mat)

    images = {}
    # base_color and normal were the only passes until the adversarial review. TRELLIS.2 also
    # emits a metallic-roughness map, and the bake threw it away and shipped a constant 0.5 -
    # for Rowan the generator had asked for 0.92 (near-matte cloth), and Wren's leather, denim,
    # skin and buckles were all flattened to the same sheen. "rm" carries that data through;
    # "ao" adds the ambient occlusion every shipped game character carries, baked from the
    # high-resolution surface so clothing creases read without any lighting.
    FILL = {"base_color": (0.35, 0.33, 0.30, 1.0), "normal": (0.5, 0.5, 1.0, 1.0),
            "rm": (1.0, 0.9, 0.0, 1.0), "ao": (1.0, 1.0, 1.0, 1.0)}
    for pass_name, colorspace, noncolor in (("base_color", "sRGB", False),
                                            ("normal", "Non-Color", True),
                                            ("rm", "Non-Color", True),
                                            ("ao", "Non-Color", True)):
        im = bpy.data.images.new(f"{pass_name}", a.bake_res, a.bake_res,
                                 alpha=False, float_buffer=False, is_data=noncolor)
        # Pre-fill, then bake without clearing. A ray that misses leaves its texel untouched;
        # left black, a normal texel decodes to a garbage direction and lights up as a white
        # specular streak, and a black ao texel is a hole. The fills are the harmless defaults.
        fill = FILL[pass_name]
        buf = np.tile(np.array(fill, dtype=np.float32), a.bake_res * a.bake_res)
        im.pixels.foreach_set(buf)
        images[pass_name] = im
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = im
        node.name = pass_name
        node.select = False

    def bake(pass_name, bake_type, source):
        node = nt.nodes[pass_name]
        for n in nt.nodes:
            n.select = False
        node.select = True
        nt.nodes.active = node
        bpy.ops.object.select_all(action="DESELECT")
        source.select_set(True)
        low.select_set(True)
        bpy.context.view_layer.objects.active = low
        # Voxel remesh smooths concavities, so the low-poly can sit well away from the
        # original surface; too small a cage makes those rays miss and the texel bakes white.
        kw = {"type": bake_type, "use_selected_to_active": True, "use_clear": False,
              "cage_extrusion": a.cage_extrusion, "max_ray_distance": a.ray_distance,
              "margin": 16, "margin_type": "ADJACENT_FACES"}
        if bake_type == "DIFFUSE":
            kw.update({"pass_filter": {"COLOR"}})
        if bake_type == "EMIT":
            kw.pop("pass_filter", None)
        bpy.ops.object.bake(**kw)
        print(f"[retopo] baked {pass_name}", flush=True)

    # Bake albedo as EMIT, not DIFFUSE. TRELLIS writes a metallic channel, and a metallic
    # surface has no diffuse component, so a DIFFUSE bake returns garbage exactly where the
    # jacket is most metallic. Re-wiring the source material to emit its own base colour
    # bakes the texture itself, with no shading in the result.
    def emit_bake(pass_name, colour_of):
        """Rewire every source material to emit colour_of(node_tree, bsdf) -> socket, bake it,
        then restore the original shading. Emitting a value bakes the value itself, with no
        lighting in it - the same trick that keeps metallic regions out of the albedo bake."""
        saved = []
        for m in list(high.data.materials):
            if not m or not m.use_nodes:
                continue
            nt_h = m.node_tree
            out_node = next((n for n in nt_h.nodes if n.type == "OUTPUT_MATERIAL"), None)
            pbsdf = next((n for n in nt_h.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if out_node is None or pbsdf is None:
                continue
            link = out_node.inputs["Surface"].links[0] if out_node.inputs["Surface"].links else None
            saved.append((nt_h, out_node, link.from_socket if link else None))
            emit = nt_h.nodes.new("ShaderNodeEmission")
            emit.name = "_bake_emit"
            nt_h.links.new(colour_of(nt_h, pbsdf), emit.inputs["Color"])
            nt_h.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])
        bake(pass_name, "EMIT", high)
        for nt_h, out_node, from_sock in saved:        # restore the original shading
            if from_sock is not None:
                nt_h.links.new(from_sock, out_node.inputs["Surface"])
            for name in ("_bake_emit", "_bake_rgb", "_bake_combine"):
                n = nt_h.nodes.get(name)
                if n:
                    nt_h.nodes.remove(n)

    def socket_or_constant(nt_h, sock, name):
        """The socket feeding `sock`, or an RGB node holding its constant value."""
        if sock.links:
            return sock.links[0].from_socket
        rgb = nt_h.nodes.new("ShaderNodeRGB")
        rgb.name = name
        v = sock.default_value
        v = (v, v, v, 1.0) if isinstance(v, float) else tuple(v)
        rgb.outputs[0].default_value = v
        return rgb.outputs[0]

    def base_colour(nt_h, pbsdf):
        return socket_or_constant(nt_h, pbsdf.inputs["Base Color"], "_bake_rgb")

    def rough_metal(nt_h, pbsdf):
        # glTF packs roughness in G and metallic in B; bake straight into that layout
        comb = nt_h.nodes.new("ShaderNodeCombineColor")
        comb.name = "_bake_combine"
        comb.inputs[0].default_value = 1.0
        for chan, key in ((1, "Roughness"), (2, "Metallic")):
            s = pbsdf.inputs[key]
            if s.links:
                nt_h.links.new(s.links[0].from_socket, comb.inputs[chan])
            else:
                comb.inputs[chan].default_value = float(s.default_value)
        return comb.outputs[0]

    emit_bake("base_color", base_colour)
    emit_bake("rm", rough_metal)

    bake("normal", "NORMAL", dense or high)

    # Ambient occlusion from the high-resolution surface. The distance is local on purpose: at
    # the default of 10 units every point on the torso "sees" the arms and the whole body goes
    # grey. A few percent of body height captures creases, the underside of collars and the
    # join between jacket and trousers, which is what AO is for.
    height = float(max(low.dimensions))
    sc.world = sc.world or bpy.data.worlds.new("bake_world")
    sc.world.light_settings.distance = height * a.ao_distance
    prev_samples = sc.cycles.samples
    sc.cycles.samples = a.ao_samples

    def ao_baked_mean():
        """Mean occlusion over texels the bake actually wrote (the fill is exactly 1.0)."""
        v = np.empty(a.bake_res * a.bake_res * 4, np.float32)
        images["ao"].pixels.foreach_get(v)
        v = v[0::4]
        w = v < 0.9999
        return float(v[w].mean()) if w.any() else 1.0, float(w.mean())

    # The target itself must not occlude. A voxel-remeshed cage sat a centimetre off the source,
    # but a decimated solid lies within a millimetre of it on both sides, and AO rays leaving the
    # source hit the cage straight away (mean 0.435 on Juno, CPU and GPU alike). Hidden from
    # every ray type, the cage is still the bake target.
    # Likewise the generated mesh when a separate solid is the AO source: it hugs the solid
    # within a millimetre on both sides and occludes it everywhere.
    hide = [low] + ([high] if dense is not None and high is not dense else [])
    for ob_ in hide:
        for flag in ("visible_diffuse", "visible_glossy", "visible_shadow", "visible_transmission",
                     "visible_volume_scatter", "visible_camera"):
            if hasattr(ob_, flag):
                setattr(ob_, flag, False)
    if dense is not None and high is not dense:
        high.hide_render = True
    bake("ao", "AO", dense or high)
    ao_mean, ao_cov = ao_baked_mean()
    # Cycles on Metal can page-fault mid-bake ("Caused GPU Address Fault Error") and return a
    # corrupted image rather than an error - it happened twice here, both times on this pass,
    # leaving a mean occlusion of 0.34 with the darkest 5% at pure black. A GPU fault does not
    # raise, so the only defence is to check the result: real character AO sits well above 0.5
    # on average. If it does not, the pass is redone on the CPU.
    if ao_mean < 0.5:
        print(f"[retopo] AO bake looks corrupted (mean {ao_mean:.3f} over {ao_cov:.1%} of the atlas) "
              f"- redoing it on the CPU", flush=True)
        images["ao"].pixels.foreach_set(np.ones(a.bake_res * a.bake_res * 4, np.float32))
        sc.cycles.device = "CPU"
        bake("ao", "AO", dense or high)
        sc.cycles.device = "GPU"
        ao_mean, ao_cov = ao_baked_mean()
        if ao_mean < 0.5:
            raise SystemExit(f"[retopo] AO still implausible on CPU (mean {ao_mean:.3f}); "
                             "refusing to ship it")
    print(f"[retopo] AO mean {ao_mean:.3f} over {ao_cov:.1%} of the atlas", flush=True)
    sc.cycles.samples = prev_samples

    # wire the baked maps into the material
    tex = nt.nodes["base_color"]
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    # The normal bake is written to disk but left unwired by default: on a voxel-remeshed
    # cage the baked tangent-space values come back wrong (mean blue channel 133 where a flat
    # normal map should be ~255) and EEVEE renders blown-out specular streaks from it. The
    # albedo bake is correct. Pass --wire-normal to connect it anyway.
    if not a.no_normal_map:
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(nt.nodes["normal"].outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    # Repair unbaked texels. Where a ray missed, the texel stays black, and a black normal
    # decodes to a garbage direction that renders as a white specular streak. Replacing those
    # with a flat normal is the standard bake cleanup; measured before/after below.
    nrm = images["normal"]
    px = np.empty(len(nrm.pixels), dtype=np.float32)
    nrm.pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    # A tangent-space normal must point outward, so blue (z) >= 0.5 in 0..1 encoding. Texels
    # below that are bake errors, not detail: the voxel cage sits inside the dense surface in
    # concave regions, so the ray hits a backface and returns an inward normal. Those render as
    # white specular streaks. Flatten them and keep the two thirds that are valid.
    dead = px[:, 2] < 0.5
    n_dead = int(dead.sum())
    px[dead] = (0.5, 0.5, 1.0, 1.0)
    nrm.pixels.foreach_set(px.reshape(-1))
    nrm.update()
    print(f"[retopo] normal map: repaired {n_dead:,} unbaked texels "
          f"({100*n_dead/len(px):.1f}%)", flush=True)

    # ---- selective median on the normal map -----------------------------------------------------
    # The bake faithfully records the source's own tessellation noise as isolated outlier texels,
    # which render as speckle. This used to be a one-off command typed into a terminal and
    # applied to Rowan by hand - Wren never got it. A 5x5 median replaces only texels that
    # disagree with their neighbourhood by more than 12 levels, so creases and seams survive;
    # the result is renormalised because a median of unit vectors is not a unit vector.
    if a.normal_median:
        from numpy.lib.stride_tricks import sliding_window_view
        R2 = a.bake_res
        nv = np.empty(R2 * R2 * 4, np.float32)
        images["normal"].pixels.foreach_get(nv)
        nv = nv.reshape(R2, R2, 4)
        rgb = nv[:, :, :3]
        med = np.empty_like(rgb)
        pad = np.pad(rgb, ((2, 2), (2, 2), (0, 0)), mode="edge")
        for r0 in range(0, R2, 256):                    # chunked: a full 5x5 window stack is ~5 GB
            r1 = min(R2, r0 + 256)
            win = sliding_window_view(pad[r0:r1 + 4], (5, 5), axis=(0, 1))
            med[r0:r1] = np.median(win.reshape(r1 - r0, R2, 3, 25), axis=3)
        swap = (np.abs(rgb - med).max(axis=2) > 12 / 255)[:, :, None]
        out = np.where(swap, med, rgb)
        n3 = out * 2.0 - 1.0
        n3 /= np.maximum(np.linalg.norm(n3, axis=2, keepdims=True), 1e-6)
        n3[:, :, 2] = np.maximum(n3[:, :, 2], 0.25)     # nothing inward-facing
        n3 /= np.maximum(np.linalg.norm(n3, axis=2, keepdims=True), 1e-6)
        nv[:, :, :3] = (n3 + 1.0) * 0.5
        images["normal"].pixels.foreach_set(nv.reshape(-1))
        images["normal"].update()
        print(f"[retopo] normal median: replaced {100*swap.mean():.2f}% of texels", flush=True)

    # ---- pack occlusion / roughness / metallic into one texture, glTF's layout ---------------
    R = a.bake_res
    ao = np.empty(R * R * 4, np.float32)
    images["ao"].pixels.foreach_get(ao)
    ao = ao.reshape(R, R, 4)[:, :, 0]
    # AO at a few dozen samples is grainy; it is also a low-frequency signal, so a small blur
    # removes the grain without softening anything that should be sharp.
    k = max(1, R // 1024)
    for axis in (0, 1):
        acc = np.zeros_like(ao)
        for d in range(-k, k + 1):
            acc += np.roll(ao, d, axis=axis)
        ao = acc / (2 * k + 1)
    rm = np.empty(R * R * 4, np.float32)
    images["rm"].pixels.foreach_get(rm)
    rm = rm.reshape(R, R, 4)
    orm_px = np.stack([ao, rm[:, :, 1], rm[:, :, 2], np.ones_like(ao)], axis=2)
    orm = bpy.data.images.new("orm", R, R, alpha=False, float_buffer=False, is_data=True)
    orm.pixels.foreach_set(orm_px.reshape(-1))
    orm.update()
    images["orm"] = orm
    used = np.asarray(orm_px[:, :, 1] < 0.999) | np.asarray(orm_px[:, :, 2] > 0.001)
    print(f"[retopo] ORM: ao mean {ao.mean():.3f} (p5 {np.percentile(ao,5):.3f}), "
          f"roughness p5/p50/p95 {np.percentile(rm[:,:,1],5):.2f}/{np.percentile(rm[:,:,1],50):.2f}/"
          f"{np.percentile(rm[:,:,1],95):.2f}, metallic max {rm[:,:,2].max():.2f}", flush=True)

    orm_node = nt.nodes.new("ShaderNodeTexImage")
    orm_node.image = orm
    orm_node.name = "orm"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(orm_node.outputs["Color"], sep.inputs[0])
    nt.links.new(sep.outputs[1], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs[2], bsdf.inputs["Metallic"])
    # The glTF exporter writes occlusionTexture from a node group named "glTF Material Output"
    # with an "Occlusion" input; wired from the same image, it shares one texture with
    # metallicRoughness instead of shipping a second file.
    grp = bpy.data.node_groups.get("glTF Material Output")
    if grp is None:
        grp = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
        grp.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
    gnode = nt.nodes.new("ShaderNodeGroup")
    gnode.node_tree = grp
    nt.links.new(sep.outputs[0], gnode.inputs["Occlusion"])
    for name in ("rm", "ao"):                         # intermediates, not shipped
        n = nt.nodes.get(name)
        if n:
            nt.nodes.remove(n)

    out_dir = os.path.dirname(os.path.abspath(a.out))
    for name, im in images.items():
        im.filepath_raw = os.path.join(out_dir, f"baked_{name}.png")
        im.file_format = "PNG"
        im.save()

    for obj in (high, dense):
        if obj is not None:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except ReferenceError:
                pass
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=True)

    quads = sum(1 for p in low.data.polygons if len(p.vertices) == 4)
    info = {"method": method, "high_tris": hi_tris, "low_polys": len(low.data.polygons),
            "quads": quads, "quad_fraction": round(quads / max(1, len(low.data.polygons)), 3),
            "vertices": len(low.data.vertices), "bake_res": a.bake_res,
            "reduction": round(1 - len(low.data.polygons) / max(1, hi_tris), 3)}
    json.dump(info, open(os.path.splitext(a.out)[0] + "_retopo.json", "w"), indent=2)
    print("[retopo] " + json.dumps(info))


main()
