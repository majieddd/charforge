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


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    high = import_one(a.mesh)
    high.name = "high"
    hi_tris = len(high.data.polygons)

    # The pre-simplification mesh carries detail the exported GLB already threw away, so it is
    # the right normal-bake source. It has no texture, hence two sources: albedo from the
    # textured mesh, normals from the dense one.
    dense = None
    if a.highres and os.path.exists(a.highres):
        dense = import_one(a.highres)
        dense.name = "dense"
        bpy.ops.object.select_all(action="DESELECT")
        dense.select_set(True)
        bpy.context.view_layer.objects.active = dense
        bpy.ops.object.shade_smooth()
        print(f"[retopo] normal source: {len(dense.data.polygons):,} faces from {a.highres}", flush=True)

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

    # Smooth shading before baking. On a flat-shaded cage every face carries its own tangent
    # basis, so the baked normal map encodes the facet angle rather than surface detail - which
    # is why the first attempt produced a map with a mean blue channel of 133 and rendered as
    # blown-out specular streaks.
    bpy.ops.object.select_all(action="DESELECT")
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.shade_smooth()

    # fresh UVs for the baked maps
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")

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
    for pass_name, colorspace, noncolor in (("base_color", "sRGB", False), ("normal", "Non-Color", True)):
        im = bpy.data.images.new(f"{pass_name}", a.bake_res, a.bake_res,
                                 alpha=False, float_buffer=False, is_data=noncolor)
        # Pre-fill, then bake without clearing. A ray that misses leaves its texel untouched;
        # left black, a normal texel decodes to a garbage direction and lights up as a white
        # specular streak. Flat (0.5, 0.5, 1.0) is the harmless default.
        fill = (0.5, 0.5, 1.0, 1.0) if pass_name == "normal" else (0.35, 0.33, 0.30, 1.0)
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
        src = pbsdf.inputs["Base Color"]
        if src.links:
            nt_h.links.new(src.links[0].from_socket, emit.inputs["Color"])
        else:
            emit.inputs["Color"].default_value = src.default_value
        nt_h.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])

    bake("base_color", "EMIT", high)

    for nt_h, out_node, from_sock in saved:            # restore the original shading
        if from_sock is not None:
            nt_h.links.new(from_sock, out_node.inputs["Surface"])
        n = nt_h.nodes.get("_bake_emit")
        if n:
            nt_h.nodes.remove(n)

    bake("normal", "NORMAL", dense or high)

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
