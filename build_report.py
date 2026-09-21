"""Build the CharForge report from the measured run outputs.

Reads   charforge/work/char01/*.json + renders
Writes  charforge/report.html  (self-contained, images embedded as data URIs)
"""
from __future__ import annotations

import base64
import io
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
W = ROOT / "work" / "char01"


def img(path, width=900, quality=82):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load(p, default=None):
    f = W / p
    return json.loads(f.read_text()) if f.exists() else (default if default is not None else {})


verify = load("verify.json")
riginfo = load("rigged_riginfo.json")
parts = load("parts.json")
joints = load("joints.json")
gate = load("gate_compare.json")
clips = load("animated_clips.json")

# --- stage timings measured during the run (seconds)
STAGES = [
    ("1", "Reference image", "Krea 2 Turbo (Q5_1 GGUF) via ComfyUI, 832x1216, 8 steps",
     229, "ref_9_0.png"),
    ("2", "Prompt-match gate", "local VLM description + typed judgments (Jev 9/9, Laya 3/9)", 4, None),
    ("3", "Geometry + PBR texture", "TRELLIS.2-4B via MLX: sparse structure -> shape latent -> "
     "texture latent -> UV bake", 248, None),
    ("4", "Multi-view render", "8 orbit views + exact per-vertex visibility (Blender)", 7, None),
    ("5", "Part segmentation", "SegFormer human parser back-projected onto vertices, "
     "flood-filled and smoothed over the mesh graph", 8, None),
    ("6", "Skeleton", "ViTPose on the 0deg and 90deg renders, unprojected to 3D, "
     "symmetrised and snapped into the mesh", 10, None),
    ("7", "Skinning", "proximity skinning over the bone hierarchy + Laplacian weight smoothing, "
     "hair and accessories bound rigidly", 4, None),
    ("8", "Animation", "6 clips authored on SMPL-H-named bones, each self-contained", 1, None),
    ("9", "Deformation verification", "per-part edge stretch, rigid residual, weight hygiene", 2, None),
]
TOTAL = sum(s[3] for s in STAGES)

TOOLS = [
    ("TRELLIS.2-4B", "Microsoft", "MIT", "yes - MLX port",
     "<b>Used for the run.</b> 248 s on the M5 for 198k triangles with baked PBR. Needs DINOv3, "
     "which is gated; the ungated timm copy converts cleanly (converter included).", "ran"),
    ("Hunyuan3D-2.1", "Tencent", "Community (not EU/UK/KR)", "yes - MPS port",
     "<b>Ran on the same test image.</b> 132 s for shape on the M5 at octree 192 (the port's published "
     "M4 Pro figure is ~5.7 min), 39,885 vertices, no texture. Lower geometric detail than TRELLIS.2 "
     "- see the comparison above.", "ran"),
    ("Pixal3D", "TencentARC", "MIT", "no",
     "SIGGRAPH 2026, the paper in the brief. Main branch is built on the TRELLIS.2 backbone but "
     "adds NATTEN neighbourhood-attention CUDA kernels, which have no Metal path - so it cannot "
     "run on this Mac without writing a pure-PyTorch neighbourhood attention fallback.", "blocked"),
    ("Modly", "lightningpixel", "MIT", "yes - Apple Silicon build",
     "Desktop wrapper that installs Hunyuan3D-2-mini / TripoSG / Trellis2-GGUF as extensions and "
     "exposes a CLI. Useful as a packaging reference; the underlying engines are the ones tested here.",
     "not run"),
    ("UniRig / SkinTokens", "VAST-AI", "MIT", "untested here",
     "Learned skeleton + skinning prediction. Weights downloaded. The deterministic route was used "
     "instead because it is inspectable and cannot fail silently; UniRig is the obvious upgrade for "
     "non-humanoid characters.", "downloaded"),
    ("HY-Motion 1.0", "Tencent", "Community", "untested here",
     "1B text-to-motion producing SMPL-H clips. Weights downloaded. The rig already uses SMPL-H bone "
     "names specifically so these clips retarget by name.", "downloaded"),
]

COMPETITORS = [
    ("Meshy 7", "Aug 2026", "$0 free tier, paid tiers by credit",
     "Auto-rig in &lt;30 s, 600+ motion presets, 8K textures, quadruped rigs"),
    ("Hyper3D Rodin Gen-2", "2026", "$30/mo Creator (~60 models), $120/mo Business, ~$0.40/run API",
     "Native quad topology, 4K PBR, symmetry forcing, texture delighting"),
    ("Tripo (Smart Mesh P1.0)", "2026", "$19.90/mo Pro",
     "~8 s generation, quad or triangle low-poly output, biped and multi-leg auto-rig"),
    ("Atlas", "2026", "EUR 50/mo Pro (~250 meshes), EUR 200 Studio, EUR 2000+ Enterprise",
     "Orchestration over Tripo v3 + FLUX with retopo/UV/PBR/rig; used by SEGA, Square Enix, Krafton"),
]

AAA_GAP = [
    ("Topology", "198k unstructured triangles from marching cubes",
     "40-100k quad-dominant polys with edge loops at every joint, plus an LOD chain",
     "QuadriFlow or Quad Remesher retopology, then bake the 1.3M-vertex original as a normal map"),
    ("Hair", "an opaque shell fused to the skull, correctly rigid but geometrically solid",
     "hair cards or strands with anisotropic shading and physics",
     "use the hair part mask as the groom region and fit cards to it"),
    ("Materials", "one 1024x1024 base colour + metallic-roughness, lighting partly baked in",
     "per-material 4K sets (skin with subsurface, fabric, leather) and a delit albedo",
     "multi-view reprojection at higher resolution + a delighting pass"),
    ("Face", "geometry only, no facial rig",
     "FACS-style blendshape rig with 100+ targets, eye and jaw joints",
     "a face-specific model; this pipeline does not attempt it"),
    ("Cloth", "linear blend skinning, measured p99 stretch up to 1.18 on the run cycle",
     "corrective blendshapes or cloth simulation, separate cloth meshes",
     "split the jacket into its own mesh using the existing part labels, then simulate or correct"),
    ("Silhouette detail", "soft facial features, simplified hands",
     "sculpted detail, correct hand topology with usable finger joints",
     "higher-resolution generation pass and hand-specific refinement"),
]


def second_run_section():
    """A second character, generated with the same script, is the only evidence that the
    pipeline generalises rather than having been tuned to one lucky mesh."""
    w2 = ROOT / "work" / "char02"
    v2 = json.loads((w2 / "verify.json").read_text()) if (w2 / "verify.json").exists() else None
    p2 = json.loads((w2 / "parts.json").read_text()) if (w2 / "parts.json").exists() else None
    if not v2:
        return ""
    st2 = (p2 or {}).get("stats", {})
    worst_hair = v2["verdict"].get("worst_hair_rigid_residual_p95")
    acc_res = max((c.get("rigid_residual_p95_by_part", {}).get("accessory") or 0)
                  for c in v2["clips"].values()) if v2.get("clips") else None
    cloth = max((c["edge_stretch_p99_by_part"].get("clothing") or 0) for c in v2["clips"].values())
    body = v2["verdict"].get("worst_body_stretch_p99", 0)
    figs = ""
    for name, cap in (("hero/pair.png", "The second character, textured and by part."),
                       ("anim_sheet.png", "The same six clips on the second character.")):
        f = w2 / name
        if f.exists():
            figs += f'<figure><img src="{img(f, 860)}" alt="second generated character"><figcaption>{cap}</figcaption></figure>'
    acc_stretch = max((c["edge_stretch_p99_by_part"].get("accessory") or 0)
                      for c in v2["clips"].values())
    rows = [
        ["hair rigid residual (p95, worst clip)",
         f'<span class="pass">{worst_hair*100:.2f}%</span>' if worst_hair is not None else "&mdash;"],
        ["body edge stretch (p99, worst clip)", f"{body*100:.0f}%"],
        ["clothing edge stretch (p99, worst clip)", f"{cloth*100:.0f}%"],
        ["accessory edge stretch (p99, worst clip)",
         f'<span class="fail">{acc_stretch*100:.0f}%</span>'],
        ["vertices", f"{v2.get('vertices',0):,}"],
        ["parts found", " &middot; ".join(f"{k} {n:,}" for k, n in st2.items() if n)],
        ["max influences per vertex", str(v2.get("weights", {}).get("max_influences", "?"))],
        ["unweighted vertices", str(v2.get("weights", {}).get("unweighted_vertices", "?"))],
    ]
    return f"""
<section>
  <h2>Does it generalise? A second character</h2>
  <p>One good result can be luck. The same script was run again, unchanged, on a deliberately harder
  subject: <i>a woman in a hooded leather jacket, with a long braid over her shoulder and a canvas
  satchel on a strap across her chest</i>. A braid that hangs onto the shoulder is the worst case for
  hair weighting, and a bag on a strap is a genuine rigid accessory that must not stretch with the
  torso.</p>
  {bars_table(rows, ["Measure", "Second character"], ["left", "right"])}
  <p class="takeaway"><b>The braid works, the satchel does not &mdash; and the pipeline says so.</b>
  Hair holds at 0.00% rigid residual even hanging over a shoulder. But the bag's strap measures
  {acc_stretch*100:.0f}% edge stretch on the wave clip, and the render shows why: the strap is not a
  separate object. The generator produced one fused shell, so the strap is a painted-on band of the
  jacket that bridges the hip and the sleeve. When the arm lifts, the band is stretched between two
  bones that are moving apart.</p>
  <p class="caveat">No weighting rule fixes this, and I tried several &mdash; binding by the
  supporting body part, rigid-binding only separate mesh components, restricting accessories to
  torso bones. Each changed the number without fixing the cause, because the cause is topological:
  you cannot rigidly attach something that is not a separate object. The fix is upstream &mdash;
  generate the bag as its own mesh, or cut it out along the part boundary and re-attach it &mdash;
  which is exactly what the part labels make possible and what I would build next. The value here
  is that the failure is a measured number in a JSON file a build can gate on, not something
  discovered by a player.</p>
  {figs}
</section>"""


def engine_section():
    info = load("hunyuan_mesh_info.json")
    cmp_img = W / "engine_compare.png"
    if not info or not cmp_img.exists():
        return ""
    rows = [
        ["TRELLIS.2-4B (MLX)", "170,958 v / 197,815 f", "248 s", "yes, 1024&sup2; PBR baked",
         "sharp: zip, pockets, cuffs, separated fingers, boot laces, facial features"],
        ["Hunyuan3D-2.1 (MPS)", f"{info['vertices']:,} v / {info['faces']:,} f",
         f"{info['generate_seconds']:.0f} s (+{info['load_seconds']:.0f} s load)", "no - shape only",
         "smooth: soft face, mitten hands, plain jacket, block boots"],
    ]
    return f"""
<section>
  <h2>TRELLIS.2 against Hunyuan3D-2.1, same image</h2>
  <p>Both generators run on this Mac, so the brief's question &mdash; which one &mdash; is answerable by
  running them on the same reference image and rendering both in the same neutral clay material, where
  only geometry shows.</p>
  {bars_table(rows, ["Engine", "Output", "Time", "Texture", "Detail"], ["left", "left", "right", "left", "left"])}
  <figure><img src="{img(cmp_img, 900)}" alt="TRELLIS.2 versus Hunyuan3D-2.1 geometry">
  <figcaption>Same input image, same lighting, same clay material. Left TRELLIS.2, right Hunyuan3D-2.1.</figcaption></figure>
  <p class="takeaway"><b>TRELLIS.2 wins this comparison clearly, and it is the one the pipeline uses.</b>
  It resolves the jacket's zip and pockets, separates the fingers, and puts recognisable features on the
  face. Hunyuan3D produced a smoother, heavier figure with fused hands &mdash; in roughly half the time,
  and with no texture at all, so the honest total comparison is 248 s textured against
  {info['generate_seconds']:.0f} s untextured plus a texture pass the port documents at around 8 minutes.</p>
  <p class="caveat">One caveat in Hunyuan's favour: this ran at octree resolution
  {info['octree']} and {info['steps']} steps, the port's safe defaults for a 24&nbsp;GB machine. Higher
  settings would recover detail at the cost of time and memory, and Hunyuan3D's PBR paint stage is a
  genuinely strong piece of work that this comparison does not exercise. The claim here is narrow: for a
  single clothed human at settings that fit this laptop, TRELLIS.2 produced the better mesh.</p>
</section>"""


def retopo_section():
    r = load("retopo_retopo.json") if (W / "retopo_retopo.json").exists() else {}
    cmp_img = W / "retopo_compare.png"
    if not r:
        return ""
    rows = [["input", f"{r['high_tris']:,} triangles (marching cubes, no edge loops)"],
            ["output", f"{r['low_polys']:,} polygons, {r['quad_fraction']*100:.0f}% quads"],
            ["reduction", f"{r['reduction']*100:.1f}%"],
            ["method", r["method"].replace("_", " ")],
            ["baked maps", f"albedo + normal at {r['bake_res']}&times;{r['bake_res']}"]]
    fig = (f'<figure><img src="{img(cmp_img, 900)}" alt="generated mesh versus retopologised mesh">'
           f'<figcaption>Same lighting, same camera. The retopologised version carries the original '
           f'albedo and a repaired normal map baked onto fresh UVs.</figcaption></figure>') if cmp_img.exists() else ""
    return f"""
<section>
  <h2>Closing the topology gap</h2>
  <p>The largest single item in the table above is topology, and it is also the most tractable:
  it is deterministic geometry work, not a model capability. The pipeline now includes a
  retopology stage.</p>
  {bars_table(rows, ["", "Retopology stage"], ["left", "left"])}
  {fig}
  <p class="caveat"><b>Two findings worth recording.</b> Blender's QuadriFlow &mdash; the obvious
  choice for a quad cage &mdash; silently does nothing on this mesh and returns success; generated
  meshes carry tens of thousands of non-manifold edges and QuadriFlow needs a manifold surface.
  Voxel remesh does not care and returns 100% quads, so that is the fallback, with the voxel size
  solved for iteratively to hit the polygon budget. Second, the normal bake needed two fixes before
  it was usable. Baking against a flat-shaded low-poly cage produces garbage tangent-space values
  &mdash; mean blue channel 133 where a flat normal encodes to 255 &mdash; which renders as
  blown-out specular streaks; shading the cage smooth before baking is what makes the tangent frame
  continuous enough to bake into. That leaves texels the ray never hit, which encode as
  inward-facing normals; those are detected by blue &lt; 0.5 and rewritten flat. The map now reads
  mean blue 245.5, median 255, with no inward-facing texels remaining, and is wired into the
  material rather than written and abandoned.</p>
  <p class="takeaway">Net: <b>197,815 marching-cubes triangles become 39,134 quads &mdash; 100%
  quads, an 80.2% reduction &mdash; carrying 2K albedo and normal maps baked from the discarded
  high-resolution surface.</b> This is the stage that turns a generated blob into something with
  edge loops a rigger or an engine can work with.</p>
</section>"""


def weld_section():
    w = load("../weld_stats.json") if (W.parent / "weld_stats.json").exists() else {}
    if not w:
        return ""
    rows = []
    for c, d in w.items():
        rows.append([f"<b>{escape(c)}</b>",
                     f"{d['vertices']:,} &rarr; {d['unique_positions']:,}",
                     f"{d['components_before']:,} &rarr; {d['components_after']:,}",
                     f"{d['largest_before']/d['vertices']*100:.1f}% &rarr; "
                     f"{d['largest_after']/d['vertices']*100:.1f}%"])
    return f"""
<section>
  <h2>The bug underneath the other bugs</h2>
  <p>Part segmentation, weight smoothing, island filtering and mesh splitting all walk the mesh as a
  graph: each reads a vertex's neighbours and spreads information across the surface. All four were
  quietly producing nonsense, and the cause was one property of the file format rather than anything
  in the algorithms.</p>
  <p><b>glTF has no shared vertices across a seam.</b> Wherever a UV island ends or a normal breaks,
  the exporter duplicates the vertex, because a vertex in glTF carries exactly one UV and one normal.
  Geometrically the surface is continuous; in the index buffer it is cut. Reading adjacency straight
  from the triangle indices therefore does not give you the surface &mdash; it gives you the surface
  shredded along every texture seam.</p>
  {bars_table(rows, ["Character", "Vertices &rarr; unique positions",
                     "Connected components", "Largest single island"],
              ["left", "right", "right", "right"])}
  <p class="takeaway"><b>On char01 the largest connected region of a 170,958-vertex character was
  2,705 vertices &mdash; 1.6% of the mesh, spread over 13,077 fragments.</b> Flood fill could not
  cross a seam, so it labelled patches; smoothing averaged over islands a few hundred vertices wide;
  the "remove small islands" filter saw everything as a small island. Welding vertices that share a
  position &mdash; same geometry, merged graph &mdash; collapses those 13,077 fragments to 69 and
  puts 98.8% of the character in one piece.</p>
  <p class="caveat">Worth noting how this presented: nothing threw, nothing logged a warning, and
  every stage reported success. The symptom was a rig whose shoulder strap stretched 757% in a walk
  cycle. Three earlier fixes were aimed at the symptom &mdash; a better skinning falloff, heavier
  weight smoothing, a stricter island test &mdash; and each moved the number a little, which is
  exactly what a plausible wrong explanation does.</p>
</section>"""


def split_section():
    import json as _json
    rows, splits = [], {}
    for c in ("char01", "char02"):
        pre = W.parent / c / "verify.json"
        post = W.parent / c / "verify_split.json"
        sp = W.parent / c / "split_split.json"
        if not (pre.exists() and post.exists()):
            continue
        a = _json.load(open(pre))
        b = _json.load(open(post))
        if sp.exists():
            splits[c] = _json.load(open(sp))

        def worst(d, part):
            return max((cl.get("edge_stretch_p99_by_part", {}).get(part) or 0.0)
                       for cl in d.get("clips", {}).values()) if d.get("clips") else 0.0
        rows.append([f"<b>{escape(c)}</b>",
                     f"{worst(a,'body')*100:.1f}% &rarr; <b>{worst(b,'body')*100:.1f}%</b>",
                     f"{worst(a,'accessory')*100:.1f}% &rarr; <b>{worst(b,'accessory')*100:.1f}%</b>",
                     f"{worst(a,'clothing')*100:.0f}% &rarr; {worst(b,'clothing')*100:.0f}%"])
    if not rows:
        return ""
    sp2 = splits.get("char02", {}).get("parts", {})
    detail = ""
    if sp2:
        acc = sp2.get("accessory", {})
        detail = (f"On char02 the satchel and its strap come out as {acc.get('vertices',0):,} vertices "
                  f"in {acc.get('islands',0):,} separate islands, each bound rigidly to whichever "
                  f"wearable bone its centroid sits nearest.")
    return f"""
<section>
  <h2>Splitting the character into parts it can actually be rigged as</h2>
  <p>The previous run ended with a recommendation: the labels make it possible to separate the jacket
  and the bag into their own meshes, so stop skinning them like skin. That is now done. Body and
  clothing stay skinned; hair and accessories are cut into connected islands and each island is bound
  rigidly to one bone, so a bag travels with the hip instead of being pulled apart between hip and
  spine.</p>
  {bars_table(rows, ["Character", "Body stretch (p99, worst clip)",
                     "Accessory stretch", "Clothing stretch"],
              ["left", "right", "right", "right"])}
  <p class="takeaway"><b>char01's body stretch drops from 39.8% to 15.0% and its verdict flips from
  fail to pass; char02's drops from 31.8% to 6.1%.</b> {detail}</p>
  <p class="caveat"><b>Read the accessory column carefully.</b> Rigid binding makes intra-island edge
  stretch zero <i>by construction</i> &mdash; a rigid body cannot stretch, so 0.0% is not a quality
  score and it would read 0.0% even if every island were bound to the wrong bone. The number that
  carries information is the one it replaced: char02's strap was reaching 182% before, which is a
  visibly torn bag. What actually has to be checked after this change is bone <i>assignment</i>,
  which is why the split stage tests each island against the wearable-bone set rather than trusting
  a nearest-vertex lookup.</p>
  <p class="caveat"><b>Still unsolved:</b> clothing. The jacket is still skinned and still reaches
  ~100% p99 stretch on a run cycle at the hem. Splitting it out is what makes cloth simulation
  possible, but a split mesh is not a simulated one &mdash; that work has not been done, and the
  honest summary is that this stage fixed the bag and the hair, improved the body, and left the
  jacket where it was.</p>
</section>"""


def pixal3d_section():
    d = load("../pixal3d_probe.json") if (W.parent / "pixal3d_probe.json").exists() else {}
    if not d:
        return ""
    rows = [["modules imported on MPS",
             f"{len(d.get('imports_ok', []))}/6, none failed"],
            ["sparse conv vs dense conv3d",
             f"max abs error {d.get('conv_max_abs_error_vs_dense')} &mdash; numerically identical"],
            ["forward pass, {:,} voxels".format(d.get("mps_voxels", 0)),
             f"{d.get('mps_first_forward_s')}s first (builds the neighbour map), "
             f"{d.get('mps_cached_forward_s')}s cached"],
            ["attention", f"{d.get('attention_backend')} backend, runs"]]
    return f"""
<section>
  <h2>Pixal3D: the blocker was not the one in the README</h2>
  <p>The earlier write-up repeated what the repository says &mdash; that Pixal3D needs NATTEN
  neighbourhood attention, that NATTEN is CUDA-only, and that Apple Silicon therefore needs a Metal
  port of it. Checking that claim against the source rather than the README:
  <b>Pixal3D never imports NATTEN anywhere.</b> It is a dependency inherited from the TRELLIS.2 base
  environment, and the attention path Pixal3D actually uses dispatches to PyTorch's own
  scaled-dot-product attention.</p>
  <p>The real blocker is duller and much smaller. Pixal3D's sparse stack dispatches to a backend
  named by <code>SPARSE_CONV_BACKEND</code>; the config accepts <code>'none'</code>, meaning
  pure PyTorch, but no <code>conv_none.py</code> ships with the repository, so selecting it fails at
  the first layer. Writing that one module is the whole port.</p>
  {bars_table(rows, ["Probe", "Result"], ["left", "left"])}
  <p class="takeaway"><b>All six Pixal3D modules now import and run on MPS</b>, and the
  submanifold sparse convolution agrees with a dense <code>conv3d</code> reference to
  {d.get('conv_max_abs_error_vs_dense')} absolute error on a fully occupied grid, which is the
  condition under which the two are mathematically the same operation.</p>
  <p class="caveat">The neighbour search is the part that needed care. The reference pure-PyTorch
  implementation in the TRELLIS.2 Mac port builds a Python dictionary over every voxel and loops
  over N&times;K pairs &mdash; fine for a smoke test, hopeless at 300k voxels. Here each coordinate
  is packed into one int64 key and every kernel offset is resolved with a single
  <code>torch.searchsorted</code> against the sorted key array, which keeps the work inside GPU
  kernels. <b>What this does not yet establish is output quality:</b> the modules execute and the
  convolution is correct, but Pixal3D has not been run end-to-end to a mesh, so there is no
  comparison against TRELLIS.2 on the test character yet. Imports and a verified kernel are a
  precondition, not a result.</p>
</section>"""


def backview_section():
    rows_in = load("../backview_baseline.json") if (W.parent / "backview_baseline.json").exists() else []
    if not rows_in:
        return ""
    rows = []
    for r in rows_in:
        c = r["curvature_deg"]
        rows.append([f"<b>{escape(r['label'])}</b>",
                     f"{c['front_crease']*100:.2f}%", f"{c['back_crease']*100:.2f}%",
                     f"<b>{r['crease_back_over_front']:.2f}</b>",
                     f"{r['relief_back_over_front']:.2f}"])
    return f"""
<section>
  <h2>How much does the model actually invent behind the character?</h2>
  <p>The reference image is a front view, so everything behind the character is guessed. The obvious
  next improvement is multi-view conditioning &mdash; the MLX TRELLIS port takes several images and
  the pipeline was passing one. Before building that, it is worth measuring how bad the guess is,
  because the size of the prize decides whether the stage is worth its runtime.</p>
  <p>The measure is crease density: the fraction of edges bent past 20&deg;, computed separately over
  camera-facing and camera-averted surfaces. Mean curvature is useless here &mdash; a marching-cubes
  surface is uniformly bumpy, and that noise swamps the signal. Real garment structure is sparse and
  sharp, so the tail of the dihedral distribution is what separates a tailored back from a blank one.</p>
  {bars_table(rows, ["Mesh", "Front creased", "Back creased", "Back / front", "Relief ratio"],
              ["left", "right", "right", "right", "right"])}
  <p class="takeaway"><b>The invented back is not blank.</b> TRELLIS.2 puts 10.1% creased edges on
  char01's back against 12.2% on its front &mdash; a ratio of 0.83 &mdash; and char02 reaches 0.92.
  Hunyuan3D-2.1 is the weaker case at 0.68, on a surface that is less detailed everywhere: 6.5%
  creased edges on its <i>front</i>, about half of what TRELLIS manages.</p>
  <p class="caveat">This changes what multi-view conditioning is worth here: the headroom is roughly
  8&ndash;17% of back-side crease density, not the rescue of a featureless shell. It also exposes a
  trap in the obvious implementation. Re-conditioning on renders of the first-pass mesh feeds the
  model its own guess, which adds no information and can only sharpen or entrench it. New signal has
  to come from somewhere else &mdash; an image model repainting those renders &mdash; so the
  experiment is built with a control arm (raw renders) alongside the treatment (repainted renders),
  because otherwise the diffusion prior's work gets credited to multi-view conditioning.</p>
</section>"""


def distill_section():
    tr = load("../../models/laya-gate/eval.json") if (W.parent.parent / "models" / "laya-gate" / "eval.json").exists() else {}
    ev = load("../gate_eval.json") if (W.parent / "gate_eval.json").exists() else {}
    if not tr:
        return ""
    b, af = tr.get("before", {}), tr.get("after", {})

    set_blocks = ""
    for set_name, res in (ev.get("sets") or {}).items():
        labels = list(res)
        capped = next((r.get("uninformative_questions") or {} for r in res.values()), {})
        qs = sorted({q for r in res.values() for q in r.get("per_question", {})})
        rows = []
        for q in qs:
            why = capped.get(q)
            name = (f"<b>{escape(q.replace('_',' '))}</b>" if not why else
                    f"<span class='sub'>{escape(q.replace('_',' '))}</span>")
            rows.append([name] + [f"{res[l]['per_question'].get(q, float('nan')):.2f}" for l in labels]
                        + [f"<span class='sub'>{escape(why)}</span>" if why else ""])
        rows.append(["<b>accuracy, all questions</b>"] +
                    [f"<b>{res[l]['accuracy']:.3f}</b>" for l in labels] + [""])
        if any("accuracy_discriminating" in res[l] for l in labels):
            rows.append(["<b>accuracy, discriminating questions only</b>"] +
                        [f"<b>{res[l].get('accuracy_discriminating', float('nan')):.3f}</b>"
                         for l in labels] + [""])
        rows.append(["<span class='sub'>ms per description</span>"] +
                    [f"<span class='sub'>{res[l].get('ms_per_description') or 0:.0f}</span>"
                     for l in labels] + [""])
        head = ["Question"] + labels + ["&nbsp;"]
        set_blocks += (f"<h3>{escape(set_name)}</h3>" +
                       bars_table(rows, head, ["left"] + ["right"] * len(labels) + ["left"]))

    return f"""
<section>
  <h2>Teaching the local model the gate, with the API model as teacher</h2>
  <p>The gate result above is the one genuinely awkward finding in this project: Jev answers all
  nine questions correctly and Laya, the local model, answers three. A pipeline advertised as fully
  local cannot depend on an API call for its own verification step. The fix that Laya's own
  documentation points at is fine-tuning, and the teacher is sitting right there.</p>
  <p>So: 5,000 descriptions composed from slots that fix the ground truth, labelled by Jev in 76
  seconds for $0.13, and Laya's decision head plus its top six encoder layers trained on Jev's
  <i>probabilities</i> rather than its hard answers &mdash; soft cross-entropy, because the
  calibration is the part worth copying. 99.8M of 421.3M parameters move; the rest stay frozen.</p>
  {bars_table([["held-out agreement with Jev",
                f"{b.get('agreement_with_jev',0):.3f} &rarr; <b>{af.get('agreement_with_jev',0):.3f}</b>"],
               ["mean absolute error vs Jev's probability",
                f"{b.get('mae',0):.3f} &rarr; <b>{af.get('mae',0):.3f}</b>"],
               ["Brier score", f"{b.get('brier',0):.4f} &rarr; <b>{af.get('brier',0):.4f}</b>"],
               ["training", f"{tr.get('minutes','?')} minutes on the laptop GPU, 2 epochs"]],
              ["", "Distillation"], ["left", "left"])}
  <p class="caveat"><b>That 0.999 is not the result, it is a warning.</b> The held-out split comes
  from the same slot generator as the training set, so near-perfect agreement shows the student
  learned the teacher on the teacher's own distribution &mdash; which is partly a statement about
  how templated that distribution is. It says nothing about free prose from a vision model, and a
  model that had simply memorised the template would score exactly this well. Two harder sets
  follow, and they are reported separately for that reason.</p>
  {set_blocks}
  <p class="takeaway"><b>Distillation moves the local model most of the way, and not all the
  way.</b> On real vision-model prose the gate goes from 0.675 to 0.870 over the questions that
  discriminate, against Jev's 0.960; on the out-of-distribution probe, 0.644 to 0.785 against
  Jev's 0.989. On the specific case that started this &mdash; the actual char01 reference
  description &mdash; the tuned model now answers 9/9 where the base answered 3/9, at 392 ms on
  the laptop and no API call. It is not a yes-machine: it answers yes 39% of the time against a
  47% positive rate, and separates true from false cleanly (mean 0.66 against 0.17).</p>
  <p class="caveat"><b>Two questions in the gold set carry no signal at all, and neither is the
  decision model's fault.</b> All three models &mdash; including Jev &mdash; score at or below
  the majority-class baseline on <i>arms away from body</i> and <i>even lighting</i>, because
  qwen2.5vl does not report those facts faithfully: it describes the A-pose as "arms relaxed at
  the sides", and calls a deliberately side-lit render "even and frontal, with no shadows". The
  decision models are reading the description correctly; the description is wrong. That is worth
  stating plainly, because it relocates the next piece of work &mdash; the weakest link in this
  gate is the vision model, not the small classifier everyone would suspect. The remaining gap to
  Jev on the probe, 0.785 against 0.989, is the part that <i>is</i> the student's to close.</p>
</section>"""


def multiview_section():
    import json as _json
    f = W / "backview_multiview.json"
    if not f.exists():
        return ""
    rows_in = _json.load(open(f))
    f2 = W / "backview_stochastic.json"
    if f2.exists():
        rows_in = rows_in + _json.load(open(f2))
    by = {r["label"]: r for r in rows_in}
    order = [("pass1_ref_only", "pass 1 &mdash; reference image only"),
             ("pass2_raw_multiview", "pass 2 &mdash; concat, raw renders"),
             ("pass3_enhanced_multiview", "pass 3 &mdash; concat, repainted renders"),
             ("pass4_stochastic_weighted", "pass 4 &mdash; <b>stochastic</b>, repainted renders")]
    rows = []
    for key, lab in order:
        r = by.get(key)
        if not r:
            continue
        ig = r["integrity"]
        rows.append([f"<b>{lab}</b>",
                     f"{r['curvature_deg']['back_crease']*100:.2f}%",
                     f"{r['crease_back_over_front']:.3f}",
                     f"<b>{ig['open_edge_fraction']*100:.2f}%</b>"])
    cmp_img = W / "multiview_compare.png"
    fig = (f'<figure><img src="{img(cmp_img, 980)}" alt="back and front views of the three '
           f'conditioning passes"><figcaption>Back row above, front row below. Same seed, same '
           f'resolution, same step count, same face budget &mdash; only the conditioning images '
           f'differ.</figcaption></figure>') if cmp_img.exists() else ""
    return f"""
<section>
  <h2>Multi-view conditioning, and a metric that lied</h2>
  <p>The MLX port accepts several conditioning images and concatenates their DINOv3 feature
  tokens, and the pipeline was passing one. The obvious experiment is to render the first-pass
  mesh from the side and behind and feed those back. The obvious experiment is also circular:
  those renders <i>are</i> the model's own guess, so they carry no new information. Anything they
  buy has to come from removing ambiguity rather than adding fact. To separate that from a real
  image prior, the enhanced arm repaints the same renders through Krea 2 at denoise 0.42, and the
  raw arm is kept as the control.</p>
  {bars_table(rows, ["Conditioning", "Back creased edges", "Back / front ratio",
                     "Open (torn) edges"], ["left", "right", "right", "right"])}
  <p class="takeaway"><b>Read the last column first, because the third column is a trap.</b> By
  crease density both multi-view passes look like clear wins &mdash; the back goes from 0.83 of
  the front's detail to 0.97 and 1.00, a 16&ndash;20% gain, consistent across p95 and relief.
  The control arm is <i>torn</i>: 10.58% of its edges are holes against 2.30% for pass 1, 4.6
  times worse. Crease density cannot tell a garment seam from a rip, and every statistic in that
  table agreed with each other and with a conclusion that looking at the render refutes in a
  second.</p>
  {fig}
  <p>What the renders actually show: pass 2 is blotched with white tearing across the trousers
  and sleeves, holed at the neck, and its hair has come apart. Pass 3 is the interesting one
  &mdash; its surface is <i>cleaner</i> than the original (1.88% open edges against 2.30%), its
  jacket gains a real hem band and collar, and its back genuinely does reach parity with its
  front. It also wrecks the face.</p>
  <p><b>The diagnosis.</b> Concatenation gives every view equal weight in cross-attention, so
  three views dilute the front reference to a third of the context and the face loses the tokens
  that described it &mdash; the damage lands exactly where the reference was carrying the most
  information. It also triples the context length, from 1,029 tokens to 3,087, which is not the
  regime the checkpoint was trained in. TRELLIS's own multi-image scheme is not concatenation at
  all: it varies <i>which</i> view conditions each denoising step, so every forward pass sees the
  token layout training used while the trajectory as a whole is pulled toward agreeing with all
  the views.</p>
  <h3>Pass 4: implementing the scheme the model was actually trained for</h3>
  <p>That is now built. <code>MultiViewConditioning</code> in the MLX port builds one
  cross-attention KV cache per view and swaps which one the sampler reads at the top of each
  denoising step, applied identically across all four sampling stages. A weighted schedule keeps
  the reference image dominant &mdash; it is the only view with a real face &mdash; while the
  side and back views steer the geometry only they can see; here 3:1:1, so the reference
  conditions six of the twelve steps and the first step always.</p>
  <p class="takeaway"><b>Pass 4 is the best mesh of the four on every measure, including the
  guard.</b> Open edges fall to <b>1.10%</b> &mdash; cleaner than concatenation (1.88%), cleaner
  than the torn control (10.58%), and cleaner than the original single-view pass (2.30%).
  Back-side crease density reaches parity with the front (1.03 against pass 1's 0.83). And this
  time the renders agree with the numbers: the face is sharp where concatenation smeared it, the
  jacket keeps a defined collar and hem band, and the back is clean. Context per step is 1,029
  tokens, exactly what the single-view path uses.</p>
  <p class="caveat">Two honest limits. This is one character and one weighting; 3:1:1 was chosen
  from the diagnosis rather than swept, and the right ratio almost certainly depends on how many
  views there are and how much each one adds. And the conditioning views here were produced by
  repainting renders of the first pass through an image model, so the pipeline pays for a second
  diffusion pass plus a second 3D pass &mdash; about 4.6 minutes of img2img and 3 minutes of
  generation on this machine &mdash; to buy a cleaner surface and a back that matches the front.
  Whether that trade is worth it depends on whether the back of the character is ever seen.</p>
</section>"""


def animation_section():
    import json as _json
    cl = W / "cloth_walk.json"
    cloth = _json.load(open(cl)) if cl.exists() else {}
    fig = ""
    sheet = W / "anim2_sheet.png"
    if sheet.exists():
        fig = (f'<figure><img src="{img(sheet, 980)}" alt="six animation clips, six frames each">'
               f'<figcaption>Six frames from each clip after the rewrite. Arms hang and swing, '
               f'the pelvis bobs, the jump leaves the ground.</figcaption></figure>')
    cf = W / "cloth_compare.png"
    cloth_fig = (f'<figure><img src="{img(cf, 980)}" alt="skinned versus cloth-simulated jacket">'
                 f'<figcaption>Top row skinned, bottom row simulated, same frames and cameras. '
                 f'The holes that voxel remeshing punched through the fabric are gone; the '
                 f'simulated jacket carries a looser hem and fuller sleeves.</figcaption></figure>'
                 ) if cf.exists() else ""
    cloth_rows = ""
    if cloth:
        cloth_rows = bars_table(
            [["garment", f"{cloth.get('garment_vertices',0):,} vertices, "
                         f"{cloth.get('pinned_vertices',0):,} pinned at the collar and shoulder seam"],
             ["clip", f"{escape(str(cloth.get('clip','')))}, frames "
                      f"{cloth.get('frames',[0,0])[0]}&ndash;{cloth.get('frames',[0,0])[1]}"],
             ["p99 edge stretch, skinned", f"{cloth.get('skinned_p99_stretch',0)*100:.1f}%"],
             ["p99 edge stretch, simulated",
              f'<span class="fail">{cloth.get("simulated_p99_stretch",0)*100:,.0f}%</span> '
              f"&mdash; diverged"]],
            ["", "Cloth simulation on the generated garment"], ["left", "left"])
    return f"""
<section>
  <h2>Watching the animation, and what that exposed</h2>
  <p>The brief asked whether the character survives animation. The earlier answer leaned on a
  deformation metric; actually watching the clips gave a different and worse answer. The walk
  scissored its legs with both arms locked out in the bind A-pose, the pelvis never rose or fell,
  the feet skated, <i>jump</i> never left the ground and <i>turn</i> span like a turntable.</p>
  <p class="takeaway"><b>The cause was not subtle: the walk and run clips never keyframed the
  shoulder or elbow bones at all.</b> Which also means the rig had never been tested there. A
  deformation check only exercises the joints something actually moves, so the clean numbers were
  partly a measure of how little the animation asked of the skin.</p>
  <p>The clips are now sampled from published gait curves rather than eyeballed &mdash; hip,
  knee and ankle angles tabulated across the stride, contralateral arm swing, pelvis bob twice
  per stride and lateral sway once, trunk counter-rotation, and a jump with a real crouch,
  ballistic arc and landing absorb. Every clip drives 63 curves where the old ones drove a
  handful.</p>
  {fig}
  <p class="caveat"><b>One bug worth recording, because it is the same bug as last time.</b>
  The first rewrite swung the shoulders on the wrong local axis and the arms rose sideways
  instead of swinging fore-aft. Bone-local axes on a generated skeleton are not guessable, so
  the fix was to measure them &mdash; rotate +40&deg; about each axis, watch where the child
  joint lands. Local X moves the elbow fore-aft (dY 0.32); local Z abducts it sideways
  (dX -0.23). Hips were already on X and were right. This is the third time in this project that
  guessing an axis cost a render cycle and measuring it took one script.</p>
  <h3>Cloth</h3>
  <p>Moving the arms properly exposed a second failure the old animation had hidden completely:
  the jacket sleeve tears at the shoulder. Splitting the parts made this addressable, because the
  clothing is now its own mesh &mdash; but handing it straight to Blender's cloth solver did not
  work, and the reason turned out to be upstream of the solver entirely.</p>
  {cloth_rows}
  <p><b>The garment was never a simulable mesh.</b> Cutting the clothing out of a fused
  marching-cubes character by vertex label leaves 129,961 vertices of which <b>32.9% of the edges
  are boundary edges</b> &mdash; 91,486 open borders across 29 fragments. A solver asked to
  integrate that is not simulating a jacket, it is simulating tens of thousands of loose flaps,
  and it diverged accordingly: 13,800% edge stretch with self-collision on, 8,438% with it off,
  which rules out self-collision as the cause.</p>
  <h4>Three ways to retopologise a garment, two of which are wrong</h4>
  <p>The fix is a garment-specific retopology pass, and which remesher is used turns out to
  matter more than any cloth setting:</p>
  {bars_table([
    ["<b>voxel remesh</b>",
     "Closes the shell into a manifold solid: 0.03% boundary, 100% quads, 11,936 verts. Looks "
     "ideal on paper and is <b>visually destroyed</b> &mdash; it is volumetric, and the voxel "
     "size that hits the budget is 2.1&nbsp;cm, which is thicker than the fabric. It punches "
     "holes clean through the jacket."],
    ["<b>QuadriFlow</b>",
     "The right tool in principle: a surface remesher that keeps the neck, cuff and hem "
     "boundaries. It returned success and left the mesh byte-identical at 78,351 vertices. "
     "Caught only because the script checks the vertex count afterwards."],
    ["<b>decimate</b> (used)",
     "Weld, drop the specks, collapse to budget. Keeps the garment a <i>surface</i>: 78,351 "
     "&rarr; 15,070 verts, boundary 32.9% &rarr; 9.5% &mdash; the remaining borders are the "
     "actual neck, cuff and hem openings. All 15,070 vertices carry skin weights transferred "
     "from the original."],
  ], ["Method", "Result"], ["left", "left"])}
  <p class="takeaway"><b>On the decimated garment the solver is stable and the jacket drapes.</b>
  No divergence, and cloth moves the surface by 5&ndash;9&nbsp;cm at the median against the
  skinned pose. Side by side the simulated version reads as fabric where the skinned one reads as
  shrink-wrap: a softer hem, rounder sleeves, a fuller silhouette.</p>
  {cloth_fig}
  <p class="caveat"><b>And the metric says it got worse &mdash; because it is the wrong metric.</b>
  p99 edge stretch goes 207.4% skinned to 209.9% simulated. That number was built to catch
  skinning <i>tears</i>, and it measures any deviation from the rest mesh. Drape <i>is</i>
  deviation from the rest mesh. A cloth simulation that does its job will always look worse by it,
  so it cannot be used to judge this stage; what it can still do is tell divergence from
  stability, which is how the 8,438% failure was caught. Judging drape needs a different measure
  &mdash; body penetration depth and fold count would be the honest pair &mdash; and that is not
  built. This is the third time in this project that a metric confidently reported the opposite
  of what the render shows.</p>
  <p class="caveat"><b>What the playground now ships.</b> Cloth is baked for all five clips and
  the web build is no longer decimated: 57,699 polygons at full resolution with 2K maps, against
  18,760 and 1K in the first build. The GLB is 13.1 MB, which does not fit a single base64
  module under the 16 MB per-file limit, so it is split across three and rejoined at load. The
  stretch numbers here are still not comparable to the 60% quoted for the original garment:
  that was a different mesh with far shorter edges, and p99 over a different distribution is a
  different statistic.</p>
</section>"""


def mixamo_section():
    import json as _json
    rt = W / "retarget.json"
    bk = W / "cloth_baked.json"
    if not rt.exists():
        return ""
    r = _json.load(open(rt))
    b = _json.load(open(bk)) if bk.exists() else {}
    rows = [[f"<b>{escape(k)}</b>", f"{v['frames']} frames",
             f"{v['bones_mapped']}/20", f"{v['leg_lengths_per_sec']:.2f}"]
            for k, v in r.items()]
    fig = ""
    sh = W / "mixamo_clips.png"
    if sh.exists():
        fig = (f'<figure><img src="{img(sh, 980)}" alt="five retargeted Mixamo clips">'
               f'<figcaption>The five clips after retargeting, eight frames each.</figcaption>'
               f'</figure>')
    cloth = ""
    if b:
        per = ", ".join(f"{c['clip']} {c['targets']}" for c in b.get("clips", []))
        cloth = bars_table(
            [["garment", f"{b['garment_vertices']:,} vertices (decimated for the web)"],
             ["baked", f"{b['morph_targets']} morph targets across "
                       f"{len(b.get('clips', []))} clips &mdash; {escape(per)}"],
             ["cost", f"~{b['approx_target_bytes']/1e6:.1f} MB of vertex offsets"]],
            ["", "Cloth baked to glTF morph targets"], ["left", "left"])
    return f"""
<section>
  <h2>Captured motion, and four bugs that each looked like success</h2>
  <p>The procedural gait was a synthesis. The rig was given SMPL-H bone names precisely so
  captured motion could be dropped onto it, and the Mixamo dataset (jasongzy/Mixamo, 2,453 clips)
  is that motion. Retargeting it exposed more about how this kind of work fails than about
  animation.</p>
  {bars_table(rows, ["Clip", "Length", "Bones mapped", "Ground speed (leg-lengths/s)"],
              ["left", "right", "right", "right"])}
  {fig}
  <p class="caveat"><b>Four failures, none of which announced itself.</b></p>
  <ul class="plain">
    <li><b>Scale from the wrong measurement.</b> Hip height above ground looked like the obvious
    reference for matching stride across rigs. This rig's armature origin sits near the chest, so
    its pelvis reads z=0.005 and every clip retargeted to a 200&times; reduction &mdash; the
    character mimed walking on the spot. Bone-chain length is placement-independent.</li>
    <li><b>Units counted twice.</b> Blender's FBX importer leaves Mixamo's centimetres in the
    armature data and puts the 0.01 conversion on the object matrix, so `head_local` reports a
    leg of 84.8 while the hips genuinely stand 0.84&nbsp;m up. Dividing metres by centimetres
    shrank the motion another 100&times;.</li>
    <li><b>Armature space is not world space.</b> Mixamo is Y-up, Blender is Z-up, and the
    importer puts that correction on the object matrix. The rest-delta computed from
    armature-space matrices therefore mixed a Y-up source with a Z-up target: the retargeted run
    came out horizontal, flying like Superman.</li>
    <li><b>Name collision.</b> The target file already held procedural clips called walk, run and
    idle, so Blender silently renamed the incoming ones walk.001 and so on &mdash; and the review
    renderer dutifully played the <i>old</i> clips while the log reported five new ones baked. A
    broken procedural `run` was reviewed as though it were the retarget.</li>
  </ul>
  <p class="takeaway"><b>And one that was not a bug at all.</b> With the maths finally right, the
  idle clip still retargeted to a character crouching in mid-air &mdash; because the clip really
  is a seated idle. The content scan had classified clips by speed, flight phase and cadence, and
  a seated idle scores exactly like a standing one. Measuring posture directly &mdash; the cosine
  between the thigh and world-down &mdash; separates them, and re-picking on that basis gave the
  five clips above.</p>
  <h3>Baking cloth into the browser</h3>
  <p>glTF has no cloth solver, so a simulated garment reaches a browser only as baked vertex
  animation: one morph target per simulated frame, each holding the difference between the
  simulated and the <i>skinned</i> shape &mdash; not the simulated shape itself, because glTF
  applies morph targets before skinning and storing the posed result would apply the pose
  twice.</p>
  {cloth}
  <p class="caveat"><b>Two traps in playback, both silent.</b> First, a baked cloth simulation
  exports as its own animation carrying only morph weights, so it has to be driven alongside the
  skeletal clip it belongs to; loaded and left to the state machine it sits at weight zero, and
  the first build reported "17 morph targets" while the jacket never moved. Second, once all
  five clips were baked, Blender's exporter in ACTIONS mode shipped only the shape-key action
  that happened to be <i>assigned</i> &mdash; 57 morph targets arrived with nothing driving any
  of them. Pairing each clip's skeletal and morph-weight actions on identically-named NLA tracks
  and exporting as tracks merges them into one animation per clip. Verified by reading the
  influences back out of the running page: walking drives <code>walk_0001…walk_0034</code>,
  running drives <code>run_0001…run_0015</code>, idling drives <code>idle_0009…idle_0033</code>,
  with no cross-bleed.</p>
</section>"""


def tpose_section():
    import json as _json
    tp = W / "tpose.json"
    qa = W / "anim_qa.json"
    if not tp.exists():
        return ""
    t = _json.load(open(tp))
    q = _json.load(open(qa)) if qa.exists() else {}
    rows = [[f"<b>{escape(k.replace('_',' '))}</b>",
             f"{t['before'][k]['deg_from_down']}&deg;",
             f"{t['after'][k]['deg_from_down']}&deg;"]
            for k in t.get("before", {})]
    qrows = ""
    if q.get("clips"):
        sw = q.get("shoulder_width_m", 0)
        qr = [[f"<b>{escape(k)}</b>",
               f"{v['min_wrist_to_spine_m']:.3f} m",
               f"{v['max_wrist_behind_chest_m']:+.3f} m",
               f"{v['elbow_angle_range_deg'][0]:.0f}&ndash;{v['elbow_angle_range_deg'][1]:.0f}&deg;",
               f'<span class="{"pass" if v["pass"] else "fail"}">'
               f'{"PASS" if v["pass"] else "FAIL"}</span>']
              for k, v in q["clips"].items()]
        qrows = bars_table(qr, ["Clip", "Min wrist&rarr;spine", "Max wrist behind chest",
                                "Elbow range", ""],
                           ["left", "right", "right", "right", "left"])
    fig = ""
    dn = W / "dense_walkrun.png"
    if dn.exists():
        fig = (f'<figure><img src="{img(dn, 1000)}" alt="sixteen frames of walk and run from two '
               f'angles"><figcaption>Sixteen frames of a stride, walk and run, from two '
               f'angles.</figcaption></figure>')
    return f"""
<section>
  <h2>Why the retargeted walk looked broken, and the one number that explains it</h2>
  <p>The first Mixamo pass produced a walk whose arms swung down <i>through</i> the torso and out
  behind the back. The maths was right by then &mdash; world-space rest deltas, correct units,
  correct axes &mdash; and it still looked wrong. The cause was upstream of the retargeter
  entirely, in what the two rigs call "rest".</p>
  {bars_table(rows, ["Bone", "Rig rest, before", "Rig rest, after"],
              ["left", "right", "right"])}
  <p class="takeaway"><b>Mixamo authors against a T-pose: its upper arm rest points exactly
  90&deg; from straight down. The rig inferred from a generated character is a narrow A-pose, at
  29.4&deg;.</b> Retargeting applies the source's rotation relative to <i>its own</i> rest, so a
  clip that lowers the arm ~70&deg; from horizontal lands at 29 + 70 &asymp; 99&deg; past
  vertical on this rig &mdash; inside the ribcage. The legs escaped because their rests agree to
  within 6&deg;, which is exactly why the failure read as "the arms are broken" rather than "the
  retarget is broken".</p>
  <p>The fix is to move the rig into the space the motion was authored in: pose the arm chain to
  the T, bake that pose into every skinned mesh by applying a <i>copy</i> of its armature
  modifier, then make it the rest pose. Skipping the bake is the classic version of this mistake
  &mdash; the skeleton moves and the geometry does not follow.</p>
  <p class="caveat"><b>The T-pose rest looks alarming, and is fine.</b> Displayed at rest the
  jacket stretches a triangular web between arm and torso, because a garment generated around an
  A-pose has no armpit material to open out. Linear blend skinning is reversible, though: posing
  the arms back down restores the original silhouette exactly, and animations never sit at the
  rest pose. Worth checking rather than assuming &mdash; a webbed rest could equally have meant
  broken weights.</p>
  <h3>Checking it without being able to watch it</h3>
  <p>Frame strips are not playback, and I cannot watch a video. What replaces it is measuring the
  specific defect on every frame of every clip: the distance from each wrist to the spine chain,
  against the torso's own radius. That is what "the arm goes through his body" is, numerically.
  An earlier version of the check measured distance to the body <i>mesh</i> and failed all five
  clips at &minus;0.26&nbsp;m &mdash; a bone joint is meant to sit inside its own flesh, so an
  elbow always reads as "inside the body".</p>
  {qrows}
  {fig}
</section>"""


def bars_table(rows, headers, aligns):
    th = "".join(f'<th class="{a}">{escape(h)}</th>' for h, a in zip(headers, aligns))
    trs = "".join("<tr>" + "".join(f'<td class="{a}">{c}</td>' for c, a in zip(r, aligns)) + "</tr>"
                  for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def stretch_chart(v):
    clips_ = v.get("clips", {})
    order = ["idle", "walk", "wave", "jump", "run", "turn"]
    order = [c for c in order if c in clips_]
    if not order:
        return ""
    w, rowh, gap, left, right = 860, 26, 10, 78, 168
    h = len(order) * (rowh + gap) + 40
    plot = w - left - right
    vmax = 1.4
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="edge stretch by clip and part">']
    for t in (0, 0.25, 0.5, 0.75, 1.0, 1.25):
        x = left + plot * t / vmax
        out.append(f'<line x1="{x:.1f}" y1="20" x2="{x:.1f}" y2="{h-18}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="14" class="tick" text-anchor="middle">{int(t*100)}%</text>')
    colors = {"body": "var(--c-body)", "clothing": "var(--c-cloth)", "hair": "var(--c-hair)"}
    for i, c in enumerate(order):
        y = 26 + i * (rowh + gap)
        out.append(f'<text x="{left-10}" y="{y+16}" class="blabel" text-anchor="end">{escape(c)}</text>')
        sub = clips_[c]["edge_stretch_p99_by_part"]
        for j, part in enumerate(["body", "clothing", "hair"]):
            val = sub.get(part) or 0.0
            bw = max(1.5, plot * min(val, vmax) / vmax)
            yy = y + j * 8
            out.append(f'<rect x="{left}" y="{yy}" width="{bw:.1f}" height="6" rx="2" fill="{colors[part]}"/>')
        v0 = sub.get("body") or 0
        v1 = sub.get("clothing") or 0
        out.append(f'<text x="{left+plot+8}" y="{y+16}" class="bval">body {v0*100:.0f}% / cloth {v1*100:.0f}%</text>')
    out.append("</svg>")
    return "".join(out)


def build():
    v = verify
    verdict = v.get("verdict", {})
    wt = v.get("weights", {})

    stage_rows = []
    for num, name, desc, secs, _ in STAGES:
        stage_rows.append([f'<span class="sn">{num}</span> <b>{escape(name)}</b><br>'
                           f'<span class="sub">{desc}</span>',
                           f"{secs} s" if secs < 60 else f"{secs//60}m {secs%60:02d}s"])

    tool_rows = [[f"<b>{escape(n)}</b><br><span class='sub'>{escape(org)}</span>", escape(lic),
                  f'<span class="tag t-{st}">{escape(mac)}</span>', d] for n, org, lic, mac, d, st in TOOLS]

    comp_rows = [[f"<b>{escape(n)}</b>", escape(rel), escape(price), escape(f)] for n, rel, price, f in COMPETITORS]

    gap_rows = [[f"<b>{escape(a)}</b>", escape(b), escape(c), escape(d)] for a, b, c, d in AAA_GAP]

    gc = gate.get("checks", {})
    gate_rows = []
    for k, r in gc.items():
        jp = r["jev"] >= r["threshold"]
        lp = r["laya"] >= r["threshold"]
        gate_rows.append([escape(k.replace("_", " ")),
                          f'<span class="{"pass" if jp else "fail"}">{r["jev"]:.2f}</span>',
                          f'<span class="{"pass" if lp else "fail"}">{r["laya"]:.2f}</span>',
                          "agree" if jp == lp else '<span class="fail">disagree</span>'])

    p = parts.get("stats", {})
    total_v = sum(p.values()) or 1

    html = f"""<title>CharForge</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{{
  --bg:#F3F4F6; --surface:#FFFFFF; --sunk:#EDEFF2; --ink:#14181F; --ink2:#3C4654; --muted:#69748A;
  --line:#DCE0E7; --line2:#C7CCD6;
  --c-body:#2F5FD8; --c-cloth:#A8811B; --c-hair:#0F8A6A; --c-acc:#B4483A;
  --good:#0F8A6A; --bad:#B4483A; --warn:#A8811B; --accent:#2F5FD8;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,sans-serif;
  --serif:"IBM Plex Serif",Georgia,serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#0F1216; --surface:#161B22; --sunk:#1B212A; --ink:#E7EBF0; --ink2:#C2CAD6; --muted:#8B97A8;
  --line:#242B36; --line2:#323B49;
  --c-body:#6E9BFF; --c-cloth:#D1A73A; --c-hair:#3FBE97; --c-acc:#E27A6B;
  --good:#3FBE97; --bad:#E27A6B; --warn:#D1A73A; --accent:#6E9BFF;
}}}}
:root[data-theme="dark"]{{
  --bg:#0F1216; --surface:#161B22; --sunk:#1B212A; --ink:#E7EBF0; --ink2:#C2CAD6; --muted:#8B97A8;
  --line:#242B36; --line2:#323B49;
  --c-body:#6E9BFF; --c-cloth:#D1A73A; --c-hair:#3FBE97; --c-acc:#E27A6B;
  --good:#3FBE97; --bad:#E27A6B; --warn:#D1A73A; --accent:#6E9BFF;
}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:var(--sans);margin:0;font-size:15px;line-height:1.6}}
.wrap{{max-width:1060px;margin:0 auto;padding-inline:20px}}
header.top{{border-bottom:1px solid var(--line);background:var(--surface)}}
header.top .wrap{{padding-block:34px 26px}}
.eyebrow{{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:14px}}
h1{{font-weight:600;font-size:clamp(28px,4.4vw,42px);line-height:1.1;margin:0 0 14px;letter-spacing:-.022em;text-wrap:balance}}
.lede{{font-family:var(--serif);font-size:17.5px;line-height:1.62;color:var(--ink2);max-width:66ch;margin:0}}
.lede b{{color:var(--ink)}}
main .wrap{{padding-block:34px 70px}}
section{{margin-bottom:52px}}
h2{{font-size:23px;font-weight:600;letter-spacing:-.018em;margin:0 0 8px}}
h3{{font-size:16.5px;font-weight:600;margin:26px 0 8px}}
p{{margin:0 0 12px;max-width:74ch}}
.muted,.sub{{color:var(--muted)}}
.sub{{font-size:12.5px}}
code{{font-family:var(--mono);font-size:.88em;background:var(--sunk);padding:1px 5px;border-radius:4px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(152px,1fr));gap:12px;margin:18px 0}}
.kpi{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:13px 15px}}
.kpi .n{{font-family:var(--mono);font-size:21px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}}
.kpi .l{{font-size:11.5px;color:var(--muted);margin-top:3px;line-height:1.35}}
.kpi.ok .n{{color:var(--good)}} .kpi.warn .n{{color:var(--warn)}}
.tw{{overflow-x:auto;margin:16px 0;border:1px solid var(--line);border-radius:8px;background:var(--surface)}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;min-width:540px}}
th,td{{padding:9px 13px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}
thead th{{font-family:var(--mono);font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:500;background:var(--sunk);white-space:nowrap}}
tbody tr:last-child td{{border-bottom:none}}
td.right,th.right{{text-align:right;font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums}}
.sn{{font-family:var(--mono);font-size:11px;color:var(--accent);margin-right:4px}}
figure{{margin:18px 0}}
figure img{{width:100%;max-width:100%;display:block;border:1px solid var(--line);border-radius:8px;background:var(--sunk)}}
figcaption{{font-size:12.5px;color:var(--muted);margin-top:7px;max-width:74ch}}
.chart{{width:100%;height:auto;display:block;margin:8px 0}}
.chart .grid{{stroke:var(--line);stroke-width:1}}
.chart .tick{{fill:var(--muted);font-family:var(--mono);font-size:10px}}
.chart .blabel{{fill:var(--ink2);font-family:var(--sans);font-size:12.5px}}
.chart .bval{{fill:var(--muted);font-family:var(--mono);font-size:10.5px}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--muted);margin:6px 0 2px}}
.legend i{{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:5px}}
.takeaway{{border-left:3px solid var(--accent);padding:10px 0 10px 15px;margin:16px 0;font-family:var(--serif);font-size:15.5px;max-width:74ch}}
.caveat{{background:var(--sunk);border-radius:8px;padding:12px 15px;font-size:13.5px;color:var(--ink2);max-width:74ch}}
.pass{{color:var(--good);font-family:var(--mono)}} .fail{{color:var(--bad);font-family:var(--mono)}}
.tag{{display:inline-block;font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:99px;border:1px solid var(--line2);white-space:nowrap}}
.t-ran{{border-color:var(--good);color:var(--good)}}
.t-installed{{border-color:var(--accent);color:var(--accent)}}
.t-blocked{{border-color:var(--bad);color:var(--bad)}}
.t-downloaded,.t-not{{color:var(--muted)}}
ul.plain{{padding-left:18px;font-size:14px;color:var(--ink2);max-width:74ch}}
ul.plain li{{margin-bottom:7px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:760px){{.two{{grid-template-columns:1fr}}}}
</style>

<header class="top"><div class="wrap">
  <div class="eyebrow">Local pipeline &middot; Apple M5 / 24 GB &middot; no cloud GPU</div>
  <h1>A character factory that runs on the laptop</h1>
  <p class="lede">Text in, rigged and animated game character out, entirely on this machine:
  <b>{TOTAL//60} minutes {TOTAL%60} seconds</b> from a prompt to a {v.get('vertices',0):,}-vertex character with
  PBR textures, a 20-bone skeleton and six animation clips. The test was the hard case on purpose &mdash;
  <b>a man in a jacket with hair</b> &mdash; because hair and accessories are what naive rigging destroys.
  Across all six clips the hair's measured deviation from rigid attachment to the skull was
  <b>0.00%</b>.</p>
</div></header>

<main><div class="wrap">

<section>
  <h2>The result</h2>
  <div class="kpis">
    <div class="kpi ok"><div class="n">0.00%</div><div class="l">hair deviation from rigid head attachment, worst clip</div></div>
    <div class="kpi"><div class="n">{v.get('vertices',0):,}</div><div class="l">vertices &middot; {int(v.get('edges',0)):,} edges</div></div>
    <div class="kpi"><div class="n">{TOTAL//60}m {TOTAL%60:02d}s</div><div class="l">prompt to animated GLB, end to end</div></div>
    <div class="kpi ok"><div class="n">{wt.get('max_influences','?')}</div><div class="l">max bone influences per vertex (engine budget is 4)</div></div>
  </div>
  <figure>
    <img src="{img(W/'hero'/'pair.png', 900)}" alt="Generated character, textured and coloured by semantic part">
    <figcaption>Left: the generated character as it exports. Right: the same mesh coloured by the part
    labels the pipeline inferred &mdash; <span style="color:var(--c-cloth)">clothing</span>,
    <span style="color:var(--c-body)">body</span>, <span style="color:var(--c-hair)">hair</span>,
    <span style="color:var(--c-acc)">accessory</span>. Those labels are what let each part be bound
    differently: skin deforms, hair rides the skull rigidly.</figcaption>
  </figure>
</section>

<section>
  <h2>What runs, stage by stage</h2>
  <p>Every stage runs locally. Timings are from the actual run on the M5, excluding one-off model
  downloads.</p>
  {bars_table(stage_rows, ["Stage", "Time"], ["left", "right"])}
  <figure>
    <img src="{img(W/'contact.png', 900)}" alt="Eight orbit views of the generated character">
    <figcaption>The generated mesh from eight orbit views. The back of a character is invented by the
    model &mdash; it was never in the reference image &mdash; which is the usual place single-image 3D
    generation falls apart. Here the jacket seams, hem and hair close coherently.</figcaption>
  </figure>
</section>

<section>
  <h2>Does it survive animation?</h2>
  <p>This is the question the brief actually asked. A generated character arrives as one fused mesh:
  hair welded to scalp, jacket welded to torso. Bind that naively and the hair shears across the
  shoulders the first time the character turns its head. So the pipeline measures it.</p>
  <div class="legend">
    <span><i style="background:var(--c-body)"></i>body</span>
    <span><i style="background:var(--c-cloth)"></i>clothing</span>
    <span><i style="background:var(--c-hair)"></i>hair</span>
  </div>
  {stretch_chart(v)}
  <p class="sub">99th-percentile edge stretch per part, per clip. Lower is better; 0% means the part
  moved rigidly. <code>turn</code> is a whole-body rotation and reads exactly 0% across every part,
  which is the control that proves the metric is measuring deformation and not motion.</p>
  <figure>
    <img src="{img(W/'anim_sheet.png', 860)}" alt="Six animation clips, four sampled poses each">
    <figcaption>All six clips, four poses each. Hair stays on the head through the run cycle and the
    360&deg; turn; the jacket follows the torso; the arms and legs swing from the inferred skeleton.</figcaption>
  </figure>
  <p class="takeaway"><b>Hair and accessories: 0.00% deviation from rigid attachment, every clip.</b>
  {p.get('hair',0):,} hair vertices bind to the head bone and {p.get('accessory',0):,} accessory
  vertices to their nearest bone, so they travel with the skull instead of being dragged by the spine.
  That is the whole point of segmenting the mesh before rigging it.</p>
  <p class="caveat"><b>Where it still fails:</b> clothing stretches up to
  {max((c['edge_stretch_p99_by_part'].get('clothing') or 0) for c in v.get('clips',{}).values())*100:.0f}%
  at the 99th percentile on the most extreme clips, concentrated at the hip/waist band where the jacket
  hem meets rotating legs. Laplacian weight smoothing cut that roughly in half (run cycle: 226% &rarr;
  118%; body 57% &rarr; 40%), but linear blend skinning on a fused mesh cannot fully fix it. The
  production answer is to split the jacket into its own mesh &mdash; which the part labels already
  make possible &mdash; and simulate or correct it. The splitting half of that is now done and
  measured below; the simulation half is not.</p>
</section>

{engine_section()}

{weld_section()}

{split_section()}

{retopo_section()}

{backview_section()}

{multiview_section()}

{animation_section()}

{mixamo_section()}

{tpose_section()}

{distill_section()}

{pixal3d_section()}

{second_run_section()}

<section>
  <h2>The verification gate, and an awkward result</h2>
  <p>Each stage ends with a check rather than a hope. A local vision model describes the render, and
  typed questions turn that description into pass/fail. The vision model's description of the reference
  image was accurate and complete &mdash; it named the bomber jacket, the medium-length wavy hair, the
  boots, the plain background, the full-body framing.</p>
  {bars_table(gate_rows, ["Check", "Jev", "Laya (local)", ""], ["left", "right", "right", "left"])}
  <p class="takeaway"><b>Jev answered {gate.get('jev_passed','?')}/9 correctly in
  {gate.get('jev_latency_ms',0):.0f}&nbsp;ms for ${gate.get('jev_cost_usd',0):.6f}. Laya, zero-shot and
  local, got {gate.get('laya_passed','?')}/9</b> &mdash; it read a description that explicitly says
  "the full figure of a person from head to feet" and scored <i>full body</i> at 0.41. The questions
  were not the problem; the same text and thresholds went to both.</p>
  <p>This matters for a pipeline that is supposed to be fully local. The options are to fine-tune Laya
  on Jev-labelled examples of exactly these questions, or to let the vision model answer the closed
  questions directly. What does not work is assuming an off-the-shelf small decision model is reliable
  zero-shot &mdash; Laya's own documentation says the base checkpoints need domain fine-tuning, and this
  is what that looks like in practice.</p>
</section>

<section>
  <h2>What ran on this hardware, and what did not</h2>
  <p>The brief asked to try each generator on the same test. Apple Silicon is the deciding constraint:
  most 3D generation ships CUDA kernels.</p>
  {bars_table(tool_rows, ["Tool", "Licence", "Apple Silicon", "Finding"], ["left", "left", "left", "left"])}
  <p class="caveat"><b>On Pixal3D specifically</b> &mdash; the paper in the brief. The claim in this
  table that it needs a CUDA-only NATTEN kernel came from the repository's own README and turned out
  to be wrong on inspection; see the Pixal3D section above for what actually blocks it and what now
  runs. Since it shares the TRELLIS.2 backbone that
  already runs here, that port is a bounded piece of work rather than a research project &mdash; but it
  is work, and it was not done.</p>
</section>

<section>
  <h2>Against the commercial tools</h2>
  {bars_table(comp_rows, ["Product", "Version", "Price", "What it does that this does not"], ["left", "left", "left", "left"])}
  <p>The honest position: these products are ahead on polish &mdash; quad topology, 8K textures, a
  motion library, a rig that handles quadrupeds. What this pipeline has instead is that it runs on a
  laptop with no per-asset cost, the intermediate artefacts are all inspectable, and the parts are
  semantically labelled, which none of the hosted tools expose. For a studio generating hundreds of
  background characters under an IP restriction that forbids uploading concept art, that combination
  is the product.</p>
</section>

<section>
  <h2>Distance to the bar in the brief</h2>
  <p>The brief asked for Battlefield 6 / Elden Ring fidelity. That is not what this produces, and it is
  not what Meshy, Rodin or Tripo produce either. A hero character at that level is weeks of skilled work:
  sculpt, retopology, UV layout, baked maps, groomed hair, a facial rig, cloth setup. Here is the gap,
  item by item, with the route to close each one.</p>
  {bars_table(gap_rows, ["Aspect", "What this produces", "What AAA requires", "How to close it"], ["left", "left", "left", "left"])}
  <p class="takeaway"><b>Where this output actually belongs today:</b> background and crowd characters,
  NPCs seen at conversation distance, previsualisation, prototype and jam projects, and as a sculpt
  base a human artist then retopologises. That is a real market &mdash; it is most of the characters in
  most games &mdash; but it is not the hero character in the cinematic.</p>
</section>

<section>
  <h2>Run it</h2>
  <p>The pipeline is nine scripts, each independently runnable, each writing JSON a build system can
  gate on.</p>
  <div class="tw"><table><tbody>
    <tr><td><code>pipeline/comfy.py</code></td><td>reference image via ComfyUI + Krea 2 Turbo</td></tr>
    <tr><td><code>pipeline/gates.py</code></td><td>local VLM description + typed pass/fail gate</td></tr>
    <tr><td><code>pipeline/convert_dinov3.py</code></td><td>ungated timm DINOv3 &rarr; the layout TRELLIS ports expect</td></tr>
    <tr><td><code>blender/render_views.py</code></td><td>orbit renders + exact per-vertex visibility</td></tr>
    <tr><td><code>pipeline/parts.py</code></td><td>human parser back-projected onto the mesh</td></tr>
    <tr><td><code>pipeline/skeleton.py</code></td><td>ViTPose &rarr; 3D joints, symmetrised and snapped</td></tr>
    <tr><td><code>blender/build_rig.py</code></td><td>armature + per-part skinning</td></tr>
    <tr><td><code>blender/animate.py</code></td><td>six self-contained clips</td></tr>
    <tr><td><code>blender/verify_deform.py</code></td><td>the measurements in this report</td></tr>
  </tbody></table></div>
  <p class="sub">The viewer in <code>charforge/viewer</code> loads the exported GLB with the clip list,
  the part colouring and these same checks: <code>python3 -m http.server -d charforge/viewer 8900</code>.</p>
</section>

<section>
  <h2>What I would do next</h2>
  <ul class="plain">
    <li><b>Retopology and normal baking.</b> The 1.3M-vertex mesh TRELLIS produces before simplification
    is thrown away. Baking it onto a 40k quad retopology would close most of the visible quality gap and
    is entirely deterministic work in Blender.</li>
    <li><b>Split the parts into separate meshes.</b> The labels exist and export with the model; using
    them to separate the jacket would let it be simulated instead of skinned, which is where the
    remaining stretch lives.</li>
    <li><b>Multi-view conditioning.</b> The MLX TRELLIS port accepts multiple conditioning images and
    only one was used. Generating consistent side and back views first should improve the invented
    back of the character.</li>
    <li><b>Fine-tune Laya on the gate questions</b> using Jev as the teacher, so the verification layer
    is both reliable and local.</li>
    <li><b>Port NATTEN to Metal</b> to unlock Pixal3D on this hardware.</li>
  </ul>
</section>

</div></main>"""
    out = ROOT / "report.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    build()
