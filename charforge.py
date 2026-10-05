#!/usr/bin/env python3
"""CharForge: a text prompt or an image in, a game-ready animated character out.

    python charforge.py make --prompt "a knight in weathered plate armour" --name knight
    python charforge.py make --image concept.png --name knight --height 1.85
    python charforge.py make --name wren --from rig          # rerun from a stage onward
    python charforge.py stages                               # list the stages
    python charforge.py move --name wren --move punch_combo  # a new move, from a generated video
    python charforge.py move --name aoi --move punch_combo --performer wren   # ...from Wren's video
    python charforge.py studio                               # the same, from a browser: localhost:8830

This replaces run_pipeline.sh, which ran a pipeline that had been thrown out - skinning the raw
generated surface and animating it with procedural gait curves - while every character that
actually shipped was built by chaining scripts by hand. There is now one path, and it is the
path the published characters were built on.

Every stage writes into work/<name>/ and is skipped when its output already exists, so a run
that fails at stage 12 resumes at stage 12. The finished character lands in out/<name>/: glTF,
FBX, loose textures and a manifest, plus a compressed build for the web.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRELLIS = ROOT / "vendor" / "trellis2mlx"
HUNYUAN = ROOT / "vendor" / "hunyuan3d-2.1-mac-rocm"      # tools/setup_hunyuan.sh; optional (texture views)


def hunyuan_shape_ready() -> bool:
    return ((HUNYUAN / "venv" / "bin" / "python").exists()
            and (HUNYUAN / "weights" / "Hunyuan3D-2.1" / "hunyuan3d-dit-v2-1").is_dir())


def hunyuan_ready() -> bool:
    return ((HUNYUAN / "venv" / "bin" / "python").exists()
            and (HUNYUAN / "weights" / "Hunyuan3D-2.1" / "hunyuan3d-paintpbr-v2-1").is_dir())
DWPOSE = ROOT / "models" / "dwpose" / "dw-ll_ucoco_384.onnx"    # optional: the hands (tools/dwpose.py)
MIXAMO_HF = Path.home() / ".cache/huggingface/hub/datasets--jasongzy--Mixamo/snapshots"


# Art styles. The style is a property of the character, not of the renderer: it decides how the
# reference image is asked for, how the side and back views are repainted, and it is recorded in
# the package so a viewer or an engine picks the matching shading (a cel-shaded character lit
# like a photograph looks like neither). Every other stage is the same pipeline.
STYLES = {
    "realistic": {
        "look": "photorealistic game character, realistic skin, hair and fabric detail",
        "negative": "cartoon, anime, illustration, cel shading",
        "views": "photorealistic, detailed fabric and seams",
    },
    "anime": {
        "look": ("anime style 3D game character, cel-shaded, clean flat colours with crisp shadows, "
                 "anime face with large expressive eyes, stylised hair in clean clumps, Genshin Impact style"),
        "negative": "photorealistic, photo, realistic skin pores, western cartoon, blurry",
        "views": "anime style, cel-shaded, clean flat colours, crisp outlines",
    },
    "stylized": {
        "look": ("stylised 3D cartoon game character, chunky appealing proportions, slightly larger head, "
                 "hands and feet, simple clean shapes, hand-painted textures, Fortnite and Overwatch style"),
        "negative": "photorealistic, photo, anime, thin realistic proportions, blurry",
        "views": "stylised cartoon 3D, hand-painted textures, clean shapes",
    },
}


def _recorded(r, key, given, default):
    """A per-character setting: given on this run (and recorded), else as recorded when the
    character was first made, else the default. A rerun from a late stage without --height used to
    fall back to 1.75 m and silently resize a 1.70 m character."""
    f = r.work / "style.json"
    rec = json.load(open(f)) if f.exists() else {}
    if given is not None:
        if rec.get(key) != given:
            rec[key] = given
            r.work.mkdir(parents=True, exist_ok=True)
            json.dump(rec, open(f, "w"))
        return given
    return rec.get(key, default)


def style_of(r) -> str:
    """The character's style: given now, or recorded when it was first generated."""
    return _recorded(r, "style", getattr(r.a, "style", None), "realistic")


def option_of(r, key, default):
    """An opt-in build option (--texture-views, --head, --rest): given now and recorded, or as recorded. A rebuild from
    a later stage without it used to drop it: v0.13's benchmark rebuild gave Pip back the generator's blotchy sides
    where v0.12 had built him with Hunyuan3D-Paint's."""
    return _recorded(r, key, getattr(r.a, key, None), default)


def image_model_of(r) -> str:
    """The image model the reference is made with: given now, or recorded when first made."""
    return _recorded(r, "image_model", getattr(r.a, "image_model", None), "qwen21")


def prompt_of(r):
    """The character's description: given now, or recorded when it was first made."""
    return _recorded(r, "prompt", getattr(r.a, "prompt", None), None)


def height_of(r) -> float:
    """The character's standing height in metres: given now, or recorded when first made, else the
    height the description implies (a child 1.2-1.5 m - "a boy scout" came out a grown man at the
    Studio's 1.72), else 1.75."""
    f = r.work / "style.json"
    auto = (json.load(open(f)) if f.exists() else {}).get("height_auto")
    return float(_recorded(r, "height", getattr(r.a, "height", None), auto or 1.75))


def _record(r, **kv):
    f = r.work / "style.json"
    rec = json.load(open(f)) if f.exists() else {}
    rec.update(kv)
    r.work.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(f, "w"), indent=1)


def description_of(r):
    """What the reference image is asked for: the prompt written out (pipeline/describe.py) - an age and a
    build, skin and hair, every garment with its colour - made once and recorded with the character.
    A one-line prompt left all of that to the image model: "a boy scout" came back a grown man, "a gray
    alien, extremely muscular" a grey clay sculpture with no clothes. --literal keeps the prompt as
    written; --description gives the written-out one (the Studio's edited text)."""
    prompt = prompt_of(r)
    if not prompt:
        return None
    f = r.work / "style.json"
    rec = json.load(open(f)) if f.exists() else {}
    given = getattr(r.a, "description", None)
    if given:
        given = " ".join(given.split())
        if rec.get("description") != given or rec.get("description_for") != prompt:
            _record(r, description=given, description_for=prompt, description_engine="given", literal=False)
        return given
    if getattr(r.a, "literal", False):
        _record(r, literal=True)
        return prompt
    if rec.get("literal"):
        return prompt
    if rec.get("description") and rec.get("description_for") == prompt:
        return rec["description"]
    sys.path.insert(0, str(ROOT / "pipeline"))
    from describe import describe  # noqa: E402
    with gpu("writing the prompt out"):
        d = describe(prompt, style_of(r), getattr(r.a, "height", None))
    _record(r, description=d["description"], description_for=prompt, description_engine=d["engine"],
            age=d["age"], height_auto=d["height_m"])
    print(f"      described ({d['engine']}): {d['age']}, {d['height_m']:.2f} m - {d['description']}", flush=True)
    return d["description"]


_GPU = {"depth": 0, "f": None}
GPU_SCRIPTS = {"pose_gate.py", "parts.py", "skeleton.py", "face_landmarks.py", "refine_pose.py", "head_crop.py"}


@contextlib.contextmanager
def gpu(what):
    """One GPU job at a time across every CharForge process on this Mac. Two refines at once corrupted
    each other - Aoi's silhouette went 0.672 -> 0.647 and Mara's kick 0.733 -> 0.485, where alone they
    reach 0.82-0.86 - and a TRELLIS pass beside another was stopped by macOS twice. Held for TRELLIS, the
    image model, the prompt model and the refine; a second process waits here (reentrant within one)."""
    if _GPU["depth"]:
        _GPU["depth"] += 1
        try:
            yield
        finally:
            _GPU["depth"] -= 1
        return
    import fcntl
    p = ROOT / "work" / ".gpu.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a+")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"      waiting for the GPU (another CharForge job is using it) before {what}", flush=True)
            t0 = time.time()
            fcntl.flock(f, fcntl.LOCK_EX)
            print(f"      the GPU is free after {time.time() - t0:.0f}s", flush=True)
        _GPU.update(depth=1, f=f)
        yield
    finally:
        _GPU.update(depth=0, f=None)
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


def blender_bin() -> str:
    for cand in (os.environ.get("BLENDER"), shutil.which("blender"),
                 "/Applications/Blender.app/Contents/MacOS/Blender"):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit("Blender not found: install Blender 5.x or set BLENDER=/path/to/blender")


# ---- the stages ------------------------------------------------------------------------------
# name, what it produces, one line of what it is for. Order is execution order.
STAGES = [
    ("reference", "reference.png", "the image everything is generated from"),
    ("generate",  "pass1.glb",     "TRELLIS.2: geometry and PBR texture from the reference"),
    ("multiview", "mesh.glb",      "the 3D model, its surface filtered (--quality best: a second pass on repainted views)"),
    ("views",     "views/projection.json", "orbit renders for segmentation and pose"),
    ("parts",     "parts.json",    "per-vertex body / clothing / hair / accessory labels"),
    ("skeleton",  "joints.json",   "3D joint positions from the orbit renders"),
    ("solidify",  "solid.npz",     "one clean solid: flakes closed, inner shells and shards gone"),
    ("joints",    "joints_refined.json", "every joint traced onto the centre line of its limb"),
    ("hands",     "solid_hands.glb", "generated paddle hands replaced by modelled hands with fingers"),
    ("retopo",    "retopo.glb",    "decimated to 60k triangles; colour, normal, roughness/metal, AO baked"),
    ("labels",    "labels.json",   "part labels carried onto the clean mesh"),
    ("texclean",  "albedo_clean.png", "skin the generator painted onto clothing, removed"),
    ("texture",   "albedo.png",    "the reference, delit, and painted side and back views projected onto the mesh"),
    ("weights",   "weights.npz",   "skin weights by distance measured through the body"),
    ("rig",       "rig.blend",     "skeleton with finger bones; weights applied"),
    ("springs",   "rig_s.blend",   "bone chains for what hangs - braids, ponytails, bags"),
    ("frame",     "rig_m.blend",   "metres, soles on the floor, origin under the pelvis"),
    ("tpose",     "rig_t.blend",   "the T the clips are retargeted to, hands squared (kept as the rest with --rest T)"),
    ("face",      "rig_f.blend",   "face rig: jaw bone, blink / smile / brows / pucker shapes"),
    ("animate",   "final.blend",   "Mixamo captures retargeted onto the rig, fingers included"),
    ("refine",    "refine.json",   "clips made from the character's own videos re-posed to lie on them"),
    ("package",   "PACKAGE",       "glTF + FBX + textures + manifest, Mixamo bone names"),
    ("web",       "WEB",           "compressed build for the browser"),
]
NAMES = [s[0] for s in STAGES]


class Run:
    def __init__(self, a):
        self.a = a
        self.work = ROOT / "work" / a.name
        self.out = ROOT / "out" / a.name
        self.logs = self.work / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.blender = None
        self.t0 = time.time()

    def path(self, rel):
        if rel == "PACKAGE":
            return self.out / f"{self.a.name}.glb"
        if rel == "WEB":
            return self.out / f"{self.a.name}_web.glb"
        return self.work / rel

    def sh(self, stage, cmd, cwd=None, env=None, keep=None):
        """Run one command, logging everything, echoing only the lines that carry a result."""
        log = self.logs / f"{stage}.log"
        with open(log, "w") as fh:
            fh.write("$ " + " ".join(str(c) for c in cmd) + "\n")     # the command, to replay a stage by hand
            fh.flush()
            p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=fh,
                               stderr=subprocess.STDOUT)
        lines = log.read_text(errors="replace").splitlines()
        if keep:
            for ln in lines:
                if any(k in ln for k in keep):
                    print(f"      {ln.strip()[:150]}", flush=True)
        if p.returncode != 0:
            tail = "\n".join("      " + x for x in lines[-15:])
            raise SystemExit(f"\n[{stage}] failed (exit {p.returncode}). Last lines of {log}:\n{tail}")

    def bl(self, stage, script, *args, keep):
        self.blender = self.blender or blender_bin()
        with gpu(f"{stage} (Blender)"):                   # renders and bakes: the GPU
            self.sh(stage, [self.blender, "-b", "-noaudio", "--python", ROOT / "blender" / script,
                            "--", *args], keep=keep)

    def py(self, stage, script, *args, keep):
        # the scripts that run a model on the GPU (PyTorch): beside another, Pip's refine started from a
        # silhouette term 2.33 where alone it starts at 0.0013, and went to NaN
        with gpu(stage) if script in GPU_SCRIPTS else contextlib.nullcontext():
            self.sh(stage, [sys.executable, ROOT / "pipeline" / script, *args], keep=keep)


# ---- stage bodies ----------------------------------------------------------------------------
def reference_prompt(desc: str, style: str, model: str = "krea2"):
    """The reference image's prompt and negative for a description and style. Qwen-Image 2.1
    runs at cfg 1, which ignores a negative, so what the negative excludes is asked for in the
    positive; its background is its own alpha, not a grey field."""
    st = STYLES[style]
    neg = ("cropped, close-up, multiple people, text, watermark, dramatic shadows, blurry, sitting, "
           "holding objects, " + st["negative"])
    body = (f"full body character reference sheet of {desc}, standing in a relaxed A-pose, "
            "arms slightly away from the body, hands open with the fingers relaxed, facing the camera "
            f"directly, neutral expression, {st['look']}, even neutral studio lighting, no shadows")
    if model == "qwen21":
        # Qwen follows "arms slightly away from the body" literally: its hands hung 0.28-0.37 torso
        # lengths from the hips, against Krea's 0.61-0.72, and hands that near the thighs fuse to
        # them in the 3D model. The angle is spelled out.
        body = body.replace("standing in a relaxed A-pose, arms slightly away from the body",
                            "standing in an A-pose with both arms held straight and angled down and out at "
                            "about 35 degrees from the body, the hands well clear of the hips and thighs")
        return (f"{body}, a single character alone, painted in full colour, no text, no props, full figure "
                "visible head to toe, centered, orthographic view"), neg
    return (f"{body}, plain flat light grey background, full figure visible head to toe, centered, "
            "orthographic view"), neg


ARM_GAP_MIN = 0.45      # torso lengths between each wrist and its hip below which the hands fuse to the body


def reference_arm_gap(r: Run, path) -> float | None:
    """How far the reference's hands hang clear of the body (pipeline/pose_gate.py), None if unread."""
    out = r.work / "reference_check.json"
    out.unlink(missing_ok=True)
    try:
        r.py("reference_check", "pose_gate.py", "--image", path, "--out", out, keep=())
        return json.load(open(out)).get("arm_gap")
    except (SystemExit, OSError, ValueError):
        return None


def s_reference(r: Run):
    out = r.path("reference.png")
    if r.a.image:
        src = Path(r.a.image).expanduser()
        if not src.exists():
            raise SystemExit(f"--image {src} does not exist")
        shutil.copy(src, out)
        print(f"      using your image {src.name}", flush=True)
        gap = reference_arm_gap(r, out)
        if gap is not None and gap < ARM_GAP_MIN:
            print(f"      warning: the hands are close to the body ({gap:.2f} torso lengths from the hips) - they will "
                  f"fuse to it in the 3D model; a picture with the arms held away from the body works better",
                  flush=True)
        return
    desc = description_of(r)                 # the prompt written out (given now, or remembered)
    if not desc:
        raise SystemExit("give --prompt or --image")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import comfy  # noqa: E402  (needs ComfyUI with Krea 2 on :8188)
    full, neg = reference_prompt(desc, style_of(r), image_model_of(r))
    qwen = image_model_of(r) == "qwen21"
    # Drawn again, on a new seed, while the hands touch the body: however the prompt spells out the
    # A-pose, an image model sometimes draws the character's habitual stance instead - Bo the chef
    # came with his fists on his hips (0.33 torso lengths), and the 3D model fused them to his belt.
    best = None
    for k in range(3):
        seed = r.a.seed + 1000 * k
        if qwen:
            wf = comfy.qwen21(full, width=832, height=1216, steps=30, seed=seed, transparent=True)
            with gpu("drawing the reference"):
                paths, dt = comfy.run(wf, r.work, prefix=f"ref{k}" if k else "ref", timeout=3600)
        else:
            wf = comfy.krea2_t2i(full, neg, width=832, height=1216, steps=8, cfg=1.0, seed=seed, gguf=True)
            with gpu("drawing the reference"):
                paths, dt = comfy.run(wf, r.work, prefix=f"ref{k}" if k else "ref", timeout=2400)
        gap = reference_arm_gap(r, paths[0])
        print(f"      prompt -> image in {dt:.0f}s ({'Qwen-Image 2.1, with its own alpha' if qwen else 'Krea 2'}, "
              f"seed {seed}); hands {'unread' if gap is None else f'{gap:.2f} torso lengths'} from the hips",
              flush=True)
        if best is None or (gap if gap is not None else 9) > (best[1] if best[1] is not None else 9):
            best = (paths[0], gap, seed)
        if gap is None or gap >= ARM_GAP_MIN:
            break
        if k < 2:
            print("      the hands touch the body and would fuse to it in the 3D model - drawing it again", flush=True)
    src, gap, seed = best
    if gap is not None and gap < ARM_GAP_MIN:
        print(f"      three draws and the hands still touch the body: keeping the clearest ({gap:.2f}, seed {seed})",
              flush=True)
    if qwen:
        # Qwen-Image 2.1 draws the figure with its own alpha: TRELLIS takes that cut-out as it is
        # (no rembg), the texture stage takes the mask from it, and the stages that read the
        # reference as a picture get it on the same plain light grey Krea is asked for.
        from PIL import Image
        rgba = Image.open(src).convert("RGBA")
        rgba.save(r.path("reference_rgba.png"))
        grey = Image.new("RGBA", rgba.size, (205, 205, 205, 255))
        Image.alpha_composite(grey, rgba).convert("RGB").save(out)
        rgba.getchannel("A").point(lambda v: 255 if v >= 128 else 0).save(r.path("reference_mask.png"))
    else:
        shutil.copy(src, out)


def free_comfy():
    """Ask ComfyUI to drop its cached models before TRELLIS runs. On Apple Silicon the GPU shares
    system memory, and ComfyUI keeps the image model resident after a job: the next TRELLIS pass
    then runs against it and the machine swaps (13.8 GB of swap in use on a 24 GB laptop, a
    reference image that took 259 s). The model is the one this pipeline just used, and ComfyUI
    reloads it on the next request; --keep-comfy-loaded turns this off. Best effort - no ComfyUI,
    nothing to free."""
    import urllib.request
    try:
        req = urllib.request.Request("http://127.0.0.1:8188/free", method="POST",
                                     data=b'{"unload_models": true, "free_memory": true}',
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:                                        # noqa: BLE001
        pass


def trellis(r: Run, stage, images, out, seed=None):
    if not r.a.keep_comfy_loaded:
        free_comfy()
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH=".", TRELLIS_EVAL_EVERY="2")
    dino = ROOT / "models" / "dinov3-vitl16-hf"
    if dino.exists():
        env["TRELLIS_DINOV3_PATH"] = str(dino)
    py = TRELLIS / ".venv" / "bin" / "python"
    if not py.exists():
        raise SystemExit(f"TRELLIS environment missing at {py} - see README, 'Running it'")
    cmd = [py, "-u", "generate.py", "--image", *images, "--output", out,
           "--resolution", "1024", "--steps", "12", "--target-faces", str(getattr(r.a, "trellis_faces", 1_000_000)),
           # its colours at 2048: at the default 1024 a million faces' charts are a texel or two across (E140)
           "--texture-size", "2048",
           "--seed", str(r.a.seed if seed is None else seed)]
    try:
        with gpu("TRELLIS"):
            r.sh(stage, cmd, cwd=TRELLIS, env=env, keep=("Total:", "Saved:"))
    except SystemExit:
        # macOS kills a GPU command buffer that holds the display up too long ("Impacting Interactivity"):
        # Knight's and Kaito's 1024 stages (~8k tokens, six transformer blocks a buffer) both died of it
        # while Bo's smaller one passed. Once more, a block a buffer and fewer operations in each.
        if "Impacting Interactivity" not in (r.logs / f"{stage}.log").read_text(errors="replace"):
            raise
        print("      the GPU took too long in one go and macOS stopped it - trying again in smaller pieces",
              flush=True)
        env.update(TRELLIS_EVAL_EVERY="1", MLX_MAX_OPS_PER_BUFFER="4")
        with gpu("TRELLIS, again"):
            r.sh(stage, cmd, cwd=TRELLIS, env=env, keep=("Total:", "Saved:"))


def reference_for_3d(r: Run, route=None):
    """The reference as TRELLIS should see it: the cut-out with the image model's own alpha when
    there is one (TRELLIS then skips its background remover), else the picture - or the picture
    anyway where that cut-out gave a board (generate.json's "input": "rembg")."""
    if route is None:
        g = r.work / "generate.json"
        route = (json.load(open(g)) if g.exists() else {}).get("input", "alpha")
    rgba = r.path("reference_rgba.png")
    return rgba if rgba.exists() and route == "alpha" else r.path("reference.png")


FRONT_IOU_MIN = 0.8     # a model's front silhouette on its reference's: sound models 0.90-0.94


def front_iou(r: Run, glb, tag) -> float:
    """How the model's front silhouette lies on the reference's (align.similarity: best scale and shift).
    TRELLIS now and then builds a flat board into the model with the figure - in front of it (Gray,
    0.44), behind it (Knight, 0.39) or through it (Kaito, 0.35) - the reference's grey backdrop, made
    solid; the rig then cut it into membranes and loose pieces."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT / "pipeline"))
    from align import image_mask, similarity  # noqa: E402
    d = r.work / "gate" / tag
    r.bl("gate", "render_views.py", "--mesh", glb, "--out", d, "--views", 1, "--res", 512, "--elev", 0,
         "--ortho", 2.15, keep=())
    rm = np.asarray(Image.open(d / "view_00.png").convert("RGBA"))[..., 3] > 127
    mask = r.path("reference_mask.png")
    im = (np.asarray(Image.open(mask).convert("L")) > 127 if mask.exists()
          else image_mask(np.asarray(Image.open(r.path("reference.png")).convert("RGB"))))
    return float(similarity(rm, im)[1]) if rm.any() and im.any() else 0.0


def s_generate(r: Run):
    """TRELLIS on the reference, checked from the front: a model whose silhouette does not lie on the
    reference's (a board built in with the figure) is made again - first from the picture through
    TRELLIS's own background remover instead of the image model's cut-out, then on a new seed.
    Knight's cut-out gave a board on three seeds (IoU 0.32-0.40); through the remover, 0.92."""
    out = r.path("pass1.glb")
    (r.work / "generate.json").unlink(missing_ok=True)
    have_alpha = r.path("reference_rgba.png").exists()
    tries = ([("alpha", r.a.seed)] if have_alpha else []) + [("rembg", r.a.seed), ("rembg", r.a.seed + 1000)]
    best = None
    for k, (route, seed) in enumerate(tries):
        cand = r.work / f"pass1_{route}_s{seed}.glb"
        trellis(r, "generate", [reference_for_3d(r, route)], cand, seed=seed)
        iou = front_iou(r, cand, f"pass1_{route}_s{seed}")
        how = "the image model's cut-out" if route == "alpha" else "TRELLIS's background remover"
        print(f"      front silhouette on the reference: IoU {iou:.2f} ({how}, seed {seed})", flush=True)
        if best is None or iou > best[1]:
            best = (cand, iou, seed, route)
        if iou >= FRONT_IOU_MIN:
            break
        if k < len(tries) - 1:
            print("      the model does not match the reference from the front (a board built in?) - "
                  "generating it again", flush=True)
    cand, iou, seed, route = best
    if iou < FRONT_IOU_MIN:
        print(f"      no try matches from the front: keeping the closest (IoU {iou:.2f}, seed {seed})", flush=True)
    shutil.copy(cand, out)
    json.dump({"seed": seed, "front_iou": round(iou, 3), "input": route}, open(r.work / "generate.json", "w"))


def s_multiview(r: Run):
    """Stochastic multi-view: the second pass sees side and back views that a diffusion prior has
    repainted from renders of the first. Needs ComfyUI; without it the first pass is used. Off by
    default since E106: the repaint invents what the reference does not show and the second model
    follows it (front fit worse on 12 of 12 characters), for a median 16 minutes."""
    out = r.path("mesh.glb")
    if r.a.quality == "fast":
        shutil.copy(r.path("pass1.glb"), out)
        print("      one pass (the default): the first model is the mesh", flush=True)
        head_swap(r)
        smooth_source(r)
        return
    try:
        sys.path.insert(0, str(ROOT / "pipeline"))
        import comfy  # noqa: E402
        comfy._get("/system_stats")          # importing the client proves nothing; ask the server
    except Exception as e:                                   # noqa: BLE001
        shutil.copy(r.path("pass1.glb"), out)
        print(f"      ComfyUI unavailable ({e}); single-view mesh kept", flush=True)
        return
    mv = r.work / "mv"
    mv.mkdir(exist_ok=True)
    r.bl("multiview", "render_views.py", "--mesh", r.path("pass1.glb"), "--out", mv / "src",
         "--views", 4, "--res", 1024, "--elev", 0, "--ortho", 2.15, keep=("[render_views]",))
    spec = description_of(r) or "the character in the reference image"
    neg = ("blurry, low detail, flat shading, melted geometry, extra limbs, text, watermark, "
           "duplicated figure")
    enhanced = []
    for idx, view in ((1, "side"), (2, "back")):
        src = mv / "src" / f"view_{idx:02d}.png"
        name = f"{r.a.name}_{view}.png"
        comfy.upload_image(src, name=name)
        prompt = (f"{view} view of {spec}, full body, same outfit and colours, "
                  f"{STYLES[style_of(r)]['views']}, even studio lighting, plain light grey background")
        wf = comfy.krea2_i2i(name, prompt, negative=neg, denoise=0.42, steps=10, seed=r.a.seed)
        with gpu(f"repainting the {view} view"):
            paths, secs = comfy.run(wf, mv, prefix=view)
        enhanced.append(paths[0])
        print(f"      enhanced {view} view in {secs:.0f}s", flush=True)
    g = json.load(open(r.work / "generate.json")) if (r.work / "generate.json").exists() else {}
    seed = g.get("seed", r.a.seed)
    trellis(r, "multiview", [reference_for_3d(r), *enhanced], out, seed=seed)
    iou = front_iou(r, out, "mesh")
    print(f"      front silhouette on the reference: IoU {iou:.2f}", flush=True)
    if iou < FRONT_IOU_MIN and iou < g.get("front_iou", 0) - 0.05:
        trellis(r, "multiview", [reference_for_3d(r), *enhanced], out, seed=seed + 1)
        iou2 = front_iou(r, out, "mesh2")
        print(f"      again on seed {seed + 1}: IoU {iou2:.2f}", flush=True)
        if iou2 < FRONT_IOU_MIN and iou2 < g.get("front_iou", 0) - 0.05:
            shutil.copy(r.path("pass1.glb"), out)
            print("      the second pass fits the reference worse than the first - keeping the first", flush=True)
    head_swap(r)
    smooth_source(r)


HEAD_FIT = {"head_iou": 0.80, "surface_median_mm": 15.0, "icp_turn_deg": 12.0}     # a fit worse than these is not used


def head_swap(r: Run):
    """The head made again on its own, from a close crop of the reference (--head hunyuan, E139).

    In a whole-figure model the head is a few dozen voxels across: Boyscout's eyes came out as ragged holes and his
    grin as a slit, so the reference's face, projected on, lay on a shape unlike the picture's and slid off it seen
    from the side. Hunyuan3D 2.1's shape model on a crop of the head alone (pipeline/head_crop.py; 7-8 minutes)
    sculpts brows, lids, lips and ears that follow the picture. It is placed on the old head through the picture
    both were made from and ICP (pipeline/head_align.py), and replaces it above the neck, coloured from the old
    surface until the texture stage paints the face (blender/head_merge.py). A fit outside HEAD_FIT keeps the old head."""
    (r.work / "head" / "swapped.json").unlink(missing_ok=True)
    if option_of(r, "head", "none") != "hunyuan":
        return
    if not hunyuan_shape_ready():
        print("      --head hunyuan needs Hunyuan3D 2.1 (tools/setup_hunyuan.sh): the generated head is kept", flush=True)
        return
    hd = r.work / "head"
    hd.mkdir(exist_ok=True)
    r.py("head_crop", "head_crop.py", "--work", r.work, "--out-dir", hd, keep=("[head]",))
    box = json.load(open(hd / "head_box.json"))["box"]
    import hashlib
    digest = hashlib.md5((hd / "head_rgba.png").read_bytes()).hexdigest()
    made = json.load(open(hd / "head_hy.json")) if (hd / "head_hy.json").exists() else {}
    if made.get("crop_md5") == digest and (hd / "head_hy.glb").exists() and not r.a.force:
        print("      the head's shape: made before from the same crop", flush=True)
    else:
        with gpu("the head's shape (Hunyuan3D 2.1)"):
            r.sh("head_shape", [HUNYUAN / "venv" / "bin" / "python", ROOT / "tools" / "hunyuan_shape.py", "--name", r.a.name,
                                "--image", hd / "head_rgba.png", "--out", hd / "head_hy.glb"], keep=("[hy_shape]",))
        json.dump({"crop_md5": digest}, open(hd / "head_hy.json", "w"))
    g = json.load(open(r.work / "generate.json")) if (r.work / "generate.json").exists() else {}
    if r.a.quality == "best" and (r.work / "gate" / "mesh" / "meta.json").exists():
        body, gate_dir = "mesh.glb", r.work / "gate" / "mesh"             # the second pass, with its own front render
    else:
        body, gate_dir = "pass1.glb", r.work / "gate" / f"pass1_{g.get('input', 'alpha')}_s{g.get('seed', r.a.seed)}"
    try:
        r.py("head_align", "head_align.py", "--work", r.work, "--body", body, "--gate", gate_dir,
             "--head", hd / "head_hy.glb", "--crop", hd / "head_rgba.png", "--box", ",".join(str(v) for v in box),
             "--out", hd / "head_aligned.glb", "--debug", hd / "align.png", "--json", hd / "align.json", keep=("[head]",))
    except SystemExit:
        print("      the new head could not be placed: the generated head is kept", flush=True)
        return
    fit = json.load(open(hd / "align.json"))
    bad = [k for k, lim in HEAD_FIT.items() if (fit[k] < lim if k == "head_iou" else fit[k] > lim)]
    if bad:
        print(f"      the new head does not sit on the old one ({', '.join(f'{k} {fit[k]}' for k in bad)}): "
              "the generated head is kept", flush=True)
        return
    shutil.copy(r.path("mesh.glb"), r.work / "mesh_old_head.glb")
    r.bl("head_merge", "head_merge.py", "--body", r.work / "mesh_old_head.glb", "--head", hd / "head_aligned.glb",
         "--out", r.path("mesh.glb"), "--report", hd / "merge.json", keep=("[head]",))
    json.dump({"fit": fit, "box": box}, open(hd / "swapped.json", "w"))


def smooth_source(r: Run):
    """Bilateral normal filtering of the generated surface (blender/mesh_denoise.py), in place on mesh.glb.

    solidify puts the solid's vertices back onto the generated surface to keep its detail, so the surface's
    flaws reach the normal map: a decimated mesh's large flat facets on gently curved armour came out as a
    triangle pattern under raking light, whatever the low-poly's triangles (decimated, relaxed or remeshed).
    Filtering turns facets a few degrees apart into one smooth surface and keeps creases tens of degrees
    sharp; positions only move (a median 0.05 mm on a 1M-face model), so UVs and texture stay (E135)."""
    if getattr(r.a, "no_denoise", False):
        return
    raw = r.path("mesh_raw.glb")
    shutil.copy(r.path("mesh.glb"), raw)
    r.bl("multiview", "mesh_denoise.py", "--mesh", raw, "--out", r.path("mesh.glb"), keep=("[denoise]",))


def s_views(r: Run):
    r.bl("views", "render_views.py", "--mesh", r.path("mesh.glb"), "--out", r.work / "views",
         "--views", 8, "--res", 640, keep=("[render_views]",))


def s_parts(r: Run):
    r.py("parts", "parts.py", "--views", r.work / "views", "--out", r.path("parts.json"),
         keep=("[parts]",))


def s_skeleton(r: Run):
    r.py("skeleton", "skeleton.py", "--views", r.work / "views", "--out", r.path("joints.json"),
         keep=("[skeleton]",))


def locate_eyes(r: Run):
    """Each eye opening of the generated model, for the solid to rebuild as lids over a ball (blender/eye_fill.py,
    pipeline/eye_marks.py; E140). None with --eyes none, or when DWPose cannot read the model's eyes."""
    if getattr(r.a, "eyes", "rebuild") == "none":
        return None
    py = TRELLIS / ".venv" / "bin" / "python"
    if not py.exists():
        return None
    ed = r.work / "eyes"
    ed.mkdir(exist_ok=True)
    r.bl("eyes_render", "eye_fill.py", "render", "--mesh", r.path("mesh.glb"), "--out", ed / "front.png",
         "--cam", ed / "cam.json", keep=("[eyes]",))
    mask = reference_mask(r)
    r.sh("eyes_marks", [py, ROOT / "pipeline" / "eye_marks.py", "--render", ed / "front.png", "--out", ed / "marks.json",
                        "--debug", ed / "marks.png", "--reference", texture_reference(r),
                        *(["--ref-mask", mask] if mask.exists() else [])], keep=("[eye_marks]",))
    r.bl("eyes_locate", "eye_fill.py", "locate", "--mesh", r.path("mesh.glb"), "--cam", ed / "cam.json",
         "--marks", ed / "marks.json", "--style", style_of(r), "--out", ed / "eyes.json", keep=("[eyes]",))
    found = json.load(open(ed / "eyes.json"))
    return ed / "eyes.json" if any(e.get("used") for e in found.get("eyes", [])) else None


def s_solidify(r: Run):
    r.bl("solidify", "sdf_io.py", "to-sdf", "--mesh", r.path("mesh.glb"), "--out", r.path("sdf.npz"),
         keep=("[sdf]",))
    eyes = locate_eyes(r)
    r.py("solidify", "solidify.py", "--sdf", r.path("sdf.npz"), "--mesh", r.path("mesh.glb"),
         "--parts", r.path("parts.json"), "--out", r.path("solid.npz"),
         *(["--eyes", eyes] if eyes else []), keep=("[solidify]",))


def s_joints(r: Run):
    r.py("joints", "refine_joints.py", "--joints", r.path("joints.json"), "--solid", r.path("solid.npz"),
         "--sdf", r.path("sdf.npz"), "--out", r.path("joints_refined.json"), keep=("[joints]",))
    r.bl("joints_qa", "qa_skeleton.py", "--mesh", r.path("mesh.glb"), "--joints",
         r.path("joints_refined.json"), "--frame-from", r.path("sdf.npz"), "--compare",
         r.path("joints.json"), "--out", r.work / "qa" / "skeleton.png", keep=("[qa]",))


def s_hands(r: Run):
    r.py("hands", "cut_hands.py", "--solid", r.path("solid.npz"), "--sdf", r.path("sdf.npz"),
         "--joints", r.path("joints_refined.json"), "--out", r.path("solid_cut.npz"),
         "--spec", r.path("hands_spec.json"),
         *(["--parts", r.path("parts.json")] if r.path("parts.json").exists() else []), keep=("[hands]",))
    # the arms parted from whatever they were generated against below the armpit (a vest, a hip)
    r.py("arms", "free_arms.py", "--solid", r.path("solid_cut.npz"), "--sdf", r.path("sdf.npz"),
         "--joints", r.path("joints_refined.json"), "--out", r.path("solid_free.npz"),
         "--report", r.path("free_arms.json"), keep=("[arms]",))
    r.bl("hands_mesh", "sdf_io.py", "to-mesh", "--sdf", r.path("solid_free.npz"), "--out",
         r.path("solid_cut.glb"), "--project", r.path("mesh.glb"),
         *(["--keep-out", r.path("solid_eyes.json")] if r.path("solid_eyes.json").exists() else []),
         keep=("put back", "rebuilt"))
    r.bl("hands_union", "hands.py", "--mesh", r.path("solid_cut.glb"), "--spec", r.path("hands_spec.json"),
         "--out", r.path("solid_hands.glb"), "--json", r.path("hands.json"), keep=("[hands]",))


def s_retopo(r: Run):
    # the generator's colours without the black specks its bake leaves along its charts' edges (E140)
    r.py("despeck", "despeck_source.py", "--mesh", r.path("mesh.glb"), "--out", r.work / "source_albedo.png",
         keep=("[despeck]",))
    r.bl("retopo", "retopo.py", "--source-albedo", r.work / "source_albedo.png", "--mesh", r.path("mesh.glb"), "--cage", r.path("solid_hands.glb"),
         "--out", r.path("retopo.glb"), "--bake-res", 4096, "--cage-extrusion", 0.006,
         "--ray-distance", 0.014, "--joints", r.path("joints_refined.json"),
         "--head-share", getattr(r.a, "head_share", 0.28),
         keep=("source winding", "cage:", "head share", "baked from", "legs:", "UVs:", "AO mean", "AO bake", "normal map:", "ORM:"))


def s_labels(r: Run):
    r.py("labels", "transfer_labels.py", "--retopo", r.path("retopo.glb"), "--source",
         r.path("mesh.glb"), "--parts", r.path("parts.json"), "--out", r.path("labels.json"),
         keep=("reclassified", "final:"))


def s_texclean(r: Run):
    r.py("texclean", "texture_cleanup.py", "--retopo", r.path("retopo.glb"), "--labels",
         r.path("labels.json"), "--albedo", r.work / "baked_base_color.png", "--out",
         r.path("albedo_clean.png"), "--mask", r.work / "texclean_mask.png",
         keep=("[texclean]",))


def reference_mask(r: Run):
    """The reference's own silhouette: plain backgrounds pass a colour threshold, a painting's street does not, so the
    generation stage's background remover cuts it out."""
    mask = r.path("reference_mask.png")
    py = TRELLIS / ".venv" / "bin" / "python"
    if not mask.exists() and py.exists():
        r.sh("foreground", [py, ROOT / "pipeline" / "foreground.py", "--image", r.path("reference.png"),
                            "--out", mask], keep=("[foreground]",))
    return mask


def texture_reference(r: Run):
    """The picture projected onto the front: the reference itself, or with its lighting taken out when
    tools/delight.py has made work/<name>/reference_delit.png (E133), so that the front's colours are an
    albedo like the sides' and back's."""
    delit = r.path("reference_delit.png")
    return delit if delit.exists() else r.path("reference.png")


def s_texture(r: Run):
    mask = reference_mask(r)
    if option_of(r, "texture_views", "none") == "hunyuan" and hunyuan_ready():
        # sides and back from Hunyuan3D-Paint's multi-view model instead of TRELLIS's own blotchy colours (E134),
        # and the reference's lighting taken out so the front is an albedo like them (E133; a drawn anime style
        # keeps its painted shading)
        hy_py = HUNYUAN / "venv" / "bin" / "python"
        with gpu("Hunyuan3D-Paint views"):
            r.sh("texture_paint", [hy_py, ROOT / "tools" / "hunyuan_views.py", "--name", r.a.name], keep=("[hy_views]",))
        if style_of(r) != "anime":
            with gpu("delighting"):
                r.sh("texture_delight", [hy_py, ROOT / "tools" / "delight.py", "--name", r.a.name, "--model", "hunyuan",
                                         "--out", r.path("reference_delit.png")], keep=("[delight]",))
    hy = option_of(r, "texture_views", "none") == "hunyuan" and (r.work / "hy_views" / "views.json").exists()
    r.bl("texture_maps", "uv_maps.py", "--mesh", r.path("retopo.glb"), "--out-dir", r.work / "texproj",
         "--face-joints", r.path("joints_refined.json"), "--face-frame", r.path("sdf.npz"),
         *(["--views", "0,90,180,270"] if hy else []), keep=("[uv_maps]",))
    if hy:
        # side and back views painted by Hunyuan3D-Paint's multi-view model (tools/hunyuan_views.py), fitted
        # onto our cameras: they follow the reference where TRELLIS's own colours, and the old repaints of
        # them, carried blotches and invented boots (E134)
        r.py("texture_views", "hy_views_to_mv.py", "--work", r.work,
             *(["--front", texture_reference(r), "--front-mask", mask] if mask.exists() else []), keep=("[hy_mv]",))
    views = [f"000:{texture_reference(r)}" + (f":{mask}" if mask.exists() else "")]
    for az, stem in (("090", "side"), ("180", "back"), ("270", "side2")):
        hy_view = r.work / "mv" / f"{stem}_hy.png"
        hits = sorted(p for p in (r.work / "mv").glob(f"{stem}_*.png") if not p.name.endswith("_hy.png"))
        if hy and hy_view.exists():
            views.append(f"{az}:{hy_view}")
        elif hits:
            views.append(f"{az}:{hits[-1]}")
    fd = face_detail(r, mask)
    args = ["--dir", r.work / "texproj", "--base", r.path("albedo_clean.png"), "--out", r.path("albedo.png"),
            "--hands", r.path("hands_spec.json"), "--edge-guard", 0.02, "--ao", r.work / "baked_ao.png"]
    if r.path("solid_cut.npz").exists():                      # the modelled hand is what stands outside it
        args += ["--hands-solid", r.path("solid_cut.npz")]
    if fd:
        args += ["--face-detail", fd, "--face-box", r.work / "texproj" / "face_src.json"]
        if style_of(r) == "realistic":
            # a realistic face the picture cannot be laid on keeps the generated one (Knight's came
            # out with every feature twice); drawn faces have large, simple features and take it as before
            args += ["--keep-base-face"]
    for v in views:
        args += ["--view", v]
    if (r.work / "head" / "swapped.json").exists() or style_of(r) != "realistic":
        # The face flow lays the picture's features onto the generator's own; on a drawn face those disagree by more
        # than a warp can mend, and its 2.5% gate does not catch it: on v0.12's finer mesh Pip's flow came in under it
        # and dragged his eye into a second one, where projected without it his face is as drawn (E138). A head made on
        # its own sits where the picture puts it and has no features of its own to align to (E139). Realistic faces keep
        # it: there the generator draws the face as the picture does (juno 1.0%, mara 1.2%, rowan 1.7%).
        args += ["--no-face-flow"]
    if (r.work / "head" / "swapped.json").exists() and (r.work / "head" / "merge.json").exists():
        args += ["--front-only", r.work / "head" / "merge.json"]
    marks = r.work / "eyes" / "marks.json"
    if marks.exists() and "picture" in json.load(open(marks)) and style_of(r) != "anime":
        # the face laid on in parts: each brow, eye, the nose, mouth and jaw onto the model's own (E140). Not on anime:
        # a drawn face's shape does not say how big its eyes are, only the generator's paint does, and fitted to it
        # Kaito's eyes came out a fifth wider than his picture's
        args += ["--face-marks", marks, "--face-cam", r.work / "eyes" / "cam.json"]
    if r.path("solid_free.npz").exists():                     # the arms were cut free: colour what that opened
        args += ["--solid", r.path("solid.npz"), "--mesh", r.path("retopo.glb")]
    r.py("texture", "project_texture.py", *args, keep=("[texproj]",))
    if style_of(r) != "anime":
        # one skin, one tone: the picture's shadows and the generator's shading taken out of the skin at the 1-2 cm
        # scale they come at, its fine detail kept - no ring at a bare wrist (Mara's dE 16 -> 2), fewer blotches on
        # the face (E138). A cel-shaded style draws its skin's shading on purpose.
        r.py("texture_skin", "skin_tone.py", "--albedo", r.path("albedo.png"), "--dir", r.work / "texproj",
             "--hands", r.path("hands_spec.json"), "--labels", r.path("labels.json"), "--retopo", r.path("retopo.glb"),
             "--out", r.path("albedo.png"),
             *(["--keep", r.work / "texproj" / "face_keep.png"] if (r.work / "texproj" / "face_keep.png").exists() else []),
             keep=("[skin]",))
    if style_of(r) != "anime":
        # a physically based renderer multiplies the light by the albedo: black cloth painted at sRGB 5-15 renders
        # as a hole without folds (Rowan's jeans; 17-38% of five characters' textures), so non-metals are eased
        # into the PBR guides' 30-240 (E132). A cel-shaded style draws with its colours as painted.
        r.py("texture_range", "albedo_range.py", "--albedo", r.path("albedo.png"), "--orm", r.work / "baked_orm.png",
             "--out", r.path("albedo.png"), keep=("[albedo]",))


def face_detail(r: Run, mask):
    """A detailed image of the reference's face region, for the projection (pipeline/face_detail.py).
    The reference's face region, enlarged to the camera's resolution, goes through image-to-image
    at low strength: the enlargement fixes where everything is, the diffusion puts back the eyes,
    lashes, lips and skin a 5x enlargement blurs away. Without ComfyUI the face is painted from
    the full-body reference alone, as before."""
    tp = r.work / "texproj"
    try:
        r.py("face_detail", "face_detail.py", "--dir", tp, "--reference", texture_reference(r),
             *(["--mask", mask] if mask.exists() else []), keep=("[face_detail]",))
    except SystemExit:
        return None
    src, out = tp / "face_src.png", tp / "face_detail.png"
    if not src.exists():
        return None
    import hashlib
    digest = hashlib.md5(src.read_bytes()).hexdigest()
    stamp = tp / "face_detail.json"
    same = stamp.exists() and json.load(open(stamp)).get("source_md5") == digest
    if r.a.force or not out.exists() or not same:
        try:
            sys.path.insert(0, str(ROOT / "pipeline"))
            import comfy  # noqa: E402
            comfy._get("/system_stats")
        except Exception as e:                                   # noqa: BLE001
            print(f"      ComfyUI unavailable ({e}); face painted from the full-body reference", flush=True)
            return None
        from PIL import Image
        st = STYLES[style_of(r)]
        spec = description_of(r) or "the character in the reference image"
        name = f"{r.a.name}_face.png"
        comfy.upload_image(src, name=name)
        prompt = (f"close-up of the face of {spec}, head and shoulders, facing the camera, the same face, "
                  f"the same expression and hairstyle, sharp focus, even studio light, {st['views']}")
        neg = st["negative"] + ", blurry, soft focus, a different person, different face, open mouth, text"
        # 0.28: at 0.35 the eyes came back a different shape on juno - detail, but not her face
        wf = comfy.krea2_i2i(name, prompt, negative=neg, denoise=0.28, steps=10, seed=r.a.seed)
        with gpu("detailing the face"):
            paths, secs = comfy.run(wf, tp, prefix="face")
        res = Image.open(src).size
        Image.open(paths[-1]).convert("RGB").resize(res, Image.LANCZOS).save(out)
        json.dump({"source_md5": digest, "denoise": 0.28, "seconds": round(secs)}, open(stamp, "w"))
        print(f"      face detail in {secs:.0f}s", flush=True)
    return out


def s_weights(r: Run):
    # distance through the solid as the arms were cut free of it (free_arms.py): through the
    # generated solid, a vest glued to the arm was a short walk from the arm's bone and took its weight
    solid = r.path("solid_free.npz") if r.path("solid_free.npz").exists() else r.path("solid.npz")
    r.py("weights", "geodesic_weights.py", "--mesh", r.path("retopo.glb"), "--solid", solid,
         "--sdf", r.path("sdf.npz"), "--joints", r.path("joints_refined.json"),
         "--out", r.path("weights.npz"), keep=("[weights]",))


def s_rig(r: Run):
    r.bl("rig", "rig_build.py", "--mesh", r.path("retopo.glb"), "--joints", r.path("joints_refined.json"),
         "--frame-from", r.path("sdf.npz"), "--hands", r.path("hands.json"), "--weights",
         r.path("weights.npz"), "--albedo", r.path("albedo.png"), "--out", r.path("rig.blend"),
         "--labels", r.path("labels.json"),
         "--json", r.work / "rig.json", keep=("[rig]",))


def s_springs(r: Run):
    r.bl("springs", "springs.py", "--blend", r.path("rig.blend"), "--labels", r.path("labels.json"),
         "--out", r.path("rig_s.blend"), "--json", r.path("springs.json"), keep=("[springs]",))


def s_frame(r: Run):
    r.bl("frame", "normalize_frame.py", "--blend", r.path("rig_s.blend"), "--out",
         r.path("rig_m.blend"), "--height", height_of(r), keep=("[frame] now", "[frame] origin"))


def s_tpose(r: Run):
    keep_rest = ["--keep-rest"] if option_of(r, "rest", "A") == "A" else []
    r.bl("tpose", "tpose.py", "--blend", r.path("rig_m.blend"), "--out", r.path("rig_t.blend"), *keep_rest,
         keep=("rest after", "squared"))


def s_face(r: Run):
    fd = r.work / "face"
    r.bl("face_render", "face_render.py", "--blend", r.path("rig_t.blend"), "--out-dir", fd, keep=("[face]",))
    try:
        r.py("face_landmarks", "face_landmarks.py", "--dir", fd, "--out", r.path("face.json"),
             "--style", style_of(r), keep=("[face]",))
        r.bl("face_rig", "face_rig.py", "--blend", r.path("rig_t.blend"), "--face", r.path("face.json"),
             "--out", r.path("rig_f.blend"), keep=("[face]",))
    except SystemExit as e:
        # No face the landmark finder trusts (a helmet, a mask, a creature): the character ships
        # without a face rig rather than with a wrong one. The QA image shows what was found.
        print(f"      no face rig: {str(e).strip().splitlines()[0][:120]}", flush=True)
        shutil.copy(r.path("rig_t.blend"), r.path("rig_f.blend"))


def clips_manifest(r: Run) -> Path:
    """The default clip set, resolved against the local Mixamo mirror."""
    spec = json.load(open(ROOT / "animations" / "default_clips.json"))
    snaps = sorted(MIXAMO_HF.glob("*/animation")) if MIXAMO_HF.exists() else []
    if not snaps:
        raise SystemExit("animation clips not found: this stage retargets Mixamo captures from a "
                         "local mirror. Download clips from mixamo.com (FBX, 'without skin') and "
                         "point animations/default_clips.json at them.")
    # clips added to this character alone - pipeline/video_motion.py --add-to writes them, from a
    # video of the motion
    extra = r.work / "extra_clips.json"
    if extra.exists():
        more = json.load(open(extra))
        spec["clips"].update(more)
        print(f"      extra clips for this character: {', '.join(more)}", flush=True)
    resolved = {}
    for clip, entry in spec["clips"].items():
        opts = dict(entry) if isinstance(entry, dict) else {"file": entry}
        # a clip made from a video (pipeline/video_motion.py --fit) is a file of its own
        p = Path(opts["file"]) if Path(opts["file"]).is_absolute() else snaps[-1] / opts["file"]
        if not p.exists():
            raise SystemExit(f"clip {clip!r}: {opts['file']} not in {snaps[-1]}")
        opts["file"] = str(p)
        resolved[clip] = opts
    out = r.work / "clips.json"
    json.dump(resolved, open(out, "w"), indent=1)
    return out


def s_animate(r: Run):
    # a clip made from a video turns the head to the video's face with the character's own face points
    calib = ["--calib", r.work / "joint_calib.json"] if (r.work / "joint_calib.json").exists() else []
    r.bl("animate", "retarget.py", "--rig", r.path("rig_f.blend"), "--clips", clips_manifest(r),
         "--out", r.work / "animated.glb", "--blend-out", r.path("final.blend"),
         "--json", r.work / "retarget.json", *calib,
         keep=("heading", "feet:", "leg-lengths/s", "WARNING", "clips ->", "aimed along", "the head"))
    # a capture was performed by someone of one build: on a thicker character the same motion folds a forearm
    # into a sleeve or hangs the arms inside the torso. Measure how far this character's elbows, knees and
    # arms can go before one limb sinks into another (its own mesh), and hold the clips inside that
    # (experiment E128: Bo's frames with >2 cm penetration 62% -> 31%, foot slip unchanged)
    r.bl("limits", "joint_limits.py", "--blend", r.path("final.blend"), "--out", r.work / "joint_limits.json",
         keep=("limit",))
    r.bl("clearance", "clearance.py", "--blend", r.path("final.blend"), "--limits", r.work / "joint_limits.json",
         "--json", r.work / "clearance.json", keep=("[clearance]",))


def ensure_hand_calib(r: Run):
    """Where DWPose sees this character's hands against its bones (tools/calibrate_hands.py: its own wave,
    idle and jump from three angles), made once per rig; None without DWPose's model."""
    out, rig = r.work / "hand_calib.json", r.work / "rig_f.blend"
    if not DWPOSE.exists():
        return None
    if not out.exists() or (rig.exists() and out.stat().st_mtime < rig.stat().st_mtime):
        r.sh("hand_calib", [sys.executable, ROOT / "tools" / "calibrate_hands.py", "--name", r.a.name], keep=("[hands]",))
    return out if out.exists() else None


def s_refine(r: Run):
    """Each clip made from a video of this character (charforge.py move) re-posed, frame by frame, so
    its own mesh lies on the video: blender/export_skin.py -> pipeline/refine_pose.py (silhouette,
    joints and face against the video, small turns on twenty bones; with DWPose's reading of the
    video, the hands too - wrists and fingers) -> blender/apply_pose_corrections.py into final.blend.
    A clip from another character's video keeps its fit (a different body cannot lie on that figure),
    and a correction that does not bring the mesh closer is not applied."""
    import numpy as np
    extra = r.work / "extra_clips.json"
    report = {}
    clips = json.load(open(extra)) if extra.exists() else {}
    for clip, spec in clips.items():
        src = Path(spec.get("from_video") or "")
        if not (spec.get("fitted") and src.suffix == ".mp4"):
            continue
        performer = src.parent.parent.name
        masks, pose = src.with_name(f"{clip}_masks.npz"), src.with_name(f"{clip}_match_pose.npz")
        if performer != r.a.name:
            report[clip] = {"skipped": f"made from {performer}'s video"}
            continue
        if not (masks.exists() and pose.exists() and "kp17" in np.load(pose).files):
            report[clip] = {"skipped": "no figure masks or 17-point reading (charforge.py move --refit)"}
            continue
        n = len(np.load(masks)["masks"])
        tmp = r.work / "refine"
        tmp.mkdir(exist_ok=True)
        calib = r.work / "joint_calib.json"
        hcal = ensure_hand_calib(r) if src.with_name(f"{clip}_dw.npz").exists() else None
        r.bl("refine_skin", "export_skin.py", "--blend", r.path("final.blend"), "--clip", clip, "--frames", n,
             "--points", 30000, *(["--calib", calib] if calib.exists() else []),
             *(["--hand-calib", hcal] if hcal else []), "--out", tmp / f"{clip}_skin.npz", keep=("[skin]",))
        dw = src.with_name(f"{clip}_dw.npz")                         # DWPose's hands, when charforge.py move made them
        with gpu(f"refining {clip}"):
            r.sh("refine", [sys.executable, ROOT / "pipeline" / "refine_pose.py", "--skin", tmp / f"{clip}_skin.npz",
                            "--masks", masks, "--kp", pose, "--yaw", spec["fit"]["preview_yaw_deg"],
                            "--out", tmp / f"{clip}_corr.npz", *(["--hands", dw] if dw.exists() else [])],
                 keep=("silhouette IoU", "hands (", "elbows:"))
        res = json.load(open(tmp / f"{clip}_corr.json"))
        report[clip] = res
        if res["iou_points_refined"] <= res["iou_points_clip"]:
            report[clip]["applied"] = False
            continue
        r.bl("refine_apply", "apply_pose_corrections.py", "--blend", r.path("final.blend"), "--clip", clip,
             "--corr", tmp / f"{clip}_corr.npz", "--out", r.path("final.blend"), "--masks", masks, keep=("[apply]",))
        report[clip]["applied"] = True
    json.dump(report, open(r.work / "refine.json", "w"), indent=1)


def s_package(r: Run):
    args = ["--blend", r.path("final.blend"), "--out-dir", r.out, "--name", r.a.name,
            "--retarget", r.work / "retarget.json", "--springs", r.path("springs.json")]
    if prompt_of(r):
        args += ["--prompt", prompt_of(r)]
    args += ["--image", r.path("reference.png"), "--style", style_of(r)]
    r.bl("package", "package.py", *args,
         keep=("renamed", "glTF ->", "FBX  ->", "FBX LODs", "manifest:", "capsule:", "[pkg]    "))


def s_web(r: Run):
    r.sh("web", [sys.executable, ROOT / "tools" / "optimize_glb.py", "--in", r.path("PACKAGE"),
                 "--out", r.path("WEB"), "--res", 2048], keep=("->",))
    r.bl("thumb", "thumbnail.py", "--blend", r.path("final.blend"),
         "--out", r.out / f"{r.a.name}_thumb.png", keep=("[thumb]",))


BODY = {n: globals()[f"s_{n}"] for n in NAMES}


def make(a):
    r = Run(a)
    # a new prompt or picture under a name already taken: every stage was cached and the new prompt
    # ignored ("a young knight in steel armor" gave back the existing Knight in 0.0 min)
    # (and it recorded that prompt, style and height over the existing character's own)
    f = r.work / "style.json"
    rec = json.load(open(f)) if f.exists() else {}
    old = rec.get("prompt")
    changed = [k for k, v in (("style", a.style), ("height", a.height)) if v is not None and rec.get(k) not in (None, v)]
    if (r.work / "reference.png").exists() and a.from_stage is None and not a.force and (
            a.image or changed or (a.prompt and " ".join(a.prompt.split()) != " ".join((old or "").split()))):
        raise SystemExit(f"a character named {a.name} already exists (made from "
                         f"{repr(old) if old else 'a picture'}): pick another name, or add --from reference to "
                         "make it again from what you gave")
    r.work.mkdir(parents=True, exist_ok=True)
    style_of(r), height_of(r), prompt_of(r), image_model_of(r)   # record what this run was given
    start = NAMES.index(a.from_stage) if a.from_stage else 0
    stop = NAMES.index(a.until) if a.until else len(NAMES) - 1
    print(f"CharForge  {a.name}  ->  {r.out}", flush=True)
    for i, (name, produces, why) in enumerate(STAGES):
        if i > stop:
            break
        target = r.path(produces)
        if i < start and target.exists():
            continue
        if i >= start and target.exists() and not a.force and a.from_stage is None:
            print(f"  [{i+1:2d}/{len(STAGES)}] {name:<9s} cached", flush=True)
            continue
        t = time.time()
        print(f"  [{i+1:2d}/{len(STAGES)}] {name:<9s} {why}", flush=True)
        BODY[name](r)
        if not target.exists():
            raise SystemExit(f"[{name}] finished without producing {target}")
        print(f"               {time.time()-t:5.0f}s", flush=True)
    print(f"\ndone in {(time.time()-r.t0)/60:.1f} min -> {r.out}", flush=True)
    m = r.out / f"{a.name}.json"
    if m.exists():
        man = json.load(open(m))
        print(f"  {man['height_m']:.2f} m, {man['triangles']:,} triangles, "
              f"{man['skeleton']['bones']} bones ({man['skeleton']['convention']}), "
              f"{len(man['clips'])} clips", flush=True)


def ensure_calib(name, force=False):
    """work/<name>/joint_calib.json: where the pose model sees the character's joints and face
    (tools/calibrate_joints.py) - a fit on a video of it needs its body in the pose model's terms."""
    out = ROOT / "work" / name / "joint_calib.json"
    rig = ROOT / "work" / name / "rig_f.blend"
    if force or not out.exists() or (rig.exists() and rig.stat().st_mtime > out.stat().st_mtime):
        subprocess.run([sys.executable, str(ROOT / "tools" / "calibrate_joints.py"), "--names", name], check=True)
    return out


def move(a):
    """New moves for a finished character, each from a video of a character doing it.

    video    pipeline/motion_video.py: the performer's reference image and a move from
             animations/motion_prompts.json -> MiniMax H3 in ComfyUI -> 5 s of the performer doing it
    motion   pipeline/video_motion.py --fit: the pose model reads the video, the nearest library
             capture is found and, where it misses the video, turned to follow it -> a new clip
             (a Mixamo skeleton, so one video's clip serves every character)
    rig      the animate, package and web stages again, once, with every new clip in the set
    preview  each video and the character doing its clip, side by side

    --move takes one move or several (punch_combo,spell_cast), --all every move in
    motion_prompts.json. The performer is the character itself unless --performer names another:
    one video of Mara punching gives every character the punch. A video or a fit already made is
    reused (--force makes the videos again).
    """
    work = ROOT / "work" / a.name
    if not (work / "final.blend").exists():
        raise SystemExit(f"{a.name} is not built yet: charforge.py make --name {a.name} ...")
    moves = (list(json.load(open(ROOT / "animations" / "motion_prompts.json"))["moves"]) if a.all else
             [m.strip() for m in (a.move or "").split(",") if m.strip()])
    if not moves:
        raise SystemExit("give --move (one, or several separated by commas) or --all")
    if a.video and len(moves) != 1:
        raise SystemExit("--video goes with a single --move")
    # move@performer takes that move from another character's video (punch_combo@mara)
    who = {m.split("@")[0]: (m.split("@")[1] if "@" in m else (a.performer or a.name)) for m in moves}
    moves = list(who)
    t0 = time.time()
    print(f"CharForge move  {a.name}: " + ", ".join(m + (f" (from {who[m]}'s video)" if who[m] != a.name else "")
                                                     for m in moves), flush=True)
    specs = {}
    for mv in moves:
        performer = who[mv]
        vids = ROOT / "work" / performer / "motion_videos"
        vids.mkdir(parents=True, exist_ok=True)
        mp4, match = vids / f"{mv}.mp4", vids / f"{mv}_match.json"
        if not a.preview_only:
            if a.video:
                shutil.copy(a.video, mp4)
            elif not mp4.exists() or a.force:
                cmd = [sys.executable, ROOT / "pipeline" / "motion_video.py", "--name", performer, "--move", mv,
                       "--seed", str(a.seed), "--size", a.size]
                if a.keep_comfy_loaded:
                    cmd.append("--keep-comfy-loaded")
                with gpu(f"the {mv} video"):
                    subprocess.run([str(c) for c in cmd], check=True)
            print(f"  video    {mp4}", flush=True)
            if not match.exists() or a.force or a.video or a.refit:
                # the figure cut out of every frame (the pose model's boxes, and the fidelity score's
                # silhouettes): a colour threshold takes half of a studio backdrop for the figure
                masks = vids / f"{mv}_masks.npz"
                if not masks.exists() or a.force or a.video:
                    with gpu(f"the {mv} figure masks"):
                        subprocess.run([str(TRELLIS / ".venv" / "bin" / "python"), str(ROOT / "pipeline" / "foreground.py"),
                                        "--video", str(mp4), "--out", str(masks)], check=True)
                cmd = [sys.executable, ROOT / "pipeline" / "video_motion.py", "--video", mp4, "--library",
                       ROOT / "work" / "motion_library.npz", "--fps", "24", "--as", mv, "--out", match,
                       "--masks", masks, "--calib", ensure_calib(performer)]
                if not a.no_fit:
                    cmd.append("--fit")
                with gpu(f"reading the {mv} video"):
                    subprocess.run([str(c) for c in cmd], check=True)
            # the hands, which the pose model's 17 points do not reach: DWPose's 21 points a hand, for the
            # refine stage to turn the wrists and fingers to (skipped without its model)
            dw, masks = vids / f"{mv}_dw.npz", vids / f"{mv}_masks.npz"
            if DWPOSE.exists() and masks.exists() and (not dw.exists() or a.force or a.video):
                with gpu(f"the {mv} hands"):
                    subprocess.run([str(TRELLIS / ".venv" / "bin" / "python"), str(ROOT / "tools" / "dwpose.py"),
                                    "--video", str(mp4), "--masks", str(masks), "--out", str(dw)], check=True)
        specs[mv] = json.load(open(match))["clip_spec"]
    if not a.preview_only:
        ensure_calib(a.name)                          # the head's face points, for the retarget
        extra = work / "extra_clips.json"
        have = json.load(open(extra)) if extra.exists() else {}
        have.update(specs)
        json.dump(have, open(extra, "w"), indent=1)
        make(argparse.Namespace(name=a.name, prompt=None, image=None, height=None, style=None, quality="best",
                                seed=7, keep_comfy_loaded=a.keep_comfy_loaded, from_stage="animate", until=None,
                                force=False))
    for mv in moves:
        preview(a.name, mv, specs[mv], ROOT / "work" / who[mv] / "motion_videos" / f"{mv}.mp4", who[mv])
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)


def preview(name, mv, spec, mp4, performer):
    """A new clip seen from the video's camera angle (relative to the body's average heading,
    as retarget.py turns it), next to the video: work/<name>/motion_videos/<move>_compare.mp4."""
    work = ROOT / "work" / name
    yaw = (spec.get("fit") or {}).get("preview_yaw_deg", 30.0)
    frames = work / "motion_videos" / f"{mv}_frames"
    shutil.rmtree(frames, ignore_errors=True)
    subprocess.run([blender_bin(), "-b", "-noaudio", "--python", str(ROOT / "blender" / "render_clip_seq.py"), "--",
                    str(work / "final.blend"), mv, str(frames), "512", f"{yaw:.1f}", "2"],
                   check=True, capture_output=True)
    out = work / "motion_videos" / f"{mv}_compare.mp4"
    subprocess.run([sys.executable, str(ROOT / "tools" / "side_by_side.py"), "--left", str(mp4), "--right", str(frames),
                    "--fps", "30", "--out", str(out), "--labels", f"video ({performer})", name], check=True)
    print(f"  preview  {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="build a character")
    m.add_argument("--name", required=True, help="folder name under work/ and out/")
    src = m.add_mutually_exclusive_group()
    src.add_argument("--prompt", help="describe the character")
    src.add_argument("--image", help="or give a full-body image of the character")
    m.add_argument("--height", type=float, default=None,
                   help="standing height in metres (default: what the description implies - a child 1.2-1.5, "
                        "an adult 1.75 - or whatever the character was made at)")
    m.add_argument("--literal", action="store_true",
                   help="ask the image model for the prompt exactly as written (by default a short prompt is "
                        "written out first: age and build, skin and hair, each garment with its colour)")
    m.add_argument("--description", default=None,
                   help="the written-out description to draw from, instead of making one from --prompt")
    m.add_argument("--style", choices=sorted(STYLES), default=None,
                   help="art style (default: realistic, or whatever the character was made in)")
    m.add_argument("--image-model", choices=("qwen21", "krea2"), default=None,
                   help="what makes the reference image from --prompt (default: qwen21, or whatever the "
                        "character was made with). Qwen-Image 2.1 draws its own transparent cut-out but is "
                        "under a research-only licence; use krea2 for commercial work")
    m.add_argument("--quality", choices=("best", "fast"), default="fast",
                   help="fast (default): one TRELLIS pass on the reference. best: a second pass on the reference plus "
                        "side and back views the image model repaints (needs ComfyUI; a median 16 more minutes). On 12 "
                        "characters the second pass fitted the reference's front worse every time and drifted from it "
                        "(Boyscout grew knee socks, Vex's trainers became boots, Mara's braid moved behind), though "
                        "a back can come out more plausible (experiment E106)")
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--trellis-faces", type=int, default=1_000_000,
                   help="faces TRELLIS.2 keeps of its surface (default 1M: at 200k, large flat facets and sawtooth "
                        "plate edges reached the normal map; experiment E135)")
    m.add_argument("--no-denoise", action="store_true",
                   help="skip the bilateral normal filtering of the generated surface (E135)")
    m.add_argument("--texture-views", choices=("hunyuan", "none"), default=None,
                   help="hunyuan (needs tools/setup_hunyuan.sh): side and back views painted by Hunyuan3D-Paint 2.1 and "
                        "the reference delit (E133, E134) - a cleaner back where TRELLIS's is blotchy (Pip), but on 6 of 8 "
                        "other characters it painted what the reference does not show or lost a pattern; none "
                        "(default): the sides and back the generator painted")
    m.add_argument("--head", choices=("hunyuan", "none"), default=None,
                   help="hunyuan (needs tools/setup_hunyuan.sh): the head made again on its own from a close crop of the "
                        "reference by Hunyuan3D 2.1's shape model - a sculpted face where the whole-figure model's has a few "
                        "rough facets, ~8 more minutes (experiment E139); none: the head as generated with the body")
    m.add_argument("--eyes", choices=("rebuild", "none"), default="rebuild",
                   help="rebuild (default): each eye the generator modelled as an opening full of shards becomes lids "
                        "over a ball - found by DWPose on the model's own paint, rebuilt in the solid; eyes it cannot "
                        "read (most anime faces) stay as generated (experiment E140). none: every eye as generated")
    m.add_argument("--head-share", type=float, default=0.28,
                   help="the share of the triangle budget the head keeps (default 0.28; 0 leaves it to the "
                        "decimation, which gave Mara's 17%% and crumpled her eyes and lips; experiment E140)")
    m.add_argument("--rest", choices=("T", "A"), default=None,
                   help="the rest pose the character ships in: A (default) keeps the pose it was modelled in and "
                        "records the T only for retargeting, so no shoulder is skinned up to a T and down again - on "
                        "Cadet that halved crushed faces and self-intersections and took frames with deep "
                        "penetration from 40%% to 28%%, the clips unchanged (experiment E137); T bakes the mesh to "
                        "a T-pose, the animation library's own rest, for tools that insist on one")
    m.add_argument("--keep-comfy-loaded", action="store_true",
                   help="do not ask ComfyUI to unload its cached models before a TRELLIS pass "
                        "(use this if you are working in ComfyUI while CharForge runs)")
    m.add_argument("--from", dest="from_stage", choices=NAMES, help="rerun from this stage onward")
    m.add_argument("--until", choices=NAMES, help="stop after this stage")
    m.add_argument("--force", action="store_true", help="ignore cached stage outputs")
    sub.add_parser("stages", help="list the stages")
    st = sub.add_parser("studio", help="CharForge Studio: make characters from a browser (http://localhost:8830)")
    st.add_argument("--port", type=int, default=8830)
    st.add_argument("--no-browser", action="store_true", help="do not open the browser")
    mv = sub.add_parser("move", help="a new move for a finished character, from a generated video")
    mv.add_argument("--name", required=True)
    mv.add_argument("--move", default=None,
                    help="a move in animations/motion_prompts.json (the clip's name), or several, comma-separated")
    mv.add_argument("--all", action="store_true", help="every move in animations/motion_prompts.json")
    mv.add_argument("--performer", default=None,
                    help="another character whose video of the move to use (made if missing)")
    mv.add_argument("--video", default=None, help="use this video instead of generating one")
    mv.add_argument("--seed", type=int, default=1)
    mv.add_argument("--size", default="512x768")
    mv.add_argument("--no-fit", action="store_true", help="use the nearest library clip as it is")
    mv.add_argument("--keep-comfy-loaded", action="store_true")
    mv.add_argument("--force", action="store_true", help="generate the video again")
    mv.add_argument("--refit", action="store_true", help="fit the clip to its video again (keeps the video)")
    mv.add_argument("--preview-only", action="store_true", help="only render the side-by-side preview again")
    a = ap.parse_args()
    if a.cmd == "move":
        move(a)
        return
    if a.cmd == "studio":
        sys.path.insert(0, str(ROOT / "studio"))
        import server
        server.main(a.port, not a.no_browser)
        return
    if a.cmd == "stages":
        for i, (n, p, w) in enumerate(STAGES):
            print(f"{i+1:2d}  {n:<9s} {p:<22s} {w}")
        return
    make(a)


if __name__ == "__main__":
    main()
