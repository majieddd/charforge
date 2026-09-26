"""A character performing a move, on video: MiniMax H3 (reference-to-video) in ComfyUI.

    python pipeline/motion_video.py --name mara --move punch_combo
    python pipeline/motion_video.py --name mara --move custom --action "[00:00-00:02] {subject} ..."
    python pipeline/motion_video.py --moves                      # the moves animations/motion_prompts.json holds

The video step of text/image -> video -> motion: the character's reference image goes in as the
reference, a move from animations/motion_prompts.json as the prompt, and out comes a 5-second
video of the character doing it (work/<name>/motion_videos/<move>.mp4, with <move>.json: the
prompt, settings and timings). pipeline/video_motion.py then reads the motion off it.

What is fixed, and why:
  prompt    MiniMax's six-section format (subject_definitions ... non_diegetic_music) with the move
            as an action chain with timings - H3 was trained on prompts rewritten into this form,
            and a one-word action gives a vague, drifting move
  camera    locked off, the whole body in frame, a plain studio: a pose model has to see every
            joint in every frame, and a moving camera turns into motion that is not there
  size      512 x 768 portrait, 124 frames (5.2 s at 24 fps): the shortest length H3 was trained
            on, at a third of its 768p area - on an M5 with 24 GB, attention over the video's
            tokens is the cost, and it grows with the square of the area
  memory    two jobs: the text encoder (Qwen3-VL 32B, 15.7 GB, on the CPU - its float8 scales
            cannot go to the Mac's GPU) encodes the prompt and the reference; ComfyUI then drops
            it, keeps the encoding in its cache, and loads the video model for the second job

The model files (ComfyUI/models): Minimax-h3_Singularity_ref2va_v1.3_Pruned_w4a8 (a community
fine-tune of MiniMax H3's reference-to-video model) made loadable here in two steps -
tools/f8_scales_to_f32.py (the Mac's GPU has no float8) and tools/merge_lora_quantized.py, which
bakes in Comfy-Org's 4-step turbo LoRA without requantizing (ComfyUI patching it at load kept a
second copy of every weight and swapped a 24 GB Mac for 20 minutes) - and from
Comfy-Org/MiniMax-H3 the text encoder and the two VAEs. MiniMax H3 is under the MiniMax H3 Community
License: its territory excludes the US, EU, UK and South Korea, and a commercial product must show
"MiniMax H3" in its interface.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comfy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIT = "Minimax-h3_Singularity_ref2va_v1.3_Pruned_w4a8_turbo4_mps.safetensors"
TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

LOOK = {
    "realistic": "Photorealistic live-action footage, natural skin and fabric detail.",
    "anime": "Anime-style cel-shaded 3D animation that keeps the exact look of <Picture 1>: flat colours, clean outlines.",
    "stylized": "Stylized 3D animated-film look that keeps the exact look of <Picture 1>: its proportions, colours and materials.",
}
CAMERA = ("A locked-off camera on a tripod at waist height, about four metres away, three-quarter front "
          "view from {subject}'s front left, framing the whole body from head to feet with space above "
          "the head and below the feet in every frame. The camera never moves, pans, zooms or cuts; "
          "one continuous shot. The setting is an empty seamless light-grey photo studio, floor and "
          "backdrop the same grey, lit evenly and softly, with a soft contact shadow under the feet. "
          "Nothing else is in the frame.")


def moves():
    return json.load(open(ROOT / "animations" / "motion_prompts.json"))["moves"]


def prompt_for(style: str, desc: str | None, move: dict) -> str:
    """The move as MiniMax's six-section prompt, for a character shown in <Picture 1>."""
    subj = "<Subject 1>"
    who = (f"{subj} is the character in <Picture 1>, {desc}." if desc else
           f"{subj} is the character in <Picture 1>, with the face, hair, clothing, colours and body "
           f"proportions shown there.")
    beats = "\n".join(b.replace("{subject}", subj) for b in move["beats"])
    return "\n\n".join([
        "subject_definitions:\n" + who + " <Picture 1> is a full-body reference of the character "
        "standing in an A-pose; it defines appearance only and is not the first frame.",
        f"summary:\n[reference generation] One continuous static full-body shot of {subj} performing "
        f"{move['summary']}, alone in an empty studio.",
        f"retention_analysis:\n{subj}'s face, hair, clothing, colours and body proportions are fully "
        f"preserved from <Picture 1> in every frame; the A-pose itself is not kept.",
        "detailed_description:\n[Shot 1] " + LOOK.get(style, LOOK["realistic"]) + " "
        + CAMERA.replace("{subject}", subj) + "\n" + beats,
        "overall_soundscape:\n" + move.get("sound", "Quiet studio ambience, footsteps and cloth movement."),
        "non_diegetic_music:\nNone.",
    ])


def workflow(prompt: str, ref_name: str, width: int, height: int, length: int, steps: int, seed: int,
             prefix: str, encode_only: bool = False) -> dict:
    """ComfyUI API graph after Comfy-Org's video_minimax_h3_r2v template (res_multistep, simple,
    4 steps; the turbo LoRA is baked into the model file). Node ids are fixed so the second job
    finds the first job's encoding in ComfyUI's cache."""
    g = {
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "minimax", "device": "cpu"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "6": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "7": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0], "prompt": prompt,
            "width": width, "height": height, "length": length, "ref_image_size": "match",
            "ref_images.ref_image_0": ["6", 0]}},
    }
    if encode_only:
        g["20"] = {"class_type": "PreviewAny", "inputs": {"source": ["7", 1]}}
        return g
    g.update({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": DIT, "weight_dtype": "default"}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple",
                                                          "steps": steps, "denoise": 1.0}},
        "11": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["7", 0]}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0],
            "latent_image": ["7", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": 24.0}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": prefix,
                                                     "format": "mp4", "format.codec": "h264"}},
    })
    return g


def unload():
    """Drop every model ComfyUI holds - not its cache of node outputs (that would be free_memory)."""
    req = urllib.request.Request(f"{comfy.HOST}/free", data=json.dumps({"unload_models": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60).read()
    time.sleep(3)


def run(g: dict, timeout: float) -> tuple[dict, float]:
    t0 = time.time()
    entry = comfy.wait(comfy.submit(g), timeout=timeout)
    status = entry.get("status", {})
    if status.get("status_str") not in (None, "success"):
        raise SystemExit(f"ComfyUI job failed: {json.dumps(status)[:1500]}")
    return entry, time.time() - t0


def main(a):
    if a.moves:
        for k, v in moves().items():
            print(f"{k:18s} {v['summary']}")
        return
    work = ROOT / "work" / a.name
    ref = work / "reference.png"
    if not ref.exists():
        raise SystemExit(f"{ref} not found: make the character first (charforge.py make --name {a.name} ...)")
    rec = json.load(open(work / "style.json")) if (work / "style.json").exists() else {}
    style, desc = rec.get("style", "realistic"), rec.get("prompt")
    if a.move == "custom":
        if not a.action:
            raise SystemExit("--move custom needs --action (the beats, with {subject} for the character)")
        move = {"summary": a.summary or "the move described below", "beats": [a.action]}
    else:
        move = moves().get(a.move) or sys.exit(f"no move {a.move!r}; --moves lists them")
    w, h = (int(x) for x in a.size.lower().split("x"))
    length = max(5, round(a.seconds * 24))
    length += (5 - length % 17) % 17                                  # H3's 17k + 5 frame grid
    prompt = prompt_for(style, desc, move)
    out = work / "motion_videos"
    out.mkdir(parents=True, exist_ok=True)
    ref_name = f"cf_{a.name}_reference.png"
    comfy.upload_image(ref, ref_name)
    prefix = f"charforge_h3/{a.name}_{a.move}"
    if not a.keep_comfy_loaded:
        unload()
    print(f"[video] {a.name}: {a.move} - {move['summary']}; {w}x{h}, {length} frames, {a.steps} steps, seed {a.seed}",
          flush=True)
    _, t_enc = run(workflow(prompt, ref_name, w, h, length, a.steps, a.seed, prefix, encode_only=True), 3600)
    print(f"[video] prompt and reference encoded in {t_enc:.0f} s (text encoder on the CPU)", flush=True)
    if not a.keep_comfy_loaded:
        unload()
    entry, t_gen = run(workflow(prompt, ref_name, w, h, length, a.steps, a.seed, prefix), a.timeout)
    files = [im for o in entry["outputs"].values() for im in o.get("images", []) if im["filename"].endswith(".mp4")]
    if not files:
        raise SystemExit(f"no video in the job's outputs: {json.dumps(entry['outputs'])[:800]}")
    q = urllib.parse.urlencode({"filename": files[0]["filename"], "subfolder": files[0].get("subfolder", ""),
                                "type": files[0].get("type", "output")})
    mp4 = out / f"{a.move}.mp4"
    mp4.write_bytes(urllib.request.urlopen(f"{comfy.HOST}/view?{q}", timeout=300).read())
    # H3 makes the sound with the picture (whooshes, impacts, breath): kept as the move's sound effect
    import shutil
    import subprocess
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vn", "-c:a", "pcm_s16le",
                        str(out / f"{a.move}.wav")], check=False)
    meta = {"character": a.name, "move": a.move, "summary": move["summary"], "prompt": prompt,
            "model": {"diffusion": DIT, "text_encoder": TE, "vae": VAE},
            "width": w, "height": h, "frames": length, "fps": 24, "steps": a.steps, "seed": a.seed,
            "seconds_encode": round(t_enc, 1), "seconds_generate": round(t_gen, 1),
            "license": "MiniMax H3 Community License (territory excludes US, EU, UK, KR)"}
    json.dump(meta, open(out / f"{a.move}.json", "w"), indent=1)
    print(f"[video] {mp4} in {t_gen / 60:.1f} min (+{t_enc:.0f} s encoding)", flush=True)
    if not a.keep_comfy_loaded:
        unload()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--move", default="punch_combo")
    ap.add_argument("--action", default=None, help="for --move custom: the timed beats")
    ap.add_argument("--summary", default=None, help="for --move custom: one line")
    ap.add_argument("--size", default="512x768")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=4 * 3600)
    ap.add_argument("--keep-comfy-loaded", action="store_true",
                    help="do not unload ComfyUI's models before, between and after the two jobs")
    ap.add_argument("--moves", action="store_true")
    a = ap.parse_args()
    if not a.moves and not a.name:
        ap.error("--name is required")
    main(a)
