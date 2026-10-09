"""A 2x high-resolution pass over the texture's pictures, as a ComfyUI workflow. Written, not run: ComfyUI is down.

    python pipeline/tex_hires.py --name mara --what reference --print    # write the graph(s) to inspect; no network
    python pipeline/tex_hires.py --name mara --what views --submit       # run them, when ComfyUI is up (GPU lock held)

What it does. The projection reads the picture (work/<name>/reference.png, or reference_delit.png) at 832 x 1216 and
the side and back views at 1024 px (work/<name>/mv/*.png). The atlas takes about 3.7 texels to a picture pixel (3,500-
3,900 texels across a figure's height), so those pixels are the albedo's detail limit: E140 found that more texels
alone do not help. This pass doubles each picture (Lanczos, in pixel space) and then redraws it at low denoise with
Krea 2 Turbo - the models and node pattern pipeline/comfy.py's krea2_i2i already uses to turn the first-pass render
into a photographic conditioning view - so the projection reads a 1664 x 2432 (or 2048 x 2048) picture whose extra
detail is drawn at that scale rather than interpolated. The denoise strength is the knob: at 0.2-0.3 the figure, its
pose and its colours stay as they are; higher, the detail grows and so does the drift.

Not wired yet. The projection maps the picture onto the atlas through its silhouette fit and its face frames
(face_src.json, the face-detail box, reference_mask.png), all in the picture's own pixels, so a 2x reference needs
those scaled by 2 where project_texture.py reads them; charforge.py's texture stage would also need to point at the
2x file. The views need no such change (they are read at their own size). Outputs land in
work/<name>/texwork/hires/ as <stem>_11_0.png (ComfyUI's SaveImage node is 11); copy them over the 1x picture only
after tools/texture_sharpness.py (and tools/image_fidelity.py) say the 2x version helps.

Graph: LoadImage -> ImageScaleBy (lanczos x2) -> VAEEncode -> KSampler (denoise) -> VAEDecode -> SaveImage, with the
Krea 2 Turbo loader, the Qwen3-VL text encoder and the Qwen-Image VAE, as pipeline/comfy.py uses them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
NEGATIVE = "blurry, low detail, text, watermark, extra limbs, deformed, cropped"


def hires_workflow(image_name: str, prompt: str, negative: str = NEGATIVE, scale: float = 2.0,
                   denoise: float = 0.25, steps: int = 10, cfg: float = 1.0, seed: int = 0, gguf: bool = True,
                   prefix: str = "charforge_hires") -> dict:
    """The 2x refine graph for one picture already in ComfyUI's input folder (comfy.upload_image)."""
    if not 0.0 < denoise <= 0.6:
        raise ValueError("denoise should stay low (0-0.6): above that the refine redraws the figure")
    loader = ({"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "krea2_turbo_bf16-Q5_1.gguf"}}
              if gguf else
              {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2_turbo_fp8_scaled.safetensors",
                                                      "weight_dtype": "default"}})
    return {
        "1": loader,
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
                                                     "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
        "6": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "7": {"class_type": "ImageScaleBy", "inputs": {"image": ["6", 0], "upscale_method": "lanczos",
                                                       "scale_by": float(scale)}},
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["8", 0],
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
            "denoise": float(denoise)}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


OUTPUTS = {"LoadImage": 2, "CLIPLoader": 1, "VAELoader": 1, "UnetLoaderGGUF": 1, "UNETLoader": 1,
           "CLIPTextEncode": 1, "ImageScaleBy": 1, "VAEEncode": 1, "KSampler": 1, "VAEDecode": 1, "SaveImage": 0}


def check_graph(wf: dict) -> None:
    """Every link must point at a node that exists and at an output that node has; a wiring error would otherwise
    only show up inside ComfyUI, after the GPU was taken."""
    for nid, node in wf.items():
        for key, val in node["inputs"].items():
            if isinstance(val, list):
                src, slot = val
                if src not in wf:
                    raise ValueError(f"node {nid} input {key} links to missing node {src}")
                if slot >= OUTPUTS[wf[src]["class_type"]]:
                    raise ValueError(f"node {nid} input {key} links to output {slot} of node {src}")


def pictures(work: Path, what: str) -> list[Path]:
    if what == "reference":
        delit = work / "reference_delit.png"
        return [delit if delit.exists() else work / "reference.png"]
    return sorted(q for q in (work / "mv").glob("*.png") if not q.name.endswith("_hy.png"))


def prompt_for(work: Path) -> str:
    style = json.loads((work / "style.json").read_text()) if (work / "style.json").exists() else {}
    desc = style.get("description") or style.get("prompt") or "a character"
    return (f"{desc}, full body character reference, photographic fabric and skin detail, even neutral studio "
            "light, no shadows, sharp focus")


def upload_name(work: Path, src: Path) -> str:
    """The name a picture has in ComfyUI's input folder, and so in its graph's LoadImage."""
    return f"charforge_{work.name}_{src.stem}.png"


def submit(work: Path, what: str, graphs: dict) -> dict:
    """Upload each picture to ComfyUI, run its graph under the GPU lock, and fetch the result into texwork/hires/.
    Not run yet: ComfyUI is down on this machine."""
    import charforge
    import comfy
    out = work / "texwork" / "hires"
    results = {}
    with charforge.gpu(f"high-resolution pass ({what})"):
        for src in pictures(work, what):
            comfy.upload_image(src, name=upload_name(work, src))
            paths, secs = comfy.run(graphs[src.name], out, prefix=src.stem)
            results[src.name] = [str(p) for p in paths]
            print(f"[hires] {src.name}: {secs:.0f}s -> {paths}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--what", choices=("reference", "views"), default="reference")
    ap.add_argument("--denoise", type=float, default=0.25)
    ap.add_argument("--scale", type=float, default=2.0)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print", action="store_true", help="write the workflow JSON and stop: no ComfyUI call")
    mode.add_argument("--submit", action="store_true", help="run the workflows in ComfyUI (not yet run)")
    a = ap.parse_args()
    work = ROOT / "work" / a.name
    out = work / "texwork" / "hires"
    out.mkdir(parents=True, exist_ok=True)
    prompt = prompt_for(work)
    graphs = {}
    for src in pictures(work, a.what):
        wf = hires_workflow(upload_name(work, src), prompt, scale=a.scale, denoise=a.denoise,
                            prefix=f"charforge_hires_{src.stem}")
        check_graph(wf)
        graphs[src.name] = wf
    path = out / f"workflow_{a.what}.json"
    path.write_text(json.dumps(graphs, indent=1))
    print(f"[hires] {len(graphs)} graph(s) for {a.what} -> {path}", flush=True)
    if a.submit:
        print(json.dumps(submit(work, a.what, graphs), indent=1))


if __name__ == "__main__":
    main()
