"""Minimal ComfyUI API client: submit a workflow, wait, fetch the images.

ComfyUI is already installed on this machine with Krea 2 Turbo, so the reference-image stage
drives it over its HTTP API rather than re-implementing a sampler.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HOST = "http://127.0.0.1:8188"


def _post(path, payload):
    req = urllib.request.Request(f"{HOST}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def _get(path):
    return json.loads(urllib.request.urlopen(f"{HOST}{path}", timeout=60).read())


def submit(workflow: dict, client_id: str | None = None) -> str:
    client_id = client_id or str(uuid.uuid4())
    return _post("/prompt", {"prompt": workflow, "client_id": client_id})["prompt_id"]


def wait(prompt_id: str, timeout=3600, poll=3.0, log_every=60):
    t0 = time.time()
    last = 0
    while time.time() - t0 < timeout:
        h = _get(f"/history/{prompt_id}")
        if prompt_id in h:
            entry = h[prompt_id]
            st = entry.get("status", {})
            if st.get("status_str") == "error" or st.get("completed") is False and st.get("messages"):
                for m in st.get("messages", []):
                    if m[0] in ("execution_error", "execution_interrupted"):
                        raise RuntimeError(json.dumps(m[1])[:800])
            if entry.get("outputs"):
                return entry
        el = time.time() - t0
        if el - last > log_every:
            last = el
            print(f"  ... {el:.0f}s", flush=True)
        time.sleep(poll)
    raise TimeoutError(f"workflow {prompt_id} did not finish in {timeout}s")


def fetch_images(entry, out_dir: Path, prefix="out"):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for node_id, out in entry["outputs"].items():
        for i, im in enumerate(out.get("images", [])):
            q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                        "type": im.get("type", "output")})
            data = urllib.request.urlopen(f"{HOST}/view?{q}", timeout=120).read()
            p = out_dir / f"{prefix}_{node_id}_{i}.png"
            p.write_bytes(data)
            paths.append(p)
    return paths


def run(workflow: dict, out_dir: Path, prefix="out", timeout=3600):
    pid = submit(workflow)
    t0 = time.time()
    entry = wait(pid, timeout=timeout)
    paths = fetch_images(entry, out_dir, prefix)
    return paths, time.time() - t0


def krea2_t2i(prompt: str, negative: str = "", width=1024, height=1024, steps=8, cfg=1.0,
              seed=0, gguf=True, sampler="euler", scheduler="simple"):
    """Text -> image with Krea 2 Turbo (Qwen-Image-family: qwen3vl text encoder + qwen image VAE)."""
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
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler,
            "scheduler": scheduler, "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "charforge"}},
    }


def krea2_i2i(image_name: str, prompt: str, negative: str = "", denoise: float = 0.45,
              steps: int = 10, cfg: float = 1.0, seed: int = 0, gguf: bool = True):
    """Image-to-image with Krea 2 Turbo.

    Used to turn a clean render of the first-pass mesh back into something photographic before
    it is fed to the 3D model as a conditioning view: the render fixes the silhouette and the
    colours, the diffusion pass puts plausible fabric, hair and shading detail back in.
    `image_name` must already be in ComfyUI's input folder (see upload_image).
    """
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
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0],
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "simple", "denoise": denoise}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "charforge_mv"}},
    }


def upload_image(path, name=None, overwrite=True):
    """Put a file into ComfyUI's input folder so LoadImage can reference it."""
    import mimetypes
    import uuid as _uuid
    name = name or Path(path).name
    boundary = "----charforge" + _uuid.uuid4().hex
    body = []
    body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                f"Content-Type: {mimetypes.guess_type(name)[0] or 'image/png'}\r\n\r\n".encode())
    body.append(Path(path).read_bytes())
    body.append(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\n"
                f"{'true' if overwrite else 'false'}\r\n--{boundary}--\r\n".encode())
    data = b"".join(body)
    req = urllib.request.Request(f"{HOST}/upload/image", data=data,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())
