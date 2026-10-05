"""The reference picture with its lighting taken out, for the front projection (E133).

    vendor/hunyuan3d-2.1-mac-rocm/venv/bin/python tools/delight.py --name pip --model hunyuan|marigold

The reference is projected onto the front of the model with the picture's lighting drawn in, while the sides
and back carry an unlit albedo, so the same garment is two colours. A delighting model returns the picture
unlit, but at its own resolution (Hunyuan3D's delighting model 512 px, Marigold's appearance model 768 px)
and free to drift in colour. So only what it took away is kept: the ratio of the picture's lightness to its
unlit lightness, smoothed, is the lighting; the full-resolution picture divided by it keeps every pixel of
the picture's detail and its own hues.

Writes work/<name>/reference_delit_<model>.png (full resolution, outside the figure unchanged) and the
model's raw output beside it (_raw), for inspection.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
HF = Path.home() / ".cache" / "huggingface" / "hub"


def load_reference(w, image=None, mask=None):
    if image:
        # another picture (the face close-up, E138): the whole frame is the figure unless a mask says otherwise
        ref = Image.open(image).convert("RGB")
        m = Image.open(mask).convert("L").resize(ref.size) if mask else Image.new("L", ref.size, 255)
        return np.asarray(ref, np.float32) / 255, np.asarray(m, np.float32) / 255
    ref = Image.open(w / "reference.png").convert("RGB")
    if (w / "reference_rgba.png").exists():
        m = Image.open(w / "reference_rgba.png").getchannel("A").resize(ref.size)
    else:
        m = Image.open(w / "reference_mask.png").convert("L").resize(ref.size)
    return np.asarray(ref, np.float32) / 255, np.asarray(m, np.float32) / 255


def square_crop(mask, margin=0.06):
    ys, xs = np.nonzero(mask > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    return cy - side // 2, cx - side // 2, side


def crop(img, y, x, side, fill):
    h, w = img.shape[:2]
    out = np.full((side, side) + img.shape[2:], fill, np.float32)
    ys, xs = max(0, y), max(0, x)
    ye, xe = min(h, y + side), min(w, x + side)
    out[ys - y:ye - y, xs - x:xe - x] = img[ys:ye, xs:xe]
    return out


def run_hunyuan(img_rgb, res=512, seed=42):
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
    path = next((HF / "models--tencent--Hunyuan3D-2" / "snapshots").glob("*/hunyuan3d-delight-v2-0"))
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(str(path), torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("mps")
    im = Image.fromarray((img_rgb * 255).astype(np.uint8)).resize((res, res), Image.LANCZOS)
    out = pipe(prompt="", image=im, generator=torch.manual_seed(seed), height=res, width=res,
               num_inference_steps=50, image_guidance_scale=1.5, guidance_scale=1.0).images[0]
    return np.asarray(out.convert("RGB"), np.float32) / 255, np.asarray(im, np.float32) / 255


def run_marigold(img_rgb, res=768, seed=42):
    from diffusers import MarigoldIntrinsicsPipeline
    pipe = MarigoldIntrinsicsPipeline.from_pretrained("prs-eth/marigold-iid-appearance-v1-1", variant="fp16",
                                                      torch_dtype=torch.float16).to("mps")
    im = Image.fromarray((img_rgb * 255).astype(np.uint8)).resize((res, res), Image.LANCZOS)
    out = pipe(im, num_inference_steps=4, ensemble_size=1, generator=torch.manual_seed(seed))
    vis = pipe.image_processor.visualize_intrinsics(out.prediction, pipe.target_properties)
    return np.asarray(vis[0]["albedo"].convert("RGB"), np.float32) / 255, np.asarray(im, np.float32) / 255


def lum(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="hunyuan", choices=("hunyuan", "marigold"))
    ap.add_argument("--blur", type=float, default=3.0, help="smoothing of the lighting at the model's resolution (px)")
    ap.add_argument("--out", default=None, help="where to write (default work/<name>/reference_delit_<model>.png)")
    ap.add_argument("--work", default=None, help="the character's work dir (default work/<name>)")
    ap.add_argument("--image", default=None, help="delight this picture instead of the reference (needs --out)")
    ap.add_argument("--mask", default=None, help="the figure in --image (default: all of it)")
    ap.add_argument("--reuse-raw", action="store_true", help="take the model's output saved by an earlier run")
    ap.add_argument("--keep-level", action=argparse.BooleanOptionalAction, default=True,
                    help="take away only the lighting's variation, keeping the picture's overall lightness")
    a = ap.parse_args()
    w = Path(a.work) if a.work else ROOT / "work" / a.name
    ref, mask = load_reference(w, a.image, a.mask)
    y, x, side = square_crop(mask)
    rc = crop(ref, y, x, side, 1.0)
    mc = crop(mask, y, x, side, 0.0)
    on_white = rc * mc[..., None] + (1 - mc[..., None])          # the figure on white, as the models expect
    raw_path = (Path(a.out).with_name(Path(a.out).stem + "_raw.png") if a.image
                else w / f"reference_delit_{a.model}_raw.png")
    if a.reuse_raw and raw_path.exists():
        raw = np.asarray(Image.open(raw_path).convert("RGB"), np.float32) / 255
        src = np.asarray(Image.fromarray((on_white * 255).astype(np.uint8)).resize(raw.shape[1::-1], Image.LANCZOS),
                         np.float32) / 255
    else:
        raw, src = (run_hunyuan if a.model == "hunyuan" else run_marigold)(on_white)
        Image.fromarray((raw * 255).astype(np.uint8)).save(raw_path)
    # the lighting the model took away, as a smooth lightness ratio inside the figure
    eps = 0.02
    ratio = (lum(src) + eps) / (lum(raw) + eps)
    m_small = np.asarray(Image.fromarray((mc * 255).astype(np.uint8)).resize(ratio.shape[::-1], Image.BILINEAR),
                         np.float32) / 255
    ratio = np.where(m_small > 0.5, ratio, 1.0)
    rimg = Image.fromarray(np.clip(ratio * 100, 0, 255).astype(np.uint8))   # 0..2.55 in 1/100 steps
    rimg = rimg.filter(ImageFilter.GaussianBlur(a.blur)).resize((side, side), Image.BICUBIC)
    shade = np.asarray(rimg, np.float32) / 100
    shade = np.clip(shade, 0.35, 2.5)
    inside = mc > 0.5
    level = float(np.median(shade[inside]))
    if a.keep_level:
        # the model also moves the picture's exposure (Juno's red jacket and Knight2's steel came out darker):
        # the drawn mid-tones are the design's colours, so only the lighting's variation about them is removed
        shade = shade / level
    delit = np.clip(rc / shade[..., None], 0, 1)
    delit = rc * (1 - mc[..., None]) + delit * mc[..., None]      # outside the figure: as it was
    out = ref.copy()
    H, W = ref.shape[:2]
    ys, xs, ye, xe = max(0, y), max(0, x), min(H, y + side), min(W, x + side)
    out[ys:ye, xs:xe] = delit[ys - y:ye - y, xs - x:xe - x]
    dst = Path(a.out) if a.out else w / f"reference_delit_{a.model}.png"
    Image.fromarray((out * 255).astype(np.uint8)).save(dst)
    kept = " - the picture's kept" if a.keep_level else ""
    print(f"[delight] {a.name} ({a.model}): the model's overall level {level:.2f}{kept}; "
          f"lighting removed - lightness ratio p10/p50/p90 "
          f"{np.quantile(shade[inside], 0.1):.2f}/{np.median(shade[inside]):.2f}/{np.quantile(shade[inside], 0.9):.2f} "
          f"-> {dst}", flush=True)


if __name__ == "__main__":
    main()
