"""Convert timm's ungated DINOv3 ViT-L/16 checkpoint into the HF key layout TRELLIS ports expect.

facebook/dinov3-vitl16-pretrain-lvd1689m is gated behind a manual access request. timm publishes
the same weights ungated as timm/vit_large_patch16_dinov3.lvd1689m, in timm's own key layout.
This rewrites the keys (and splits the fused QKV) so the MLX/PyTorch ports load it unchanged.

DINOv3-L uses no QKV bias, so q/v biases are written as zeros - which is what the HF checkpoint
stores for those tensors too.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

TIMM_REPO = "timm/vit_large_patch16_dinov3.lvd1689m"
HID, LAYERS = 1024, 24


def find_timm():
    hits = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--timm--vit_large_patch16_dinov3.lvd1689m/snapshots/*/model.safetensors"))
    if not hits:
        raise SystemExit(f"timm checkpoint not found - run: hf download {TIMM_REPO}")
    return hits[0]


def convert(src: str, out_dir: str):
    out = {}
    with safe_open(src, framework="numpy") as f:
        t = {k: f.get_tensor(k) for k in f.keys()}

    out["embeddings.cls_token"] = t["cls_token"]
    out["embeddings.register_tokens"] = t.get("reg_token", t.get("storage_tokens"))
    out["embeddings.patch_embeddings.weight"] = t["patch_embed.proj.weight"]
    out["embeddings.patch_embeddings.bias"] = t["patch_embed.proj.bias"]
    out["norm.weight"] = t["norm.weight"]
    out["norm.bias"] = t["norm.bias"]

    zero = np.zeros((HID,), dtype=t["cls_token"].dtype)
    for i in range(LAYERS):
        s, d = f"blocks.{i}", f"layer.{i}"
        qkv = t[f"{s}.attn.qkv.weight"]
        q, k, v = qkv[:HID], qkv[HID:2 * HID], qkv[2 * HID:]
        qkv_b = t.get(f"{s}.attn.qkv.bias")
        qb, vb = (qkv_b[:HID], qkv_b[2 * HID:]) if qkv_b is not None else (zero, zero)
        out[f"{d}.attention.q_proj.weight"] = q
        out[f"{d}.attention.q_proj.bias"] = qb
        out[f"{d}.attention.k_proj.weight"] = k
        out[f"{d}.attention.v_proj.weight"] = v
        out[f"{d}.attention.v_proj.bias"] = vb
        out[f"{d}.attention.o_proj.weight"] = t[f"{s}.attn.proj.weight"]
        out[f"{d}.attention.o_proj.bias"] = t[f"{s}.attn.proj.bias"]
        out[f"{d}.norm1.weight"] = t[f"{s}.norm1.weight"]
        out[f"{d}.norm1.bias"] = t[f"{s}.norm1.bias"]
        out[f"{d}.norm2.weight"] = t[f"{s}.norm2.weight"]
        out[f"{d}.norm2.bias"] = t[f"{s}.norm2.bias"]
        out[f"{d}.layer_scale1.lambda1"] = t[f"{s}.gamma_1"]
        out[f"{d}.layer_scale2.lambda1"] = t[f"{s}.gamma_2"]
        out[f"{d}.mlp.up_proj.weight"] = t[f"{s}.mlp.fc1.weight"]
        out[f"{d}.mlp.up_proj.bias"] = t[f"{s}.mlp.fc1.bias"]
        out[f"{d}.mlp.down_proj.weight"] = t[f"{s}.mlp.fc2.weight"]
        out[f"{d}.mlp.down_proj.bias"] = t[f"{s}.mlp.fc2.bias"]

    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, str(od / "model.safetensors"))
    json.dump({"model_type": "dinov3_vit", "hidden_size": HID, "num_hidden_layers": LAYERS,
               "num_attention_heads": 16, "patch_size": 16, "num_register_tokens": 4,
               "intermediate_size": 4096, "converted_from": TIMM_REPO},
              open(od / "config.json", "w"), indent=2)
    print(f"[dinov3] {len(out)} tensors -> {od/'model.safetensors'}")
    return od


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "models" / "dinov3-vitl16-hf"))
    a = ap.parse_args()
    convert(find_timm(), a.out)
