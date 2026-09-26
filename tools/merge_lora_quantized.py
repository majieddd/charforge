"""Bake a LoRA into a ComfyUI-quantized checkpoint, once, without requantizing it.

    python tools/merge_lora_quantized.py MODEL.safetensors LORA.safetensors OUT.safetensors [--strength 1.0]

Why not let ComfyUI apply it: on a Mac, ComfyUI loads a model whole (its shared-memory mode) and
patches a LoRA into every weight at load, keeping a backup copy of each - MiniMax H3's 15.4 GB of
W4A8 weights plus the copies swapped for 20 minutes on a 24 GB machine without finishing the load.
And each patched weight is requantized with new scales, which added 4.5% error to every layer
(measured) to carry a change of 0.02% (the turbo LoRA's median, 0.19% at most).

What this does instead: every weight keeps its layer's grid - the 16-level codebook, the group
and channel scales stay exactly as they are - and only the codes move. A weight the LoRA nudges
by a fraction of the way to the next level moves there with that probability (stochastic
rounding), so each layer's expected weight is the base plus the LoRA - W + strength * alpha/rank
* B @ A, ComfyUI's arithmetic (comfy/weight_adapter/lora.py) - and the noise added is only what
representing a sub-step change on a 4-bit grid costs. The grid lives in the rotated (ConvRot)
basis, so the change is rotated into it first.

Every tensor keeps its shape and dtype; the output is the input file with the code bytes
rewritten in place (on APFS the 15 GB copy is a clone and takes no time).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import time

import numpy as np
import torch
from safetensors import safe_open

from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight as rotate


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def generator(key, dev):
    g = torch.Generator(device=dev)
    g.manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
    return g


def unpack(q):
    """(n, k/2) int8, two 4-bit codes a byte (low nibble first) -> (n, k) codes 0-15."""
    p = q.to(torch.int32) & 0xFF
    c = torch.empty(q.shape[0], q.shape[1] * 2, dtype=torch.int32, device=q.device)
    c[:, 0::2], c[:, 1::2] = p & 0xF, (p >> 4) & 0xF
    return c


def pack(c):
    return ((c[:, 0::2] & 0xF) | ((c[:, 1::2] & 0xF) << 4)).to(torch.uint8).view(torch.int8)


def repick_w4a8(q, s_rel, s_ch, cb, group, rot_group, delta, gen):
    """New codes on the unchanged grid, for the weight plus delta (physical basis)."""
    n, k = q.shape[0], q.shape[1] * 2
    groups = k // group
    codes = unpack(q)
    levels = (cb.view(1, 1, 16) * s_rel.float().unsqueeze(-1)).round().clamp(-127, 127)   # (n, groups, 16)
    now = torch.gather(levels, 2, codes.view(n, groups, group).long())                    # the int8 grid values
    t = now + rotate(delta, rot_group).view(n, groups, group) / s_ch.view(n, 1, 1)
    lv = levels.reshape(n * groups, 16).contiguous()
    tg = t.reshape(n * groups, group).contiguous()
    pos = torch.searchsorted(lv, tg)
    lo = (pos - 1).clamp(0, 14)
    lo_v, hi_v = torch.gather(lv, 1, lo), torch.gather(lv, 1, lo + 1)
    frac = ((tg - lo_v) / (hi_v - lo_v).clamp_min(1e-9)).clamp(0, 1)
    up = torch.rand(tg.shape, generator=gen, device=tg.device) < frac
    new = (lo + up.long()).view(n, k).to(torch.int32)
    return pack(new), float((new != codes).float().mean())


def repick_int8(q, scale, rot_group, delta, gen):
    """INT8 row-wise ConvRot: the same, on its integer grid."""
    t = q.float() + rotate(delta, rot_group) / scale.view(-1, 1)
    new = torch.floor(t + torch.rand(t.shape, generator=gen, device=t.device)).clamp(-127, 127).to(torch.int8)
    return new, float((new != q).float().mean())


def main(a):
    dev = torch.device(a.device)
    hdr, base = header(a.model)
    hdr.pop("__metadata__", None)
    t0 = time.time()
    shutil.copyfile(a.model, a.out + ".part")
    lora = safe_open(a.lora, "pt")
    mods = sorted({k.rsplit(".", 2)[0] for k in lora.keys() if k.endswith(".lora_A.weight")})
    if a.only:
        mods = [m for m in mods if m in set(a.only)]
    src = safe_open(a.model, "pt")
    out = open(a.out + ".part", "r+b")
    done, skipped, moved, gain, noise = 0, [], [], [], []
    for m in mods:
        pre = "model." + m + "."
        if pre + "weight" not in hdr or pre + "comfy_quant" not in hdr:
            skipped.append(m)
            continue
        conf = json.loads(bytes(src.get_tensor(pre + "comfy_quant").tolist()).decode())
        A = lora.get_tensor(m + ".lora_A.weight").to(dev, torch.float32)
        B = lora.get_tensor(m + ".lora_B.weight").to(dev, torch.float32)
        alpha = float(lora.get_tensor(m + ".alpha")) if m + ".alpha" in lora.keys() else float(A.shape[0])
        delta = (B @ A) * (a.strength * alpha / A.shape[0])
        q = src.get_tensor(pre + "weight").to(dev)
        gen = generator(pre, dev)
        rot_group = conf.get("convrot_groupsize", 256)
        if conf["format"] == "asym_w4a8_int8":
            s_rel = src.get_tensor(pre + "weight_s_rel").to(dev)
            s_ch = src.get_tensor(pre + "weight_s_channel").to(dev)
            cb = src.get_tensor(pre + "weight_codebook").to(dev)
            group = conf.get("group_size", 16)
            q2, frac = repick_w4a8(q, s_rel, s_ch, cb, group, rot_group, delta, gen)

            def phys(qq):
                n_, k_ = qq.shape[0], qq.shape[1] * 2
                lv = (cb.view(1, 1, 16) * s_rel.float().unsqueeze(-1)).round().clamp(-127, 127)
                v = torch.gather(lv, 2, unpack(qq).view(n_, k_ // group, group).long()).view(n_, k_)
                return rotate(v * s_ch.view(-1, 1), rot_group)
        elif conf["format"] == "int8_tensorwise" and conf.get("convrot"):
            scale = src.get_tensor(pre + "weight_scale").to(dev)
            q2, frac = repick_int8(q, scale, rot_group, delta, gen)

            def phys(qq):
                return rotate(qq.float() * scale.view(-1, 1), rot_group)
        else:
            skipped.append(f"{m} ({conf['format']})")
            continue
        W0 = phys(q)
        applied = phys(q2) - W0
        # unbiased: the change made carries the LoRA's change (projection ~1); the rest is noise
        gain.append(float((applied * delta).sum() / (delta * delta).sum()))
        noise.append(float((applied - delta).norm() / W0.norm()))
        moved.append(frac)
        info = hdr[pre + "weight"]
        buf = q2.to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
        s, e = info["data_offsets"]
        assert len(buf) == e - s, pre
        out.seek(base + s)
        out.write(buf)
        done += 1
        if done % 25 == 0:
            print(f"[merge] {done}/{len(mods)} layers, {time.time() - t0:.0f} s", flush=True)
    out.close()
    os.replace(a.out + ".part", a.out)
    print(f"[merge] {done} layers ({len(skipped)} LoRA modules with no layer here); codes moved: "
          f"{np.mean(moved):.3%} (mean); the change made carries the LoRA's by {np.median(gain):.2f} "
          f"(median projection, 1 = exact) with {np.median(noise):.2%} noise of the weight (median) "
          f"-> {a.out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("lora")
    ap.add_argument("out")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--only", nargs="*", default=None, help="merge just these LoRA modules (a test)")
    main(ap.parse_args())
