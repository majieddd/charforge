"""Make a ComfyUI-quantized checkpoint loadable on Apple GPUs: rewrite its float8 tensors as float32.

    python tools/f8_scales_to_f32.py IN.safetensors OUT.safetensors

PyTorch's MPS backend has no float8 type ("Trying to convert Float8_e4m3fn to the MPS backend"),
so a W4A8 checkpoint whose per-group scales are float8 (weight_s_rel) cannot be moved to the
GPU, and ComfyUI fails on the first layer. comfy_kitchen's portable W4A8 kernels also accept
float32 scales, and every e4m3 value is exact in float32, so this is lossless: only the float8
tensors change (MiniMax-H3 Singularity w4a8: 1.2 G scales, file 11.8 -> 15.4 GB).

Streams tensor by tensor (never holds the checkpoint in memory) and keeps every other tensor's
bytes and the order of the original file.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

F8 = {"F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}


def main(a):
    src = Path(a.src)
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None)
    base = 8 + n
    order = sorted(header, key=lambda k: header[k]["data_offsets"][0])

    out_header, offset = {}, 0
    for k in order:
        t = header[k]
        count = int(np.prod(t["shape"])) if t["shape"] else 1
        dtype, size = t["dtype"], t["data_offsets"][1] - t["data_offsets"][0]
        if dtype in F8:
            dtype, size = "F32", count * 4
        out_header[k] = {"dtype": dtype, "shape": t["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    if meta is not None:
        out_header["__metadata__"] = meta
    blob = json.dumps(out_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # safetensors aligns the data section to 8 bytes

    converted = 0
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    tmp = Path(a.dst + ".part")
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for k in order:
            t = header[k]
            s, e = t["data_offsets"]
            raw = np.asarray(mm[base + s: base + e])
            if t["dtype"] in F8:
                x = torch.frombuffer(bytearray(raw.tobytes()), dtype=F8[t["dtype"]]).float()
                raw = x.numpy().view(np.uint8)
                converted += 1
            out.write(raw.tobytes())
    tmp.rename(a.dst)
    print(f"{converted} float8 tensors -> float32; {src.stat().st_size / 1e9:.2f} GB -> {Path(a.dst).stat().st_size / 1e9:.2f} GB: {a.dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    main(ap.parse_args())
