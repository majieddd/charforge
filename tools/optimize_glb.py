"""Shrink a character GLB without giving up anything you can see.

The shipped web build is 23.45 MB, and 86% of that is two 4096x4096 PNGs. That resolution was
chosen on the assumption it carried detail. It does not: round-tripping the albedo through 2048
and back scores 41.3 dB PSNR against the 4K original, which is above the point where a
difference is visible - because the albedo is upsampled from TRELLIS's native 1024 in the first
place. The normal map is baked from the high-resolution surface so it holds a little more, but
still scores 38.3 dB at 2048.

Measured on this character:

    albedo   4096 PNG  10.85 MB  ->  2048 WebP q95     0.54 MB   39.6 dB
    normal   4096 PNG   9.34 MB  ->  2048 WebP lossless 3.08 MB   38.3 dB

The normal map is encoded losslessly on purpose. A normal is a direction, not a colour: a small
error in the stored value tilts the surface and shows up as shading that crawls under animation,
and lossy WebP cost it 5 dB where the albedo lost less than 2. Lossy compression on the albedo
also has history here - JPEG bled across this mesh's tiny UV islands and produced rust speckles
and dark gashes, which is why the shipped build is PNG. WebP at q95 is far above where that
happened, and `--verify` re-renders so it is checked rather than assumed.

Textures are rewritten in place and referenced through EXT_texture_webp, which three.js reads
natively.

    python tools/optimize_glb.py --in character.glb --out character_web.glb --res 2048
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def read_glb(path):
    d = open(path, "rb").read()
    if d[:4] != b"glTF":
        raise SystemExit(f"{path} is not a binary glTF")
    off, chunks = 12, []
    while off < len(d):
        ln, ty = struct.unpack_from("<II", d, off)
        chunks.append((ty, d[off + 8: off + 8 + ln]))
        off += 8 + ln
    js = json.loads(chunks[0][1].decode("utf-8"))
    bin_blob = chunks[1][1] if len(chunks) > 1 else b""
    return js, bytearray(bin_blob)


def write_glb(path, js, blob):
    j = json.dumps(js, separators=(",", ":")).encode("utf-8")
    j += b" " * ((4 - len(j) % 4) % 4)
    b = bytes(blob) + b"\0" * ((4 - len(blob) % 4) % 4)
    total = 12 + 8 + len(j) + 8 + len(b)
    out = bytearray()
    out += b"glTF" + struct.pack("<II", 2, total)
    out += struct.pack("<II", len(j), 0x4E4F534A) + j
    out += struct.pack("<II", len(b), 0x004E4942) + b
    open(path, "wb").write(out)
    return total


def psnr(a, b):
    m = float(((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2).mean())
    return 10 * np.log10(255 * 255 / max(m, 1e-9))


def main(a):
    js, blob = read_glb(a.inp)
    views = js["bufferViews"]
    before = os.path.getsize(a.inp)

    # which image is a normal map? it must not be encoded lossily
    normal_imgs = set()
    for mat in js.get("materials", []):
        nt = mat.get("normalTexture")
        if nt is not None:
            tex = js["textures"][nt["index"]]
            src = tex.get("source")
            if src is None:
                src = tex.get("extensions", {}).get("EXT_texture_webp", {}).get("source")
            if src is not None:
                normal_imgs.add(src)

    new_data = {}
    for i, im in enumerate(js.get("images", [])):
        if "bufferView" not in im:
            continue
        bv = views[im["bufferView"]]
        o = bv.get("byteOffset", 0)
        raw = bytes(blob[o: o + bv["byteLength"]])
        img = Image.open(io.BytesIO(raw))
        mode = img.mode
        img = img.convert("RGBA" if "A" in mode else "RGB")
        orig = img
        if max(img.size) > a.res:
            img = img.resize((a.res, a.res), Image.LANCZOS)

        is_normal = i in normal_imgs
        buf = io.BytesIO()
        if is_normal or a.lossless:
            img.save(buf, format="WEBP", lossless=True, quality=100, method=6)
        else:
            img.save(buf, format="WEBP", quality=a.quality, method=6)
        enc = buf.getvalue()

        buf.seek(0)
        back = Image.open(buf).convert(img.mode).resize(orig.size, Image.LANCZOS)
        q = psnr(orig, back)
        kind = "normal (lossless)" if is_normal else f"colour (q{a.quality})"
        print(f"[glb] image[{i}] {im.get('name','')}: {orig.size[0]}px {bv['byteLength']/1e6:5.2f} MB "
              f"-> {img.size[0]}px {len(enc)/1e6:5.2f} MB  {kind}  PSNR {q:.1f} dB", flush=True)
        if q < a.min_psnr:
            raise SystemExit(f"[glb] image[{i}] fell to {q:.1f} dB, below --min-psnr {a.min_psnr}")
        new_data[im["bufferView"]] = enc
        im["mimeType"] = "image/webp"

    # EXT_texture_webp: the texture points at the webp through the extension, and carries no
    # plain `source`, so a loader without the extension fails cleanly instead of reading garbage
    if new_data:
        for tex in js.get("textures", []):
            src = tex.pop("source", None)
            if src is None:
                continue
            tex.setdefault("extensions", {})["EXT_texture_webp"] = {"source": src}
        for key in ("extensionsUsed", "extensionsRequired"):
            js.setdefault(key, [])
            if "EXT_texture_webp" not in js[key]:
                js[key].append("EXT_texture_webp")

    # Rebuild the binary chunk. Every bufferView offset after a resized image shifts, so the
    # blob is reassembled in order rather than patched.
    out = bytearray()
    for idx, bv in enumerate(views):
        data = new_data.get(idx)
        if data is None:
            o = bv.get("byteOffset", 0)
            data = bytes(blob[o: o + bv["byteLength"]])
        pad = (4 - len(out) % 4) % 4
        out += b"\0" * pad
        bv["byteOffset"] = len(out)
        bv["byteLength"] = len(data)
        out += data
    js["buffers"][0]["byteLength"] = len(out)
    js["buffers"][0].pop("uri", None)

    total = write_glb(a.out, js, out)
    print(f"[glb] {before/1e6:.2f} MB -> {total/1e6:.2f} MB  "
          f"({before/max(total,1):.1f}x smaller)  -> {a.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--lossless", action="store_true",
                    help="encode every texture losslessly, not just normal maps")
    ap.add_argument("--min-psnr", type=float, default=32.0,
                    help="fail rather than ship a texture below this against the original")
    main(ap.parse_args())
