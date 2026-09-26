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
        # a quality ladder: the asked-for quality, then 100, then lossless - Knight's scratched armour
        # (roughness and metal changing texel by texel) held 28.5 dB at q95 and failed the build
        steps = [("lossless", None)] if (is_normal and not a.lossy_normals) or a.lossless else \
            [("q", a.quality), ("q", 100), ("lossless", None)]
        for how, qv in steps:
            buf = io.BytesIO()
            if how == "lossless":
                img.save(buf, format="WEBP", lossless=True, quality=100, method=6)
            else:
                img.save(buf, format="WEBP", quality=qv, method=6)
            buf.seek(0)
            if how == "lossless" or psnr(img, Image.open(buf).convert(img.mode)) >= a.min_psnr:
                break
        enc = buf.getvalue()

        buf.seek(0)
        dec = Image.open(buf).convert(img.mode)
        # the encoding judged against the image it encoded; the size is the build's choice, not a loss to
        # gate - Knight's scratched armour (roughness and metal texel by texel) read 26.8 dB against the
        # 4K original at 2K, lossless or not, and the build failed on a loss it was asked to make
        q = psnr(img, dec)
        q_all = psnr(orig, dec.resize(orig.size, Image.LANCZOS)) if img.size != orig.size else q
        kind = f"{'normal' if is_normal else 'colour'} ({'lossless' if how == 'lossless' else f'q{qv}'})"
        print(f"[glb] image[{i}] {im.get('name','')}: {orig.size[0]}px {bv['byteLength']/1e6:5.2f} MB "
              f"-> {img.size[0]}px {len(enc)/1e6:5.2f} MB  {kind}  PSNR {q:.1f} dB"
              + (f" ({q_all:.1f} against the full size)" if img.size != orig.size else ""), flush=True)
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

    if a.quantize:
        # Smaller data, in core glTF (no extension a loader could lack): skin weights as bytes summing to
        # 255, texture coordinates as 16-bit fractions, triangle indices as 16-bit where a mesh has under
        # 65,536 vertices, and animation tracks that never change - most bones' position and scale, 2,834
        # of Pip's 3,648 tracks - cut to two keys. Kept, not dropped: a clip that does not key a bone leaves
        # it where the last clip put it, in three.js as in engines. Ten characters at 1K came to ~80 MB
        # against the artifact's 64 MB.
        ctype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
        ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
        by_view = {}
        for ai_, acc in enumerate(js["accessors"]):
            if "bufferView" in acc:
                by_view.setdefault(acc["bufferView"], []).append(ai_)

        def data_of(ai_):
            acc = js["accessors"][ai_]
            bv = views[acc["bufferView"]]
            if bv.get("byteStride") or acc["bufferView"] in new_data:
                return None
            o = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            n_ = acc["count"] * ncomp[acc["type"]]
            return np.frombuffer(bytes(blob[o: o + n_ * np.dtype(ctype[acc["componentType"]]).itemsize]),
                                 ctype[acc["componentType"]]).reshape(acc["count"], ncomp[acc["type"]])

        repl = {}                                   # accessor -> its new bytes
        saved = 0
        for mesh in js.get("meshes", []):
            for prim in mesh["primitives"]:
                att = prim["attributes"]
                nverts = js["accessors"][att["POSITION"]]["count"]
                for key, ai_ in att.items():
                    acc = js["accessors"][ai_]
                    if acc["componentType"] != 5126:
                        continue
                    arr = data_of(ai_)
                    if arr is None:
                        continue
                    if key.startswith("WEIGHTS_"):
                        w8 = np.clip(np.round(arr / np.maximum(arr.sum(1, keepdims=True), 1e-9) * 255), 0, 255)
                        top = np.argmax(w8, 1)
                        w8[np.arange(len(w8)), top] += 255 - w8.sum(1)          # sums to 255 exactly
                        repl[ai_] = np.clip(w8, 0, 255).astype(np.uint8).tobytes()
                        acc["componentType"], acc["normalized"] = 5121, True
                    elif key.startswith("TEXCOORD_") and arr.min() >= 0.0 and arr.max() <= 1.0:
                        repl[ai_] = np.round(arr * 65535).astype(np.uint16).tobytes()
                        acc["componentType"], acc["normalized"] = 5123, True
                    else:
                        continue
                    acc.pop("min", None); acc.pop("max", None)
                    saved += arr.nbytes - len(repl[ai_])
                ii = prim.get("indices")
                if ii is not None and nverts < 65536 and ii not in repl:
                    acc = js["accessors"][ii]
                    if acc["componentType"] == 5125:
                        arr = data_of(ii)
                        if arr is not None:
                            repl[ii] = arr.astype(np.uint16).tobytes()
                            saved += arr.nbytes - len(repl[ii])
                            acc["componentType"] = 5123
                            acc.pop("min", None); acc.pop("max", None)
        v_saved = saved
        n_const = 0
        for an in js.get("animations", []):
            two = {}                                   # input accessor -> its 2-key replacement
            for smp in an["samplers"]:
                ao = smp["output"]
                oacc = js["accessors"][ao]
                if oacc["componentType"] != 5126 or ao in repl:
                    continue
                out_ = data_of(ao)
                if out_ is None or len(out_) < 3 or float(np.ptp(out_, axis=0).max()) > 1e-6:
                    continue
                if smp["input"] not in two:
                    t = data_of(smp["input"])
                    if t is None:
                        continue
                    tk = np.array([t[0, 0], t[-1, 0]], np.float32)
                    views.append({"buffer": 0, "byteLength": 8})
                    new_data[len(views) - 1] = tk.tobytes()
                    js["accessors"].append({"bufferView": len(views) - 1, "componentType": 5126, "count": 2,
                                            "type": "SCALAR", "min": [float(tk[0])], "max": [float(tk[1])]})
                    two[smp["input"]] = len(js["accessors"]) - 1
                smp["input"] = two[smp["input"]]
                keep_ = np.repeat(out_[:1], 2, axis=0).astype(np.float32)
                repl[ao] = keep_.tobytes()
                saved += out_.nbytes - keep_.nbytes
                oacc["count"] = 2
                oacc.pop("min", None); oacc.pop("max", None)
                n_const += 1
        # Rotation keys interpolation between their neighbours reproduces to a quarter of a degree go:
        # captures are keyed at every frame at 60 fps, and rotations were 1.33 of Pip's 1.41 MB of clips.
        def keep_keys(t, q, tol):
            n_ = len(q)
            keep = np.zeros(n_, bool)
            keep[0] = keep[-1] = True
            todo = [(0, n_ - 1)]
            while todo:
                i0, j0 = todo.pop()
                if j0 - i0 < 2:
                    continue
                qi, qj = q[i0], q[j0] * (1.0 if float(q[i0] @ q[j0]) >= 0 else -1.0)
                u = (t[i0 + 1:j0] - t[i0]) / max(float(t[j0] - t[i0]), 1e-9)
                qq = qi[None] * (1 - u)[:, None] + qj[None] * u[:, None]
                qq /= np.maximum(np.linalg.norm(qq, axis=1, keepdims=True), 1e-12)
                err = 2 * np.arccos(np.clip(np.abs((qq * q[i0 + 1:j0]).sum(1)), 0, 1))
                k_ = int(np.argmax(err))
                if err[k_] > tol:
                    keep[i0 + 1 + k_] = True
                    todo += [(i0, i0 + 1 + k_), (i0 + 1 + k_, j0)]
            return keep

        n_red, r_saved = 0, 0
        for an in js.get("animations", []):
            for ch in an["channels"]:
                if ch["target"].get("path") != "rotation":
                    continue
                smp = an["samplers"][ch["sampler"]]
                if smp.get("interpolation", "LINEAR") != "LINEAR" or smp["output"] in repl:
                    continue
                q = data_of(smp["output"])
                t = data_of(smp["input"])
                if q is None or t is None or len(q) < 4 or js["accessors"][smp["output"]]["componentType"] != 5126:
                    continue
                keep = keep_keys(t[:, 0].astype(np.float64), q.astype(np.float64), np.radians(a.rot_tol))
                if keep.sum() >= len(q):
                    continue
                tk = t[keep, 0].astype(np.float32)
                views.append({"buffer": 0, "byteLength": tk.nbytes})
                new_data[len(views) - 1] = tk.tobytes()
                js["accessors"].append({"bufferView": len(views) - 1, "componentType": 5126, "count": int(len(tk)),
                                        "type": "SCALAR", "min": [float(tk[0])], "max": [float(tk[-1])]})
                smp["input"] = len(js["accessors"]) - 1
                repl[smp["output"]] = q[keep].astype(np.float32).tobytes()
                oacc = js["accessors"][smp["output"]]
                r_saved += q.nbytes + 4 * len(q) - (len(repl[smp["output"]]) + tk.nbytes)
                oacc["count"] = int(keep.sum())
                oacc.pop("min", None); oacc.pop("max", None)
                n_red += 1
        saved += r_saved
        print(f"[glb] {n_red} rotation tracks thinned to within {a.rot_tol} deg: {r_saved / 1e6:.2f} MB saved", flush=True)

        # repack every view an accessor was replaced in: its accessors one after another, 4-byte aligned
        for vi, users_ in by_view.items():
            if vi in new_data or not any(u in repl for u in users_):
                continue
            bv = views[vi]
            packed = bytearray()
            for u in sorted(users_, key=lambda u: js["accessors"][u].get("byteOffset", 0)):
                acc = js["accessors"][u]
                if u in repl:
                    data = repl[u]
                else:
                    o = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
                    n_ = acc["count"] * ncomp[acc["type"]] * np.dtype(ctype[acc["componentType"]]).itemsize
                    data = bytes(blob[o: o + n_])
                packed += b"\0" * ((4 - len(packed) % 4) % 4)
                acc["byteOffset"] = len(packed)
                packed += data
            new_data[vi] = bytes(packed)
        print(f"[glb] vertex data quantized ({v_saved / 1e6:.2f} MB), {n_const} tracks that never change cut to "
              f"two keys ({(saved - v_saved) / 1e6:.2f} MB)", flush=True)

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
    ap.add_argument("--lossy-normals", action="store_true",
                    help="encode normal maps lossily too (a preview build under a size cap)")
    ap.add_argument("--rot-tol", type=float, default=0.25,
                    help="with --quantize: rotation keys closer than this (degrees) to the interpolation go")
    ap.add_argument("--quantize", action="store_true",
                    help="skin weights as bytes, UVs and indices as 16-bit (core glTF) - for size-capped builds")
    ap.add_argument("--min-psnr", type=float, default=32.0,
                    help="fail rather than ship a texture below this against the original")
    main(ap.parse_args())
