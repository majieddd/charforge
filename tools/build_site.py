"""Build the playground and the release packages from the characters in out/.

Until this existed, every published playground was assembled by hand: thumbnails rendered and
cropped in a scratch script, base64 chunks regenerated with a one-off command, the roster
edited inline, glb files copied into docs/ one at a time. That is how the published page drifted
from the characters it claimed to show. Everything is now generated from web/roster.json and
the packages charforge.py writes:

  docs/          the GitHub Pages build - playground + each character's compressed .glb, streamed
  web/dist/      the same playground for hosts that will not serve .glb (claude.ai artifacts):
                 each character as a lazily imported base64 module, split under the 16 MB file cap
  dist/          one zip per character for a GitHub release - glTF, FBX, textures, manifest -
                 kept out of the repository so the git history does not carry 60 MB per character

    python tools/build_site.py
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 7_000_000


def thumb_uri(png: Path) -> str:
    """Crop to the figure by alpha, fit a 124x168 card, encode as a small WebP data URI."""
    im = Image.open(png).convert("RGBA")
    a = np.asarray(im)[:, :, 3]
    ys, xs = np.where(a > 8)
    pad = 6
    im = im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                  min(im.width, xs.max() + pad), min(im.height, ys.max() + pad)))
    tw, th = 124, 168
    s = min(tw / im.width, th / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    card = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    card.paste(im, ((tw - im.width) // 2, (th - im.height) // 2), im)
    buf = io.BytesIO()
    card.save(buf, format="WEBP", quality=86, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def page_manifest(m: dict) -> dict:
    """The subset the playground reads: clip timing and speeds, and the facts on the stats panel."""
    return {"height_m": m["height_m"], "triangles": m["triangles"], "capsule": m.get("capsule"),
            "skeleton": {k: m["skeleton"][k] for k in ("convention", "bones")},
            "clips": m["clips"], "prompt": (m.get("source") or {}).get("prompt"),
            "springs": m.get("springs"), "face": m.get("face"),
            "style": (m.get("style") or {}).get("name", "realistic")}


def roster_block(entries, sources) -> str:
    return ("const CHARACTER_SOURCES = {\n" + "".join(f"  {k}: {v},\n" for k, v in sources.items())
            + "};\nconst CHARACTERS = " + json.dumps(entries, indent=1) + ";\n")


def inject(template: str, block: str) -> str:
    a, b = template.index("/*@ROSTER@*/"), template.index("/*@END_ROSTER@*/")
    return template[:a] + "/*@ROSTER@*/\n" + block + template[b:]


def main(a):
    cfg = json.load(open(ROOT / "web" / "roster.json"))
    template = (ROOT / "web" / "playground.html").read_text()
    release = f"https://github.com/{cfg['repo']}/releases/download/{cfg['release']}"
    entries, pages_src, art_src = [], {}, {}
    docs, art, dist = ROOT / "docs", ROOT / "web" / "dist", ROOT / "dist"
    for d in (art, dist):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    docs.mkdir(exist_ok=True)
    for stale in docs.glob("*.glb"):
        stale.unlink()

    for c in cfg["characters"]:
        cid = c["id"]
        pkg = ROOT / "out" / cid
        man = json.load(open(pkg / f"{cid}.json"))
        web_glb = pkg / f"{cid}_web.glb"
        entries.append({"id": cid, "name": c["name"], "blurb": c["blurb"],
                        "prompt": man["source"]["prompt"] or c.get("prompt", ""),
                        "thumb": thumb_uri(pkg / f"{cid}_thumb.png"),
                        "download": f"{release}/{cid}.zip", "manifest": page_manifest(man)})

        # Pages: stream the compressed glb
        shutil.copy(web_glb, docs / f"{cid}.glb")
        pages_src[cid] = f"(p) => fetchGLB('./{cid}.glb', p)"

        # artifact: base64 modules, lazily imported, split under the per-file cap. An artifact holds
        # 64 MB a version, and seven characters at 2K textures came to 77, so it can take its own
        # smaller build (--artifact-res); Pages keeps the 2K one.
        art_glb = web_glb
        if a.artifact_res:
            import subprocess
            import sys as _sys
            art_glb = ROOT / "work" / "_artifact_glb" / f"{cid}.glb"
            art_glb.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([_sys.executable, str(ROOT / "tools" / "optimize_glb.py"), "--in", str(pkg / f"{cid}.glb"),
                            "--out", str(art_glb), "--res", str(a.artifact_res),
                            # a preview copy: 1K loses detail against the 4K bake by design
                            "--min-psnr", "26"], check=True,
                           stdout=subprocess.DEVNULL)
        b64 = base64.b64encode(art_glb.read_bytes()).decode()
        n = math.ceil(len(b64) / CHUNK)
        for i in range(n):
            (art / f"{cid}_{i}.js").write_text(f'export const P = "{b64[i*CHUNK:(i+1)*CHUNK]}";\n')
        imports = "\n".join(f'import {{ P as P{i} }} from "./{cid}_{i}.js";' for i in range(n))
        (art / f"{cid}.js").write_text(f"{imports}\nexport const GLB_B64 = "
                                       + " + ".join(f"P{i}" for i in range(n)) + ";\n")
        art_src[cid] = f"() => import('./{cid}.js').then(m => b64ToArrayBuffer(m.GLB_B64))"

        # release zip: the full-quality package
        z = dist / f"{cid}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(pkg.rglob("*")):
                if f.is_file() and not f.name.endswith("_web.glb") and not f.name.endswith("_thumb.png"):
                    zf.write(f, f"{cid}/{f.relative_to(pkg)}")
        print(f"  {cid:6s} web {web_glb.stat().st_size/1e6:5.2f} MB ({n} artifact chunk"
              f"{'s' if n > 1 else ''}), release zip {z.stat().st_size/1e6:5.1f} MB", flush=True)

    (docs / "index.html").write_text(inject(template, roster_block(entries, pages_src)))
    (art / "index.html").write_text(inject(template, roster_block(entries, art_src)))
    print(f"  pages    -> {docs/'index.html'}\n  artifact -> {art/'index.html'}\n"
          f"  release  -> {dist}/  (upload with: gh release create {cfg['release']} dist/*.zip)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--artifact-res", type=int, default=0,
                    help="texture size for the single-file artifact build (default: the web build's)")
    main(ap.parse_args())
