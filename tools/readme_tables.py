"""The README's roster and foot-slip tables, written from the packages and audits - not typed.

    python tools/readme_tables.py            # prints the markdown
    python tools/readme_tables.py --write    # replaces the tables between their markers in README.md

Reads web/roster.json (who is shown, in order), out/<id>/<id>.json (height, triangles, LODs,
style, face, springs) and work/<id>/foot_audit.json (blender/foot_audit.py --json).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAITS = ("walk", "jog", "run", "sprint", "walk_back", "jog_back", "strafe_left", "strafe_right", "crouch_walk")


def roster(cfg):
    rel = f"https://github.com/{cfg['repo']}/releases/download/{cfg['release']}"
    rows = ["| | style | made from | height | triangles (LOD0 / 1 / 2) | face | package |",
            "|---|---|---|---|---|---|---|"]
    for c in cfg["characters"]:
        mf = ROOT / "out" / c["id"] / f"{c['id']}.json"
        if not mf.exists():
            continue
        m = json.load(open(mf))
        lods = " / ".join(f"{x['triangles']:,}" for x in (m.get("lods") or {}).get("fbx", [])) or f"{m['triangles']:,}"
        prompt = (m.get("source") or {}).get("prompt") or c.get("prompt") or ""
        made = f"*\"{prompt}\"*" if (m.get("source") or {}).get("prompt") else prompt
        face = m.get("face") or {}
        parts = (["jaw"] if face.get("jaw_bone") else []) + [x.replace("_L", "").replace("_R", "")
                                                              for x in face.get("morphs", [])]
        face_s = ", ".join(dict.fromkeys(parts)) or "-"
        rows.append(f"| **{c['name']}** | {(m.get('style') or {}).get('name', 'realistic')} | {made} | {m['height_m']:.2f} m | {lods} | {face_s} | "
                    f"[{c['id']}.zip]({rel}/{c['id']}.zip) |")
    return "\n".join(rows)


def feet(cfg):
    names, data = [], []
    for c in cfg["characters"]:
        f = ROOT / "work" / c["id"] / "foot_audit.json"
        if f.exists():
            names.append(c["name"])
            data.append(json.load(open(f)))
    rows = ["| contact slip, share of ground speed (worse foot) | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names)]
    for g in GAITS:
        cells = []
        for d in data:
            r = d.get(g)
            if not r:
                cells.append("-")
                continue
            s = [r[k]["slip"] for k in ("left", "right") if r[k]["slip"] is not None]
            cells.append(f"{max(s) * 100:.1f}%" if s else "-")
        rows.append(f"| {g.replace('_', ' ')} | " + " | ".join(cells) + " |")
    deep = [min(min(r[k]["deepest_cm"] for k in ("left", "right")) for g, r in d.items() if g in GAITS) for d in data]
    rows.append("| soles through the floor, worst | " + " | ".join(f"{-x:.1f} cm" for x in deep) + " |")
    return "\n".join(rows)


def main(a):
    cfg = json.load(open(ROOT / "web" / "roster.json"))
    blocks = {"ROSTER": roster(cfg), "FEET": feet(cfg)}
    if not a.write:
        for k, v in blocks.items():
            print(f"<!--{k}-->\n{v}\n")
        return
    p = ROOT / "README.md"
    s = p.read_text()
    for k, v in blocks.items():
        pat = re.compile(rf"<!--{k}-->.*?<!--/{k}-->", re.S)
        rep = f"<!--{k}-->\n{v}\n<!--/{k}-->"
        s = pat.sub(rep, s) if pat.search(s) else s.replace(f"<!--{k}-->", rep)
    p.write_text(s)
    print("README tables written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args())
