"""Build the living paper: docs/paper/index.html, research/TRACKER.md and paper/references.bib.

    python paper/collect.py      # after an experiment: refresh paper/data/metrics.json from the pipeline's files
    python paper/build.py        # anywhere: assemble the paper from its sources

Sources (all in the repository):
  paper/meta.json                title, version, changelog
  paper/sections/NN-*.html       the prose, in order
  paper/references.json          the bibliography (checked against each source)
  paper/data/metrics.json        numbers measured by the pipeline (paper/collect.py)
  paper/data/baselines.json      numbers measured before a change the files no longer hold
  research/experiments.json      the experiment tracker
  paper/media/                   figures and videos

In the prose:
  {{kick.joints}}       a number from the data (build.py: values()) - a missing name stops the build
  [@key] [@a; @b]       citations, numbered by first use
  [#fig-x] [#tab-x] [#sec-x]   "Figure 3", "Table 2", "Section 4.2", linked
  [E021]                an experiment, linked to its row in Appendix A
  <!--TABLE:name-->     a table generated from the data (build.py: TABLES)

The build refuses to write a page with an unresolved name, citation, reference or missing file.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
OUT = ROOT / "docs" / "paper"
META = json.load(open(P / "meta.json"))
REFS = {r["key"]: r for r in json.load(open(P / "references.json"))["refs"]}
EXP = json.load(open(ROOT / "research" / "experiments.json"))
X = {e["id"]: e for e in EXP["experiments"]}
M = json.load(open(P / "data" / "metrics.json"))
B = json.load(open(P / "data" / "baselines.json"))
SHORT = {"mara/punch_combo": "punch", "mara/roundhouse_kick": "kick", "aoi/spell_cast": "spell", "pip/victory_cheer": "cheer"}
STATUS = {"done": "done", "dropped": "dropped", "open": "open", "planned": "planned", "running": "running"}
esc = html.escape


def fmt(v, nd=None):
    if isinstance(v, float):
        if nd is not None:
            return f"{v:.{nd}f}"
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.1f}".rstrip("0").rstrip(".")
    return str(v)


def values():
    """Every name the prose may use."""
    v = {"version": META["version"], "date": META["date"]}
    for mv in M["moves"]:
        k = SHORT[f"{mv['character']}/{mv['move']}"]
        for f, x in mv.items():
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                v[f"{k}.{f}"] = x
        v[f"{k}.misread_n"] = len(mv.get("misread_frames") or [])
        v[f"{k}.palm_pct"] = round(100 * mv["palm_facing"]) if mv.get("palm_facing") is not None else None
    for c in M["roster"]:
        for f, x in c.items():
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                v[f"{c['id']}.{f}"] = x
        v[f"{c['id']}.lod0"] = c["triangles_lods"][0]
    v["n.characters"] = len(M["roster"])
    v["n.playground"] = sum(c["in_playground"] for c in M["roster"])
    v["n.face"] = sum(bool(c["face"]) for c in M["roster"])
    v["n.experiments_done"] = sum(e["status"] in ("done", "dropped") for e in EXP["experiments"])
    v["n.experiments_planned"] = sum(e["status"] in ("planned", "open", "running") for e in EXP["experiments"])
    v["n.refs"] = len(REFS)
    for k, s in M["stage_times"].items():
        v[f"stage.{k}.min"] = round(s["median_s"] / 60, 1)
        v[f"stage.{k}.s"] = s["median_s"]
    v["stage.total.min"] = round(sum(s["median_s"] for s in M["stage_times"].values()) / 60, 1)
    top4 = sum(M["stage_times"][k]["median_s"] for k in ("multiview", "reference", "generate", "texture") if k in M["stage_times"])
    v["stage.top4.min"] = round(top4 / 60, 1)
    v["feet.median"] = M["feet_summary"]["median_slip_pct"]
    v["feet.max"] = M["feet_summary"]["max_slip_pct"]
    v["feet.n"] = M["feet_summary"]["n"]
    for k, s in M["fit_selftest"].items():
        for f, x in s.items():
            if isinstance(x, (int, float)):
                v[f"st.{k}.{f}"] = x
    for m, s in M["prompt_models"].items():
        v[f"pm.{m}.score"] = s["score"]
        v[f"pm.{m}.secs"] = s["secs"]

    def flat(prefix, d):
        for k, x in d.items():
            if k.startswith("_"):
                continue
            if isinstance(x, dict):
                flat(f"{prefix}.{k}", x)
            elif isinstance(x, list):
                for i, y in enumerate(x):
                    v[f"{prefix}.{k}.{i}"] = y
            else:
                v[f"{prefix}.{k}"] = x
    flat("b", B)
    moves = M["moves"]
    v["moves.iou.min"] = min(m["iou"] for m in moves)
    v["moves.iou.max"] = max(m["iou"] for m in moves)
    v["moves.joints.min"] = min(m["joints"] for m in moves)
    v["moves.joints.max"] = max(m["joints"] for m in moves)
    ious0 = [B["moves_start"][f"{m['character']}/{m['move']}"]["iou"] for m in moves]
    v["moves.iou0.min"], v["moves.iou0.max"] = min(ious0), max(ious0)
    return v


# ---- tables generated from the data --------------------------------------------------------------
def t_moves():
    rows = []
    for m in M["moves"]:
        key = f"{m['character']}/{m['move']}"
        s0, s1 = B["moves_start"][key], B["moves_mid"][key]
        trusted = (f"<br><span class='muted'>{fmt(m['joints_trusted'], 3)} without {len(m['misread_frames'])} misread frames</span>"
                   if m.get("misread_frames") else "")
        rows.append(f"<tr><td>{esc(m['label'])}</td><td class='num'>{fmt(s0['joints'])} · {fmt(s0['iou'])}</td>"
                    f"<td class='num'>{fmt(s1['joints'], 3)} · {fmt(s1['iou'], 3)}</td>"
                    f"<td class='num good'>{fmt(m['joints'], 3)} · {fmt(m['iou'], 3)}{trusted}</td>"
                    f"<td class='num'>{fmt(m['face'], 3)}</td><td class='num'>{fmt(m['hands'], 2)} <span class='muted'>({fmt(m['hands_median'], 2)})</span></td>"
                    f"<td class='num'>{round(100 * m['palm_facing'])}%</td><td class='num'>{fmt(m['worst_elbow_deg'], 0)}°</td></tr>")
    return ("<table><thead><tr><th>move</th><th class='num'>start</th><th class='num'>own body + aimed limbs</th>"
            "<th class='num'>now: joints · IoU</th><th class='num'>face</th><th class='num'>hands mean (median)</th>"
            "<th class='num'>palm facing</th><th class='num'>max elbow</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def t_stages():
    rows = []
    for m in M["moves"]:
        rows.append(f"<tr><td>{esc(m['label'])}</td><td class='num'>{fmt(m['stage_fit'], 3)}</td><td class='num'>{fmt(m['stage_fbx'], 3)}</td>"
                    f"<td class='num'>{fmt(m['stage_rig'], 3)}</td><td class='num'>{fmt(m['stage_render'], 3)}</td>"
                    f"<td class='num'>{len(m.get('misread_frames') or [])}</td></tr>")
    return ("<table><thead><tr><th>move</th><th class='num'>fit</th><th class='num'>FBX</th><th class='num'>rig skeleton</th>"
            "<th class='num'>render (read)</th><th class='num'>misread frames</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def t_roster():
    rows = []
    for c in sorted(M["roster"], key=lambda c: (not c["in_playground"], c["id"])):
        if c["id"] == "juno":          # the first Juno, superseded by juno3 - not a character anyone sees
            continue
        if c["from_image"] or not c.get("prompt"):
            made = "an image"
        else:
            full = c["prompt"]
            short = full if len(full) <= 70 else full[:full.rfind(" ", 0, 68)].rstrip(",") + "…"
            made = f"<i title='{esc(full)}'>“{esc(short)}”</i>"
        face = "jaw + " + str(len(c["face"])) + " shapes" if c["face"] else "<span class='muted'>refused</span>"
        tri = " / ".join(f"{t:,}" for t in c["triangles_lods"])
        extra = []
        if c.get("front_iou"):
            extra.append(f"front {fmt(c['front_iou'])}")
        if c.get("hand_bulk") and c["hand_bulk"] > 1.01:
            extra.append(f"hands ×{fmt(c['hand_bulk'], 2)}")
        rows.append(f"<tr><td><b>{esc(c['name'])}</b>{'' if c['in_playground'] else ' <span class=muted>(local)</span>'}</td><td>{esc(c['style'] or '')}</td>"
                    f"<td class='made'>{made}</td><td class='num'>{fmt(c['height_m'], 2)} m</td><td class='num'>{tri}</td>"
                    f"<td>{face}</td><td class='num'>{c['clips']}</td><td class='muted small'>{', '.join(extra)}</td></tr>")
    return ("<table><thead><tr><th>character</th><th>style</th><th>made from</th><th class='num'>height</th>"
            "<th class='num'>triangles LOD0 / 1 / 2</th><th>face rig</th><th class='num'>clips</th><th>checks</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def t_feet():
    ids = [c["id"] for c in M["roster"] if c["id"] in M["feet"] and c["in_playground"]]
    names = {c["id"]: c["name"] for c in M["roster"]}
    gaits = ["walk", "jog", "run", "sprint", "walk_back", "jog_back", "strafe_left", "strafe_right", "crouch_walk"]
    head = "".join(f"<th class='num'>{esc(names[i])}</th>" for i in ids)
    rows = []
    for g in gaits:
        cells = []
        for i in ids:
            x = M["feet"][i].get(g, {}).get("slip_pct")
            cls = " bad" if x is not None and x > 10 else ""
            cells.append(f"<td class='num{cls}'>{'' if x is None else fmt(x, 1) + '%'}</td>")
        rows.append(f"<tr><td>{g.replace('_', ' ')}</td>{''.join(cells)}</tr>")
    deep = "".join(f"<td class='num'>{fmt(min(v.get('deepest_cm', 0) for v in M['feet'][i].values()), 1)} cm</td>" for i in ids)
    rows.append(f"<tr><td>deepest sole through the floor</td>{deep}</tr>")
    return f"<table><thead><tr><th>gait</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def t_prompts():
    rows = []
    for m, s in M["prompt_models"].items():
        if s["errors"]:
            continue
        rows.append(f"<tr><td class='mono'>{esc(m)}</td><td class='num'>{s['score']} / {s['of']}</td><td class='num'>{fmt(s['secs'], 1)} s</td></tr>")
    failed = [m for m, s in M["prompt_models"].items() if s["errors"]]
    note = (f"<tr><td colspan='3' class='muted small'>No answer (errors on every prompt): {esc(', '.join(failed))}</td></tr>" if failed else "")
    return ("<table><thead><tr><th>local model (Ollama)</th><th class='num'>rules kept</th><th class='num'>time a prompt</th></tr></thead><tbody>"
            + "".join(rows) + note + "</tbody></table>")


def t_selftest():
    s = M["fit_selftest"]
    def row(label, k_own, k_cap):
        a, b_, c = s[k_own]["pose_error_capture"], s[k_cap]["pose_error_fit_gated"], s[k_own]["pose_error_fit_gated"]
        return (f"<tr><td>{label}</td><td class='num'>{fmt(a)}</td><td class='num'>{fmt(b_)}</td><td class='num'>{fmt(c)}</td></tr>")
    return ("<table><thead><tr><th>40 moves each</th><th class='num'>capture as retrieved</th><th class='num'>fit, capture's bone lengths</th>"
            "<th class='num'>fit, the video's own lengths</th></tr></thead><tbody>"
            + row("in the library, the actor's own body", "in_library", "capture_lengths_in_library")
            + row("left out of the library, the actor's own body", "left_out", "capture_lengths_left_out")
            + row("in the library, another body", "body_in_library", "capture_lengths_body_in_library")
            + row("left out of the library, another body", "body_left_out", "capture_lengths_body_left_out")
            + "</tbody></table>")


def t_times():
    st = M["stage_times"]
    order = ["reference", "generate", "multiview", "views", "parts", "skeleton", "solidify", "joints", "hands", "retopo",
             "labels", "texclean", "texture", "weights", "rig", "springs", "frame", "tpose", "face", "animate", "refine", "package", "web"]
    total = sum(st[k]["median_s"] for k in order if k in st)
    rows = []
    for k in order:
        if k not in st:
            continue
        s = st[k]
        share = s["median_s"] / total if total else 0
        bar = f"<span class='bar' style='width:{max(1, round(share * 100 * 2.2))}px'></span>"
        rows.append(f"<tr><td class='mono'>{k}</td><td class='num'>{fmt(s['median_s'] / 60, 1)}</td><td class='num'>{fmt(s['p90_s'] / 60, 1)}</td>"
                    f"<td class='num'>{s['n']}</td><td>{bar} <span class='muted small'>{round(100 * share)}%</span></td></tr>")
    return ("<table><thead><tr><th>stage</th><th class='num'>median min</th><th class='num'>p90 min</th><th class='num'>builds</th>"
            "<th>share of the median total</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def t_comparison():
    # what each system does, from its own paper or page (see references); ours from the code
    cols = ["input", "3D from", "pose handling", "rig", "skin weights", "fingers", "face", "motion", "runs"]
    rows = [
        ("CharForge (ours)", "few words or one image", "TRELLIS.2 [@trellis2]", "reference drawn in A-pose, checked (E028)",
         "joints traced in the solid, Mixamo names", "measured through the body [@geodesicvoxel]", "modelled, 15 bones a hand",
         "jaw + 5 shapes when eyes and mouth are found", "18 captures + moves from generated video", "one laptop, local"),
        ("CharacterGen [@charactergen]", "one image", "own sparse-view reconstruction", "canonicalised to A-pose by multi-view diffusion",
         "AccuRIG [@accurig]", "AccuRIG", "AccuRIG", "-", "retargeted", "GPU; <1 min"),
        ("Make-A-Character [@mach]", "text", "MetaHuman template fit to a generated portrait", "template",
         "MetaHuman", "MetaHuman", "yes", "52 blendshapes", "engine animations", "server; ~2 min"),
        ("Make-A-Character 2 [@mach2]", "one portrait", "template + face reconstruction", "template",
         "template, facial skeleton calibrated", "template", "yes", "blendshapes + facial skeleton", "co-speech (transformer)", "server; <2 min"),
        ("DrawingSpinUp [@drawingspinup]", "one drawing", "image-to-3D, thinned", "front A/T pose assumed",
         "Mixamo auto-rigger (8 keypoints)", "nearest bone", "-", "-", "Mixamo via Rokoko", "GPU"),
        ("Hunyuan3D (Cinevva) [@cinevvatool]", "text, image, up to 4 views", "Hunyuan3D [@hunyuan3d21]", "-",
         "automatic", "automatic", "-", "-", "preset clips", "hosted"),
        ("AccuRIG [@accurig]", "a mesh", "-", "A or T pose", "markers, refined", "professional-style paint", "0-5 fingers", "-", "ActorCore library", "desktop"),
        ("PINOC [@pinoc]", "a mesh", "-", "-", "guide markers, mirrored", "automatic", "yes", "-", "library + retarget", "hosted"),
        ("Rig Lab [@riglab]", "A/T-pose GLB", "-", "-", "automatic joints", "automatic", "no (hand = one joint)", "-", "4 procedural clips", "hosted"),
        ("UniRig [@unirig]", "a mesh", "-", "any", "autoregressive skeleton", "predicted", "yes", "-", "-", "GPU, open weights"),
    ]
    head = "<th>system</th>" + "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='wide'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def t_tracker():
    groups = [("Planned and open", ("running", "planned", "open")), ("Done and dropped", ("done", "dropped"))]
    out = []
    for title, sts in groups:
        rows = []
        for e in EXP["experiments"]:
            if e["status"] not in sts:
                continue
            res = esc(e.get("result") or "") or "<span class='muted'>-</span>"
            det = (f"<details><summary>question, method, metric</summary><p><b>Question.</b> {esc(e['question'])}</p>"
                   f"<p><b>Method.</b> {esc(e['method'])}</p><p><b>Metric.</b> {esc(e['metric'])}</p>"
                   + (f"<p><b>Decision.</b> {esc(e['decision'])}</p>" if e.get("decision") else "")
                   + (f"<p class='mono small'>{esc(', '.join(e['files']))}</p>" if e.get("files") else "") + "</details>")
            rows.append(f"<tr id='{e['id']}'><td class='mono'>{e['id']}</td><td><b>{esc(e['title'])}</b>{det}</td>"
                        f"<td>{esc(e['area'])}</td><td><span class='pill {e['status']}'>{e['status']}</span></td>"
                        f"<td class='small'>{res}</td><td class='mono small'>{e['date']}</td></tr>")
        out.append(f"<h3 class='plain'>{title}</h3><div class='tscroll'><table class='tracker'><thead><tr><th>id</th><th>experiment</th>"
                   f"<th>area</th><th>status</th><th>result</th><th>date</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    return "\n".join(out)


def t_changelog():
    return "".join(f"<div class='cl'><div class='clv'><b>v{c['version']}</b><br><span class='mono small'>{c['date']}</span></div><ul>"
                   + "".join(f"<li>{esc(x)}</li>" for x in c["changes"]) + "</ul></div>" for c in META["changelog"])


def bibtex_key(k):
    return re.sub(r"[^A-Za-z0-9]", "", k)


def t_cite():
    au = " and ".join(a["name"] for a in META["authors"])
    return (f"<pre class='bib'>@misc{{charforge{META['date'][:4]},\n  title  = {{{META['title']}}},\n  author = {{{au}}},\n"
            f"  year   = {{{META['date'][:4]}}},\n  note   = {{Living paper, version {META['version']} ({META['date']})}},\n"
            f"  url    = {{{META['url']}}}\n}}</pre>")


TABLES = {"moves": t_moves, "stages": t_stages, "roster": t_roster, "feet": t_feet, "prompts": t_prompts,
          "selftest": t_selftest, "times": t_times, "comparison": t_comparison, "tracker": t_tracker,
          "changelog": t_changelog, "cite": t_cite}


# ---- assembly -------------------------------------------------------------------------------------
def main():
    V = values()
    body = "\n".join(f.read_text() for f in sorted((P / "sections").glob("*.html")))

    for name, fn in TABLES.items():
        body = body.replace(f"<!--TABLE:{name}-->", fn())

    def sub_value(m):
        k = m.group(1).strip()
        spec = None
        if "|" in k:
            k, spec = [s.strip() for s in k.split("|", 1)]
        if k not in V or V[k] is None:
            raise SystemExit(f"[paper] unknown value {{{{{k}}}}}")
        x = V[k]
        if spec == "pct":
            return f"{round(100 * x)}%"
        if spec and spec.isdigit():
            return fmt(float(x), int(spec))
        if spec == "int":
            return f"{int(round(x)):,}"
        return fmt(x)
    body = re.sub(r"\{\{([^}]+)\}\}", sub_value, body)

    # citations, numbered by first use
    order = []
    def cite(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        links = []
        for k in keys:
            if k not in REFS:
                raise SystemExit(f"[paper] unknown citation @{k}")
            if k not in order:
                order.append(k)
            links.append(f"<a href='#ref-{k}' title='{esc(REFS[k]['title'])}'>{order.index(k) + 1}</a>")
        return "<span class='cite'>[" + ", ".join(links) + "]</span>"
    body = re.sub(r"\[(@[^\]]+)\]", cite, body)

    # section numbers
    sec_num, n2, n3, toc = {}, 0, 0, []
    def number_heading(m):
        nonlocal n2, n3
        level, attrs, text = m.group(1), m.group(2), m.group(3)
        idm = re.search(r'id="([^"]+)"', attrs)
        if "appendix" in attrs:
            label = text
            if idm:
                sec_num[idm.group(1)] = text.split(" ")[0]
            toc.append((level, idm.group(1) if idm else "", label, ""))
            return f"<h{level}{attrs}>{text}</h{level}>"
        if "plain" in attrs or not idm:
            return m.group(0)
        if level == "2":
            n2, n3 = n2 + 1, 0
            num = f"{n2}"
        else:
            n3 += 1
            num = f"{n2}.{n3}"
        sec_num[idm.group(1)] = num
        toc.append((level, idm.group(1), text, num))
        return f"<h{level}{attrs}><span class='secnum'>{num}</span> {text}</h{level}>"
    body = re.sub(r"<h([23])((?:\s[^>]*)?)>(.*?)</h\1>", number_heading, body)

    # figures and tables
    fig_num, tab_num = {}, {}
    def number_figure(m):
        attrs, inner = m.group(1), m.group(2)
        idm = re.search(r'id="((?:fig|tab)-[^"]+)"', attrs)
        if not idm:
            return m.group(0)
        fid = idm.group(1)
        store, word = (tab_num, "Table") if fid.startswith("tab-") else (fig_num, "Figure")
        store[fid] = len(store) + 1
        inner = inner.replace("<figcaption>", f"<figcaption><b>{word} {store[fid]}.</b> ", 1)
        return f"<figure{attrs}>{inner}</figure>"
    body = re.sub(r"<figure((?:\s[^>]*)?)>(.*?)</figure>", number_figure, body, flags=re.S)

    def xref(m):
        k = m.group(1)
        if k in fig_num:
            return f"<a href='#{k}'>Figure {fig_num[k]}</a>"
        if k in tab_num:
            return f"<a href='#{k}'>Table {tab_num[k]}</a>"
        if k in sec_num:
            return f"<a href='#{k}'>Section {sec_num[k]}</a>"
        raise SystemExit(f"[paper] unknown cross-reference [#{k}]")
    body = re.sub(r"\[#([a-z0-9-]+)\]", xref, body)

    def exref(m):
        k = m.group(1)
        if k not in X:
            raise SystemExit(f"[paper] unknown experiment [{k}]")
        return f"<a class='exp' href='#{k}' title='{esc(X[k]['title'])}'>{k}</a>"
    body = re.sub(r"\[(E\d{3})\]", exref, body)
    # experiment ids written inside generated tables stay plain; linked ones inside the tracker are anchors

    refs_html = "".join(
        f"<li id='ref-{k}'><span class='rn'>[{i + 1}]</span> {esc(', '.join(REFS[k]['authors']))}. "
        f"<a href='{esc(REFS[k]['url'])}'>{esc(REFS[k]['title'])}</a>. <i>{esc(REFS[k]['venue'])}</i>, {REFS[k]['year']}."
        f" <span class='role {REFS[k]['role']}'>{REFS[k]['role']}</span> <span class='muted small'>{esc(REFS[k].get('note', ''))}</span></li>"
        for i, k in enumerate(order))
    body = body.replace("<!--REFERENCES-->", f"<ol class='refs'>{refs_html}</ol>")

    toc_html = "".join(
        f"<li class='l{lv}'><a href='#{i}'>{('<span class=secnum>' + n + '</span> ') if n else ''}{re.sub('<[^>]+>', '', t)}</a></li>"
        for lv, i, t, n in toc)

    for bad in (r"\{\{", r"\[@", r"<!--TABLE:"):
        if re.search(bad, body):
            raise SystemExit(f"[paper] unresolved {bad} in the page")

    media = set(re.findall(r'(?:src|poster|href)="(media/[^"]+)"', body))
    missing = [m for m in media if not (P / m).exists()]
    if missing:
        raise SystemExit(f"[paper] missing media: {missing}")

    page = TEMPLATE.format(
        title=esc(META["title"]), subtitle=esc(META["subtitle"]), version=META["version"], date=META["date"],
        authors=", ".join(esc(a["name"]) + (f" <span class='muted'>({esc(a['affiliation'])})</span>" if a.get("affiliation") else "")
                          for a in META["authors"]),
        ack=esc(META.get("acknowledgement", "")), repo=META["repo"], toc=toc_html, body=body,
        nrefs=len(order), nexp=len(EXP["experiments"]), css=CSS)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page)

    # the media the paper and the notebook use, and nothing else
    nb = OUT / "notebook.html"
    used = set(media)
    if nb.exists():
        used |= set(re.findall(r'(?:src|poster)="(media/[^"]+)"', nb.read_text()))
    (OUT / "media").mkdir(exist_ok=True)
    for m in sorted(used):
        src, dst = P / m, OUT / m
        if src.exists() and (not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime or dst.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dst)
    for f in (OUT / "media").iterdir():
        if f"media/{f.name}" not in used:
            f.unlink()

    write_bib(order)
    write_tracker()
    print(f"[paper] v{META['version']}: {len(order)} references cited of {len(REFS)}, {len(fig_num)} figures, {len(tab_num)} tables, "
          f"{len(EXP['experiments'])} experiments, {len(used)} media files -> {OUT.relative_to(ROOT)}/index.html ({len(page) // 1024} KB)")


def write_bib(order):
    lines = []
    for k in list(order) + [k for k in REFS if k not in order]:
        r = REFS[k]
        kind = {"article": "article", "inproceedings": "inproceedings"}.get(r["type"], "misc")
        fields = {"title": r["title"], "author": " and ".join(r["authors"]), "year": str(r["year"]), "url": r["url"],
                  ("journal" if kind == "article" else "booktitle" if kind == "inproceedings" else "howpublished"): r["venue"]}
        lines.append(f"@{kind}{{{bibtex_key(k)},\n" + ",\n".join(f"  {a} = {{{b}}}" for a, b in fields.items()) + "\n}")
    (P / "references.bib").write_text("% Generated by paper/build.py from paper/references.json - edit that file.\n\n" + "\n\n".join(lines) + "\n")


def write_tracker():
    R = EXP["resume"]
    L = ["# CharForge experiment tracker", "",
         "Generated by `paper/build.py` from [`research/experiments.json`](experiments.json) - edit that file, then rebuild. "
         f"Rendered in the paper as [Appendix A]({META['url']}#appendix-tracker).", "",
         "## Resume here", "", f"*Updated {R['updated']}.* {R['state']}", "",
         "**Next:** " + ", ".join(f"[{i}](#{i.lower()}) {X[i]['title']}" for i in R["next"] if i in X), "",
         R["how"], "",
         "## How to add an experiment", "",
         "1. Add an entry to `research/experiments.json` with the next free id (`E1xx` for the current programme), "
         "status `planned`, and its question, method and metric - before running it.",
         "2. Run it; record the measurement in `result` with the file it came from, and the `decision`.",
         "3. If it changes a number the paper quotes, re-run `python paper/collect.py`; then `python paper/build.py`, "
         "add a line to the changelog in `paper/meta.json`, and commit the code, data and paper together.", ""]
    for title, sts in (("Planned, running and open", ("running", "planned", "open")), ("Done and dropped", ("done", "dropped"))):
        L += [f"## {title}", "", "| id | experiment | area | status | result |", "|---|---|---|---|---|"]
        for e in EXP["experiments"]:
            if e["status"] in sts:
                res = (e.get("result") or "").replace("|", "/").replace("\n", " ")
                L.append(f"| [{e['id']}](#{e['id'].lower()}) | {e['title']} | {e['area']} | {e['status']} | {res} |")
        L.append("")
    L += ["## Details", ""]
    for e in EXP["experiments"]:
        L += [f"### {e['id']}", "", f"**{e['title']}** · {e['area']} · {e['status']} · {e['date']}", "",
              f"- **Question.** {e['question']}", f"- **Method.** {e['method']}", f"- **Metric.** {e['metric']}"]
        if e.get("result"):
            L.append(f"- **Result.** {e['result']}")
        if e.get("decision"):
            L.append(f"- **Decision.** {e['decision']}")
        if e.get("files"):
            L.append("- **Files.** " + ", ".join(f"`{f}`" for f in e["files"]))
        L.append("")
    (ROOT / "research" / "TRACKER.md").write_text("\n".join(L))


CSS = r"""
:root{
  --bg:#FBFBFC; --paper:#FFFFFF; --ink:#1A1F26; --ink2:#46505E; --muted:#6C7685; --line:#E1E5EB; --soft:#F3F5F8;
  --accent:#1F5BB8; --accent2:#0E7A5C; --bad:#B4382F; --warn:#9A6A0A; --bar:#9DB6DD;
  --serif:"Source Serif 4","Source Serif Pro",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  color-scheme:light;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0F1216; --paper:#151A20; --ink:#E6EAF0; --ink2:#B3BCC8; --muted:#8A95A5; --line:#28303A; --soft:#1B2129;
  --accent:#7FB0FF; --accent2:#5CCBA4; --bad:#F0877E; --warn:#E0B356; --bar:#3E5F92; color-scheme:dark;}}
:root[data-theme="dark"]{
  --bg:#0F1216; --paper:#151A20; --ink:#E6EAF0; --ink2:#B3BCC8; --muted:#8A95A5; --line:#28303A; --soft:#1B2129;
  --accent:#7FB0FF; --accent2:#5CCBA4; --bad:#F0877E; --warn:#E0B356; --bar:#3E5F92; color-scheme:dark;}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:17.5px/1.68 var(--serif);-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
.layout{display:grid;grid-template-columns:minmax(0,1fr);max-width:1180px;margin:0 auto;padding:0 20px 80px;gap:40px}
@media (min-width:1100px){.layout{grid-template-columns:230px minmax(0,1fr)}}
nav.toc{display:none;font:13px/1.45 var(--sans)}
@media (min-width:1100px){nav.toc{display:block;position:sticky;top:18px;align-self:start;max-height:calc(100vh - 36px);overflow:auto;padding-top:28px}}
nav.toc ol{list-style:none;margin:0;padding:0}
nav.toc li{margin:2px 0}
nav.toc li.l3{padding-left:14px;font-size:12.5px}
nav.toc a{color:var(--ink2);text-decoration:none}
nav.toc a:hover{color:var(--accent)}
nav.toc .tt{font:600 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
main{min-width:0;max-width:1000px}
main > section > p, main > section > ul, main > section > ol, .subtitle{max-width:70ch}
header.top{padding:34px 0 8px;border-bottom:1px solid var(--line);margin-bottom:8px}
.status{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;font:13px var(--sans);color:var(--ink2);margin-bottom:18px}
.status .live{display:inline-flex;gap:6px;align-items:center;font:600 11.5px var(--mono);letter-spacing:.06em;text-transform:uppercase;
  color:var(--accent2);border:1px solid color-mix(in srgb,var(--accent2) 45%,transparent);border-radius:999px;padding:3px 9px}
.status .live i{width:7px;height:7px;border-radius:50%;background:var(--accent2)}
h1{font:700 clamp(28px,4vw,40px)/1.15 var(--sans);letter-spacing:-.015em;margin:0 0 10px;text-wrap:balance}
.subtitle{font:italic 19px/1.45 var(--serif);color:var(--ink2);margin:0 0 14px;max-width:62ch}
.byline{font:14px/1.5 var(--sans);color:var(--ink2)}
.links{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 4px;font:13.5px var(--sans)}
.links a{border:1px solid var(--line);background:var(--paper);border-radius:7px;padding:6px 11px;text-decoration:none;color:var(--ink)}
.links a:hover{border-color:var(--accent);color:var(--accent)}
h2{font:700 25px/1.25 var(--sans);letter-spacing:-.01em;margin:52px 0 12px;scroll-margin-top:16px;text-wrap:balance}
h3{font:600 19px/1.3 var(--sans);margin:30px 0 8px;scroll-margin-top:16px;text-wrap:balance}
h3.plain{font-size:16px;margin-top:22px}
.secnum{font-family:var(--mono);font-weight:500;color:var(--muted);margin-right:4px}
p{margin:0 0 14px;max-width:70ch;hyphens:auto}
.abstract{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:18px 22px;margin:22px 0 8px}
.abstract .lbl{font:600 11.5px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.abstract p:last-child{margin-bottom:0}
ul,ol{max-width:68ch;padding-left:22px;margin:0 0 14px}
li{margin:3px 0}
.cite{font-family:var(--sans);font-size:.82em;white-space:nowrap}
.cite a{text-decoration:none}
a.exp{font:500 .8em var(--mono);text-decoration:none;border:1px solid var(--line);border-radius:4px;padding:0 4px;background:var(--soft)}
figure{margin:24px 0;max-width:100%}
figure.wide{margin-left:0;margin-right:0}
figure img,figure video{display:block;width:100%;height:auto;border-radius:8px;border:1px solid var(--line);background:var(--soft)}
figcaption{font:14px/1.5 var(--sans);color:var(--ink2);margin-top:8px;max-width:72ch}
figcaption b{color:var(--ink)}
.tscroll{overflow-x:auto;margin:0;border:1px solid var(--line);border-radius:8px;background:var(--paper)}
table{border-collapse:collapse;width:100%;font:13.5px/1.45 var(--sans);min-width:560px}
table.wide{min-width:980px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font:600 11px/1.3 var(--mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted);background:var(--soft)}
tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.made{min-width:220px;max-width:320px;font-size:12.5px}
.good{color:var(--accent2)} .bad{color:var(--bad)} .muted{color:var(--muted)} .small{font-size:12.5px}
.mono{font-family:var(--mono)}
.bar{display:inline-block;height:9px;background:var(--bar);border-radius:2px;vertical-align:middle}
.pill{display:inline-block;font:600 10.5px/1 var(--mono);letter-spacing:.04em;text-transform:uppercase;padding:4px 6px;border-radius:4px;white-space:nowrap}
.pill.done{background:color-mix(in srgb,var(--accent2) 16%,transparent);color:var(--accent2)}
.pill.dropped{background:color-mix(in srgb,var(--bad) 14%,transparent);color:var(--bad)}
.pill.open{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.pill.planned,.pill.running{background:color-mix(in srgb,var(--accent) 14%,transparent);color:var(--accent)}
table.tracker td{font-size:13px}
table.tracker td:last-child,table.tracker td:first-child{white-space:nowrap}
table.tracker details{margin-top:4px;font-size:12.5px;color:var(--ink2)}
table.tracker details summary{cursor:pointer;color:var(--muted)}
table.tracker details p{margin:6px 0;max-width:none}
.callout{border-left:3px solid var(--accent);background:var(--paper);border-radius:0 8px 8px 0;padding:12px 16px;margin:18px 0;font-size:16px}
.callout.open{border-color:var(--warn)}
.callout b.lbl{font:600 11.5px var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--muted);display:block;margin-bottom:2px}
.pipeline{display:grid;gap:14px;margin:18px 0}
.lane{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.lane h4{margin:0 0 10px;font:600 15px var(--sans)}
.lane ol{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;max-width:none}
.lane li{background:var(--soft);border:1px solid var(--line);border-radius:7px;padding:8px 10px;font:12.5px/1.4 var(--sans);color:var(--ink2);margin:0}
.lane li b{display:block;font-size:13.5px;color:var(--ink);margin-bottom:2px}
.lane li code{font:11.5px var(--mono);color:var(--muted)}
ol.refs{list-style:none;padding:0;max-width:none;font:14px/1.5 var(--sans)}
ol.refs li{margin:0 0 10px;padding-left:40px;text-indent:-40px}
ol.refs .rn{display:inline-block;width:36px;text-indent:0;font-family:var(--mono);color:var(--muted)}
.role{font:600 10px var(--mono);text-transform:uppercase;letter-spacing:.05em;padding:2px 5px;border-radius:3px;background:var(--soft);color:var(--muted)}
.role.used{color:var(--accent2)} .role.compared{color:var(--accent)}
.cl{display:grid;grid-template-columns:90px minmax(0,1fr);gap:12px;border-top:1px solid var(--line);padding:12px 0;font:14.5px/1.5 var(--sans)}
.cl ul{margin:0;padding-left:18px;max-width:none}
pre.bib{font:12.5px/1.5 var(--mono);background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow-x:auto}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}
.two figure{margin:0}
footer{margin-top:60px;padding-top:16px;border-top:1px solid var(--line);font:13px/1.5 var(--sans);color:var(--muted)}
@media (max-width:640px){body{font-size:16.5px}h2{font-size:22px}.abstract{padding:14px 16px}}
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>CharForge — a living research paper</title>
<meta name="description" content="{subtitle}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>{css}</style>
</head>
<body>
<div class="layout">
<nav class="toc" aria-label="Contents"><div class="tt">Contents</div><ol>{toc}</ol></nav>
<main>
<header class="top">
  <div class="status"><span class="live"><i></i>living paper</span><span>version {version}</span><span>updated {date}</span>
    <a href="#appendix-changelog">changelog</a><span>{nexp} experiments tracked</span></div>
  <h1>{title}</h1>
  <p class="subtitle">{subtitle}</p>
  <div class="byline">{authors}</div>
  <div class="links"><a href="{repo}">Code</a><a href="../">Playground</a><a href="notebook.html">Lab notebook</a>
    <a href="{repo}/blob/main/research/TRACKER.md">Experiment tracker</a><a href="{repo}/blob/main/paper/references.bib">BibTeX</a>
    <a href="#appendix-cite">How to cite</a></div>
</header>
{body}
<footer>
  <div>{ack}</div>
  <div>Built by <code>paper/build.py</code> from the repository; every table is regenerated from the measurement files
  (<code>paper/collect.py</code>). Version {version}, {date}. {nrefs} references.</div>
</footer>
</main>
</div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
