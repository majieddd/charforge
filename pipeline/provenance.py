"""PROVENANCE.json for a built character: which models and data made it, and what each one's terms say (E153).

OpenMeshy ships a record of the licences behind every output; CharForge's terms are scattered through the README and the paper,
and a package leaves the repository. This writes the same record beside the package. Only terms the project has
already verified are stated, each with where and when; every other component a build used is listed as not assessed, and the
summary says what that leaves open. It describes the build, it is not legal advice, and "unknown permissions remain unknown".

    python pipeline/provenance.py --work work/<name> --out out/<name> --name <name>
"""
import argparse
import datetime
import json
from pathlib import Path

TERMS = {
    "trellis2": {
        "name": "TRELLIS.2 (Microsoft)", "role": "3D generation of the character's shape and texture",
        "terms": "The project's official page says its materials are solely for academic/research purposes and not intended for "
                 "commercial exploitation. This is distinct from the code repository's licence and does not by itself settle "
                 "rights in generated output.",
        "source": "https://microsoft.github.io/TRELLIS.2/", "checked": "2026-09-26"},
    "qwen21": {
        "name": "Qwen-Image 2.1", "role": "the reference picture the character was made from",
        "terms": "Qwen Research License: non-commercial research and evaluation only; commercial use needs a separate licence from Qwen.",
        "source": "https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE", "checked": "2026-09-26"},
    "krea2": {
        "name": "Krea 2", "role": "the reference picture the character was made from",
        "terms": "Krea 2 Community License Agreement: commercial use without an enterprise licence for organisations whose total "
                 "company-wide annual revenue is below US$1,000,000 (trailing twelve months); you own the outputs you generate, subject to the "
                 "agreement; a distributor of the model must keep a licence notice and begin the model's name with \"Krea\"; content-filter, "
                 "export-control and no-circumvention conditions apply.",
        "source": "https://www.krea.ai/krea-2-licensing", "checked": "2026-10-07"},
    "hunyuan": {
        "name": "Hunyuan3D 2.1 (Tencent)", "role": "side and back views painted onto the model, and/or the head's shape (opt-in)",
        "terms": "Distributed under Tencent's Hunyuan3D 2.1 Community License, whose territory excludes the EU, the UK and South Korea, "
                 "with commercial limits; read the licence file shipped with the model before any use. "
                 "(Stated here from the project's own notes and a second pipeline's README, 2026-10-07; the licence text was not re-read for this file.)",
        "source": "model repository's LICENSE file", "checked": "2026-10-07"},
    "mixamo": {
        "name": "Mixamo motion captures (Adobe), via the pinned jasongzy/Mixamo mirror", "role": "the animation clips",
        "terms": "Adobe's FAQ says Mixamo characters and animations may be used royalty-free in personal, commercial and non-profit "
                 "projects, including video games. That does not by itself settle redistribution of the mirror or of the raw clips, "
                 "and the mirror separately requires accepting access conditions.",
        "source": "https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html", "checked": "2026-09-26"},
    "unimate": {
        "name": "UniMate", "role": "moves made from words",
        "terms": "Code and weights are MIT, with google/flan-t5-base (Apache-2.0) as its text encoder. Its training set, UniML3D, keeps each "
                 "source's own terms (Mixamo, Objaverse-XL and the commercial Truebones ZOO pack); generated motion is not thereby cleared of them.",
        "source": "https://github.com/Friedrich-M/UniMate", "checked": "2026-09-26"},
    "h3": {
        "name": "MiniMax H3 (Ref2VA community fine-tune)", "role": "the video a move from video was read from",
        "terms": "H3's Community License defines an applicable territory that excludes the United States, the EU, the UK and South Korea, "
                 "and restricts using or distributing H3 outputs outside it; commercial products must display \"MiniMax H3\", and products above "
                 "$20M yearly revenue need written authorisation. The community fine-tune used is separately labelled Apache-2.0 while the "
                 "upstream agreement defines H3 Works to include derivatives: how these interact is unresolved.",
        "source": "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE", "checked": "2026-09-30"},
    "mia": {
        "name": "Make-It-Animatable v2 (Guo et al., CVPR 2025)", "role": "the skin weights (opt-in, experimental)",
        "terms": "Code MIT; the model card on Hugging Face says Apache-2.0 for the weights. The v2 models are built on the Hunyuan3D-2.1 "
                 "ShapeVAE (Tencent's community licence, whose territory excludes the EU, the UK and South Korea; the licence text was not "
                 "re-read for this file) and were trained on Mixamo characters, whose own terms this record does not settle.",
        "source": "https://huggingface.co/jasongzy/Make-It-Animatable", "checked": "2026-10-07"},
}
NOT_ASSESSED = [
    "DWPose and ViTPose (pose and face-point reading), DINOv3 (identity features and the generator's image encoder), the "
    "face parser and the background remover - used to measure, segment and align, and in TRELLIS.2's own pipeline",
    "the language model that wrote the character's description from a prompt (a local Ollama model or the Jev API)",
    "Blender and its add-ons' terms for the exported FBX/glTF",
]


def build(work: Path, name: str, out: Path) -> dict:
    sj = work / "style.json"
    style = json.load(open(sj)) if sj.exists() else {}
    used = ["trellis2"]
    not_assessed = list(NOT_ASSESSED)
    evidence = {"trellis2": "every build generates its shape and texture with it"}
    if style.get("prompt") or not (work / "reference.png").exists():
        im = style.get("image_model") or "qwen21"
        if im == "qwen21":
            used.append("qwen21")
            evidence["qwen21"] = ("recorded image model" if style.get("image_model") else
                                  "the default reference image model; not recorded for this character, so assumed")
        elif im == "krea2":
            used.append("krea2")
            evidence["krea2"] = "recorded image model"
        else:
            not_assessed.append(f"the reference image model '{im}'")
    else:
        evidence["reference"] = "a picture supplied by the user: its rights are the user's to clear"
    if style.get("texture_views") == "hunyuan" or style.get("head") == "hunyuan":
        used.append("hunyuan")
        evidence["hunyuan"] = ", ".join(f"{k}={style[k]}" for k in ("texture_views", "head") if style.get(k) == "hunyuan")
    clips = json.load(open(work / "clips.json")) if (work / "clips.json").exists() else {}
    if any("unimate" not in str(v.get("file")) and "from_video" not in v for v in clips.values()):
        used.append("mixamo")
        evidence["mixamo"] = "the default clips"
    if any("unimate" in str(v.get("file")) for v in clips.values()):
        used.append("unimate")
        evidence["unimate"] = "clips: " + ", ".join(k for k, v in clips.items() if "unimate" in str(v.get("file")))
    if any("from_video" in v for v in clips.values()):
        used += [k for k in ("h3", "mixamo") if k not in used]
        evidence["h3"] = "clips: " + ", ".join(k for k, v in clips.items() if "from_video" in v) + " (a generated video, matched to a Mixamo capture)"
    wc = work / "weights_choice.json"
    if wc.exists() and json.load(open(wc)).get("picked") == "mia":
        used.append("mia")
        evidence["mia"] = "--skin-weights mia"
    comps = [{**TERMS[k], "evidence": evidence.get(k, "")} for k in used]
    return {
        "character": name, "written": datetime.date.today().isoformat(),
        "code": "CharForge's code is MIT (LICENSE in the repository).",
        "summary": "Nothing here clears this character for commercial use. It was made with components whose own terms are research-only "
                   "or restricted (listed below); use that goes beyond research and evaluation needs each to be read and cleared, and "
                   "the components not assessed to be checked too.",
        "components": comps, "not_assessed": not_assessed, "user_supplied": evidence.get("reference"),
        "note": "A record of what this build used and what its terms say, from sources the project has verified; not legal advice.",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    rec = build(Path(a.work), a.name, Path(a.out))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(Path(a.out) / "PROVENANCE.json", "w"), indent=1)
    print(f"[provenance] {a.name}: {len(rec['components'])} components with terms, {len(rec['not_assessed'])} not assessed", flush=True)
