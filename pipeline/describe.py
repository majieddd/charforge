"""A short prompt written out as a full character description for the reference image - on this Mac.

    python pipeline/describe.py --prompt "a boy scout" [--style stylized] [--json out.json]

What a one-line prompt leaves open, the image model fills in however it likes: "a boy scout" came back
a grown man at the Studio's default adult height, "a gray alien that is extremely muscular" as a grey
clay sculpture with no clothes and no colour (and TRELLIS built a board into it). A character needs the
same things settled every time: an age and a build (the proportions, and the height the rig is scaled
to), skin and hair, and each garment named with its own colour - TRELLIS separates what differs in
colour, and the rig's armhole and hem rules tell a vest from a sleeve and a jacket from the trousers
by colour.

A local language model (Ollama; unloaded as soon as it has answered - it would sit on memory the GPU
stages need) fills those in as short fields, keeping every detail given, and the sentence is put
together here. Asked for free prose under a list of rules, 3B and 7B models broke them one prompt in
two - a scout "holding a wooden badge", a knight with "a small shield", "standing at 1.5 meters tall" -
so a field that names a prop, a pose or a height is dropped. Without a model, a fixed template asks
the image model itself for the same things.

Prints and returns {"description", "age", "height_m", "engine", "fields"}.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

OLLAMA = os.environ.get("CF_OLLAMA", "http://127.0.0.1:11434")
# the first installed of these. Compared on five prompts against the rules below (charforge_notes/prompts):
# gemma4 e4b 19/20 at 9 s a prompt (thinking switched off - with it on, Gemma 4 answered nothing), gemma2
# 9B 20/20 at 15 s, gemma4 e2b 19/20 at 5 s, gemma3 4B 18/20 at 8 s, qwen2.5vl 7B 16/20 (a garment colour
# repeated three times in five), qwen2.5 7B 15/20; ministral 8B and mistral-nemo 12B timed out
DEFAULT_MODELS = "gemma4:e4b,gemma2:9b-instruct-q5_K_M,gemma2:9b,gemma4:e2b,gemma3:4b,qwen2.5vl:7b,llama3.2:3b"
SETTINGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json")


def config():
    """(models in order of preference, OpenAI-API server or ""): the environment (CF_DESCRIBE_MODEL,
    CF_DESCRIBE_URL), else charforge/settings.json (the Studio's Setup page writes it), else Ollama with
    the defaults. Any server with the OpenAI chat API works too - LM Studio http://127.0.0.1:1234/v1,
    mlx_lm.server, llama.cpp - with the model id it lists. Read at each call: a change needs no restart."""
    try:
        st = json.load(open(SETTINGS))
    except (OSError, ValueError):
        st = {}
    models = os.environ.get("CF_DESCRIBE_MODEL") or st.get("describe_model") or DEFAULT_MODELS
    url = os.environ.get("CF_DESCRIBE_URL") or st.get("describe_url") or ""
    return [m.strip() for m in models.split(",") if m.strip()], url.rstrip("/")


MODELS, OPENAI_URL = config()
HEIGHTS = {"child": 1.35, "teen": 1.62, "adult": 1.75, "elder": 1.70}
FIELDS = ("who", "build", "skin", "hair", "face", "top", "bottom", "shoes")
# what cannot be modelled from one picture, or is not the character's to say
BANNED = re.compile(r"\b(hold(s|ing)?|carr(y|ies|ying)|wield(s|ing)?|grip(s|ping)?|shield|sword|axe|spear|staff|"
                    r"wand|bow|gun|rifle|pistol|weapon|backpack|rucksack|bag|satchel|cape|cloak|wings?|tail|"
                    r"stance|standing|stands|pose[sd]?|posing|a-pose|t-pose|meters?|metres?|cm|feet tall|"
                    r"background|lighting|camera|render(ed)?|statue|sculpt(ure)?|clay)\b", re.I)

SYSTEM = """You complete the description of ONE game character from a user's short idea. Fill in the JSON \
fields; keep every detail the user gave, in their meaning, and invent only what is missing.

Fields (each a short phrase, no full sentences):
- who: the character in a few words with the user's own words - species or kind, age (a child 7-12 \
for boy, girl, kid, little, scout; a teenager 13-17), role.
- age: one of child, teen, adult, elder.
- build: body type and proportions.
- skin: the skin's colour and texture (for a creature: its skin, fur or scales). Never grey stone or clay.
- hair: style and colour. People have hair unless the user says bald; only a creature that has none \
(an alien, a lizard, a robot) is "bald".
- face: one or two notable features.
- top, bottom, shoes: each garment WITH a concrete colour; the top, the bottom and the shoes each a \
DIFFERENT colour (never all black), and different from the skin. The character is dressed unless the user says otherwise - a \
creature can wear armour, a harness, a sleeveless top, shorts.
- extras: at most two close-fitting accessories with colours (a belt, a badge, a neckerchief, gloves, \
glasses, a hat that leaves the face clear). Nothing held in the hands, no bags, capes, cloaks, wings, \
tails, weapons or shields.
- height_m: standing height in metres (child 1.2-1.45, teen 1.5-1.75, adult 1.55-1.95, a giant or \
hulking creature up to 2.2).

Never mention a pose, a stance, the camera, the background, lighting or an art style.

Examples:
idea: a pirate
{"who": "a seasoned adult pirate captain", "age": "adult", "build": "wiry and weathered", "skin": "tanned olive skin", "hair": "long black hair tied back", "face": "a scar across the left cheek and a short beard", "top": "a faded red long-sleeved shirt under a brown leather waistcoat", "bottom": "dark blue striped trousers", "shoes": "black knee-high boots", "extras": ["a wide brown leather belt with a brass buckle"], "height_m": 1.78}
idea: a lizard warrior
{"who": "an adult lizardfolk warrior", "age": "adult", "build": "tall and lean with a long neck", "skin": "green scales with a pale yellow belly", "hair": "bald", "face": "amber slit-pupil eyes and a short snout", "top": "a sleeveless bronze scale-armour vest", "bottom": "a dark brown leather kilt over grey leggings", "shoes": "bare clawed feet with brown leather wraps", "extras": ["a red cloth sash"], "height_m": 1.9}
idea: little girl astronaut
{"who": "a seven-year-old girl astronaut", "age": "child", "build": "small child proportions with a round face", "skin": "light brown skin", "hair": "two short black puffs", "face": "big brown eyes and freckles", "top": "a white padded space-suit top with orange panels", "bottom": "white padded space-suit trousers", "shoes": "chunky grey moon boots", "extras": ["a small blue mission patch", "white gloves"], "height_m": 1.22}

Answer with the JSON object only."""


_THINKS = {}


def _thinks(model: str) -> bool:
    """Whether a model thinks before it answers (Ollama's capabilities). Such a model - Gemma 4, ministral 3 -
    spent the answer's token budget thinking and came back empty or cut off; it is asked not to. A model
    that cannot think answers nothing at all when told not to (llama3.2), so it is not told."""
    if model not in _THINKS:
        try:
            req = urllib.request.Request(f"{OLLAMA}/api/show", data=json.dumps({"model": model}).encode(),
                                         headers={"Content-Type": "application/json"})
            _THINKS[model] = "thinking" in (json.loads(urllib.request.urlopen(req, timeout=10).read()).get("capabilities") or [])
        except Exception:                                # noqa: BLE001
            _THINKS[model] = False
    return _THINKS[model]


def _ollama(model: str, idea: str, style: str, age: str, timeout: float = 240.0) -> dict:
    words = {"child": "a child of 7-12", "teen": "a teenager of 13-17", "elder": "an older adult", "adult": "an adult"}
    req = {"model": model, "system": SYSTEM,
           "prompt": f"idea: {idea}\n(the character is {words[age]}; the game's art style: {style})",
           "format": "json", "stream": False, "keep_alive": 0,
           "options": {"temperature": 0.3, "num_predict": 500}}
    if _thinks(model):
        req["think"] = False
    body = json.dumps(req).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body, headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return json.loads(out["response"])


def _openai(model: str, idea: str, style: str, age: str, timeout: float = 180.0, url: str = "") -> dict:
    """The same request to a server with the OpenAI chat API; the JSON is read from the reply's text."""
    words = {"child": "a child of 7-12", "teen": "a teenager of 13-17", "elder": "an older adult", "adult": "an adult"}
    body = json.dumps({"model": model, "temperature": 0.3, "max_tokens": 500, "stream": False,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": f"idea: {idea}\n(the character is {words[age]}; "
                                                                f"the game's art style: {style})"}]}).encode()
    req = urllib.request.Request(f"{url or OPENAI_URL}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    text = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["choices"][0]["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)          # reasoning models think aloud first
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in the reply: {text[:200]}")
    return json.loads(m.group(0))


def _installed(url: str | None = None) -> list[str]:
    url = OPENAI_URL if url is None else url
    if url:
        try:
            ms = json.loads(urllib.request.urlopen(f"{url}/models", timeout=3).read())
            return [m["id"] for m in ms.get("data", [])]
        except Exception:                                # noqa: BLE001
            return []
    try:
        tags = json.loads(urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3).read())
        return [m["name"] for m in tags.get("models", [])]
    except Exception:                                    # noqa: BLE001  (no Ollama: the template)
        return []


def age_of(text: str) -> str:
    t = f" {text.lower()} "
    m = re.search(r"\b(\d{1,2})[- ]year[- ]old\b", t)
    if m:
        n = int(m.group(1))
        return "child" if n < 13 else "teen" if n < 18 else "adult"
    if re.search(r"\b(boy|girl|kid|kids|child|children|little|toddler|schoolboy|schoolgirl|scout)\b", t):
        return "child"
    if re.search(r"\b(teen|teenage|teenager|adolescent|high[- ]school)\b", t):
        return "teen"
    if re.search(r"\b(old|elderly|grandpa|grandma|grandfather|grandmother|senior)\b", t):
        return "elder"
    return "adult"


def _clean(v) -> str:
    v = " ".join(str(v or "").split()).strip(" .,;")
    if v.lower() in ("none", "no hair", "none hair", "n/a"):
        return "bald"
    return "" if not v or BANNED.search(v) else v


def compose(f: dict) -> str:
    """The sentence the image model gets, from the fields: who, build, skin, hair, face, then the clothes."""
    who = _clean(f.get("who")) or "a character"
    if not re.match(r"^(a|an|the)\b", who, re.I):
        who = ("an " if who[:1].lower() in "aeiou" else "a ") + who
    parts = [who]
    for k in ("build", "skin", "hair", "face"):
        v = _clean(f.get(k))
        if v:
            parts.append(v if k != "hair" or "hair" in v.lower() or v.lower() == "bald" else f"{v} hair")
    wear = []
    for v in (_clean(f.get(k)) for k in ("top", "bottom", "shoes")):
        if v and v.lower() not in (w.lower() for w in wear):          # a 7B model gave the witch boots twice
            wear.append(v)
    extras = [v for v in (_clean(x) for x in (f.get("extras") or [])[:2]) if v and v.lower() not in
              (w.lower() for w in wear)]
    s = ", ".join(parts)
    if wear:
        s += ", wearing " + (", ".join(wear[:-1]) + " and " + wear[-1] if len(wear) > 1 else wear[0])
    if extras:
        s += ", with " + " and ".join(extras)
    if len(wear) < 3:                                                   # the rest of the outfit, left to the image model
        s += ", in a complete outfit - a top, trousers or a skirt, and shoes - each in its own colour"
    return s


def template(prompt: str) -> str:
    """No language model: ask the image model itself to settle what the prompt leaves open."""
    age = age_of(prompt)
    who = {"child": "a child of about ten with child proportions", "teen": "a teenager",
           "elder": "an older person", "adult": ""}[age]
    base = prompt.strip().rstrip(".")
    lead = f"{base}, {who}" if who else base
    return (f"{lead}, fully dressed in a complete outfit - a top, trousers or shorts or a skirt, and shoes - "
            "each garment in its own clearly different colour, with natural skin and hair colours, painted in "
            "full colour, nothing held in the hands")


def describe(prompt: str, style: str = "realistic", height: float | None = None) -> dict:
    """The description for the reference image, its age group and a height (the one given wins)."""
    prompt = " ".join(prompt.split())
    long_enough = len(prompt.split()) >= 20              # already written out (the roster's run 20-40 words): used as it is
    result = None
    if not long_enough:
        models, url = config()
        have = _installed(url)
        for model in models:
            if model not in have:
                continue
            try:
                f = (_openai(model, prompt, style, age_of(prompt), url=url) if url
                     else _ollama(model, prompt, style, age_of(prompt)))
                if sum(bool(_clean(f.get(k))) for k in FIELDS) < 6:
                    raise ValueError(f"too few fields: {f}")
                # the prompt's own words decide the age group, whatever the model says: asked to fill in
                # "a witch" it made her a child, and "a young knight" a child too
                age = age_of(prompt)
                result = {"description": compose(f), "age": age, "height_m": f.get("height_m"),
                          "engine": f"{'openai-api' if url else 'ollama'} {model}", "fields": f}
                break
            except Exception as e:                       # noqa: BLE001  (a model that fails: the next, then the template)
                print(f"[describe] {model}: {e}", file=sys.stderr, flush=True)
    if result is None:
        result = {"description": prompt if long_enough else template(prompt), "age": age_of(prompt),
                  "height_m": None, "engine": "as given" if long_enough else "template", "fields": None}
    # the height: given, else the model's within the age group's range, else the age group's
    lo, hi = {"child": (1.1, 1.5), "teen": (1.45, 1.8), "elder": (1.5, 1.9), "adult": (1.5, 2.2)}[result["age"]]
    try:
        h = float(result["height_m"])
    except (TypeError, ValueError):
        h = HEIGHTS[result["age"]]
    result["height_m"] = round(float(height) if height else min(max(h, lo), hi), 2)
    result["prompt"] = prompt
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--style", default="realistic")
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = describe(a.prompt, a.style, a.height)
    print(f"[describe] {r['engine']}: {r['age']}, {r['height_m']:.2f} m - {r['description']}", flush=True)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
