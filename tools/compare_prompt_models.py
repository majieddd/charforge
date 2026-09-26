"""Which local model writes prompts out best: each writes five prompts out, scored by the four rules
pipeline/describe.py asks for (7+ fields filled; no prop, pose or height; top, bottom and shoes each their own
colour; no bald human) - 20 checks a model. Results: research/data/prompt_models/ (experiment E015).

    python tools/compare_prompt_models.py gemma4:e4b gemma4:e2b qwen2.5:7b     (models pulled in Ollama)
"""
import json, re, sys, time
from pathlib import Path
ARGS = sys.argv[1:]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.argv = ["x"]
import charforge
import describe as D

PROMPTS = ["a boy scout", "Gray Alien That is Extremely Muscular and looks like a gigachad", "a young knight in steel armor",
           "a witch", "a robot chef"]
MODELS = ARGS if ARGS else ["qwen2.5vl:7b", "gemma2:9b-instruct-q5_K_M", "ministral-3:8b", "gemma3:4b", "llama3.2:3b"]
COLOURS = r"(black|white|grey|gray|silver|gold|golden|red|crimson|maroon|orange|yellow|green|olive|khaki|tan|beige|brown|blue|navy|teal|cyan|purple|violet|pink|magenta|cream|ivory|bronze|copper)"
out = {}
with charforge.gpu("comparing prompt models"):
    for m in MODELS:
        rows = []
        for p in PROMPTS:
            t = time.time()
            try:
                f = D._ollama(m, p, "stylized", D.age_of(p))
                raw = " ".join(str(f.get(k, "")) for k in D.FIELDS) + " " + " ".join(map(str, f.get("extras") or []))
                filled = sum(bool(D._clean(f.get(k))) for k in D.FIELDS)
                banned = bool(D.BANNED.search(raw))
                cols = [set(re.findall(COLOURS, str(f.get(k, "")).lower())) for k in ("top", "bottom", "shoes")]
                distinct = len({frozenset(c) for c in cols if c}) == 3
                human = not re.search(r"alien|robot|lizard|creature|monster", p.lower())
                bald_wrong = human and str(f.get("hair", "")).lower().strip() in ("bald", "none", "no hair")
                hair_in_who = "hair" in str(f.get("who", "")).lower()
                rows.append({"prompt": p, "secs": round(time.time() - t, 1), "filled": filled, "banned": banned,
                             "distinct_colours": distinct, "bald_wrong": bald_wrong, "text": D.compose(f), "fields": f})
            except Exception as e:
                rows.append({"prompt": p, "error": str(e)[:200]})
        ok = [r for r in rows if "error" not in r]
        score = sum((r["filled"] >= 7) + (not r["banned"]) + r["distinct_colours"] + (not r["bald_wrong"]) for r in ok)
        out[m] = {"score": score, "of": 4 * len(PROMPTS), "secs": round(sum(r["secs"] for r in ok) / max(len(ok), 1), 1), "rows": rows}
        print(f"{m:34s} score {score}/{4 * len(PROMPTS)}  {out[m]['secs']} s a prompt  errors {len(rows) - len(ok)}", flush=True)
OUT = ROOT / "research" / "data" / "prompt_models"
OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / ("compare_" + "_".join(m.split(":")[0] + m.split(":")[-1] for m in MODELS)[:80] + ".json"), "w"), indent=1)
