"""Quality gates for the character pipeline, run entirely locally.

A generation pipeline that never checks its own output wastes an hour of GPU time on a
character that was framed wrong at step one. Each stage ends with a gate:

  vision  - a local VLM (Ollama qwen2.5vl) describes what is actually in the render
  judge   - Laya turns that description into typed, calibrated answers to the questions
            that decide whether the stage passed

Laya is the right tool here precisely because it does not generate: the questions are
closed, the answers are probabilities, and the thresholds live in code where they can be
tuned per stage. Everything runs on the machine - no data leaves it.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import httpx

OLLAMA = "http://127.0.0.1:11434"
VLM = "qwen2.5vl:7b"

DESCRIBE = (
    "Describe this character image for a 3D artist. State plainly: whether the whole figure is "
    "visible from head to feet or is cropped; the pose (arms position, facing direction); what "
    "clothing is worn; the hair length and style; any accessories; the background; the lighting; "
    "and whether more than one figure is present. Be literal and specific."
)

# Each gate: question -> (Laya question, pass rule)
REFERENCE_QUESTIONS = {
    "full_body": {"type": "noul", "instructions":
        "According to `description`, is the character's entire body visible from head to feet, uncropped?"},
    "single_figure": {"type": "noul", "instructions":
        "According to `description`, is exactly one character shown (not two or more)?"},
    "arms_away_from_body": {"type": "noul", "instructions":
        "According to `description`, are the character's arms held away from the torso (an A-pose or T-pose), "
        "rather than hanging straight down, crossed, or hidden?"},
    "plain_background": {"type": "noul", "instructions":
        "According to `description`, is the background plain and empty (a flat studio backdrop), "
        "rather than a scene with objects or environment?"},
    "front_facing": {"type": "noul", "instructions":
        "According to `description`, is the character facing the camera from the front?"},
    "even_lighting": {"type": "noul", "instructions":
        "According to `description`, is the lighting flat and even, without strong shadows or dramatic contrast?"},
}

PASS_RULES = {"full_body": 0.5, "single_figure": 0.5, "arms_away_from_body": 0.4,
              "plain_background": 0.5, "front_facing": 0.5, "even_lighting": 0.35}


def describe(image_path: str, model=VLM, prompt=DESCRIBE, timeout=600) -> str:
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    r = httpx.post(f"{OLLAMA}/api/generate", timeout=timeout, json={
        "model": model, "prompt": prompt, "images": [b64], "stream": False,
        "options": {"temperature": 0}, "keep_alive": "10m"})
    r.raise_for_status()
    return r.json()["response"].strip()


def spec_questions(spec: dict) -> dict:
    """Questions generated from the character spec itself, so the gate checks what was asked for."""
    q = {}
    for name, thing in spec.items():
        q[f"has_{name}"] = {"type": "noul", "instructions":
            f"According to `description`, does the character have {thing}?"}
    return q


def judge(description: str, questions: dict, agent=None):
    import laya
    agent = agent or laya.load("convaiinnovations/laya", device="mps")
    t0 = time.perf_counter()
    res = agent.predict({"description": description}, questions)
    return res["answers"], (time.perf_counter() - t0) * 1000, agent


def gate_reference(image_path: str, spec: dict | None = None, agent=None, out: str | None = None):
    """Gate for stage 1: is this reference image usable as the basis for a 3D character?"""
    desc = describe(image_path)
    qs = {**REFERENCE_QUESTIONS, **(spec_questions(spec) if spec else {})}
    answers, ms, agent = judge(desc, qs, agent)
    checks = {}
    for k, a in answers.items():
        thr = PASS_RULES.get(k, 0.5)
        checks[k] = {"value": round(a["noul"], 3), "threshold": thr, "pass": a["noul"] >= thr}
    failed = [k for k, v in checks.items() if not v["pass"]]
    payload = {"image": str(image_path), "description": desc, "checks": checks,
               "failed": failed, "passed": not failed, "laya_ms": round(ms, 1)}
    if out:
        json.dump(payload, open(out, "w"), indent=2)
    return payload, agent


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spec = {"jacket": "a jacket", "long_or_medium_hair": "hair that is at least medium length",
            "boots": "boots or shoes"}
    p, _ = gate_reference(a.image, spec, out=a.out)
    print(f"[gate] {'PASS' if p['passed'] else 'FAIL'}  ({p['laya_ms']:.0f} ms of Laya)")
    for k, v in p["checks"].items():
        print(f"   {'ok ' if v['pass'] else 'FAIL'} {k:24s} {v['value']:.3f} (>= {v['threshold']})")
    print("\n--- VLM description ---\n" + p["description"][:900])
