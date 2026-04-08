"""Grading functions — all return float strictly in (0.01, 0.99)."""
from __future__ import annotations
import os
import re
from openai import OpenAI


def clamp_score(score: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return round(max(lo, min(hi, float(score))), 6)


def _client() -> OpenAI:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    api_key  = groq_key if groq_key else hf_token
    base_url = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
    return OpenAI(api_key=api_key or "dummy", base_url=base_url)


def _model() -> str:
    return os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")


# ── easy ──────────────────────────────────────────────────────────────────────
PASS_KW = ["valid", "correct", "200", "success", "ok", "matches", "present"]
FAIL_KW = ["invalid", "error", "mismatch", "wrong", "missing", "null", "empty"]


def grade_easy(action: str, ground_truth: str) -> float:
    text  = action.lower()
    truth = ground_truth.lower()
    score = 0.10

    t_words = set(re.findall(r"\w+", truth))
    a_words = set(re.findall(r"\w+", text))
    if t_words:
        score += (len(t_words & a_words) / len(t_words)) * 0.50

    is_pass = any(k in truth for k in PASS_KW)
    if is_pass and any(k in text for k in PASS_KW):
        score += 0.25
    elif not is_pass and any(k in text for k in FAIL_KW):
        score += 0.25

    return clamp_score(score, lo=0.05, hi=0.88)


# ── medium ────────────────────────────────────────────────────────────────────
STRUCT_KW = ["status", "schema", "field", "type", "required", "response", "null", "empty", "format"]


def grade_medium(action: str, ground_truth: str) -> float:
    text  = action.lower()
    truth = ground_truth.lower()
    score = 0.10

    t_words = set(re.findall(r"\w+", truth))
    a_words = set(re.findall(r"\w+", text))
    score += (len(t_words & a_words) / max(len(t_words), 1)) * 0.40
    score += (sum(1 for k in STRUCT_KW if k in text) / len(STRUCT_KW)) * 0.30

    wc = len(a_words)
    if wc >= 30:
        score += 0.10
    elif wc >= 15:
        score += 0.05

    return clamp_score(score, lo=0.05, hi=0.87)


# ── hard ──────────────────────────────────────────────────────────────────────
HARD_PROMPT = """You are a strict evaluator scoring an AI agent API validation response.

Expected answer:
{ground_truth}

Agent response:
{action}

Output ONLY valid JSON like this: {{"score": 0.72}}
The score must be a float strictly between 0.05 and 0.93.
"""


def _fallback(action: str, ground_truth: str) -> float:
    t = set(re.findall(r"\w+", ground_truth.lower()))
    a = set(re.findall(r"\w+", action.lower()))
    return clamp_score(0.10 + (len(t & a) / max(len(t), 1)) * 0.60, lo=0.05, hi=0.93)


def grade_hard(action: str, ground_truth: str) -> float:
    try:
        resp = _client().chat.completions.create(
            model=_model(),
            messages=[{"role": "user", "content": HARD_PROMPT.format(
                ground_truth=ground_truth, action=action
            )}],
            temperature=0.0,
            max_tokens=64,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw)
        if m:
            return clamp_score(float(m.group(1)), lo=0.05, hi=0.93)
        m2 = re.search(r'\b([0-9]+\.[0-9]+)\b', raw)
        if m2:
            return clamp_score(float(m2.group(1)), lo=0.05, hi=0.93)
    except Exception:
        pass
    return _fallback(action, ground_truth)
