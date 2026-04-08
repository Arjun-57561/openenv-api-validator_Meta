"""
Grading functions for easy / medium / hard tasks.
All grade_* functions return float strictly in open interval (0, 1).
Uses Groq API (llama-3.1-8b-instant) with HF_TOKEN fallback.
"""
import os
import re
from openai import OpenAI


# ── helpers ───────────────────────────────────────────────────────────────────

def clamp_score(score: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, float(score)))


def _client() -> OpenAI:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    api_key  = groq_key if groq_key else hf_token
    base_url = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _model() -> str:
    return os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")


# ── easy grader (rule-based, max 0.88) ───────────────────────────────────────

EASY_PASS_KW = ["valid", "correct", "200", "success", "ok", "matches", "schema"]
EASY_FAIL_KW = ["invalid", "error", "mismatch", "wrong", "missing", "null", "undefined"]


def grade_easy(action_content: str, ground_truth: str) -> float:
    text  = action_content.lower()
    truth = ground_truth.lower()

    score = 0.10  # base — never starts at 0

    truth_words  = set(re.findall(r"\w+", truth))
    action_words = set(re.findall(r"\w+", text))
    if truth_words:
        score += (len(truth_words & action_words) / len(truth_words)) * 0.50

    is_pass = any(k in truth for k in EASY_PASS_KW)
    is_fail = any(k in truth for k in EASY_FAIL_KW)

    if is_pass and any(k in text for k in EASY_PASS_KW):
        score += 0.20
    elif is_fail and any(k in text for k in EASY_FAIL_KW):
        score += 0.20
    else:
        score -= 0.05

    return clamp_score(score, lo=0.05, hi=0.88)


# ── medium grader (hybrid, max 0.87) ─────────────────────────────────────────

MEDIUM_STRUCT_KW = ["status", "schema", "field", "type", "required", "response"]


def grade_medium(action_content: str, ground_truth: str) -> float:
    text  = action_content.lower()
    truth = ground_truth.lower()

    score = 0.10

    truth_words  = set(re.findall(r"\w+", truth))
    action_words = set(re.findall(r"\w+", text))
    score += (len(truth_words & action_words) / max(len(truth_words), 1)) * 0.40
    score += (sum(1 for k in MEDIUM_STRUCT_KW if k in text) / len(MEDIUM_STRUCT_KW)) * 0.30

    wc = len(action_words)
    if wc >= 30:
        score += 0.10
    elif wc >= 15:
        score += 0.05

    return clamp_score(score, lo=0.05, hi=0.87)


# ── hard grader (LLM-as-judge, max 0.93) ─────────────────────────────────────

HARD_PROMPT = """You are a strict JSON-output judge evaluating an AI agent's API validation response.

Expected answer:
{ground_truth}

Agent response:
{action}

Output ONLY valid JSON: {{"score": <float>}}
Rules:
- score must be strictly between 0.05 and 0.93 (never 0 or 1)
- 0.05-0.30: completely wrong
- 0.31-0.60: partially correct
- 0.61-0.93: substantially correct with good reasoning
"""


def _fallback(action_content: str, ground_truth: str) -> float:
    t = set(re.findall(r"\w+", ground_truth.lower()))
    a = set(re.findall(r"\w+", action_content.lower()))
    return clamp_score(0.10 + (len(t & a) / max(len(t), 1)) * 0.60, lo=0.05, hi=0.93)


def grade_hard(action_content: str, ground_truth: str) -> float:
    try:
        resp = _client().chat.completions.create(
            model=_model(),
            messages=[{"role": "user", "content": HARD_PROMPT.format(
                ground_truth=ground_truth, action=action_content
            )}],
            temperature=0.0,
            max_tokens=64,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        if m:
            return clamp_score(float(m.group(1)), lo=0.05, hi=0.93)
        m2 = re.search(r'\b([0-9]\.[0-9]+)\b', raw)
        if m2:
            return clamp_score(float(m2.group(1)), lo=0.05, hi=0.93)
    except Exception:
        pass
    return _fallback(action_content, ground_truth)
