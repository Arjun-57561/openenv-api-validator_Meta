"""
Grading functions for easy / medium / hard tasks.
All public grade_* functions MUST return a float in the OPEN interval (0, 1).
"""
import os
import re
from openai import OpenAI


# ── helpers ──────────────────────────────────────────────────────────────────

def clamp_score(score: float, lo: float = 0.05, hi: float = 0.95) -> float:
    """Clamp to (0,1)-safe defaults so 0.0 and 1.0 are never returned."""
    return max(lo, min(hi, float(score)))


def safe_score(raw: float) -> float:
    return clamp_score(raw)


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("HF_TOKEN", ""),
        base_url=os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1"),
    )


def _model() -> str:
    return os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")


# ── easy grader (rule-based, max 0.88) ───────────────────────────────────────

EASY_KEYWORDS_PASS = [
    "valid", "correct", "200", "success", "ok", "matches", "schema",
]
EASY_KEYWORDS_FAIL = [
    "invalid", "error", "mismatch", "wrong", "missing", "null", "undefined",
]


def grade_easy(action_content: str, ground_truth: str) -> float:
    """Rule-based scoring. Returns float in (0.05, 0.88)."""
    text = action_content.lower()
    truth = ground_truth.lower()

    score = 0.10  # base – never starts at 0

    truth_words = set(re.findall(r"\w+", truth))
    action_words = set(re.findall(r"\w+", text))
    overlap = len(truth_words & action_words)
    if truth_words:
        overlap_ratio = overlap / len(truth_words)
        score += overlap_ratio * 0.50

    is_pass_truth = any(k in truth for k in EASY_KEYWORDS_PASS)
    is_fail_truth = any(k in truth for k in EASY_KEYWORDS_FAIL)

    if is_pass_truth and any(k in text for k in EASY_KEYWORDS_PASS):
        score += 0.20
    elif is_fail_truth and any(k in text for k in EASY_KEYWORDS_FAIL):
        score += 0.20
    else:
        score -= 0.05

    return clamp_score(score, lo=0.05, hi=0.88)


# ── medium grader (hybrid, max 0.87) ─────────────────────────────────────────

MEDIUM_STRUCTURE_KEYS = ["status", "schema", "field", "type", "required", "response"]


def grade_medium(action_content: str, ground_truth: str) -> float:
    """Hybrid keyword + structure scoring. Returns float in (0.05, 0.87)."""
    text = action_content.lower()
    truth = ground_truth.lower()

    score = 0.10

    truth_words = set(re.findall(r"\w+", truth))
    action_words = set(re.findall(r"\w+", text))
    overlap = len(truth_words & action_words) / max(len(truth_words), 1)
    score += overlap * 0.40

    struct_hits = sum(1 for k in MEDIUM_STRUCTURE_KEYS if k in text)
    score += (struct_hits / len(MEDIUM_STRUCTURE_KEYS)) * 0.30

    word_count = len(action_words)
    if word_count >= 30:
        score += 0.10
    elif word_count >= 15:
        score += 0.05

    return clamp_score(score, lo=0.05, hi=0.87)


# ── hard grader (LLM-as-judge with deterministic fallback, max 0.93) ─────────

HARD_JUDGE_PROMPT = """You are a strict JSON-output judge evaluating an AI agent's API validation response.

TASK: Score how well the agent's response matches the expected answer.

Expected answer:
{ground_truth}

Agent response:
{action}

Rules:
- Output ONLY a JSON object: {{"score": <float>}}
- score must be strictly between 0.05 and 0.93 (never 0 or 1)
- 0.05-0.30: completely wrong / irrelevant
- 0.31-0.60: partially correct
- 0.61-0.93: substantially correct with good reasoning
"""


def _deterministic_fallback(action_content: str, ground_truth: str) -> float:
    truth_words = set(re.findall(r"\w+", ground_truth.lower()))
    action_words = set(re.findall(r"\w+", action_content.lower()))
    overlap = len(truth_words & action_words) / max(len(truth_words), 1)
    return clamp_score(0.10 + overlap * 0.60, lo=0.05, hi=0.93)


def grade_hard(action_content: str, ground_truth: str) -> float:
    """LLM-as-judge scoring. Returns float in (0.05, 0.93)."""
    try:
        client = _client()
        prompt = HARD_JUDGE_PROMPT.format(
            ground_truth=ground_truth, action=action_content
        )
        response = client.chat.completions.create(
            model=_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        if match:
            return clamp_score(float(match.group(1)), lo=0.05, hi=0.93)
        match2 = re.search(r'\b([0-9]\.[0-9]+)\b', raw)
        if match2:
            return clamp_score(float(match2.group(1)), lo=0.05, hi=0.93)
    except Exception:
        pass
    return _deterministic_fallback(action_content, ground_truth)