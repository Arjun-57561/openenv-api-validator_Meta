from __future__ import annotations

import json
import re
from typing import Any, Dict

from openai import OpenAI

MIN_SCORE = 0.001
MAX_SCORE = 0.95


def clamp_score(x: float) -> float:
    """Keep all scores strictly inside the allowed reward interval."""
    try:
        value = float(x)
    except Exception:
        value = 0.5
    return max(MIN_SCORE, min(MAX_SCORE, round(value, 4)))


def safe_score(x: float) -> float:
    return clamp_score(x)


def client() -> OpenAI:
    import os

    base = (os.getenv("APIBASEURL", "") or os.getenv("API_BASE_URL", "")).strip()
    key = (os.getenv("HFTOKEN", "") or os.getenv("HF_TOKEN", "")).strip()
    return OpenAI(base_url=base, api_key=key or "dummy")


def model_name() -> str:
    import os

    return (os.getenv("MODELNAME", "") or os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")).strip()


def grade_easy(agent_text: str, groundtruth: Dict[str, Any]) -> float:
    """Rule-based scoring for easy tasks with max theoretical score of 0.90."""
    text = (agent_text or "").lower()
    score = MIN_SCORE

    expstatus = int(groundtruth.get("expected_status", 0) or groundtruth.get("expectedstatus", 0))
    if expstatus and str(expstatus) in agent_text:
        score += 0.18

    if (
        "application/json" in text
        or "content-type" in text
        or " json" in text
        or text.startswith("json")
    ):
        score += 0.18

    req = groundtruth.get("required_fields", groundtruth.get("requiredfields", []))
    if req:
        hits = sum(1 for f in req if f.lower() in text)
        score += 0.22 * (hits / len(req))

    must_note_ok = groundtruth.get("must_note_ok", groundtruth.get("mustnoteok", False))
    if must_note_ok and (
        "valid" in text or "acceptable" in text or "correct" in text or "ok" in text
    ):
        score += 0.18

    false_error = bool(re.search(r"\b(500|404|400)\b", agent_text or "")) and expstatus == 200
    if not false_error:
        score += 0.14

    return max(MIN_SCORE, min(MAX_SCORE, round(score, 4)))


def grade_medium(agent_text: str, groundtruth: Dict[str, Any]) -> float:
    """Hybrid keyword/structure score with max theoretical score of 0.87."""
    text = (agent_text or "").lower()
    score = 0.30

    keywords = groundtruth.get("expected_keywords", groundtruth.get("expectedkeywords", []))
    if keywords:
        hits = sum(1 for k in keywords if k.lower() in text)
        score += 0.40 * (hits / len(keywords))

    must_mention_nested = groundtruth.get(
        "must_mention_nested", groundtruth.get("mustmentionnested", False)
    )
    if must_mention_nested and re.search(r"user\.address|nested|address\.zip|zip", text):
        score += 0.10

    must_mention_null = groundtruth.get(
        "must_mention_null", groundtruth.get("mustmentionnull", False)
    )
    if must_mention_null and ("null" in text or "missing" in text or "omit" in text):
        score += 0.07

    if len((agent_text or "").strip()) < 40:
        score -= 0.15

    return max(MIN_SCORE, min(MAX_SCORE, round(score, 4)))


def grade_hard(agent_text: str, groundtruth: Dict[str, Any]) -> float:
    """LLM-as-judge plus deterministic fallback, both hard-clamped to safe bounds."""
    rubric = groundtruth.get("rubric", "")
    reference = groundtruth.get("reference_verdict", groundtruth.get("referenceverdict", ""))

    payload = {
        "rubric": rubric,
        "reference_verdict": reference,
        "agent_verdict": (agent_text or "")[:8000],
    }

    system = (
        "You grade how well the agent's API validation verdict matches the rubric and reference. "
        "Reply ONLY with compact JSON: "
        "{\"score\": <float strictly between 0 and 1, never 0.0 or 1.0>, "
        "\"reason\": <short str>}. "
        "Use fine-grained scoring and never emit boundary values."
    )
    user = json.dumps(payload)

    try:
        resp = client().chat.completions.create(
            model=model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.35,
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            data = json.loads(match.group())
            llm_score = float(data.get("score", 0.5))
            return max(MIN_SCORE, min(MAX_SCORE, llm_score))
    except Exception:
        pass

    text = (agent_text or "").lower()
    reftoks = set(re.findall(r"[a-z]{4,}", reference.lower()))
    if not reftoks:
        return clamp_score(0.45)

    overlap = sum(1 for token in reftoks if token in text)
    base = 0.25 + 0.55 * min(0.999, overlap / max(6, len(reftoks) * 0.3))

    if "pii" in text or "gdpr" in text or "privacy" in text:
        base += 0.08
    if "rate" in text and "limit" in text:
        base += 0.06
    if "requestid" in text or "request_id" in text:
        base += 0.04
    if "retryafter" in text or "retry_after" in text:
        base += 0.04

    return clamp_score(base)


def gradeeasy(agent_text: str, groundtruth: Dict[str, Any]) -> float:
    return grade_easy(agent_text, groundtruth)


def grademedium(agent_text: str, groundtruth: Dict[str, Any]) -> float:
    return grade_medium(agent_text, groundtruth)


def gradehard(agent_text: str, groundtruth: Dict[str, Any]) -> float:
    return grade_hard(agent_text, groundtruth)