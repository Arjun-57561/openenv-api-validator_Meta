from __future__ import annotations

import json
import re
from typing import Any, Dict

from openai import OpenAI


def _client() -> OpenAI:
    import os

    base = os.getenv("API_BASE_URL", "").strip()
    key = os.getenv("HF_TOKEN", "").strip()
    return OpenAI(base_url=base, api_key=key or "dummy")


def _model_name() -> str:
    import os

    return os.getenv("MODEL_NAME", "llama-3.3-70b-versatile").strip()


def grade_easy(agent_text: str, ground_truth: Dict[str, Any]) -> float:
    """
    Rule-based: expected HTTP status, required top-level JSON fields, content-type mention.
    Partial credit from independent checks → varied scores in [0, 1].
    """
    text = agent_text.lower()
    score = 0.0
    weights: list[float] = []

    exp_status = int(ground_truth.get("expected_status", 0))
    weights.append(0.2 if exp_status and str(exp_status) in agent_text else 0.0)
    weights.append(0.2 if ("application/json" in text or "content-type" in text or " json" in text) else 0.0)

    req = ground_truth.get("required_fields", [])
    if req:
        hits = sum(1 for f in req if f.lower() in text)
        weights.append(0.25 * (hits / len(req)))
    else:
        weights.append(0.0)

    if ground_truth.get("must_note_ok"):
        weights.append(0.2 if ("valid" in text or "acceptable" in text or "correct" in text or "ok" in text) else 0.0)
    else:
        weights.append(0.0)

    # Penalize falsely claiming a client/server error for a successful response
    false_error = ("500" in agent_text or "404" in agent_text or " 400 " in agent_text) and exp_status == 200
    weights.append(0.15 if not false_error else 0.0)

    score = sum(weights)
    return max(0.0, min(1.0, round(score, 4)))


def grade_medium(agent_text: str, ground_truth: Dict[str, Any]) -> float:
    """
    Keyword + light structure: penalties for missing risk flags, bonus for schema wording.
    """
    text = agent_text.lower()
    score = 0.35
    keywords = ground_truth.get("expected_keywords", [])
    if keywords:
        hits = sum(1 for k in keywords if k.lower() in text)
        score += 0.45 * (hits / len(keywords))
    if ground_truth.get("must_mention_nested") and re.search(
        r"user\.address|nested|address\.zip|zip", text
    ):
        score += 0.12
    if ground_truth.get("must_mention_null") and ("null" in text or "missing" in text or "omit" in text):
        score += 0.08
    if len(agent_text.strip()) < 40:
        score -= 0.15
    return max(0.0, min(1.0, round(score, 4)))


def grade_hard(agent_text: str, ground_truth: Dict[str, Any]) -> float:
    """
    LLM-as-judge with strict JSON output; falls back to blended heuristic if the API errors.
    """
    rubric = ground_truth.get("rubric", "")
    reference = ground_truth.get("reference_verdict", "")
    payload = {
        "rubric": rubric,
        "reference_verdict": reference,
        "agent_verdict": agent_text[:8000],
    }
    system = (
        "You grade how well the agent's API validation verdict matches the rubric and reference. "
        "Reply ONLY with compact JSON: {\"score\": <float between 0 and 1>, \"reason\": <short str>} "
        "Use fine-grained scoring; avoid 0.0 or 1.0 unless truly deserved."
    )
    user = json.dumps(payload)
    try:
        resp = _client().chat.completions.create(
            model=_model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.35,
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            data = json.loads(m.group())
            s = float(data.get("score", 0.5))
            return max(0.0, min(1.0, round(s, 4)))
    except Exception:
        pass

    # Fallback: partial overlap with reference keywords (still variable, not constant)
    text = agent_text.lower()
    ref_toks = set(re.findall(r"[a-z]{4,}", reference.lower()))
    if not ref_toks:
        return 0.45
    overlap = sum(1 for t in ref_toks if t in text)
    base = 0.25 + 0.55 * min(1.0, overlap / max(6, len(ref_toks) * 0.3))
    if "pii" in text or "gdpr" in text:
        base += 0.08
    if "rate" in text and "limit" in text:
        base += 0.06
    return max(0.0, min(1.0, round(base, 4)))
