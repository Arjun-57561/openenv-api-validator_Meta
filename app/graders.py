from __future__ import annotations

import json
import re
from typing import Any, Dict

from openai import OpenAI


def clamp_score(x: float) -> float:
    """Force every score to stay strictly inside the safe reward band."""
    try:
        value = float(x)
    except Exception:
        value = 0.5

    return max(0.001, min(0.95, round(value, 4)))


def safe_score(x: float) -> float:
    return clamp_score(x)


def client() -> OpenAI:
    import os

    base = os.getenv("APIBASEURL", "").strip()
    key = os.getenv("HFTOKEN", "").strip()
    return OpenAI(base_url=base, api_key=key or "dummy")


def model_name() -> str:
    import os

    return os.getenv("MODELNAME", "llama-3.3-70b-versatile").strip()


def grade_easy(agent_text: str, groundtruth: Dict[str, Any]) -> float:
    """
    Rule-based: expected HTTP status, required top-level JSON fields, content-type mention.
    Partial credit from independent checks -> varied scores strictly inside (0, 1).
    """
    text = agent_text.lower()
    score = 0.0

    expstatus = int(groundtruth.get("expected_status", 0) or groundtruth.get("expectedstatus", 0))
    score += 0.18 if expstatus and str(expstatus) in agent_text else 0.0

    score += (
        0.18
        if (
            "application/json" in text
            or "content-type" in text
            or " json" in text
            or text.startswith("json")
        )
        else 0.0
    )

    req = groundtruth.get("required_fields", groundtruth.get("requiredfields", []))
    if req:
        hits = sum(1 for f in req if f.lower() in text)
        score += 0.22 * (hits / len(req))

    must_note_ok = groundtruth.get("must_note_ok", groundtruth.get("mustnoteok", False))
    if must_note_ok:
        score += (
            0.18
            if ("valid" in text or "acceptable" in text or "correct" in text or "ok" in text)
            else 0.0
        )

    false_error = (
        ("500" in agent_text or "404" in agent_text or " 400 " in agent_text)
        and expstatus == 200
    )
    score += 0.14 if not false_error else 0.0

    return clamp_score(score)


def grade_medium(agent_text: str, groundtruth: Dict[str, Any]) -> float:
    """
    Keyword + light structure: penalties for missing risk flags, bonus for schema wording.
    """
    text = agent_text.lower()
    score = 0.3

    keywords = groundtruth.get("expected_keywords", groundtruth.get("expectedkeywords", []))
    if keywords:
        hits = sum(1 for k in keywords if k.lower() in text)
        score += 0.4 * (hits / len(keywords))

    must_mention_nested = groundtruth.get(
        "must_mention_nested", groundtruth.get("mustmentionnested", False)
    )
    if must_mention_nested and re.search(r"user\.address|nested|address\.zip|zip", text):
        score += 0.1

    must_mention_null = groundtruth.get(
        "must_mention_null", groundtruth.get("mustmentionnull", False)
    )
    if must_mention_null and ("null" in text or "missing" in text or "omit" in text):
        score += 0.07

    if len(agent_text.strip()) < 40:
        score -= 0.15

    return clamp_score(score)


def grade_hard(agent_text: str, groundtruth: Dict[str, Any]) -> float:
    """
    LLM-as-judge with strict JSON output; falls back to blended heuristic if the API errors.
    """
    rubric = groundtruth.get("rubric", "")
    reference = groundtruth.get("reference_verdict", groundtruth.get("referenceverdict", ""))

    payload = {
        "rubric": rubric,
        "reference_verdict": reference,
        "agent_verdict": agent_text[:8000],
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
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            data = json.loads(m.group())
            s = float(data.get("score", 0.5))
            return clamp_score(s)
    except Exception:
        pass

    text = agent_text.lower()
    reftoks = set(re.findall(r"[a-z]{4,}", reference.lower()))
    if not reftoks:
        return clamp_score(0.45)

    overlap = sum(1 for t in reftoks if t in text)
    base = 0.25 + 0.55 * min(1.0, overlap / max(6, len(reftoks) * 0.3))

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