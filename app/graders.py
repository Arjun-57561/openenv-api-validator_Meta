"""
Grading functions — pure rule-based, no LLM calls.
All return float strictly in open interval (0.05, 0.95).
"""
from __future__ import annotations
import re


def _clamp(v: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return round(max(lo, min(hi, float(v))), 4)


def _words(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


EASY_PASS = ["valid", "correct", "200", "201", "ok", "matches", "schema", "success"]
EASY_FAIL = ["invalid", "error", "mismatch", "wrong", "missing", "null", "404", "500"]
MEDIUM_KW = ["status", "schema", "field", "type", "required", "null", "violation", "partial", "invalid"]
HARD_KW   = ["payment", "3ds", "webhook", "batch", "stream", "207", "requires_action", "redirect", "cents", "signature"]


def grade_easy(action: str, ground_truth: str) -> float:
    aw, tw = _words(action), _words(ground_truth)
    score = 0.12 + (len(aw & tw) / max(len(tw), 1)) * 0.45
    truth_is_pass = any(k in ground_truth.lower() for k in EASY_PASS)
    truth_is_fail = any(k in ground_truth.lower() for k in EASY_FAIL)
    if truth_is_pass and any(k in action.lower() for k in EASY_PASS):
        score += 0.20
    elif truth_is_fail and any(k in action.lower() for k in EASY_FAIL):
        score += 0.20
    return _clamp(score)


def grade_medium(action: str, ground_truth: str) -> float:
    aw, tw = _words(action), _words(ground_truth)
    score = 0.12
    score += (len(aw & tw) / max(len(tw), 1)) * 0.38
    score += (sum(1 for k in MEDIUM_KW if k in action.lower()) / len(MEDIUM_KW)) * 0.28
    if len(aw) >= 15:
        score += 0.08
    return _clamp(score)


def grade_hard(action: str, ground_truth: str) -> float:
    aw, tw = _words(action), _words(ground_truth)
    score = 0.12 + (len(aw & tw) / max(len(tw), 1)) * 0.45
    score += (sum(1 for k in HARD_KW if k in action.lower()) / len(HARD_KW)) * 0.25
    if len(aw) >= 10:
        score += 0.06
    return _clamp(score)
