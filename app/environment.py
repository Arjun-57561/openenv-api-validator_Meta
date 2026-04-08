from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from app.graders import grade_easy, grade_hard, grade_medium
from app.models import Action, State, StepResult

Difficulty = Literal["easy", "medium", "hard"]
MIN_REWARD = 0.001
MAX_REWARD = 0.95


def _safe_score(x: float) -> float:
    """Force every reward to stay strictly inside (0.001, 0.95)."""
    try:
        value = float(x)
    except Exception:
        value = 0.5
    return max(MIN_REWARD, min(MAX_REWARD, round(value, 6)))


def _scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "name": "easy_task",
            "difficulty": "easy",
            "prompt": (
                "You are validating this HTTP API response.\n\n"
                "GET /users/42 - HTTP 200\n"
                "Content-Type: application/json\n"
                "Body: {\"id\": 42, \"name\": \"Ada\", \"role\": \"admin\"}\n\n"
                "State whether the response is acceptable for a user profile endpoint, "
                "mention the status code, and list which required fields for a minimal "
                "profile (id, name) are present or missing."
            ),
            "ground_truth": {
                "expected_status": 200,
                "required_fields": ["id", "name"],
                "must_note_ok": True,
                "must_note_error": False,
            },
        },
        {
            "name": "medium_task",
            "difficulty": "medium",
            "prompt": (
                "Validate this response for internal consistency and schema risks.\n\n"
                "POST /orders - HTTP 201\n"
                "Body: {\n"
                '  "order_id": "A-100",\n'
                '  "user": {"id": 7, "address": {"zip": null}}\n'
                "}\n\n"
                "Call out nested null issues, any implied contract problems, and whether "
                "clients could break if they assume zip is always a string."
            ),
            "ground_truth": {
                "expected_keywords": ["nested", "null", "schema", "zip", "risk"],
                "must_mention_nested": True,
                "must_mention_null": True,
            },
        },
        {
            "name": "hard_task",
            "difficulty": "hard",
            "prompt": (
                "Review this public API documentation snippet and sample error payload.\n\n"
                "Docs: 'Errors return JSON {code, message, request_id}. "
                "Do not log request_id in client telemetry.'\n\n"
                "Sample: HTTP 429\n"
                "Body: {\"code\":\"rate_limited\",\"message\":\"Slow down\","
                "\"request_id\":\"req_9f3\",\"retry_after\":2}\n\n"
                "Assess compliance with the stated contract, privacy/logging risk for "
                "request_id, and whether retry_after is documented/consistent."
            ),
            "ground_truth": {
                "rubric": (
                    "Reward nuanced identification of: (1) field presence vs docs, "
                    "(2) privacy risk of request_id in clients, "
                    "(3) undocumented retry_after, "
                    "(4) partial credit if reasoning is weak but directionally right."
                ),
                "reference_verdict": (
                    "The payload matches the documented shape. request_id should not be "
                    "logged in client telemetry per docs; teams should hash or drop it. "
                    "retry_after is not mentioned in the contract excerpt, so it is an "
                    "undocumented extension that could surprise integrators."
                ),
            },
        },
    ]


class APIResponseValidatorEnv:
    """Single-step episodes: reset loads one scenario, step grades one action."""

    def __init__(self) -> None:
        self._scenarios = {s["difficulty"]: s for s in _scenarios()}
        self._state: Optional[State] = None
        self._active: Optional[Dict[str, Any]] = None
        self._last_difficulty_request: Optional[Difficulty] = None

    def reset(self, difficulty: Optional[Difficulty] = None) -> State:
        if difficulty is None:
            order: Tuple[Difficulty, ...] = ("easy", "medium", "hard")
            if self._last_difficulty_request is None:
                difficulty = "easy"
            else:
                idx = order.index(self._last_difficulty_request)
                difficulty = order[(idx + 1) % len(order)]

        self._last_difficulty_request = difficulty
        spec = self._scenarios[difficulty]
        self._active = spec
        self._state = State(
            difficulty=spec["difficulty"],
            step_count=0,
            current_input=spec["prompt"],
            last_reward=MIN_REWARD,
            task_name=spec["name"],
            done=False,
        )
        return self._state

    def state(self) -> State:
        if self._state is None:
            return self.reset("easy")
        return self._state

    def step(self, action: Action) -> StepResult:
        if self._state is None or self._active is None:
            self.reset("easy")
        assert self._state is not None and self._active is not None

        difficulty = self._active["difficulty"]
        ground_truth = self._active["ground_truth"]

        if difficulty == "easy":
            raw_reward = grade_easy(action.content, ground_truth)
        elif difficulty == "medium":
            raw_reward = grade_medium(action.content, ground_truth)
        else:
            raw_reward = grade_hard(action.content, ground_truth)

        reward = _safe_score(raw_reward)
        new_state = self._state.model_copy(
            update={
                "step_count": self._state.step_count + 1,
                "last_reward": reward,
                "done": True,
            }
        )
        self._state = new_state
        return StepResult(state=new_state, reward=reward, done=True)