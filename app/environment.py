"""
APIResponseValidatorEnv – OpenEnv-compatible RL environment.
"""
import random
from typing import Optional
from app.models import Action, State, StepResult, clamp_reward
from app.graders import grade_easy, grade_medium, grade_hard

SCENARIOS = {
    "easy": [
        {
            "input": "HTTP 200 OK\nContent-Type: application/json\n{\'user_id\': 42, \'name\': \'Alice\', \'email\': \'alice@example.com\'}",
            "ground_truth": "VALID – status 200 with correct JSON body matching user schema. Fields user_id (int), name (str), email (str) all present.",
        },
        {
            "input": "HTTP 404 Not Found\nContent-Type: application/json\n{\'error\': \'User not found\'}",
            "ground_truth": "INVALID – expected 200 OK but received 404. Response body contains error message, not user object.",
        },
        {
            "input": "HTTP 200 OK\nContent-Type: application/json\n{\'items\': [], \'total\': 0, \'page\': 1}",
            "ground_truth": "VALID – status 200 with empty items list. Pagination fields total and page are present.",
        },
        {
            "input": "HTTP 500 Internal Server Error\nContent-Type: text/html\n<html>Server Error</html>",
            "ground_truth": "INVALID – server error 500, HTML body instead of JSON. Indicates backend crash.",
        },
        {
            "input": "HTTP 201 Created\nContent-Type: application/json\n{\'id\': 99, \'created_at\': \'2024-01-15T10:30:00Z\'}",
            "ground_truth": "VALID – 201 Created with resource id and ISO timestamp. Correct status for POST resource creation.",
        },
    ],
    "medium": [
        {
            "input": "Endpoint: POST /api/v1/orders\nResponse 200: {\'order_id\': \'ORD-789\', \'status\': \'pending\', \'total\': null}",
            "ground_truth": "PARTIAL VALID – order created but total field is null. Schema violation: total must be a positive float.",
        },
        {
            "input": "Endpoint: GET /api/v2/products/XYZ\nResponse 200: {\'product_id\': \'XYZ\', \'price\': -5.0, \'stock\': 100}",
            "ground_truth": "INVALID – price is -5.0 which violates business rule (price >= 0).",
        },
        {
            "input": "Endpoint: GET /api/v1/users/123\nResponse 200: {\'id\': 123, \'username\': \'bob\', \'role\': \'admin\', \'last_login\': \'2024-01-15\'}",
            "ground_truth": "VALID – complete user object with correct types. last_login in ISO format. Role is valid enum value.",
        },
        {
            "input": "Endpoint: DELETE /api/v1/items/55\nResponse 200: {\'deleted\': true, \'id\': 55}",
            "ground_truth": "VALID but non-standard – DELETE should return 204 No Content. Status 200 with body is acceptable.",
        },
        {
            "input": "Endpoint: POST /api/v1/auth/login\nResponse 200: {\'token\': \'\', \'expires_in\': 3600}",
            "ground_truth": "INVALID – token field is empty string. Login must return a non-empty bearer token. Critical schema error.",
        },
    ],
    "hard": [
        {
            "input": "Endpoint: POST /api/v3/payments\nResponse 200: {\'payment_id\': \'pay_abc\', \'status\': \'requires_action\', \'amount\': 15000, \'currency\': \'usd\', \'next_action\': {\'type\': \'redirect_to_url\', \'url\': \'https://bank.example/3ds\'}}",
            "ground_truth": "VALID – amount as integer cents correct. Status requires_action indicates 3DS. next_action URL present. Currency lowercase usd is Stripe convention.",
        },
        {
            "input": "Endpoint: GET /api/v1/reports/summary\nResponse 200: {\'metrics\': {\'revenue\': 1250000, \'orders\': 4820, \'avg_order\': 259.33, \'refund_rate\': 0.023}}",
            "ground_truth": "VALID – avg_order = revenue/orders = 259.33 correct. refund_rate 0.023 is 2.3% ratio within normal range.",
        },
        {
            "input": "Endpoint: POST /api/v2/batch/users\nResponse 207: {\'results\': [{\'index\': 0, \'status\': 201}, {\'index\': 2, \'status\': 422, \'error\': \'Invalid email format\'}]}",
            "ground_truth": "VALID – 207 Multi-Status correct for batch with partial failures. Invalid email returns 422. Exemplary REST batch design.",
        },
        {
            "input": "Endpoint: GET /api/v1/stream/events\nResponse 200\nContent-Type: text/event-stream\ndata: {\'event\': \'connected\', \'client_id\': \'clnt_xyz\'}",
            "ground_truth": "VALID SSE stream – correct Content-Type text/event-stream. Connected event with client_id present.",
        },
        {
            "input": "Endpoint: POST /api/v1/webhooks/github\nHeaders: X-Hub-Signature-256: sha256=abc123\nResponse 200: {\'received\': true, \'event\': \'push\', \'processed_commits\': 3}",
            "ground_truth": "VALID – webhook acknowledgment with 200 status. received: true confirms delivery. Signature verification should occur server-side.",
        },
    ],
}


class APIResponseValidatorEnv:
    def __init__(self):
        self._state: Optional[State] = None
        self._scenario: Optional[dict] = None

    def reset(self, difficulty: str = "easy", seed: Optional[int] = None) -> State:
        if difficulty not in SCENARIOS:
            difficulty = "easy"
        rng = random.Random(seed)
        self._scenario = rng.choice(SCENARIOS[difficulty])
        self._state = State(
            difficulty=difficulty,
            step_count=0,
            current_input=self._scenario["input"],
            ground_truth=self._scenario["ground_truth"],
            done=False,
            metadata={"total_scenarios": len(SCENARIOS[difficulty])},
        )
        return self._state

    def get_state(self) -> State:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call /reset first.")
        return self._state

    def step(self, action: Action) -> StepResult:
        if self._state is None or self._state.done:
            raise RuntimeError("Call /reset before /step.")

        difficulty   = self._state.difficulty
        ground_truth = self._scenario["ground_truth"]

        if difficulty == "easy":
            raw = grade_easy(action.content, ground_truth)
        elif difficulty == "medium":
            raw = grade_medium(action.content, ground_truth)
        else:
            raw = grade_hard(action.content, ground_truth)

        reward = clamp_reward(raw)

        self._state.step_count += 1
        self._state.done = True

        return StepResult(
            state=self._state,
            reward=reward,
            done=True,
            info={"difficulty": difficulty, "raw_reward": raw},
        )
