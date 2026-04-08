"""
APIResponseValidatorEnv – OpenEnv-compatible environment.
"""
import random
from typing import Optional
from app.models import Action, State, StepResult, clamp_reward
from app.grader import grade_easy, grade_medium, grade_hard

SCENARIOS = {
    "easy": [
        {
            "input": "HTTP 200 OK\nContent-Type: application/json\n{'user_id': 42, 'name': 'Alice', 'email': 'alice@example.com'}",
            "ground_truth": "VALID – status 200 with correct JSON body matching user schema. Fields user_id (int), name (str), email (str) all present.",
        },
        {
            "input": "HTTP 404 Not Found\nContent-Type: application/json\n{'error': 'User not found'}",
            "ground_truth": "INVALID – expected 200 OK but received 404. Response body contains error message, not user object.",
        },
        {
            "input": "HTTP 200 OK\nContent-Type: application/json\n{'items': [], 'total': 0, 'page': 1}",
            "ground_truth": "VALID – status 200 with empty items list. Pagination fields total and page present.",
        },
        {
            "input": "HTTP 500 Internal Server Error\nContent-Type: text/html\n<html>Server Error</html>",
            "ground_truth": "INVALID – server error 500, HTML body instead of JSON. Indicates backend crash, not a valid API response.",
        },
        {
            "input": "HTTP 201 Created\nContent-Type: application/json\n{'id': 99, 'created_at': '2024-01-15T10:30:00Z'}",
            "ground_truth": "VALID – 201 Created with resource id and ISO timestamp. Correct status for POST resource creation.",
        },
    ],
    "medium": [
        {
            "input": "Endpoint: POST /api/v1/orders\nRequest: {'product_id': 'ABC123', 'qty': 2}\nResponse 200: {'order_id': 'ORD-789', 'status': 'pending', 'total': null, 'items': [{'sku': 'ABC123', 'quantity': 2}]}",
            "ground_truth": "PARTIAL VALID – order created with correct status. ISSUE: total field is null; should be a positive float. Schema violation: total must not be null for a valid order.",
        },
        {
            "input": "Endpoint: GET /api/v2/products/XYZ\nResponse 200: {'product_id': 'XYZ', 'name': 'Widget', 'price': -5.0, 'stock': 100}",
            "ground_truth": "INVALID – price field is -5.0 which violates business rule (price >= 0). All other fields are present and correctly typed.",
        },
        {
            "input": "Endpoint: GET /api/v1/users/123\nResponse 200: {'id': 123, 'username': 'bob', 'role': 'admin', 'last_login': '2024-01-15'}",
            "ground_truth": "VALID – complete user object with correct types. last_login is date string in ISO format. Role is valid enum value.",
        },
        {
            "input": "Endpoint: DELETE /api/v1/items/55\nResponse 200: {'deleted': true, 'id': 55, 'timestamp': '2024-02-10T08:00:00Z'}",
            "ground_truth": "VALID but non-standard – DELETE should return 204 No Content. Body is acceptable but status code 204 preferred over 200.",
        },
        {
            "input": "Endpoint: POST /api/v1/auth/login\nResponse 200: {'token': '', 'expires_in': 3600}",
            "ground_truth": "INVALID – token field is empty string. Login must return a non-empty bearer token. Critical schema error.",
        },
    ],
    "hard": [
        {
            "input": "Endpoint: POST /api/v3/payments\nRequest: {'amount': 150.00, 'currency': 'USD', 'source': 'tok_visa'}\nResponse 200: {'payment_id': 'pay_abc', 'status': 'requires_action', 'amount': 15000, 'currency': 'usd', 'next_action': {'type': 'redirect_to_url', 'url': 'https://bank.example/3ds'}, 'created': 1705312200}",
            "ground_truth": "VALID with caveats – amount stored as integer cents (15000 = $150.00), correct. Status 'requires_action' indicates 3DS authentication needed. next_action URL present as required. Currency lowercase 'usd' is Stripe convention, acceptable.",
        },
        {
            "input": "Endpoint: GET /api/v1/reports/summary?from=2024-01-01&to=2024-12-31\nResponse 200: {'period': {'from': '2024-01-01', 'to': '2024-12-31'}, 'metrics': {'revenue': 1250000, 'orders': 4820, 'avg_order': 259.33, 'refund_rate': 0.023}, 'breakdown': [{'month': 'Jan', 'revenue': 95000}, '... 11 more items ...']}",
            "ground_truth": "VALID – well-structured analytics response. Period matches query params. avg_order = revenue/orders = 259.33 correct. refund_rate 0.023 is ratio (2.3%), within normal range.",
        },
        {
            "input": "Endpoint: POST /api/v2/batch/users\nRequest: [{'email': 'a@test.com'}, {'email': 'b@test.com'}, {'email': 'invalid-email'}]\nResponse 207: {'results': [{'index': 0, 'status': 201, 'id': 1}, {'index': 1, 'status': 201, 'id': 2}, {'index': 2, 'status': 422, 'error': 'Invalid email format'}]}",
            "ground_truth": "VALID – 207 Multi-Status is correct for batch with partial failures. Each result indexed correctly. Invalid email returns 422 with clear error. Exemplary REST batch API design.",
        },
        {
            "input": "Endpoint: GET /api/v1/stream/events (SSE)\nResponse 200:\nContent-Type: text/event-stream\ndata: {'event': 'connected', 'client_id': 'clnt_xyz'}\ndata: {'event': 'data', 'payload': {'temp': 72.3, 'unit': 'F'}}\ndata: {'event': 'heartbeat', 'ts': 1705312200}",
            "ground_truth": "VALID SSE stream – correct Content-Type text/event-stream. Connected event with client_id, data events with payload. Heartbeat present for connection keepalive.",
        },
        {
            "input": "Endpoint: POST /api/v1/webhooks/github\nHeaders: X-Hub-Signature-256: sha256=abc123, X-GitHub-Event: push\nResponse 200: {'received': true, 'event': 'push', 'processed_commits': 3, 'branch': 'main'}",
            "ground_truth": "VALID – webhook acknowledgment with correct 200 status. received: true confirms delivery. Echoes event type and processing details. Signature verification should occur server-side.",
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

        difficulty = self._state.difficulty
        ground_truth = self._scenario["ground_truth"]

        if difficulty == "easy":
            raw_reward = grade_easy(action.content, ground_truth)
        elif difficulty == "medium":
            raw_reward = grade_medium(action.content, ground_truth)
        else:
            raw_reward = grade_hard(action.content, ground_truth)

        reward = clamp_reward(raw_reward)  # final safety clamp

        self._state.step_count += 1
        self._state.done = True

        return StepResult(
            state=self._state,
            reward=reward,
            done=True,
            info={"difficulty": difficulty, "raw_reward": raw_reward},
        )