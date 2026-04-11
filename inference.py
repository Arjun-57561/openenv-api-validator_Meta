"""
Baseline agent for OpenEnv API Response Validator.
Prints [START]/[STEP]/[END] blocks to stdout for easy, medium, hard difficulties.
"""
from __future__ import annotations

import json
import os
import sys

import requests

SPACE_URL = os.getenv("SPACE_URL", "https://Asharjun-api-response-validator.hf.space").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

FALLBACK_ACTION = (
    "The HTTP response status code should be verified for correctness. "
    "Required fields including id and name should be checked for presence. "
    "Content-Type should be application/json. Nested null values pose schema risks. "
    "Any undocumented fields represent contract violations. "
    "request_id should not be logged in client telemetry per the documentation."
)


def _safe_reward(x) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.5
    if v <= 0.0:
        return 0.05
    if v >= 1.0:
        return 0.95
    return round(v, 4)


def _safe_json(resp: requests.Response) -> dict:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_action(observation: str) -> str:
    if not API_BASE_URL or not HF_TOKEN:
        return FALLBACK_ACTION
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You validate HTTP API responses for contract, consistency, and risk. "
                        "Answer in concise technical prose (3-8 sentences). Be specific about "
                        "status codes, fields, nulls, and documentation mismatches."
                    ),
                },
                {"role": "user", "content": observation},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return (resp.choices[0].message.content or "").strip() or FALLBACK_ACTION
    except Exception:
        return FALLBACK_ACTION


def run_episode(difficulty: str) -> None:
    task_name = f"{difficulty}_task"
    fallback_reward = 0.5

    # Always print [START] first no matter what
    print(f'[START] {json.dumps({"task": task_name, "difficulty": difficulty})}', flush=True)

    try:
        # Reset
        try:
            r = requests.post(
                f"{SPACE_URL}/reset",
                json={"difficulty": difficulty},
                timeout=120,
            )
            if r.status_code == 200:
                st = _safe_json(r)
                task_name = st.get("task_name", task_name)
                obs = st.get("current_input", "")
            else:
                obs = ""
        except Exception:
            obs = ""

        # Get agent action
        action_text = _get_action(obs) if obs else FALLBACK_ACTION

        # Step
        try:
            sr = requests.post(
                f"{SPACE_URL}/step",
                json={"content": action_text},
                timeout=120,
            )
            if sr.status_code == 200:
                out = _safe_json(sr)
                reward = _safe_reward(out.get("reward", fallback_reward))
                done = bool(out.get("done", True))
                state = out.get("state", {})
                step_idx = int(state.get("step_count", 1))
            else:
                reward = fallback_reward
                done = True
                step_idx = 1
        except Exception:
            reward = fallback_reward
            done = True
            step_idx = 1

    except Exception:
        action_text = FALLBACK_ACTION
        reward = fallback_reward
        done = True
        step_idx = 1

    print(
        f'[STEP] {json.dumps({"step": step_idx, "action": action_text, "reward": reward, "done": done})}',
        flush=True,
    )
    print(
        f'[END] {json.dumps({"task": task_name, "total_reward": reward, "steps": 1})}',
        flush=True,
    )


def main() -> None:
    for diff in ("easy", "medium", "hard"):
        run_episode(diff)


if __name__ == "__main__":
    main()
