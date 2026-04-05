"""
Baseline agent: calls the deployed Space (or local server) and uses the OpenAI-compatible
client for all LLM generations. Set SPACE_URL to your Hugging Face Space URL when validating remotely.
"""

from __future__ import annotations

import json
import os
import sys

import requests
from openai import OpenAI


def _llm() -> OpenAI:
    base = os.getenv("API_BASE_URL", "").strip()
    key = os.getenv("HF_TOKEN", "").strip()
    if not base or not key:
        print("Set API_BASE_URL and HF_TOKEN (Groq key) for the baseline agent.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(base_url=base, api_key=key)


def _model() -> str:
    return os.getenv("MODEL_NAME", "llama-3.3-70b-versatile").strip()


def _space_url() -> str:
    return os.getenv("SPACE_URL", "http://127.0.0.1:7860").rstrip("/")


def _agent_action(observation: str) -> str:
    client = _llm()
    messages = [
        {
            "role": "system",
            "content": (
                "You validate HTTP API responses for contract, consistency, and risk. "
                "Answer in concise technical prose (3–8 sentences). Be specific about "
                "status codes, fields, nulls, and documentation mismatches."
            ),
        },
        {"role": "user", "content": observation},
    ]
    resp = client.chat.completions.create(
        model=_model(),
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    return (resp.choices[0].message.content or "").strip()


def run_episode(difficulty: str) -> tuple[str, float, int]:
    base = _space_url()
    r = requests.post(f"{base}/reset", json={"difficulty": difficulty}, timeout=120)
    r.raise_for_status()
    st = r.json()
    task_name = st["task_name"]
    obs = st["current_input"]

    print(f'[START] {json.dumps({"task": task_name, "difficulty": st["difficulty"]})}', flush=True)

    action_text = _agent_action(obs)
    body = {"content": action_text}
    sr = requests.post(f"{base}/step", json=body, timeout=120)
    sr.raise_for_status()
    out = sr.json()
    reward = float(out["reward"])
    done = bool(out["done"])
    step_idx = int(out["state"]["step_count"])

    print(
        f'[STEP] {json.dumps({"step": step_idx, "action": action_text, "reward": reward, "done": done})}',
        flush=True,
    )

    total = reward
    steps = 1
    print(f'[END] {json.dumps({"task": task_name, "total_reward": total, "steps": steps})}', flush=True)
    return task_name, total, steps


def main() -> None:
    for diff in ("easy", "medium", "hard"):
        run_episode(diff)


if __name__ == "__main__":
    main()
