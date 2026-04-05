from __future__ import annotations

import json
import os
import sys
from typing import Optional

import requests
from openai import OpenAI


def _llm() -> OpenAI:
    base = os.getenv("API_BASE_URL", "").strip()
    key = os.getenv("HF_TOKEN", "").strip()
    if not base or not key:
        print("Set API_BASE_URL and HF_TOKEN.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(base_url=base, api_key=key)


def _model() -> str:
    return os.getenv("MODEL_NAME", "llama-3.3-70b-versatile").strip()


def _space_url() -> str:
    return os.getenv("SPACE_URL", "http://127.0.0.1:7860").rstrip("/")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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
    env_name = "api-response-validator"
    model_name = _model()

    r = requests.post(f"{base}/reset", json={"difficulty": difficulty}, timeout=120)
    r.raise_for_status()
    st = r.json()
    task_name = st["task_name"]
    obs = st["current_input"]

    log_start(task=task_name, env=env_name, model=model_name)

    all_rewards = []
    steps = 0
    done = False
    error = None

    while not done and steps < 8:
        try:
            action_text = _agent_action(obs)
        except Exception as e:
            error = str(e)
            action_text = ""

        body = {"content": action_text}
        sr = requests.post(f"{base}/step", json=body, timeout=120)
        sr.raise_for_status()
        out = sr.json()

        reward = float(out["reward"])
        done = bool(out["done"])
        steps += 1
        obs = out.get("state", {}).get("current_input", obs)

        log_step(step=steps, action=action_text, reward=reward, done=done, error=error)
        all_rewards.append(reward)
        error = None

    total = sum(all_rewards)
    score = total / max(steps, 1)
    success = score >= 0.1

    log_end(success=success, steps=steps, score=score, rewards=all_rewards)
    return task_name, total, steps


def main() -> None:
    for diff in ("easy", "medium", "hard"):
        run_episode(diff)


if __name__ == "__main__":
    main()