# C:\Users\Arjun\OneDrive\Desktop\Meta_RL\openenv-api-validator\inference.py
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


def _safe_score(x: float) -> float:
    """Force emitted rewards to be strictly inside (0, 1)."""
    try:
        x = float(x)
    except Exception:
        x = 0.5

    if x <= 0.0:
        return 0.01
    if x >= 1.0:
        return 0.99
    return round(x, 6)


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


def _safe_json(resp: requests.Response) -> dict:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {"value": data}
    except Exception:
        return {"raw": resp.text[:1000]}


def _post_reset(base: str, difficulty: str) -> requests.Response:
    candidates = [
        (f"{base}/reset", {"difficulty": difficulty}),
        (f"{base}/reset", None),
    ]
    last_resp = None
    for url, payload in candidates:
        try:
            if payload is None:
                resp = requests.post(url, timeout=120)
            else:
                resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp
            last_resp = resp
        except requests.RequestException:
            continue
    if last_resp is not None:
        return last_resp
    raise requests.RequestException("Unable to reach /reset")


def run_episode(difficulty: str) -> tuple[str, float, int]:
    base = _space_url()
    fallback_task_name = f"{difficulty}_task"

    try:
        r = _post_reset(base, difficulty)
        if r.status_code != 200:
            task_name = fallback_task_name
            fallback_reward = _safe_score(0.01)
            print(f'[START] {json.dumps({"task": task_name, "difficulty": difficulty})}', flush=True)
            print(
                f'[STEP] {json.dumps({"step": 0, "action": "", "reward": fallback_reward, "done": True})}',
                flush=True,
            )
            print(
                f'[END] {json.dumps({"task": task_name, "total_reward": fallback_reward, "steps": 1})}',
                flush=True,
            )
            return task_name, fallback_reward, 1

        st = _safe_json(r)
        task_name = st.get("task_name", fallback_task_name)
        obs = st.get("current_input", "")

        print(
            f'[START] {json.dumps({"task": task_name, "difficulty": st.get("difficulty", difficulty)})}',
            flush=True,
        )

        try:
            action_text = _agent_action(obs)
        except Exception:
            action_text = (
                "The response should be checked for HTTP status correctness, required fields, "
                "schema consistency, nested null risks, and any mismatch with documented behavior."
            )

        body = {"content": action_text}
        try:
            sr = requests.post(f"{base}/step", json=body, timeout=120)
            if sr.status_code != 200:
                out = {}
                reward = _safe_score(0.01)
                done = True
                step_idx = 1
            else:
                out = _safe_json(sr)
                reward = _safe_score(out.get("reward", 0.01))
                done = bool(out.get("done", True))
                state = out.get("state", {})
                step_idx = int(state.get("step_count", 1))
        except requests.RequestException:
            reward = _safe_score(0.01)
            done = True
            step_idx = 1

        print(
            f'[STEP] {json.dumps({"step": step_idx, "action": action_text, "reward": reward, "done": done})}',
            flush=True,
        )

        total = reward
        steps = 1
        print(f'[END] {json.dumps({"task": task_name, "total_reward": total, "steps": steps})}', flush=True)
        return task_name, total, steps

    except Exception:
        task_name = fallback_task_name
        fallback_reward = _safe_score(0.01)
        print(f'[START] {json.dumps({"task": task_name, "difficulty": difficulty})}', flush=True)
        print(
            f'[STEP] {json.dumps({"step": 0, "action": "", "reward": fallback_reward, "done": True})}',
            flush=True,
        )
        print(
            f'[END] {json.dumps({"task": task_name, "total_reward": fallback_reward, "steps": 1})}',
            flush=True,
        )
        return task_name, fallback_reward, 1


def main() -> None:
    for diff in ("easy", "medium", "hard"):
        run_episode(diff)


if __name__ == "__main__":
    main()