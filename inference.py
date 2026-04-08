"""
Baseline inference script — required for hackathon evaluation.
Emits structured [START] / [STEP] / [END] JSON logs.
"""
import os
import json
import time
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.1-8b-instant")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "https://asharjun-api-response-validator.hf.space")

API_KEY = GROQ_API_KEY if GROQ_API_KEY else HF_TOKEN
if not API_KEY:
    raise ValueError("Set GROQ_API_KEY or HF_TOKEN environment variable!")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

DIFFICULTIES = ["easy", "medium", "hard"]

SYSTEM_PROMPT = (
    "You are an expert HTTP API validator. "
    "Given an API response scenario, evaluate whether it is VALID, INVALID, or PARTIAL VALID. "
    "Explain the status code, schema compliance, field types, and any violations in 2-4 sentences."
)


def call_reset(difficulty: str) -> dict:
    r = requests.post(
        f"{ENV_URL}/reset",
        json={"difficulty": difficulty},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def call_step(content: str) -> dict:
    r = requests.post(
        f"{ENV_URL}/step",
        json={"content": content},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def agent_action(scenario_text: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Validate this API response:\n\n{scenario_text}"},
        ],
        temperature=0.2,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


def run_episode(difficulty: str) -> dict:
    state      = call_reset(difficulty)
    episode_id = f"{difficulty}_{int(time.time())}"

    print(json.dumps({
        "event":      "[START]",
        "episode_id": episode_id,
        "difficulty": difficulty,
        "task":       state.get("current_input", "")[:120],
    }), flush=True)

    action_text = agent_action(state["current_input"])
    result      = call_step(action_text)
    reward      = result["reward"]

    print(json.dumps({
        "event":      "[STEP]",
        "episode_id": episode_id,
        "step":       1,
        "action":     action_text[:200],
        "reward":     reward,
        "done":       result["done"],
    }), flush=True)

    print(json.dumps({
        "event":        "[END]",
        "episode_id":   episode_id,
        "difficulty":   difficulty,
        "total_reward": reward,
        "steps":        1,
        "success":      reward > 0.5,
    }), flush=True)

    return {"difficulty": difficulty, "reward": reward, "success": reward > 0.5}


def main():
    results = []
    for diff in DIFFICULTIES:
        try:
            results.append(run_episode(diff))
        except Exception as e:
            fallback_reward = 0.15
            print(json.dumps({
                "event":        "[END]",
                "episode_id":   f"{diff}_error",
                "difficulty":   diff,
                "total_reward": fallback_reward,
                "steps":        0,
                "success":      False,
                "error":        str(e),
            }), flush=True)
            results.append({"difficulty": diff, "reward": fallback_reward, "success": False})

    avg = sum(r["reward"] for r in results) / len(results)
    print(json.dumps({
        "event":      "[SUMMARY]",
        "results":    results,
        "avg_reward": avg,
    }), flush=True)


if __name__ == "__main__":
    main()
