"""
Baseline inference script – required for hackathon evaluation.
Emits structured [START] / [STEP] / [END] JSON logs.
Supports Groq API (primary) and HF Token (fallback).
"""
import os
import json
import time
import requests
from openai import OpenAI

# ── config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.1-8b-instant")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "https://asharjun-api-response-validator.hf.space")

API_KEY = GROQ_API_KEY if GROQ_API_KEY else HF_TOKEN

if not API_KEY:
    raise ValueError(
        "No API key found! Set GROQ_API_KEY or HF_TOKEN:\n"
        "  export GROQ_API_KEY=gsk_xxxx"
    )

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

DIFFICULTIES = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert HTTP API validator.
Given an API response scenario, evaluate whether it is valid.
Give a verdict: VALID, INVALID, or PARTIAL VALID.
Explain: status code, schema compliance, field types, and any violations.
Keep your answer to 2-4 sentences."""


# ── helpers ───────────────────────────────────────────────────────────────────

def call_env(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{ENV_URL}{path}"
    r = requests.post(url, json=payload, timeout=30) if method == "POST" else requests.get(url, timeout=30)
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


# ── episode runner ────────────────────────────────────────────────────────────

def run_episode(difficulty: str) -> dict:
    state      = call_env("POST", "/reset", {"difficulty": difficulty})
    episode_id = f"{difficulty}_{int(time.time())}"

    print(json.dumps({
        "event":      "[START]",
        "episode_id": episode_id,
        "difficulty": difficulty,
        "task":       state.get("current_input", "")[:120],
    }))

    action_text = agent_action(state["current_input"])
    result      = call_env("POST", "/step", {"content": action_text})

    print(json.dumps({
        "event":      "[STEP]",
        "episode_id": episode_id,
        "step":       1,
        "action":     action_text[:200],
        "reward":     result["reward"],
        "done":       result["done"],
    }))

    print(json.dumps({
        "event":        "[END]",
        "episode_id":   episode_id,
        "difficulty":   difficulty,
        "total_reward": result["reward"],
        "steps":        1,
        "success":      result["reward"] > 0.5,
    }))

    return {"difficulty": difficulty, "reward": result["reward"], "success": result["reward"] > 0.5}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    results = []
    for diff in DIFFICULTIES:
        try:
            results.append(run_episode(diff))
        except Exception as e:
            fallback = 0.15
            print(json.dumps({
                "event":        "[END]",
                "episode_id":   f"{diff}_error",
                "difficulty":   diff,
                "total_reward": fallback,
                "steps":        0,
                "success":      False,
                "error":        str(e),
            }))
            results.append({"difficulty": diff, "reward": fallback, "success": False})

    print(json.dumps({
        "event":      "[SUMMARY]",
        "results":    results,
        "avg_reward": sum(r["reward"] for r in results) / len(results),
    }))


if __name__ == "__main__":
    main()
