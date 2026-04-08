---
title: API Response Validator
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
tags:
  - reinforcement-learning
  - openenv
  - api-validation
  - rl-environment
  - meta-pytorch-hackathon
---

# API Response Validator

> OpenEnv-compatible RL environment for the **Meta × PyTorch × Scaler Hackathon – Round 1**

An AI agent learns to validate HTTP API responses against contracts, schema rules,
and REST best practices across three difficulty levels.

---

## Environment Overview

| Property | Value |
|---|---|
| Task type | Text-based API response validation |
| Difficulties | `easy` · `medium` · `hard` |
| Episodes | Single-step |
| Reward range | `(0.001, 0.999)` — strictly open interval |
| Observation | HTTP API response scenario (text) |
| Action | Validation verdict + reasoning (text) |

---

## Tasks

### easytask — Basic API Response Validation
Validate simple HTTP responses: check status codes and basic JSON structure.
- **Grader:** Rule-based keyword matching
- **Reward threshold:** 0.55

### mediumtask — Schema & Business Rule Validation
Identify schema violations, type errors, and REST convention issues.
- **Grader:** Hybrid keyword + structural vocabulary scoring
- **Reward threshold:** 0.50

### hardtask — Complex API Contract Validation
Evaluate payments, batch 207 multi-status, SSE streams, and webhooks.
- **Grader:** LLM-as-judge (Groq `llama-3.1-8b-instant`) with deterministic fallback
- **Reward threshold:** 0.45

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/reset` | Start episode — `{"difficulty": "easy"}` |
| `GET` | `/state` | Current environment state |
| `POST` | `/step` | Submit action — `{"content": "<verdict>"}` |

---

## Action & Observation Spaces

**Observation (State):**
```json
{
  "difficulty": "easy",
  "step_count": 0,
  "current_input": "HTTP 200 OK\nContent-Type: application/json\n{...}",
  "ground_truth": null,
  "done": false,
  "metadata": {"total_scenarios": 5}
}
```

**Action:**
```json
{
  "content": "VALID – status 200 with correct JSON body. Fields user_id, name, email all present and correctly typed."
}
```

**Step Result:**
```json
{
  "state": {...},
  "reward": 0.823,
  "done": true,
  "info": {"difficulty": "easy", "raw_reward": 0.823}
}
```

---

## Local Setup

### Requirements

- Python 3.11+
- Docker (for container builds)
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Environment Variables

```bash
export GROQ_API_KEY="gsk_your_groq_key"
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="hf_your_token"
export ENV_URL="http://localhost:7860"
```

### Run Server Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Run Inference Script

```bash
python inference.py
```

Expected output format:
```json
{"event": "[START]", "episode_id": "easy_1234", "difficulty": "easy", "task": "..."}
{"event": "[STEP]",  "episode_id": "easy_1234", "step": 1, "action": "...", "reward": 0.82, "done": true}
{"event": "[END]",   "episode_id": "easy_1234", "difficulty": "easy", "total_reward": 0.82, "steps": 1, "success": true}
{"event": "[SUMMARY]", "results": [...], "avg_reward": 0.72}
```

### Docker

```bash
docker build -t api-validator .
docker run -p 7860:7860 \
  -e GROQ_API_KEY=gsk_xxx \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.1-8b-instant \
  -e HF_TOKEN=hf_xxx \
  api-validator
```

---

## Project Structure

```
.
├── app/
│   ├── __init__.py       # Package init
│   ├── main.py           # FastAPI endpoints (/reset, /state, /step, /health)
│   ├── models.py         # Pydantic models + reward clamping to (0.001, 0.999)
│   ├── environment.py    # RL environment logic + 15 scenario bank (5 per difficulty)
│   └── graders.py        # easy / medium / hard scoring functions
├── server/
│   └── app.py            # Uvicorn entry point
├── inference.py           # Baseline agent — required for hackathon evaluation
├── openenv.yaml           # OpenEnv specification
├── Dockerfile             # Container config (port 7860)
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Package config
└── README.md              # This file
```

---

## HF Space Secrets

Go to **Settings → Variables and Secrets** and add:

| Name | Value |
|---|---|
| `GROQ_API_KEY` | `gsk_your_groq_key` |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | `llama-3.1-8b-instant` |
| `HF_TOKEN` | `hf_your_token` |

---

## Submission Checklist

- [x] HF Space live — `/health` returns 200
- [x] `/reset` returns valid state for `easy`, `medium`, `hard`
- [x] `/step` returns reward strictly in `(0, 1)` — never `0.0` or `1.0`
- [x] Dockerfile builds and runs on port 7860
- [x] `inference.py` at root with `[START]` / `[STEP]` / `[END]` structured logs
- [x] `openenv.yaml` with 3 tasks and correct reward range
- [x] README with environment description, action/observation spaces, setup instructions

---

## License

MIT
