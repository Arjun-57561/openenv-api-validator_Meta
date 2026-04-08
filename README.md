# API Response Validator

> OpenEnv-compatible RL environment for the **Meta × PyTorch × Scaler Hackathon – Round 1**

An AI agent learns to validate HTTP API responses against contracts, schema rules, and REST best practices across three difficulty levels.

---

## Live Space

🤗 [Asharjun/api-response-validator](https://huggingface.co/spaces/Asharjun/api-response-validator)

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

### `easytask` — Basic API Response Validation
Validate simple HTTP responses: check status codes (200/201/404/500) and basic JSON body structure.
- **Grader:** Rule-based keyword matching
- **Reward threshold:** 0.55

### `mediumtask` — Schema & Business Rule Validation
Identify schema violations (null required fields, negative prices), type errors, and REST convention issues.
- **Grader:** Hybrid keyword + structural vocabulary scoring
- **Reward threshold:** 0.50

### `hardtask` — Complex API Contract Validation
Evaluate advanced API patterns: payment responses, batch 207 multi-status, SSE streams, webhooks.
- **Grader:** LLM-as-judge (HF Inference API) with deterministic fallback
- **Reward threshold:** 0.45

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start new episode — body: `{"difficulty": "easy"}` |
| `GET` | `/state` | Current environment state |
| `POST` | `/step` | Submit action — body: `{"content": "<verdict>"}` |

### Example

```bash
# Reset
curl -X POST https://asharjun-api-response-validator.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Step
curl -X POST https://asharjun-api-response-validator.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"content": "VALID – status 200 with correct JSON schema."}'
```

---

## Local Setup

### Requirements
- Python 3.11+
- Docker

### Environment Variables

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="http://localhost:7860"
```

### Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Run inference

```bash
python inference.py
```

### Docker

```bash
docker build -t api-validator .
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_xxx \
  -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
  -e API_BASE_URL=https://api-inference.huggingface.co/v1 \
  api-validator
```

---

## Project Structure
