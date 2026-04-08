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
  - meta-pytorch-hackathon
---

# API Response Validator

OpenEnv-compatible RL environment for the **Meta × PyTorch × Scaler Hackathon – Round 1**.

An AI agent learns to validate HTTP API responses against contracts, schema rules,
and REST best practices across three difficulty levels.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start episode (body optional, defaults to easy) |
| `GET` | `/state` | Current state |
| `POST` | `/step` | Submit action |

## Local Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Run Inference

```bash
export GROQ_API_KEY="gsk_xxx"
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="hf_xxx"
export ENV_URL="http://localhost:7860"
python inference.py
```

## HF Space Secrets

| Name | Value |
|---|---|
| `GROQ_API_KEY` | `gsk_your_groq_key` |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | `llama-3.1-8b-instant` |
| `HF_TOKEN` | `hf_your_token` |
