---
title: API Response Validator
emoji: ⚙️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# API Response Validator (OpenEnv)

OpenEnv-compatible environment for validating HTTP API responses. Built with FastAPI and Docker for the Meta x PyTorch x Scaler hackathon.

## Endpoints
- `GET /health` → `{"status":"ok"}`
- `POST /reset` → returns initial task state
- `GET /state` → current environment state
- `POST /step` → submit action, returns reward
