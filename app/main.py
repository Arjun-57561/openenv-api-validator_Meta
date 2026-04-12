"""FastAPI application — OpenEnv API Response Validator."""
from __future__ import annotations
import os
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from app.models import Action, State, StepResult, ResetRequest
from app.environment import APIResponseValidatorEnv

app = FastAPI(
    title="API Response Validator",
    description="OpenEnv RL environment for HTTP API response validation.",
    version="1.0.0",
)

env = APIResponseValidatorEnv()

OPENENV_YAML = """
name: api_response_validator
version: "1.0.0"
description: >
  OpenEnv environment where an AI agent validates HTTP API responses
  against REST contracts, schema rules, and business logic.

observation_space:
  type: text
  description: "HTTP API response scenario including endpoint, status code, and response body."

action_space:
  type: text
  description: "Agent validation verdict with reasoning (VALID / INVALID / PARTIAL VALID)."

reward_range: [0.01, 0.99]

tasks:
  - id: easy_api_validation_task
    name: "Basic API Response Validation"
    difficulty: easy
    description: "Validate simple HTTP responses — status codes and JSON structure."
    reward_threshold: 0.55

  - id: medium_api_validation_task
    name: "Schema and Business Rule Validation"
    difficulty: medium
    description: "Identify schema violations, type errors, and business rule breaches."
    reward_threshold: 0.50

  - id: hard_api_validation_task
    name: "Complex API Contract Validation"
    difficulty: hard
    description: "Evaluate payments, batch multi-status, SSE streams, and webhook patterns."
    reward_threshold: 0.45

endpoints:
  reset:  "/reset"
  state:  "/state"
  step:   "/step"
  health: "/health"
""".strip()


@app.get("/health")
def health():
    return {"status": "ok", "service": "api-response-validator"}


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def openenv_yaml():
    """Serve the OpenEnv spec file."""
    return OPENENV_YAML


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    return [
        {"id": "easy_api_validation_task",   "difficulty": "easy",   "reward_range": [0.01, 0.99]},
        {"id": "medium_api_validation_task", "difficulty": "medium", "reward_range": [0.01, 0.99]},
        {"id": "hard_api_validation_task",   "difficulty": "hard",   "reward_range": [0.01, 0.99]},
    ]


@app.post("/reset", response_model=State)
async def reset(request: Request):
    difficulty = "easy"
    seed: Optional[int] = None
    try:
        body = await request.body()
        if body:
            data = await request.json()
            if isinstance(data, dict):
                difficulty = data.get("difficulty", "easy") or "easy"
                seed = data.get("seed", None)
    except Exception:
        pass
    return env.reset(difficulty=difficulty, seed=seed)


@app.get("/state", response_model=State)
def get_state():
    return env.get_state()


@app.post("/step", response_model=StepResult)
def step(action: Action):
    return env.step(action)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
