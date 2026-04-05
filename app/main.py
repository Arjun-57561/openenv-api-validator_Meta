from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Request

from app.environment import APIResponseValidatorEnv
from app.models import Action, ResetRequest, State, StepResult

app = FastAPI(title="OpenEnv API Response Validator")
_env = APIResponseValidatorEnv()


def _require_config() -> None:
    """Optional at import time; graders need env when hard task runs."""
    _ = os.getenv("API_BASE_URL", "").strip()
    _ = os.getenv("MODEL_NAME", "").strip()
    _ = os.getenv("HF_TOKEN", "").strip()


@app.on_event("startup")
def _startup() -> None:
    _require_config()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset", response_model=State)
async def reset(request: Request) -> State:
    try:
        raw = await request.json()
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    body = ResetRequest.model_validate(raw)
    return _env.reset(difficulty=body.difficulty)


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    try:
        return _env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/state", response_model=State)
def state() -> State:
    return _env.state()
