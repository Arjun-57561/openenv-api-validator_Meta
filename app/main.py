"""FastAPI application — OpenEnv API Response Validator."""
from __future__ import annotations
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.models import Action, State, StepResult, ResetRequest
from app.environment import APIResponseValidatorEnv

app = FastAPI(
    title="API Response Validator",
    description="OpenEnv RL environment for HTTP API response validation.",
    version="1.0.0",
)

env = APIResponseValidatorEnv()


@app.get("/health")
def health():
    return {"status": "ok", "service": "api-response-validator"}


@app.post("/reset", response_model=State)
async def reset(request: Request):
    """
    Reset environment.
    Accepts: no body, empty body, or JSON body with optional 'difficulty' and 'seed'.
    Defaults: difficulty=easy, seed=None
    """
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
