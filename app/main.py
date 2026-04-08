import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.models import Action, State, StepResult, ResetRequest
from app.environment import APIResponseValidatorEnv

env = APIResponseValidatorEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [k for k in ["HF_TOKEN", "API_BASE_URL", "MODEL_NAME"] if not os.environ.get(k)]
    if missing:
        print(f"[WARNING] Missing env vars: {missing}. Hard grader will use deterministic fallback.")
    env.reset("easy")
    yield


app = FastAPI(
    title="API Response Validator",
    description="OpenEnv-compatible RL environment for HTTP API response validation.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "api-response-validator"}


@app.post("/reset", response_model=State)
def reset(request: ResetRequest):
    try:
        return env.reset(difficulty=request.difficulty, seed=request.seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=State)
def get_state():
    try:
        return env.get_state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    try:
        return env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
