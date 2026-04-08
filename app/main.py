import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.models import Action, State, StepResult, ResetRequest
from app.environment import APIResponseValidatorEnv

env = APIResponseValidatorEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"[WARNING] Missing env vars: {missing}. Hard grader will use fallback.")
    env.reset("easy")
    yield


app = FastAPI(
    title="API Response Validator",
    description="OpenEnv-compatible environment for validating HTTP API responses.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "api-response-validator"}


@app.post("/reset", response_model=State)
def reset(request: ResetRequest):
    try:
        state = env.reset(difficulty=request.difficulty, seed=request.seed)
        return state
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
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))