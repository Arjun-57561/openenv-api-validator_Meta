"""FastAPI application — OpenEnv API Response Validator."""
from __future__ import annotations
import os
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from app.models import Action, State, StepResult, ResetRequest
from app.environment import APIResponseValidatorEnv

app = FastAPI(
    title="API Response Validator",
    description="OpenEnv RL environment for HTTP API response validation.",
    version="1.0.0",
)

env = APIResponseValidatorEnv()

OPENENV_YAML = """
spec_version: 1
name: api_response_validator
type: space
runtime: fastapi
app: app.main:app
port: 7860
""".strip()

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>API Response Validator — OpenEnv</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
    }
    header {
      background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
      border-bottom: 1px solid #2d3748;
      padding: 24px 40px;
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .logo { font-size: 2rem; }
    .header-text h1 { font-size: 1.6rem; font-weight: 700; color: #fff; }
    .header-text p { font-size: 0.9rem; color: #718096; margin-top: 4px; }
    .badge {
      display: inline-block;
      background: #22543d;
      color: #68d391;
      border: 1px solid #276749;
      border-radius: 999px;
      padding: 3px 12px;
      font-size: 0.75rem;
      font-weight: 600;
      margin-left: 12px;
      vertical-align: middle;
    }
    main { max-width: 1100px; margin: 0 auto; padding: 40px 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 36px; }
    .card {
      background: #1a1f2e;
      border: 1px solid #2d3748;
      border-radius: 12px;
      padding: 24px;
      transition: border-color 0.2s;
    }
    .card:hover { border-color: #4a5568; }
    .card-icon { font-size: 2rem; margin-bottom: 12px; }
    .card h3 { font-size: 1rem; font-weight: 600; color: #fff; margin-bottom: 6px; }
    .card p { font-size: 0.85rem; color: #718096; line-height: 1.5; }
    .difficulty {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
      margin-bottom: 8px;
    }
    .easy   { background: #1c4532; color: #68d391; }
    .medium { background: #744210; color: #f6ad55; }
    .hard   { background: #742a2a; color: #fc8181; }
    .endpoints { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; padding: 28px; margin-bottom: 36px; }
    .endpoints h2 { font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: 20px; }
    .endpoint-row {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 0;
      border-bottom: 1px solid #2d3748;
    }
    .endpoint-row:last-child { border-bottom: none; }
    .method {
      font-size: 0.7rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      min-width: 46px;
      text-align: center;
    }
    .get  { background: #1a365d; color: #63b3ed; }
    .post { background: #1c4532; color: #68d391; }
    .path { font-family: monospace; font-size: 0.9rem; color: #e2e8f0; flex: 1; }
    .desc { font-size: 0.8rem; color: #718096; }
    .try-btn {
      background: #2b6cb0;
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 5px 14px;
      font-size: 0.8rem;
      cursor: pointer;
      text-decoration: none;
    }
    .try-btn:hover { background: #3182ce; }
    .stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 36px; }
    .stat-card {
      flex: 1;
      min-width: 150px;
      background: #1a1f2e;
      border: 1px solid #2d3748;
      border-radius: 12px;
      padding: 20px;
      text-align: center;
    }
    .stat-card .num { font-size: 2rem; font-weight: 700; color: #63b3ed; }
    .stat-card .label { font-size: 0.8rem; color: #718096; margin-top: 4px; }
    footer {
      text-align: center;
      padding: 24px;
      color: #4a5568;
      font-size: 0.8rem;
      border-top: 1px solid #2d3748;
      margin-top: 40px;
    }
    a { color: #63b3ed; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <header>
    <div class="logo">🔍</div>
    <div class="header-text">
      <h1>API Response Validator <span class="badge">● RUNNING</span></h1>
      <p>OpenEnv RL Environment — Meta PyTorch Hackathon 2026 · Built by Bontha Mallikarjun Reddy</p>
    </div>
  </header>

  <main>
    <div class="stats">
      <div class="stat-card"><div class="num">3</div><div class="label">Tasks</div></div>
      <div class="stat-card"><div class="num">(0,1)</div><div class="label">Reward Range</div></div>
      <div class="stat-card"><div class="num">5</div><div class="label">Scenarios / Task</div></div>
      <div class="stat-card"><div class="num">v1.0</div><div class="label">OpenEnv Spec</div></div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-icon">🟢</div>
        <span class="difficulty easy">EASY</span>
        <h3>Basic API Response Validation</h3>
        <p>Validate simple HTTP responses — status codes and JSON structure matching expected schema.</p>
      </div>
      <div class="card">
        <div class="card-icon">🟡</div>
        <span class="difficulty medium">MEDIUM</span>
        <h3>Schema & Business Rule Validation</h3>
        <p>Identify schema violations, type errors, null fields, and business logic breaches.</p>
      </div>
      <div class="card">
        <div class="card-icon">🔴</div>
        <span class="difficulty hard">HARD</span>
        <h3>Complex API Contract Validation</h3>
        <p>Evaluate payments (3DS), batch multi-status (207), SSE streams, and webhook patterns.</p>
      </div>
    </div>

    <div class="endpoints">
      <h2>🔌 API Endpoints</h2>
      <div class="endpoint-row">
        <span class="method get">GET</span>
        <span class="path">/health</span>
        <span class="desc">Health check</span>
        <a class="try-btn" href="/health" target="_blank">Try</a>
      </div>
      <div class="endpoint-row">
        <span class="method get">GET</span>
        <span class="path">/tasks</span>
        <span class="desc">List all tasks</span>
        <a class="try-btn" href="/tasks" target="_blank">Try</a>
      </div>
      <div class="endpoint-row">
        <span class="method get">GET</span>
        <span class="path">/openenv.yaml</span>
        <span class="desc">OpenEnv spec</span>
        <a class="try-btn" href="/openenv.yaml" target="_blank">Try</a>
      </div>
      <div class="endpoint-row">
        <span class="method post">POST</span>
        <span class="path">/reset</span>
        <span class="desc">Reset environment — body: {"difficulty": "easy|medium|hard"}</span>
        <a class="try-btn" href="/docs#/default/reset_reset_post" target="_blank">Docs</a>
      </div>
      <div class="endpoint-row">
        <span class="method post">POST</span>
        <span class="path">/step</span>
        <span class="desc">Take a step — body: {"content": "your validation text"}</span>
        <a class="try-btn" href="/docs#/default/step_step_post" target="_blank">Docs</a>
      </div>
      <div class="endpoint-row">
        <span class="method get">GET</span>
        <span class="path">/state</span>
        <span class="desc">Get current state</span>
        <a class="try-btn" href="/state" target="_blank">Try</a>
      </div>
      <div class="endpoint-row">
        <span class="method get">GET</span>
        <span class="path">/docs</span>
        <span class="desc">Interactive Swagger UI</span>
        <a class="try-btn" href="/docs" target="_blank">Open</a>
      </div>
    </div>
  </main>

  <footer>
    OpenEnv API Response Validator &mdash; Meta PyTorch Hackathon 2026 &mdash;
    <a href="https://github.com/Arjun-57561/openenv-api-validator_Meta" target="_blank">GitHub</a>
  </footer>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
def health():
    return {"status": "ok", "service": "api-response-validator"}


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def openenv_yaml():
    return OPENENV_YAML


@app.get("/tasks")
def list_tasks():
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
