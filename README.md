---
title: API Response Validator
emoji: ⚙️
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "1.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# API Response Validator

OpenEnv submission: environment that validates JSON/API responses against a spec.


# API Response Validator (OpenEnv)

OpenEnv-compatible environment where a text agent reviews synthetic HTTP API responses for contract fit, schema risk, and documentation consistency. Rewards are in **[0.0, 1.0]** with partial credit on every difficulty tier.

## Environment overview

The setting mirrors common backend integration work: given status codes, headers, and JSON bodies (or short doc excerpts), the agent must produce a concrete validation verdict. **Easy** tasks stress factual checks (status, required fields). **Medium** tasks add nested JSON and null semantics. **Hard** tasks require reasoning about privacy, undocumented fields, and doc–sample alignment, including an **LLM-as-judge** scorer with rule-based fallbacks.

## Observation space

- **Type:** text  
- **Content:** A single scenario string describing the response (and sometimes surrounding documentation) the agent must evaluate.

## Action space

- **Type:** text  
- **Content:** The agent’s validation write-up (findings, risks, and whether the contract is satisfied). Submitted as JSON `{"content": "<verdict>"}` to `POST /step`.

## Tasks

| Difficulty | Task name    | What the agent should do |
|------------|--------------|---------------------------|
| Easy       | `easy_task`  | Confirm HTTP status, JSON nature, and presence of minimal fields (`id`, `name`) for a 200 user profile payload. |
| Medium     | `medium_task`| Call out nested `null` / schema hazards (e.g. `user.address.zip`) and client breakage risks. |
| Hard       | `hard_task`  | Compare docs to a sample error body; discuss `request_id` logging risk and undocumented `retry_after`. |

## Reward function

- **Easy:** Weighted rule-based checks (status, JSON awareness, field mentions, positive verdict for success, penalty for false errors).  
- **Medium:** Keyword coverage plus bonuses for nested/null/schema language; short answers are penalized.  
- **Hard:** Primary LLM judge (JSON `score` in \[0, 1\]); on API failure, a **variable** lexical overlap + keyword heuristic applies (not a flat constant).  

All returned rewards are clipped to **[0.0, 1.0]** with granular floats.

## Setup (local)

```bash
cd openenv-api-validator
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.3-70b-versatile
set HF_TOKEN=<your_groq_api_key>
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

> **Note:** The variable name `HF_TOKEN` matches the hackathon template. For Groq, set it to your **Groq API key**.

## Hugging Face Spaces (Docker)

1. Create a **Docker** Space and push this repository (see deployment commands below).  
2. In **Settings → Variables and secrets**, add:

| Name           | Value |
|----------------|--------|
| `API_BASE_URL` | `https://api.groq.com/openai/v1` |
| `MODEL_NAME`   | `llama-3.3-70b-versatile` |
| `HF_TOKEN`     | Your Groq API key |

3. Space listens on **7860** (default for this image).

## Environment variables

| Variable        | Purpose |
|-----------------|--------|
| `API_BASE_URL`  | OpenAI-compatible base URL (e.g. Groq). |
| `MODEL_NAME`    | Chat model id for graders/agent. |
| `HF_TOKEN`      | API key for that endpoint (Groq key when using Groq). |
| `SPACE_URL`     | *(inference only)* Base URL of the running Space or local server (default `http://127.0.0.1:7860`). |

## Running `inference.py`

With the API server running locally or your Space URL set:

```bash
set SPACE_URL=http://127.0.0.1:7860
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.3-70b-versatile
set HF_TOKEN=<your_groq_api_key>
python inference.py
```

Against a live Space:

```bash
set SPACE_URL=https://Asharjun-<your-space-name>.hf.space
python inference.py
```

Stdout uses the required **`[START]` / `[STEP]` / `[END]`** lines (JSON payloads), running **easy → medium → hard** sequentially. No GPU is required; runtime is a few API calls per task.

## API

- `GET /health` → `{"status":"ok"}`  
- `POST /reset` → optional body `{"difficulty":"easy"|"medium"|"hard"}`; returns `State`  
- `POST /step` → body `{"content":"..."}`; returns `{state, reward, done}`  
- `GET /state` → current `State`  

## Author

- **Hugging Face:** [Asharjun](https://huggingface.co/Asharjun)  
- **GitHub:** [Arjun-57561](https://github.com/Arjun-57561)

## Deploy (GitHub + Hugging Face Docker Space)

Replace `api-response-validator` if you use a different Space name.

```bash
cd openenv-api-validator
git init
git add .
git commit -m "Initial OpenEnv API Response Validator submission"
git branch -M main
git remote add origin https://github.com/Arjun-57561/openenv-api-validator.git
git push -u origin main
```

Create a **Docker** Space at [huggingface.co/new-space](https://huggingface.co/new-space), owner **Asharjun**, name e.g. `api-response-validator`, then:

```bash
git remote add space https://huggingface.co/spaces/Asharjun/api-response-validator
git push space main
```

Verify (after the build finishes):

```bash
curl https://Asharjun-api-response-validator.hf.space/health
curl -X POST https://Asharjun-api-response-validator.hf.space/reset
```

If your Space slug differs, the runtime URL is typically `https://Asharjun-<space-slug>.hf.space`.

## License

Submit for the Meta × PyTorch × Scaler OpenEnv hackathon per organizer terms.
