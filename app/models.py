from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    """Agent submits a free-text validation verdict (structured prose is fine)."""
    content: str = Field(..., min_length=1, description="Agent's evaluation of the API response")


class State(BaseModel):
    difficulty: Literal["easy", "medium", "hard"]
    step_count: int = Field(ge=0)
    current_input: str = Field(..., description="Scenario: API response the agent must validate")
    last_reward: float = Field(gt=0.0, lt=1.0)
    task_name: str
    done: bool = False


class StepResult(BaseModel):
    state: State
    reward: float = Field(gt=0.0, lt=1.0)
    done: bool


class ResetRequest(BaseModel):
    """Optional: force difficulty for scripted runs (e.g. inference baseline)."""
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None