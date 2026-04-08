from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


def clamp_reward(v: float) -> float:
    """Strictly clamp to open interval (0, 1) — never exactly 0.0 or 1.0."""
    return round(max(0.01, min(0.99, float(v))), 6)


class Action(BaseModel):
    content: str = Field(..., description="Agent validation verdict text")


class State(BaseModel):
    difficulty: str = Field("easy")
    step_count: int = Field(0)
    current_input: str = Field("")
    ground_truth: Optional[str] = Field(None)
    done: bool = Field(False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    state: State
    reward: float = Field(...)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("reward")
    @classmethod
    def reward_must_be_open(cls, v: float) -> float:
        return clamp_reward(v)


class ResetRequest(BaseModel):
    difficulty: str = Field("easy")
    seed: Optional[int] = Field(None)
