from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Action(BaseModel):
    content: str = Field(..., min_length=1, description="Agent's evaluation of the API response")


class State(BaseModel):
    difficulty: Literal["easy", "medium", "hard"]
    step_count: int = Field(ge=0)
    current_input: str = Field(..., description="Scenario: API response the agent must validate")
    last_reward: float = Field(default=0.001, gt=0.0, lt=1.0)
    task_name: str
    done: bool = False

    @field_validator("last_reward", mode="before")
    @classmethod
    def clamp_last_reward(cls, v: float) -> float:
        v = float(v)
        if v <= 0.0:
            return 0.001
        if v >= 1.0:
            return 0.999
        return v


class StepResult(BaseModel):
    state: State
    reward: float = Field(default=0.001, gt=0.0, lt=1.0)
    done: bool

    @field_validator("reward", mode="before")
    @classmethod
    def clamp_reward(cls, v: float) -> float:
        v = float(v)
        if v <= 0.0:
            return 0.001
        if v >= 1.0:
            return 0.999
        return v


class ResetRequest(BaseModel):
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None