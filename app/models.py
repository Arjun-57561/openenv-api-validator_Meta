from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

MIN_REWARD = 0.001
MAX_REWARD = 0.95


def _clamp_reward_value(value: float) -> float:
    try:
        reward = float(value)
    except Exception:
        return MIN_REWARD

    if reward <= 0.0:
        return MIN_REWARD
    if reward >= 1.0:
        return MAX_REWARD
    return reward


class Action(BaseModel):
    content: str = Field(..., min_length=1, description="Agent's evaluation of the API response")


class State(BaseModel):
    model_config = ConfigDict(validate_default=True)

    difficulty: Literal["easy", "medium", "hard"]
    step_count: int = Field(ge=0)
    current_input: str = Field(..., description="Scenario: API response the agent must validate")
    last_reward: float = Field(default=MIN_REWARD, gt=0.0, lt=1.0)
    task_name: str
    done: bool = False

    @field_validator("last_reward", mode="before")
    @classmethod
    def clamp_last_reward(cls, value: float) -> float:
        return _clamp_reward_value(value)


class StepResult(BaseModel):
    model_config = ConfigDict(validate_default=True)

    state: State
    reward: float = Field(default=MIN_REWARD, gt=0.0, lt=1.0)
    done: bool

    @field_validator("reward", mode="before")
    @classmethod
    def clamp_reward(cls, value: float) -> float:
        return _clamp_reward_value(value)


class ResetRequest(BaseModel):
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None