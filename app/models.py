from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any, Dict


def clamp_reward(v: float) -> float:
    """Strictly clamp reward to open interval (0, 1)."""
    return max(0.001, min(0.999, float(v)))


class Action(BaseModel):
    content: str = Field(..., description="Agent evaluation / verdict text")


class State(BaseModel):
    difficulty: str = Field(..., description="easy | medium | hard")
    step_count: int = Field(0)
    current_input: str = Field(..., description="API response scenario text shown to agent")
    ground_truth: Optional[str] = Field(None, description="Expected validation answer")
    done: bool = Field(False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    state: State
    reward: float = Field(..., ge=0.001, le=0.999)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("reward")
    @classmethod
    def reward_in_range(cls, v: float) -> float:
        return clamp_reward(v)


class ResetRequest(BaseModel):
    difficulty: str = Field("easy", description="easy | medium | hard")
    seed: Optional[int] = None