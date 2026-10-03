from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import MAX_TEXT


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(
        min_length=1, max_length=MAX_TEXT, examples=["I am not happy with this result."]
    )
    engine: Literal["rules", "transformer", "compare"] = "rules"
    explain: bool = False

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain visible characters")
        return value


class Evidence(BaseModel):
    token: str
    index: int
    contribution: float
    negated: bool = False
    rule: str


class Prediction(BaseModel):
    label: str
    confidence: float
    confidence_kind: str
    uncertain: bool
    scores: dict[str, float]
    evidence: list[Evidence] = []
    explanation_note: str


class Analysis(BaseModel):
    engine: str
    model: str
    sentiment: Prediction
    emotion: Prediction
    mood: str
    emojis: list[str]
    latency_ms: float
    warnings: list[str]


class AnalyzeResponse(BaseModel):
    rules: Analysis | None = None
    transformer: Analysis | None = None
    transformer_error: str | None = None
    disagreement: bool | None = None
