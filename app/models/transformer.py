from functools import lru_cache
from typing import Any, TypedDict

from app.api.schemas import Evidence, Prediction
from app.core.config import THRESHOLD
from app.models.rules import EMOTION_LABELS
from app.services.preprocessing import model_text, tokenize


class ModelSpec(TypedDict):
    id: str
    revision: str
    labels: list[str]
    license: str


# Revisions are populated from the official Hugging Face API during source preparation.
MODELS: dict[str, ModelSpec] = {
    "sentiment": {
        "id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "revision": "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
        "labels": ["negative", "neutral", "positive"],
        "license": "cc-by-4.0",
    },
    "emotion": {
        "id": "bhadresh-savani/distilbert-base-uncased-emotion",
        "revision": "ce6f4ffcde7642ca2cac02381a16da38e5498ff7",
        "labels": EMOTION_LABELS,
        "license": "apache-2.0",
    },
}


class ModelUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=2)
def load_pipeline(task: str) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ModelUnavailable("Install the ml extra to use Python transformers.") from exc
    spec = MODELS[task]
    try:
        return pipeline(
            "text-classification",
            model=spec["id"],
            tokenizer=spec["id"],
            revision=spec["revision"],
            device=-1,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise ModelUnavailable(
            "Model download or loading failed. Check network and cache."
        ) from exc


def convert(rows: list[dict[str, Any]], task: str) -> Prediction:
    labels = MODELS[task]["labels"]
    scores = {}
    for row in rows:
        label = str(row["label"]).lower()
        if label.startswith("label_"):
            label = labels[int(label.split("_")[-1])]
        if label not in labels:
            raise ModelUnavailable("Unexpected model label; check pinned configuration.")
        scores[label] = float(row["score"])
    if set(scores) != set(labels):
        raise ModelUnavailable("Incomplete model score distribution.")
    label = max(scores, key=lambda k: scores[k])
    return Prediction(
        label=label,
        confidence=scores[label],
        confidence_kind="uncalibrated softmax",
        uncertain=scores[label] < THRESHOLD,
        scores=scores,
        explanation_note=(
            "Token deletion shows sensitivity, not causal attribution. Softmax "
            "confidence can be overconfident under domain shift."
        ),
    )


def infer(text: str, task: str, explain: bool = False) -> Prediction:
    pipe = load_pipeline(task)
    prepared = model_text(text)
    result = convert(pipe(prepared, top_k=None, truncation=True, max_length=256), task)
    if explain:
        words = prepared.split()[:16]
        # Bounded to 16 word positions; deletion changes context and may create unnatural sentences.
        for i, token in enumerate(words):
            variant = " ".join(prepared.split()[:i] + prepared.split()[i + 1 :]) or " "
            altered = convert(pipe(variant, top_k=None, truncation=True, max_length=256), task)
            delta = result.confidence - altered.scores[result.label]
            result.evidence.append(
                Evidence(
                    token=token,
                    index=i,
                    contribution=round(delta, 5),
                    rule="token deletion sensitivity",
                )
            )
    if len(tokenize(text)) > 256:
        result.explanation_note += " Input may be truncated to 256 subword tokens."
    return result
