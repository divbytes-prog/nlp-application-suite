import math

from app.api.schemas import Evidence, Prediction
from app.core.config import THRESHOLD
from app.services.preprocessing import tokenize

LEXICON = {
    "good": 1.5,
    "great": 2.5,
    "excellent": 3.0,
    "amazing": 2.5,
    "happy": 2.0,
    "love": 2.5,
    "like": 1.0,
    "wonderful": 2.5,
    "best": 2.0,
    "win": 2.0,
    "success": 2.0,
    "profit": 1.5,
    "growth": 1.0,
    "improved": 1.5,
    "enjoy": 2.0,
    "excited": 2.0,
    "hope": 1.0,
    "bad": -1.5,
    "terrible": -3.0,
    "awful": -2.5,
    "sad": -2.0,
    "hate": -2.5,
    "angry": -2.0,
    "poor": -1.5,
    "worst": -3.0,
    "loss": -1.5,
    "decline": -1.5,
    "disappointed": -2.0,
    "fail": -2.0,
    "fear": -1.5,
    "worried": -1.5,
    "😊": 2.0,
    "😄": 2.0,
    "❤️": 2.5,
    "❤": 2.5,
    "😍": 2.5,
    "😢": -2.0,
    "😭": -2.5,
    "😡": -2.5,
}
NEGATIONS = {
    "not",
    "no",
    "never",
    "neither",
    "hardly",
    "without",
    "isn't",
    "wasn't",
    "don't",
    "doesn't",
    "can't",
    "cannot",
    "won't",
    "didn't",
    "couldn't",
}
INTENSIFIERS = {"very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3, "slightly": 0.5}
EMOTION_WORDS = {
    "sadness": {"sad", "cry", "crying", "lonely", "disappointed", "grief", "😢", "😭"},
    "joy": {"happy", "excited", "delighted", "enjoy", "great", "wonderful", "😊", "😄"},
    "love": {"love", "adore", "affection", "cherish", "❤", "😍"},
    "anger": {"angry", "hate", "furious", "annoyed", "rage", "😡"},
    "fear": {"fear", "scared", "afraid", "worried", "anxious", "terrified"},
    "surprise": {"surprised", "surprise", "wow", "unexpected", "shocked", "😮"},
}
EMOTION_LABELS = list(EMOTION_WORDS)
MOOD_MAP = {
    "sadness": "sad",
    "joy": "happy",
    "love": "affectionate",
    "anger": "angry",
    "fear": "anxious",
    "surprise": "surprised",
    "unknown": "unknown",
}


def distribution(values: dict[str, float]) -> dict[str, float]:
    peak = max(values.values())
    exp = {k: math.exp(v - peak) for k, v in values.items()}
    total = sum(exp.values())
    return {k: round(v / total, 6) for k, v in exp.items()}


def negated(tokens: list[str], index: int) -> bool:
    scope = []
    for token in reversed(tokens[max(0, index - 3) : index]):
        if token in {".", "!", "?", ",", ";", "but", "however"}:
            break
        scope.append(token)
    return sum(t in NEGATIONS for t in scope) % 2 == 1


def sentiment(text: str) -> Prediction:
    tokens = tokenize(text)
    evidence = []
    for i, token in enumerate(tokens):
        if token not in LEXICON:
            continue
        score = LEXICON[token]
        previous = tokens[i - 1] if i else ""
        score *= INTENSIFIERS.get(previous, 1.0)
        is_negated = negated(tokens, i)
        if is_negated:
            score *= -0.8
        score *= 1 + min(text.count("!"), 3) * 0.08
        evidence.append(
            Evidence(
                token=token,
                index=i,
                contribution=round(score, 3),
                negated=is_negated,
                rule="lexicon + local modifiers",
            )
        )
    total = sum(e.contribution for e in evidence)
    magnitude = sum(abs(e.contribution) for e in evidence)
    scores = distribution({"negative": -total, "neutral": 1.1 - abs(total), "positive": total})
    label = max(scores, key=lambda k: scores[k])
    confidence = scores[label] if evidence else 0.40
    mixed = magnitude > 0 and abs(total) / magnitude < 0.25
    return Prediction(
        label=label,
        confidence=confidence,
        confidence_kind="uncalibrated rule support",
        uncertain=confidence < THRESHOLD or mixed or not evidence,
        scores=scores,
        evidence=evidence,
        explanation_note=(
            "Matched lexicon contributions, not probabilities. Negation scope: up "
            "to 3 tokens, ending at punctuation or contrast."
        ),
    )


def emotion(text: str) -> Prediction:
    tokens = tokenize(text)
    evidence = []
    hits = {label: 0.0 for label in EMOTION_LABELS}
    for i, token in enumerate(tokens):
        for label, words in EMOTION_WORDS.items():
            if token in words:
                is_negated = negated(tokens, i)
                weight = 0.0 if is_negated else INTENSIFIERS.get(tokens[i - 1] if i else "", 1.0)
                hits[label] += weight
                evidence.append(
                    Evidence(
                        token=token, index=i, contribution=weight, negated=is_negated, rule=label
                    )
                )
    scores = distribution({label: value * 2 for label, value in hits.items()})
    label = max(scores, key=lambda k: scores[k]) if any(hits.values()) else "unknown"
    confidence = scores.get(label, 0.0)
    return Prediction(
        label=label,
        confidence=confidence,
        confidence_kind="uncalibrated rule support",
        uncertain=confidence < THRESHOLD,
        scores=scores,
        evidence=evidence,
        explanation_note=(
            "Keyword matching with negation. Unknown means no emotion evidence; "
            "the baseline has no reliable neutral detector."
        ),
    )
