import time

from app.api.schemas import Analysis
from app.models import rules, transformer
from app.services.emoji import recommend

SAFETY_NOTE = (
    "If this text reflects immediate danger, contact someone you trust or "
    "local emergency support. This tool cannot assess risk."
)


def analyze(text: str, engine: str = "rules", explain: bool = False) -> Analysis:
    started = time.perf_counter()
    if engine == "rules":
        sent, emo = rules.sentiment(text), rules.emotion(text)
        model = "custom-rule-baseline-v1"
    else:
        sent = transformer.infer(text, "sentiment", explain)
        emo = transformer.infer(text, "emotion", explain)
        model = " + ".join(str(m["id"]) for m in transformer.MODELS.values())
    warnings = ["Text emotion is an uncertain inference, not a mental-health diagnosis."]
    if sent.uncertain or emo.uncertain:
        warnings.append("Low or conflicting support: treat this result as uncertain.")
    if any(phrase in text.lower() for phrase in ("kill myself", "end my life", "suicide")):
        warnings.append(SAFETY_NOTE)
    return Analysis(
        engine=engine,
        model=model,
        sentiment=sent,
        emotion=emo,
        mood=rules.MOOD_MAP.get(emo.label, "unknown"),
        emojis=recommend(text, sent.label, emo.label),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        warnings=warnings,
    )
