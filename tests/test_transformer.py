import os

import pytest

from app.models import transformer
from app.models.rules import MOOD_MAP


def test_label_conversion():
    result = transformer.convert(
        [
            {"label": "LABEL_0", "score": 0.1},
            {"label": "LABEL_1", "score": 0.2},
            {"label": "LABEL_2", "score": 0.7},
        ],
        "sentiment",
    )
    assert result.label == "positive"
    assert not result.uncertain
    assert MOOD_MAP["fear"] == "anxious"


def test_mocked_adapter(monkeypatch):
    def pipe(text, **kwargs):
        return [
            {"label": "negative", "score": 0.1},
            {"label": "neutral", "score": 0.2},
            {"label": "positive", "score": 0.7},
        ]

    monkeypatch.setattr(transformer, "load_pipeline", lambda task: pipe)
    result = transformer.infer("good text", "sentiment", explain=True)
    assert len(result.evidence) == 2


def test_invalid_label():
    with pytest.raises(transformer.ModelUnavailable):
        transformer.convert([{"label": "alien", "score": 1}], "sentiment")


@pytest.mark.live
@pytest.mark.skipif(os.getenv("RUN_LIVE_MODELS") != "1", reason="Explicit model download opt-in")
def test_live_models():
    for task in transformer.MODELS:
        result = transformer.infer("I am happy about this great result", task)
        assert result.label in transformer.MODELS[task]["labels"]
        assert abs(sum(result.scores.values()) - 1) < 0.001
