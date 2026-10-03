import pytest

from app.models.rules import emotion, sentiment
from app.services.emoji import recommend
from app.services.preprocessing import explore, model_text


@pytest.mark.parametrize(
    "text,label",
    [
        ("This is great", "positive"),
        ("This is terrible", "negative"),
        ("This is not good", "negative"),
        ("This is not bad", "positive"),
        ("not bad but terrible", "negative"),
        ("😊", "positive"),
        ("The meeting is at nine", "neutral"),
    ],
)
def test_sentiment(text, label):
    assert sentiment(text).label == label


def test_negation_evidence():
    result = sentiment("I am not very happy")
    assert result.evidence[0].negated
    assert result.evidence[0].contribution < 0


def test_intensifier():
    assert (
        sentiment("very good!").evidence[0].contribution
        > sentiment("good").evidence[0].contribution
    )


def test_emotion():
    assert emotion("I am scared").label == "fear"
    assert emotion("not happy").label == "unknown"
    assert emotion("hello").uncertain


def test_preprocess():
    result = explore("I am not running", True, True)
    assert "not" in result["processed_tokens"]
    assert model_text("@someone https://example.com") == "@user http"


def test_emoji():
    assert recommend("birthday birthday", "positive", "joy")[0] == "🎂"
    assert len(set(recommend("birthday", "positive", "joy"))) == 3


def test_mixed_and_code_mixed():
    assert sentiment("good and bad").uncertain
    assert sentiment("bahut अच्छा").uncertain
