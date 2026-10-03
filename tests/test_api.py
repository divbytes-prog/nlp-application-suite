import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_and_static():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200
    assert client.get("/models").status_code == 200
    assert "runs" in client.get("/metrics").json()


@pytest.mark.parametrize("text", ["", "   ", "x" * 5001])
def test_validation(text):
    assert client.post("/analyze", json={"text": text}).status_code == 422


def test_unicode():
    response = client.post("/analyze", json={"text": "आज good 😊 @user https://example.com"})
    assert response.status_code == 200
    assert response.json()["rules"]["sentiment"]["label"] == "positive"


def test_compare_honest_unavailable():
    result = client.post("/analyze", json={"text": "good", "engine": "compare"}).json()
    assert result["rules"] is not None
    assert result["transformer"] is None
    assert result["transformer_error"]
    assert (
        client.post("/analyze", json={"text": "good", "engine": "transformer"}).status_code == 503
    )


def test_batch_and_formula():
    result = client.post(
        "/batch",
        files={"file": ("test.csv", b'text\n"=SUM(1,2)"\ngood\n', "text/csv")},
        data={"output": "csv"},
    )
    assert result.status_code == 200
    assert "'=SUM" in result.text
    assert 'filename="predictions.csv"' in result.headers["content-disposition"]


@pytest.mark.parametrize(
    "name,content,column",
    [
        ("a.exe", b"bad", "text"),
        ("a.csv", b"wrong\nx", "text"),
        ("a.txt", b"\xff", "text"),
        ("a.txt", b"x\x00x", "text"),
        ("a.csv", b'text\n"unterminated', "text"),
        ("a.txt", b"good\n" * 101, "text"),
        ("a.csv", b"text\na,b\n", "text"),
    ],
)
def test_invalid_batch(name, content, column):
    assert (
        client.post("/batch", files={"file": (name, content)}, data={"column": column}).status_code
        == 422
    )


def test_size_limit():
    assert client.post("/analyze", content=b"x" * 1_100_001).status_code == 413


def test_preprocess_api():
    result = client.post("/preprocess?remove_stopwords=true&n=2", json={"text": "I am not good"})
    assert "not" in result.json()["processed_tokens"]
