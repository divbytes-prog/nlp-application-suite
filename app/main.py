import json
from collections import Counter
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.schemas import AnalyzeRequest, AnalyzeResponse
from app.core import config
from app.models.transformer import MODELS, ModelUnavailable, load_pipeline
from app.services.batch import export_csv, parse_upload
from app.services.inference import analyze
from app.services.preprocessing import explore


class RequestLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Buffer only up to the cap; count actual bytes even with absent/false Content-Length.
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > config.MAX_REQUEST:
                await JSONResponse({"detail": "Request exceeds size limit"}, 413)(
                    scope, receive, send
                )
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        async def secured_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"x-frame-options", b"DENY"),
                        (b"cache-control", b"no-store"),
                    ]
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, replay, secured_send)


app = FastAPI(
    title="NLP Application Suite",
    version="1.0.0",
    description=(
        "Transparent baselines and transformer comparison. No user text stored by application code."
    ),
)
app.add_middleware(RequestLimitMiddleware)


@app.exception_handler(Exception)
async def unexpected_error(request: Any, exc: Exception) -> JSONResponse:
    # Never include submitted text, exception text, or secrets in responses/logs.
    return JSONResponse({"detail": "Unexpected processing error"}, status_code=500)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "transformers_enabled": config.ENABLE_TRANSFORMERS}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(payload: AnalyzeRequest) -> AnalyzeResponse:
    response = AnalyzeResponse()
    if payload.engine in {"rules", "compare"}:
        response.rules = analyze(payload.text)
    if payload.engine in {"transformer", "compare"}:
        try:
            if not config.ENABLE_TRANSFORMERS:
                raise ModelUnavailable(
                    "Python transformers are disabled on this server. Use the browser "
                    "transformer or enable the local ML extra."
                )
            response.transformer = analyze(payload.text, "transformer", payload.explain)
        except ModelUnavailable as exc:
            if payload.engine == "transformer":
                raise HTTPException(503, str(exc)) from exc
            response.transformer_error = str(exc)
    if response.rules and response.transformer:
        response.disagreement = (
            response.rules.sentiment.label != response.transformer.sentiment.label
            or response.rules.emotion.label != response.transformer.emotion.label
        )
    return response


@app.post("/batch")
async def batch_endpoint(
    file: Annotated[UploadFile, File()],
    column: Annotated[str, Form()] = "text",
    output: Annotated[str, Form()] = "json",
) -> Response:
    if output not in {"json", "csv"}:
        raise HTTPException(422, "Output must be json or csv")
    content = await file.read(config.MAX_UPLOAD + 1)
    await file.close()
    try:
        texts = parse_upload(content, file.filename or "", column)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, "Invalid upload: " + str(exc).split("\n")[0]) from exc
    rows = []
    for i, text in enumerate(texts):
        result = analyze(text)
        rows.append(
            {
                "row": i + 1,
                "text": text,
                "sentiment": result.sentiment.label,
                "emotion": result.emotion.label,
                "confidence": result.sentiment.confidence,
                "uncertain": result.sentiment.uncertain or result.emotion.uncertain,
            }
        )
    if output == "csv":
        return Response(
            export_csv(rows),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
        )
    return JSONResponse(
        {
            "engine": "rules",
            "rows": rows,
            "distribution": dict(Counter(str(r["sentiment"]) for r in rows)),
        }
    )


@app.get("/models")
def models() -> dict[str, object]:
    return {
        "baseline": "custom-rule-baseline-v1",
        "transformers_enabled": config.ENABLE_TRANSFORMERS,
        "loaded_count": load_pipeline.cache_info().currsize,
        "models": MODELS,
        "mood_mapping": {
            "sadness": "sad",
            "joy": "happy",
            "love": "affectionate",
            "anger": "angry",
            "fear": "anxious",
            "surprise": "surprised",
        },
        "threshold": config.THRESHOLD,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    path = config.ROOT / "reports" / "metrics.json"
    if not path.exists():
        return {"status": "Not yet measured", "runs": []}
    return dict(json.loads(path.read_text()))


@app.post("/preprocess")
def preprocess(
    payload: AnalyzeRequest,
    remove_stopwords: bool = False,
    stemming: bool = False,
    n: int = Query(2, ge=1, le=3),
) -> dict[str, object]:
    return explore(payload.text, remove_stopwords, stemming, n)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.ROOT / "public" / "index.html")


app.mount("/static", StaticFiles(directory=config.ROOT / "public"), name="static")
app.mount("/figures", StaticFiles(directory=config.ROOT / "reports" / "figures"), name="figures")
