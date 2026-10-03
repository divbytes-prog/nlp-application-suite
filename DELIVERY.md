# Delivery report — NLP Application Suite

This is a newly AI-assisted project prepared for Divyansh Singh from the supplied handoff. The source is published in the private GitHub repository [divbytes-prog/nlp-application-suite](https://github.com/divbytes-prog/nlp-application-suite). Vercel hosting was deferred by the user.

## Implemented features

- Three-class rule and pinned RoBERTa sentiment analysis.
- Six-class rule and pinned DistilBERT emotion analysis.
- Negation/modifier evidence, uncertainty flags, score distributions, latency and token-deletion explanations.
- Deterministic emoji recommendation, optional preprocessing stages and preserved negation.
- Validated UTF-8 CSV/TXT batches, column selection, aggregates and safe CSV exports.
- Responsive web interface with Python comparison and optional quantized browser sentiment.
- FastAPI health, analyze, batch, models, metrics and preprocessing endpoints plus OpenAPI.
- Download/evaluation/export scripts, fixed splits and revisions, normalized duplicate audits, class metrics, confusion matrices, ECE/reliability plots, error filters and observed qualitative probes.
- Tests, lint/type configuration, CI, container definition, dependency lock, README and model/data cards.

## Models and data

Python sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`, revision `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`, CC BY 4.0. Python emotion: `bhadresh-savani/distilbert-base-uncased-emotion`, revision `ce6f4ffcde7642ca2cac02381a16da38e5498ff7`, Apache 2.0. Browser sentiment: pinned Xenova quantized ONNX conversion, upstream CC BY 4.0 with no independent conversion license declaration; browser runtime unverified.

Sentiment data: Financial PhraseBank all-agree, CC BY-NC-SA 3.0, cleaned 70/15/15 label-stratified split with seed 42. Emotion data: DAIR-AI Emotion, educational/research restriction (`other` license), official split cleaned of normalized duplicates. Exact sources, revisions and citations are in DATA_CARD.md and MODEL_CARD.md. No raw benchmark text, credentials, private academic files, telephone numbers or handoff file are included.

## Genuine evaluation results

| Task | Engine | Test examples | Accuracy | Macro F1 | Weighted F1 | Scope |
|---|---|---:|---:|---:|---:|---|
| sentiment | rules | 340 | 0.6765 | 0.4289 | 0.6081 | full held-out test |
| sentiment | transformer | 340 | 0.7176 | 0.5831 | 0.6681 | full held-out test |
| emotion | rules | 1986 | 0.0755 | 0.1416 | 0.1313 | full held-out test |
| emotion | transformer | 1986 | 0.9325 | 0.8904 | 0.9322 | full held-out test |

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,ml]' --extra-index-url https://download.pytorch.org/whl/cpu
python -m scripts.download_data
OMP_NUM_THREADS=2 python -m scripts.evaluate --engine both --task both
python -m scripts.export_results
ENABLE_TRANSFORMERS=1 OMP_NUM_THREADS=2 uvicorn app.main:app --no-access-log
```

Windows PowerShell equivalents are in README.md. The rule API and UI need only the base dependencies.

## Checks and open items

- 39 offline tests passed.
- One optional live test covering both model heads passed.
- Ruff lint and format checks passed.
- MyPy passed on 18 source files.
- Python compile checks and JavaScript syntax checks passed.
- Both full cleaned held-out benchmarks completed.
- The live comparison/explanation API check is recorded separately in verification.json.
- Dockerfile and CI are present; Docker build and remote CI were not executed here (Docker is unavailable).
- Browser visual QA and screenshots are incomplete: Chromium launch is blocked by runtime socket restrictions. Optional browser ONNX inference is not claimed verified.
- Full source, evaluation reports and CI configuration are included in the private GitHub repository. Remote CI completion is not yet verified.
- Vercel is deliberately deferred; no deployment URL exists.
- Rule emotion is a weak baseline (7.55% accuracy); this result is disclosed rather than tuned on the test set.
- These models are English research tools, not diagnosis or risk assessment.

The project must not be described as fully complete against every handoff acceptance criterion until the above checks and publishing are done.

## Truthful professor-facing summary

I prepared an NLP workbench with AI coding assistance that compares a transparent Python rule baseline with pretrained sentiment and emotion classifiers. It includes a FastAPI service, a web interface, documented data restrictions, reproducible evaluation, uncertainty and error analysis. On the cleaned held-out experiments, sentiment RoBERTa achieved macro F1 0.5831 on 340 Financial PhraseBank examples and emotion DistilBERT achieved macro F1 0.8904 on 1,986 DAIR-AI Emotion examples. These are pretrained-model evaluation results, not claims of new model training or clinical validity.

## Repository files

```text
nlp-application-suite/
  .dockerignore
  .env.example
  .github/workflows/ci.yml
  .gitignore
  DATA_CARD.md
  Dockerfile
  LICENSE
  MODEL_CARD.md
  README.md
  app/__init__.py
  app/api/__init__.py
  app/api/schemas.py
  app/core/__init__.py
  app/core/config.py
  app/main.py
  app/models/__init__.py
  app/models/rules.py
  app/models/transformer.py
  app/services/__init__.py
  app/services/batch.py
  app/services/emoji.py
  app/services/inference.py
  app/services/preprocessing.py
  data/README.md
  data/samples/example.csv
  public/app.js
  public/favicon.svg
  public/index.html
  public/sentiment-worker.js
  public/style.css
  pyproject.toml
  reports/error_analysis.md
  reports/figures/emotion-rules-confusion.png
  reports/figures/emotion-rules-reliability.png
  reports/figures/emotion-transformer-confusion.png
  reports/figures/emotion-transformer-reliability.png
  reports/figures/sentiment-rules-confusion.png
  reports/figures/sentiment-rules-reliability.png
  reports/figures/sentiment-transformer-confusion.png
  reports/figures/sentiment-transformer-reliability.png
  reports/metrics.json
  reports/qualitative_probes.json
  reports/summary.md
  requirements-ml-lock.txt
  requirements.txt
  scripts/__init__.py
  scripts/download_data.py
  scripts/evaluate.py
  scripts/export_results.py
  tests/test_api.py
  tests/test_rules.py
  tests/test_security.py
  tests/test_transformer.py
  vercel.json
  DELIVERY.md
  verification.json
```
