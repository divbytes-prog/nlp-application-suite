"""Frozen rules/models evaluated on held-out data; no test-set tuning."""

import argparse
import json
import platform
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any

from app.core.config import ROOT, THRESHOLD
from app.models import rules, transformer
from scripts.download_data import load


def calibration(y: list[str], predicted: list[str], confidences: list[float]) -> dict[str, Any]:
    bins = []
    ece = 0.0
    for index in range(10):
        indices = [i for i, c in enumerate(confidences) if min(int(c * 10), 9) == index]
        if not indices:
            continue
        confidence = sum(confidences[i] for i in indices) / len(indices)
        accuracy = sum(y[i] == predicted[i] for i in indices) / len(indices)
        ece += len(indices) / len(y) * abs(confidence - accuracy)
        bins.append(
            {
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "count": len(indices),
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "ece": ece,
        "bins": bins,
        "note": "Rule support is not probability; ECE is descriptive only for that baseline.",
    }


def evaluate(task: str, engine: str, limit: int | None = None) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    splits, metadata = load(task)
    rows = splits["test"] if limit is None else splits["test"][:limit]
    if not rows:
        raise ValueError("Test set is empty")
    labels = ["negative", "neutral", "positive"] if task == "sentiment" else rules.EMOTION_LABELS
    if engine == "transformer":
        transformer.load_pipeline(task)
        # Warm-up excluded from inference latency, included separately as a documented behavior.
        transformer.infer(rows[0]["text"], task)
    actual, predicted, confidences, latencies, errors = [], [], [], [], []
    for i, row in enumerate(rows):
        start = time.perf_counter()
        result = (
            getattr(rules, task)(row["text"])
            if engine == "rules"
            else transformer.infer(row["text"], task)
        )
        latencies.append((time.perf_counter() - start) * 1000)
        actual.append(row["label"])
        predicted.append(result.label)
        confidences.append(result.confidence)
        if result.label != row["label"]:
            # Store indices, not copyrighted text, in tracked research outputs.
            errors.append(
                {
                    "index": i,
                    "actual": row["label"],
                    "predicted": result.label,
                    "confidence": result.confidence,
                    "uncertain": result.uncertain,
                }
            )
    matrix_labels = labels + (["unknown"] if "unknown" in predicted else [])
    matrix = confusion_matrix(actual, predicted, labels=matrix_labels).tolist()
    report = classification_report(
        actual, predicted, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "task": task,
        "engine": engine,
        "model": (
            transformer.MODELS[task]
            if engine == "transformer"
            else {
                "id": "custom-rule-baseline-v1",
                "revision": "1.0.0",
                "labels": labels,
                "license": "mit",
            }
        ),
        "dataset": metadata,
        "n": len(rows),
        "scope": "full held-out test"
        if limit is None
        else "first-n smoke subset; not a full benchmark",
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {k: report[k] for k in labels},
        "class_distribution": dict(Counter(actual)),
        "confusion_matrix": matrix,
        "matrix_labels": matrix_labels,
        "errors": errors,
        "calibration": calibration(actual, predicted, confidences),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "cold_start_excluded": True,
        },
        "uncertainty_threshold": THRESHOLD,
        "coverage": sum(c >= THRESHOLD for c in confidences) / len(confidences),
        "metric_note": (
            "All examples scored using raw argmax, including uncertain "
            "predictions. Unknown emotion counts as incorrect; "
            "macro F1 averages the six target classes."
        ),
    }


def figures(run: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = ROOT / "reports" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{run['task']}-{run['engine']}"
    fig, ax = plt.subplots(figsize=(7, 6))
    matrix = np.array(run["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(len(matrix)),
        yticks=range(len(matrix)),
        xticklabels=run["matrix_labels"],
        yticklabels=run["matrix_labels"],
        xlabel="Predicted",
        ylabel="Actual",
        title=name,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            str(count),
            ha="center",
            va="center",
            color="white" if count > matrix.max() / 2 else "black",
        )
    fig.tight_layout()
    fig.savefig(directory / f"{name}-confusion.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = run["calibration"]["bins"]
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-")
    ax.set(
        xlabel="Mean confidence / support", ylabel="Observed accuracy", title=name + " reliability"
    )
    fig.tight_layout()
    fig.savefig(directory / f"{name}-reliability.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["rules", "transformer", "both"], default="both")
    parser.add_argument("--task", choices=["sentiment", "emotion", "both"], default="both")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    tasks = ["sentiment", "emotion"] if args.task == "both" else [args.task]
    engines = ["rules", "transformer"] if args.engine == "both" else [args.engine]
    path = ROOT / "reports" / "metrics.json"
    current = json.loads(path.read_text()) if path.exists() else {"runs": []}
    for task in tasks:
        for engine in engines:
            run = evaluate(task, engine, args.limit)
            figures(run)
            current["runs"] = [
                r for r in current["runs"] if (r["task"], r["engine"]) != (task, engine)
            ] + [run]
            current.update(
                {
                    "status": "Measured runs; inspect scope for completeness",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "environment": {
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "processor": platform.processor(),
                        "packages": {
                            p: version(p)
                            for p in ["transformers", "torch", "datasets", "scikit-learn"]
                        },
                    },
                }
            )
            path.write_text(json.dumps(current, indent=2))
            print(
                task,
                engine,
                run["n"],
                "accuracy",
                run["accuracy"],
                "macro_f1",
                run["macro_f1"],
                flush=True,
            )
    from scripts.export_results import export

    export()


if __name__ == "__main__":
    main()
