"""Download licensed research benchmarks without redistributing raw data."""

import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.core.config import ROOT

SEED = 42
SOURCES = {
    "sentiment": {
        "id": "takala/financial_phrasebank",
        "revision": "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964",
        "license": "cc-by-nc-sa-3.0",
        "config": "sentences_allagree",
    },
    "emotion": {
        "id": "dair-ai/emotion",
        "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
        "license": "other: educational and research purposes only",
        "config": "split",
    },
}


def fingerprint(text: str) -> str:
    # Conservative likely-duplicate normalization; does not prove absence of semantic leakage.
    normalized = re.sub(r"[^\w]+", "", text.casefold())
    return hashlib.sha256(normalized.encode()).hexdigest()


def prepare(task: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    spec = SOURCES[task]
    initial_duplicates = 0
    if task == "emotion":
        from datasets import load_dataset

        dataset = load_dataset(spec["id"], spec["config"], revision=spec["revision"])
        labels = dataset["train"].features["label"].names
        splits = {
            split: [{"text": row["text"], "label": labels[row["label"]]} for row in dataset[split]]
            for split in ("train", "validation", "test")
        }
    else:
        url = f"https://huggingface.co/datasets/{spec['id']}/resolve/{spec['revision']}/data/FinancialPhraseBank-v1.0.zip"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            name = next(n for n in archive.namelist() if n.endswith("Sentences_AllAgree.txt"))
            lines = archive.read(name).decode("latin-1").splitlines()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen = set()
        for line in lines:
            text, label = line.rsplit("@", 1)
            key = fingerprint(text)
            if key not in seen:
                grouped[label].append({"text": text, "label": label})
                seen.add(key)
        initial_duplicates = len(lines) - len(seen)
        splits = {"train": [], "validation": [], "test": []}
        rng = random.Random(SEED)
        for rows in grouped.values():
            rng.shuffle(rows)
            a, b = int(len(rows) * 0.70), int(len(rows) * 0.85)
            splits["train"].extend(rows[:a])
            splits["validation"].extend(rows[a:b])
            splits["test"].extend(rows[b:])
    # Keep train first; remove duplicate/normalized-duplicate examples from later splits.
    seen = set()
    removed = {}
    for name, rows in splits.items():
        cleaned = []
        for row in rows:
            key = fingerprint(row["text"])
            if key not in seen:
                seen.add(key)
                cleaned.append(row)
        removed[name] = len(rows) - len(cleaned)
        splits[name] = cleaned
    metadata = {
        **spec,
        "seed": SEED,
        "removed_duplicates": removed,
        "initial_duplicates_removed": initial_duplicates,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "leakage_note": (
            "Normalized duplicates removed across splits. Semantic overlap and "
            "upstream pretraining contamination remain possible."
        ),
    }
    target = ROOT / "data" / "cache"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{task}.json").write_text(json.dumps({"metadata": metadata, "splits": splits}))
    return splits, metadata


def load(task: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    path: Path = ROOT / "data" / "cache" / f"{task}.json"
    if not path.exists():
        return prepare(task)
    data = json.loads(path.read_text())
    return data["splits"], data["metadata"]


if __name__ == "__main__":
    for task in SOURCES:
        _, metadata = prepare(task)
        print(json.dumps(metadata, indent=2))
