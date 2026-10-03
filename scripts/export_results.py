import json

from app.core.config import ROOT


def export() -> None:
    data = json.loads((ROOT / "reports" / "metrics.json").read_text())
    lines = [
        "| Task | Engine | Test examples | Accuracy | Macro F1 | Weighted F1 | Scope |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for run in data["runs"]:
        lines.append(
            f"| {run['task']} | {run['engine']} | {run['n']} | {run['accuracy']:.4f} | "
            f"{run['macro_f1']:.4f} | {run['weighted_f1']:.4f} | {run['scope']} |"
        )
    table = "\n".join(lines) if data["runs"] else "Not yet measured."
    (ROOT / "reports" / "summary.md").write_text(
        "# Generated evaluation results\n\n" + table + "\n"
    )
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text()
        start, end = "<!-- METRICS_START -->", "<!-- METRICS_END -->"
        if start in text and end in text:
            before, tail = text.split(start, 1)
            _, after = tail.split(end, 1)
            readme.write_text(before + start + "\n" + table + "\n" + end + after)


if __name__ == "__main__":
    export()
