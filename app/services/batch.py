import csv
import io

from app.api.schemas import AnalyzeRequest
from app.core.config import MAX_ROWS, MAX_UPLOAD


def parse_upload(content: bytes, filename: str, column: str) -> list[str]:
    if len(content) > MAX_UPLOAD:
        raise ValueError("Upload exceeds 1 MB")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("File must use UTF-8 encoding") from exc
    if "\x00" in text:
        raise ValueError("Binary content is not supported")
    if filename.lower().endswith(".txt"):
        rows = [line.strip() for line in text.splitlines() if line.strip()]
    elif filename.lower().endswith(".csv"):
        try:
            reader = csv.DictReader(io.StringIO(text), strict=True)
            if not reader.fieldnames or column not in reader.fieldnames:
                raise ValueError("Selected text column is missing")
            rows = []
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("CSV row does not match header")
                rows.append(row[column])
        except csv.Error as exc:
            raise ValueError("Malformed CSV") from exc
    else:
        raise ValueError("Only .csv and .txt files are supported")
    if not rows or len(rows) > MAX_ROWS:
        raise ValueError(f"Provide between 1 and {MAX_ROWS} rows")
    for text_row in rows:
        AnalyzeRequest(text=text_row)
    return rows


def safe_cell(value: object) -> str:
    cell = str(value)
    if cell.lstrip().startswith(("=", "+", "-", "@")) or cell.startswith(("\t", "\r", "\n")):
        return "'" + cell
    return cell


def export_csv(rows: list[dict[str, object]]) -> str:
    buffer = io.StringIO(newline="")
    fields = ["row", "text", "sentiment", "emotion", "confidence", "uncertain"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: safe_cell(row.get(k, "")) for k in fields})
    return buffer.getvalue()
