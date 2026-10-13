from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

TOOL_VERSION = "1.1.0"
DEFAULT_SEED = 20260722


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 JSON：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
        rows.append(value)
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    """Durably replace ``path`` without exposing a partial JSON file.

    Prediction files are rewritten after every completed request.  A process
    interruption during ``Path.write_text`` could therefore leave a truncated
    JSON/JSONL file and make an otherwise resumable run unusable.  Write into a
    sibling temporary file, flush it, and atomically replace the destination.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    _atomic_write_text(path, text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def referenced_paths(records: list[dict[str, Any]]) -> list[str]:
    values: set[str] = set()
    for record in records:
        for field in ("images", "removed_images", "mss_images"):
            values.update(str(value).replace("\\", "/") for value in record.get(field, []) or [])
    return sorted(values)


def dataset_fingerprint(dataset_dir: Path, records: list[dict[str, Any]] | None = None) -> str:
    records = records if records is not None else read_jsonl(dataset_dir / "annotations.jsonl")
    digest = hashlib.sha256()
    annotation_path = dataset_dir / "annotations.jsonl"
    digest.update(annotation_path.read_bytes())
    for relative in referenced_paths(records):
        path = dataset_dir / relative
        digest.update(relative.encode("utf-8"))
        if path.is_file():
            digest.update(bytes.fromhex(sha256_file(path)))
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


def run_directory(output_root: Path, dataset_dir: Path) -> tuple[Path, str]:
    records = read_jsonl(dataset_dir / "annotations.jsonl")
    fingerprint = dataset_fingerprint(dataset_dir, records)
    return output_root / f"dryrun_{fingerprint[:12]}", fingerprint


def safe_path_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return cleaned.strip("_") or "unnamed"
