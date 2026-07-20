"""Chunk manifest sidecar (JSONL, one record per chunk).

Everything query-time BM25 needs lives here so retrieval never has to pull
the full corpus out of Chroma. Rebuilt from the collection after every
ingest; written atomically so a crash mid-write cannot leave a torn file.
"""
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


_REPLACE_ATTEMPTS = 5
_REPLACE_RETRY_DELAY_SECONDS = 0.05


def write_manifest(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(_REPLACE_RETRY_DELAY_SECONDS)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
