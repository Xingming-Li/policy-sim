"""Run identity and artifact writing.

Records are appended as they are produced, so a crashed or interrupted run
still leaves a readable partial audit trail on disk.

Artifacts, all under `<output_dir>/<run_id>/`:

    resolved_config.json  the exact parameters that were executed
    messages.jsonl        one row per agent turn, with provenance
    rounds.jsonl          one row per round, with environment and metrics
    summary.json          final metrics and failure/fallback counters
"""

from __future__ import annotations

import json
import platform
import sys
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from models.message import utcnow
from models.records import MessageRecord, RoundRecord, RunSummary
from simulation.config import SimulationConfig

CONFIG_FILENAME = "resolved_config.json"
MESSAGES_FILENAME = "messages.jsonl"
ROUNDS_FILENAME = "rounds.jsonl"
SUMMARY_FILENAME = "summary.json"

_TRACKED_PACKAGES = ("openai", "pydantic", "python-dotenv")


def new_run_id() -> str:
    """Sortable, collision-resistant run identifier."""
    return f"{utcnow():%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def runtime_info() -> dict[str, str]:
    """Interpreter and library versions.

    Deliberately excludes environment variables so that no credential can
    reach an artifact.
    """
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in _TRACKED_PACKAGES:
        try:
            info[f"{package}_version"] = version(package)
        except PackageNotFoundError:
            info[f"{package}_version"] = "not installed"
    return info


class RunWriter:
    """Streaming writer for one run's artifacts."""

    def __init__(self, output_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self.dir = Path(output_dir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._messages: TextIO | None = None
        self._rounds: TextIO | None = None

    def __enter__(self) -> RunWriter:
        self._messages = self._open(MESSAGES_FILENAME)
        self._rounds = self._open(ROUNDS_FILENAME)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _open(self, filename: str) -> TextIO:
        return (self.dir / filename).open("w", encoding="utf-8", newline="\n")

    def close(self) -> None:
        for handle in (self._messages, self._rounds):
            if handle is not None and not handle.closed:
                handle.close()
        self._messages = None
        self._rounds = None

    # ------------------------------------------------------------------

    def write_config(self, config: SimulationConfig) -> Path:
        return self._write_json(CONFIG_FILENAME, config.to_json_dict())

    def append_message(self, record: MessageRecord) -> None:
        self._append(self._messages, MESSAGES_FILENAME, record.to_row())

    def append_round(self, record: RoundRecord) -> None:
        self._append(self._rounds, ROUNDS_FILENAME, record.to_row())

    def write_summary(self, summary: RunSummary) -> Path:
        return self._write_json(SUMMARY_FILENAME, summary.model_dump(mode="json"))

    # ------------------------------------------------------------------

    def _append(self, handle: TextIO | None, filename: str, row: dict[str, Any]) -> None:
        if handle is None:
            raise RuntimeError(
                f"RunWriter must be used as a context manager before writing {filename}"
            )
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Flushed per row: an interrupted run keeps everything already decided.
        handle.flush()

    def _write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.dir / filename
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact back. Used by tests and downstream analysis."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def python_executable() -> str:  # pragma: no cover - trivial
    return sys.executable
