"""JSONL-based store for ExecutionRecord objects.

Records are written to ``.sivo/records/<run_id>.jsonl``, one JSON object
per line, append-only (D-002).
"""

from __future__ import annotations

from pathlib import Path

from sivo.models import ExecutionRecord

DEFAULT_STORE_PATH = ".sivo"


class JsonlStore:
    """Persistent JSONL store for :class:`ExecutionRecord` objects.

    Args:
        store_path: Root directory for sivo data. Defaults to ``.sivo``
                    in the current working directory.
    """

    def __init__(self, store_path: str | Path = DEFAULT_STORE_PATH) -> None:
        self.store_path = Path(store_path)
        self.records_dir = self.store_path / "records"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, record: ExecutionRecord) -> None:
        """Append *record* to ``<records_dir>/<run_id>.jsonl``.

        Creates the directory tree if it does not exist. The write is
        immediately flushed to disk so that partial results are never lost
        on crash.

        Args:
            record: The :class:`ExecutionRecord` to persist.
        """
        self.records_dir.mkdir(parents=True, exist_ok=True)
        path = self._records_file(record.run_id)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
            fh.flush()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, run_id: str) -> list[ExecutionRecord]:
        """Return all :class:`ExecutionRecord` objects for *run_id*.

        Returns an empty list if no file exists for *run_id*.

        Args:
            run_id: The run identifier to load records for.
        """
        path = self._records_file(run_id)
        if not path.exists():
            return []

        records: list[ExecutionRecord] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(ExecutionRecord.model_validate_json(line))
        return records

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_runs(self) -> list[str]:
        """Return the run IDs for which records are stored, sorted by name.

        Returns an empty list if the records directory does not exist yet.
        """
        if not self.records_dir.exists():
            return []
        return sorted(p.stem for p in self.records_dir.glob("*.jsonl"))

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def filter(self, run_id: str, **metadata_filters: object) -> list[ExecutionRecord]:
        """Return records for *run_id* whose metadata matches all *metadata_filters*.

        Each keyword argument is matched against the corresponding key in
        ``record.metadata``. Only records that have all specified key/value
        pairs are returned.

        Example::

            store.filter("run_abc", model="claude-haiku-4-5", tag="smoke")

        Args:
            run_id: The run to load records from.
            **metadata_filters: ``key=value`` pairs to match against
                                 ``record.metadata``.
        """
        records = self.read(run_id)
        if not metadata_filters:
            return records
        return [
            r
            for r in records
            if all(r.metadata.get(k) == v for k, v in metadata_filters.items())
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _records_file(self, run_id: str) -> Path:
        return self.records_dir / f"{run_id}.jsonl"
