"""
Stored results of the heavy AI functions (requirement AI-02): one file per function and drawing in the analysis file
store (`ai/<function>/<drawing id>.json`), like reports and thumbnails, never rows in the databases (SC-03).

A stored result remembers the key it was made with (content fingerprint, analyzer, model, prompt, parameters and a hash
of the facts it was written from). While the key is unchanged the stored result is reused. A result with an older key
is still shown, marked as out of date, so the psychologist sees that it no longer matches the data.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from ai import settings
from analysis.store import AnalysisStore, ID_PATTERN


def key_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


class AiStore:
    def __init__(self, root: Optional[str] = None) -> None:
        self.root = AnalysisStore(root).root / "ai"

    def _path(self, feature: str, target_id: str) -> Path:
        if feature not in settings.STORED_FEATURES:
            raise ValueError("Invalid AI function")
        if not ID_PATTERN.match(target_id or ""):
            raise ValueError("Invalid identifier")
        folder = self.root / feature
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{target_id}.json"

    def peek(self, feature: str, target_id: str) -> Optional[dict[str, Any]]:
        """{"key": ..., "envelope": ...} of the stored result, whatever its key, or None."""

        try:
            return json.loads(self._path(feature, target_id).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None

    def get(self, feature: str, target_id: str, key: dict[str, Any]) -> Optional[dict[str, Any]]:
        """The stored envelope when it was made with exactly this key."""

        stored = self.peek(feature, target_id)
        return stored["envelope"] if stored and stored.get("key") == key else None

    def put(self, feature: str, target_id: str, key: dict[str, Any], envelope: dict[str, Any]) -> None:
        path = self._path(feature, target_id)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(json.dumps({"key": key, "envelope": envelope}).encode("utf-8"))
        os.replace(temporary, path)  # atomic: a reader never sees half a file

    def delete(self, feature: str, target_id: str) -> bool:
        path = self._path(feature, target_id)
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed

    def generated_stamps(self, target_id: str) -> dict[str, str]:
        """{function: generatedAt} of the stored heavy results of a drawing, for the staleness check of a report."""

        stamps = {}
        for feature in settings.HEAVY_FEATURES:
            stored = self.peek(feature, target_id)
            if stored and stored.get("envelope", {}).get("status") == "ok":
                stamps[feature] = stored["envelope"].get("generatedAt", "")
        return stamps
