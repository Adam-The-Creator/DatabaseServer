"""
The file store of the analysis (requirements TEC-03, REP-04): reports, thumbnails and stored analysis results are files
in one configurable directory, not rows in the databases (SC-03).

File names are only ever built from validated IDs and fixed suffixes, so a request cannot reach outside the directory.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import config

# Drawing, session and player IDs are UUIDs, or the fixed "default-session"
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
REPORT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{8,64}$")

DEFAULT_PATH = "analysis_store"


def is_valid_id(value: str) -> bool:
    return bool(ID_PATTERN.match(value or ""))


def new_report_id() -> str:
    return uuid4().hex


class AnalysisStore:
    def __init__(self, root: Optional[str] = None) -> None:
        self.root = Path(root or os.environ.get("ANALYSIS_STORE_PATH") or getattr(config, "ANALYSIS_STORE_PATH", DEFAULT_PATH))

    # --- plumbing ---

    def _folder(self, name: str) -> Path:
        folder = self.root / name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)  # atomic: a reader never sees half a file

    @staticmethod
    def _checked(value: str, pattern: re.Pattern) -> str:
        if not pattern.match(value or ""):
            raise ValueError("Invalid identifier")
        return value

    # --- stored analysis results, one per drawing and content fingerprint ---

    def get_analysis(self, drawing_id: str, fingerprint: str, version: str) -> Optional[dict[str, Any]]:
        path = self._folder("analysis") / self._analysis_name(drawing_id, fingerprint, version)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None

    def put_analysis(self, drawing_id: str, fingerprint: str, version: str, analysis: dict[str, Any]) -> None:
        folder = self._folder("analysis")
        path = folder / self._analysis_name(drawing_id, fingerprint, version)
        for old in folder.glob(f"{self._checked(drawing_id, ID_PATTERN)}.*.json"):  # results of older content
            if old != path:
                old.unlink(missing_ok=True)
        self._write(path, json.dumps(analysis).encode("utf-8"))

    def _analysis_name(self, drawing_id: str, fingerprint: str, version: str) -> str:
        return (f"{self._checked(drawing_id, ID_PATTERN)}.{self._checked(fingerprint, FINGERPRINT_PATTERN)}."
                f"{self._checked(version.replace('.', '-'), ID_PATTERN)}.json")

    # --- thumbnails ---

    def get_thumbnail(self, drawing_id: str, fingerprint: str) -> Optional[str]:
        path = self._folder("thumbnails") / self._thumbnail_name(drawing_id, fingerprint)
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def put_thumbnail(self, drawing_id: str, fingerprint: str, svg: str) -> None:
        folder = self._folder("thumbnails")
        path = folder / self._thumbnail_name(drawing_id, fingerprint)
        for old in folder.glob(f"{self._checked(drawing_id, ID_PATTERN)}.*.svg"):
            if old != path:
                old.unlink(missing_ok=True)
        self._write(path, svg.encode("utf-8"))

    def _thumbnail_name(self, drawing_id: str, fingerprint: str) -> str:
        return f"{self._checked(drawing_id, ID_PATTERN)}.{self._checked(fingerprint, FINGERPRINT_PATTERN)}.svg"

    # --- reports: <id>.json (what it is) and <id>.html (the report) ---

    def list_reports(self) -> list[dict[str, Any]]:
        reports = []
        for path in self._folder("reports").glob("*.json"):
            try:
                reports.append(json.loads(path.read_text(encoding="utf-8")))
            except ValueError:
                continue  # a damaged entry is skipped, it cannot be opened anyway
        return sorted(reports, key=lambda r: r.get("createdAt", ""), reverse=True)

    def get_report(self, report_id: str) -> Optional[dict[str, Any]]:
        path = self._folder("reports") / f"{self._checked(report_id, REPORT_ID_PATTERN)}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None

    def read_report_html(self, report_id: str) -> Optional[str]:
        path = self._folder("reports") / f"{self._checked(report_id, REPORT_ID_PATTERN)}.html"
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def find_report(self, kind: str, target_id: str) -> Optional[dict[str, Any]]:
        return next((r for r in self.list_reports() if r.get("kind") == kind and r.get("targetId") == target_id), None)

    def save_report(self, meta: dict[str, Any], html: str) -> None:
        folder = self._folder("reports")
        report_id = self._checked(meta["id"], REPORT_ID_PATTERN)
        self._write(folder / f"{report_id}.html", html.encode("utf-8"))  # the HTML first: a listed report always opens
        self._write(folder / f"{report_id}.json", json.dumps(meta).encode("utf-8"))

    def delete_report(self, report_id: str) -> bool:
        folder = self._folder("reports")
        report_id = self._checked(report_id, REPORT_ID_PATTERN)
        existed = (folder / f"{report_id}.json").exists()
        (folder / f"{report_id}.json").unlink(missing_ok=True)
        (folder / f"{report_id}.html").unlink(missing_ok=True)
        return existed
