"""
Checking the answers of a language or vision model (requirements AI-13, AI-16). A model is asked for JSON, but it may
still answer in prose, add a field, make one up or use words that do not belong into a decision support tool. Nothing
it says is shown before it passed these checks.
"""

import json
import re
from typing import Any

from ai.envelope import clean_text, contains_diagnostic_language
from ai.ollama import OllamaError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_object(content: str) -> dict[str, Any]:
    """The JSON object in a model's answer (a markdown code fence around it is tolerated)."""

    try:
        data = json.loads(_FENCE.sub("", content.strip()))
    except ValueError as error:
        raise OllamaError("invalidAnswer", "The answer is not JSON.") from error
    if not isinstance(data, dict):
        raise OllamaError("invalidAnswer", "The answer is not a JSON object.")
    return data


def text(data: dict[str, Any], field: str, limit: int = 600, required: bool = True) -> str:
    value = clean_text(data.get(field), limit)
    if value is None and required:
        raise OllamaError("invalidAnswer", f'The field "{field}" is missing.')
    return value or ""


def strings(data: dict[str, Any], field: str, max_items: int, limit: int = 240, required: bool = True) -> list[str]:
    value = data.get(field)
    if not isinstance(value, list):
        if required:
            raise OllamaError("invalidAnswer", f'The field "{field}" is not a list.')
        return []
    items = [clean_text(item, limit) for item in value]
    return [item for item in items if item][:max_items]


def reject_diagnostic_language(result: dict[str, Any]) -> None:
    if contains_diagnostic_language(result):
        raise OllamaError("diagnosticLanguage", "The answer used clinical or diagnostic wording.")
