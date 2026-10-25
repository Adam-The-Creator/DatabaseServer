"""
The envelope every AI output comes in (requirements AI-04, AI-05, SC-07). Nothing leaves the AI package without it, so
the web app and the reports can always mark an output as "AI-generated", say which model produced it with which
parameters, how sure it is, and what it is based on.

confidence is a number between 0 and 1 or None, and confidenceKind says what the number is. It is never presented as the
probability that a conclusion is right: a membership degree, a similarity or the share of a vocabulary is not that.
"""

import re
from datetime import datetime, timezone
from typing import Any, Optional

from analysis.common import number
from ai import VERSION, settings

LABEL = "AI-generated"
DISCLAIMER = ("AI-generated decision support for the psychologist. It describes and suggests, it is not a diagnosis, "
              "and it has to be reviewed by a psychologist.")

# Why an output has no result. The web app translates the ids, reports print these texts.
REASONS = {
    "switchedOff": "This AI function is switched off in the server configuration.",
    "notEnoughData": "Not enough drawings in the comparison group for this function.",
    "noStrokes": "The drawing has no drawn strokes.",
    "missingLibrary": "A Python library this function needs is not installed on the server.",
    "missingModel": "The model this function needs is not installed on the server.",
    "serviceUnreachable": "The local model server (Ollama) cannot be reached.",
    "invalidRules": "The rule base file is invalid.",
    "invalidVocabulary": "The vocabulary file is invalid.",
    "diagnosticLanguage": "The model's answer used clinical or diagnostic wording and was discarded.",
    "invalidAnswer": "The model's answer did not match the expected format and was discarded.",
    "timeout": "The model did not answer in time.",
    "failed": "The function failed.",
}

# Wording that does not belong into an AI output (AI-05). A model answer that contains one is discarded, not shown.
_DIAGNOSTIC = re.compile(
    r"diagnos|disorder|syndrome|schizo|psychos|psychot|bipolar|depress|autis|adhd|ptsd|suicid|self[- ]?harm|abus(?:e|ed|ing)\b"
    r"|neurotic|pathology|pathologic|dementia|anorexi|bulimi|ocd\b|dyslexi",
    re.IGNORECASE,
)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_info(model_id: str, name: str, version: str, licence: str, kind: str) -> dict[str, str]:
    """kind: statistical, rules, neural, language or vision-language."""

    return {"id": model_id, "name": name, "version": version, "licence": licence, "kind": kind}


def make(
        feature: str, *, model: dict[str, str], params: Optional[dict[str, Any]] = None,
        result: Optional[dict[str, Any]] = None, prompt_version: Optional[str] = None,
        confidence: Optional[float] = None, confidence_kind: Optional[str] = None,
        evidence: Optional[list[dict[str, Any]]] = None, status: str = "ok", reason: Optional[str] = None,
        notes: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    evidence items say what the output is based on: {"kind": "metric", "id", "value"}, {"kind": "strokes", "strokeIds",
    "label"}, {"kind": "view", "view", "groupId"?}, {"kind": "drawing", "id"}.
    """

    return {
        "ai": True,
        "label": LABEL,
        "disclaimer": DISCLAIMER,
        "feature": feature,
        "requirement": settings.FEATURES[feature]["requirement"],
        "status": status,
        "reason": reason,
        "evaluated": settings.feature_evaluated(feature),
        "model": model,
        "params": params or {},
        "promptVersion": prompt_version,  # prompt, rule base or vocabulary version
        "confidence": None if confidence is None else number(confidence, 3),
        "confidenceKind": confidence_kind if confidence is not None else None,
        "evidence": evidence or [],
        "notes": notes or [],
        "moduleVersion": VERSION,
        "generatedAt": now(),
        "result": result,
    }


def without_result(feature: str, status: str, reason: str, model: dict[str, str],
                   params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """An output that has no result: switched off, unavailable, not enough data, rejected."""

    return make(feature, model=model, params=params, status=status, reason=reason)


def contains_diagnostic_language(value: Any) -> bool:
    """Whether any text inside a (nested) answer uses wording that must not appear in an AI output."""

    if isinstance(value, str):
        return bool(_DIAGNOSTIC.search(value))
    if isinstance(value, dict):
        return any(contains_diagnostic_language(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_diagnostic_language(v) for v in value)
    return False


def clean_text(value: Any, limit: int) -> Optional[str]:
    """A string from a model answer: control characters removed, whitespace collapsed, cut to the limit. None if empty."""

    if not isinstance(value, str):
        return None
    text = " ".join(_CONTROL.sub(" ", value).split())
    return text[:limit].rstrip() or None
