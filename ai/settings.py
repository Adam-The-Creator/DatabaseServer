"""
The switches and thresholds of the AI functions, read from config.py (requirements AI-03, AI-07). They are read on
every call, so a test can change them, but the server only takes a changed config.py after a restart.
"""

import importlib.util
from typing import Any

import config

# id -> what it is. "kind": classical (scikit-learn, no weights), rules (the rule base file), model (needs weights).
FEATURES: dict[str, dict[str, Any]] = {
    "cohort": {"requirement": "AI-10", "kind": "classical", "models": ["scikit-learn"]},
    "groups": {"requirement": "AI-11", "kind": "classical", "models": ["scikit-learn"]},
    "indicators": {"requirement": "AI-14", "kind": "rules", "models": ["fuzzy-rules"]},
    "insights": {"requirement": "AI-15", "kind": "classical", "models": []},
    "labels": {"requirement": "AI-12", "kind": "model", "models": ["clip"]},
    "description": {"requirement": "AI-13", "kind": "model", "models": ["vision"]},
    "summary": {"requirement": "AI-16", "kind": "model", "models": ["text"]},
}
HEAVY_FEATURES = ("labels", "description", "summary")  # run as background jobs (AI-02) and are stored
STORED_FEATURES = HEAVY_FEATURES + ("groups",)  # the clustering is stored too: it is deterministic, but not free


def enabled() -> bool:
    return bool(getattr(config, "AI_ENABLED", True))


def feature_enabled(feature: str) -> bool:
    return enabled() and bool(getattr(config, "AI_FEATURES", {}).get(feature, False))


def feature_evaluated(feature: str) -> bool:
    return bool(getattr(config, "AI_EVALUATED", {}).get(feature, False))


def number(name: str, default: Any) -> Any:
    return getattr(config, name, default)


def module_available(name: str) -> bool:
    """Whether a Python package can be imported, without importing it."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
