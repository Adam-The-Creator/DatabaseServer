"""
Comparing a drawing with a group of other drawings (requirement CMP-03) and following a participant over time
(requirements PAR-01, CMP-02). Pure functions over metric values, so they work on cached analyses.
"""

from typing import Any, Optional

from analysis import catalog
from analysis.common import median, number

# Below this many drawings a percentile says too little to be shown
MIN_COHORT = 3


def percentile_rank(values: list[float], value: float) -> float:
    """The share of the values below the value, ties counting half (0..100)."""

    below = sum(1 for v in values if v < value)
    equal = sum(1 for v in values if v == value)
    return (below + equal / 2) / len(values) * 100


def session_stats(session_id: str, entries: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """
    The distribution of every metric over the drawings of a session, and the mean per participant (requirement SES-03).
    entries are (owner id, analysis) per drawing.
    """

    distributions = []
    for definition in catalog.METRICS:
        values = [
            m["value"] for _, analysis in entries for m in analysis["metrics"]
            if m["id"] == definition["id"] and m["value"] is not None
        ]
        if values:
            distributions.append({"id": definition["id"], "count": len(values), "min": number(min(values)),
                                  "median": number(median(values)), "max": number(max(values))})

    by_player: dict[str, list[dict[str, Any]]] = {}
    for owner, analysis in entries:
        by_player.setdefault(owner, []).append(analysis)
    participants = []
    for owner, analyses in by_player.items():
        means: dict[str, float] = {}
        for definition in catalog.METRICS:
            values = [m["value"] for a in analyses for m in a["metrics"]
                      if m["id"] == definition["id"] and m["value"] is not None]
            if values:
                means[definition["id"]] = number(sum(values) / len(values))
        participants.append({"playerId": owner, "drawings": len(analyses), "metrics": means})

    return {"sessionId": session_id, "drawingCount": len(entries), "distributions": distributions,
            "perParticipant": participants}


def compare_with_cohort(subject: dict[str, Optional[float]], cohort: list[dict[str, Optional[float]]]) -> dict[str, Any]:
    """
    Where the subject's metric values stand among the cohort's. Metrics the subject has no value for, or that fewer
    than MIN_COHORT cohort drawings have, come back without a percentile.
    """

    rows = []
    for definition in catalog.METRICS:
        metric_id = definition["id"]
        values = [m[metric_id] for m in cohort if m.get(metric_id) is not None]
        mine = subject.get(metric_id)
        enough = mine is not None and len(values) >= MIN_COHORT
        rows.append({
            "id": metric_id,
            "value": mine,
            "cohortSize": len(values),
            "percentile": number(percentile_rank(values, mine), 1) if enough else None,
            "min": number(min(values)) if values else None,
            "median": number(median(values)) if values else None,
            "max": number(max(values)) if values else None,
        })
    return {"cohortSize": len(cohort), "sufficient": len(cohort) >= MIN_COHORT, "metrics": rows}
