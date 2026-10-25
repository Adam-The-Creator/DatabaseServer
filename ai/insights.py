"""
Session level insight aggregation (requirement AI-15): what recurs across the drawings and the participants of a session,
for example how often a semantic label comes back, how the indicators are distributed, and which drawings are atypical.
Pure functions over the results of the other functions. Counting only: a recurring label means that a model named it for
several drawings, not that the drawings really show it.
"""

from typing import Any

from ai.fuzzy import LEVELS

MIN_DRAWINGS = 2


def aggregate(drawings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    drawings: [{"id", "name", "playerId", "indicators": [indicator results], "labels": result of the labelling or None,
    "atypical": bool or None}] of one session. Returns the insights as [{"kind", ...}].
    """

    insights: list[dict[str, Any]] = []

    # Semantic labels that come back (only labels that are not "other" and not marked unspecific)
    recurring: dict[str, dict[str, Any]] = {}
    for drawing in drawings:
        labels = drawing.get("labels")
        if not labels:
            continue
        found = set()
        for verdict in [labels["whole"], *labels.get("groups", [])]:
            if not verdict["unspecific"] and not verdict["top"][0]["other"]:
                found.add((verdict["top"][0]["label"], verdict["top"][0]["name"]))
        for label_id, name in found:
            entry = recurring.setdefault(label_id, {"kind": "recurringLabel", "label": label_id, "name": name,
                                                    "drawingIds": [], "playerIds": []})
            entry["drawingIds"].append(drawing["id"])
            if drawing["playerId"] not in entry["playerIds"]:
                entry["playerIds"].append(drawing["playerId"])
    for entry in sorted(recurring.values(), key=lambda e: (-len(e["drawingIds"]), e["label"])):
        if len(entry["drawingIds"]) >= 2:
            insights.append({**entry, "drawings": len(entry["drawingIds"]), "participants": len(entry["playerIds"])})

    # How the levels of every indicator are spread over the drawings
    by_indicator: dict[str, dict[str, Any]] = {}
    for drawing in drawings:
        for indicator in drawing.get("indicators", []):
            if not indicator["level"]:
                continue
            entry = by_indicator.setdefault(indicator["id"], {
                "kind": "indicatorLevels", "indicator": indicator["id"], "name": indicator["name"],
                "counts": {level: 0 for level in LEVELS}, "drawingIds": {level: [] for level in LEVELS}})
            entry["counts"][indicator["level"]] += 1
            entry["drawingIds"][indicator["level"]].append(drawing["id"])
    for entry in by_indicator.values():
        total = sum(entry["counts"].values())
        top = max(LEVELS, key=lambda level: (entry["counts"][level], -LEVELS.index(level)))
        entry["total"] = total
        entry["common"] = top if total >= 3 and entry["counts"][top] / total >= 0.6 else None
        insights.append(entry)

    atypical = [d["id"] for d in drawings if d.get("atypical")]
    if atypical:
        insights.append({"kind": "atypical", "drawingIds": atypical, "drawings": len(atypical)})
    return insights
