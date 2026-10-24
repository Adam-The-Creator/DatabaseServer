"""
The drawing metrics (requirements MET-01..08): numbers that describe a drawing. They support the psychologist's
judgement, they are not a diagnosis (SC-07).

Only drawn strokes count (MET-01), erased strokes are analysed as behaviour. A metric that cannot be computed has
value None and an unavailableReason, never a made-up number (MET-05).
"""

import math
from datetime import datetime, timezone
from typing import Any, Optional

from analysis import catalog
from analysis.common import (
    Drawing, IDLE_GAP_SECONDS, LEFT_HAND, all_lines, breaks_between, drawn_lines, hue_of, idle_periods, is_warm_hue,
    line_stats, merge_intervals, rgb_to_hex, stroke_times,
)

# Bump when a definition or the result format changes: stored results of other versions are not reused.
ANALYZER_VERSION = "1.0"

# Event types of the game (DrawingEventType in models/mongodb/models.py)
EVENT_MENU_OPEN = 0
EVENT_MODE_CHANGE = 2
EVENT_COLOR_CHANGE = 3
EVENT_WIDTH_CHANGE = 4
TOOL_EVENTS = (EVENT_MODE_CHANGE, EVENT_COLOR_CHANGE, EVENT_WIDTH_CHANGE)


def _value(metric_id: str, value: float) -> dict[str, Any]:
    return {"id": metric_id, "value": value}


def _missing(metric_id: str, reason: str) -> dict[str, Any]:
    return {"id": metric_id, "value": None, "unavailableReason": reason}


def _directional_change(lines: list[dict]) -> Optional[float]:
    """Mean turning angle between consecutive segments, 0 (straight) to 1 (turning back)."""

    total = 0.0
    count = 0
    for line in lines:
        p = line.get("points") or []
        for i in range(1, len(p) - 1):
            v1 = (p[i]["x"] - p[i - 1]["x"], p[i]["y"] - p[i - 1]["y"], p[i]["z"] - p[i - 1]["z"])
            v2 = (p[i + 1]["x"] - p[i]["x"], p[i + 1]["y"] - p[i]["y"], p[i + 1]["z"] - p[i]["z"])
            m1, m2 = math.hypot(*v1), math.hypot(*v2)
            if m1 < 1e-6 or m2 < 1e-6:
                continue
            cos = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (m1 * m2)
            total += math.acos(max(-1.0, min(1.0, cos)))
            count += 1
    return total / count / math.pi if count else None


def analyze_drawing(doc: Drawing, idle_gap_seconds: float = IDLE_GAP_SECONDS) -> dict[str, Any]:
    """The metrics, colour palette and data quality flags of one drawing (the 'DrawingAnalysis' of the API)."""

    metadata = doc.get("metadata") or {}
    lines = all_lines(doc)
    drawn = drawn_lines(doc)
    erased = len(lines) - len(drawn)
    nothing_drawn = not drawn
    stats = [line_stats(line) for line in drawn]
    total_length = sum(s["length"] for s in stats)

    results: dict[str, dict[str, Any]] = {}

    def put(entry: dict[str, Any]) -> None:
        results[entry["id"]] = entry

    # Counts and lengths
    put(_value("strokesDrawn", len(drawn)))
    put(_value("strokesErased", erased))
    put(_value("eraseRatio", erased / len(lines) * 100 if lines else 0.0))
    put(_value("pointCount", sum(s["pointCount"] for s in stats)))
    put(_missing("totalLength", "noStrokes") if nothing_drawn else _value("totalLength", total_length))
    put(_missing("meanStrokeLength", "noStrokes") if nothing_drawn
        else _value("meanStrokeLength", total_length / len(drawn)))

    # Space
    if nothing_drawn:
        put(_missing("boundingVolume", "noStrokes"))
        put(_missing("heightRange", "noStrokes"))
    else:
        dx = max(s["boundsMaxX"] for s in stats) - min(s["boundsMinX"] for s in stats)
        dy = max(s["boundsMaxY"] for s in stats) - min(s["boundsMinY"] for s in stats)
        dz = max(s["boundsMaxZ"] for s in stats) - min(s["boundsMinZ"] for s in stats)
        put(_value("boundingVolume", dx * dy * dz))
        put(_value("heightRange", dy))

    figure = metadata.get("sadChildCoordinates")
    if not figure:
        put(_missing("figureDistance", "noFigurePosition"))
    else:
        distances = [
            math.hypot(p["x"] - figure["x"], p["y"] - figure["y"], p["z"] - figure["z"])
            for line in drawn for p in line.get("points") or []
        ]
        put(_value("figureDistance", sum(distances) / len(distances)) if distances
            else _missing("figureDistance", "noStrokes"))

    # Time, pauses and speed
    times = [t for t in (stroke_times(line) for line in drawn) if t]
    no_time = "noStrokes" if nothing_drawn else "noTimeData"
    sittings = breaks_between(times)  # a drawing that was continued after a long break has several sittings
    if not times:
        for metric_id in ("drawingSpan", "activeTime", "idleTime", "meanSpeed"):
            put(_missing(metric_id, no_time))
    else:
        # The breaks between sittings are not part of the time span
        span = (max(t[1] for t in times) - min(t[0] for t in times) - sum(end - start for start, end in sittings)) / 1000
        active = sum(end - start for start, end in merge_intervals(times)) / 1000
        pauses = idle_periods(times, idle_gap_seconds * 1000)
        put(_value("drawingSpan", span))
        put(_value("activeTime", active))
        put(_value("idleTime", sum(end - start for start, end in pauses) / 1000))
        put(_value("meanSpeed", total_length / active) if active > 0 else _missing("meanSpeed", "noTimeData"))

    # Motion
    turn = _directional_change(drawn)
    put(_value("directionalChange", turn) if turn is not None else _missing("directionalChange", "noStrokes"))
    pressures = [p["pressure"] for line in drawn for p in line.get("points") or [] if p.get("pressure") is not None]
    put(_value("meanPressure", sum(pressures) / len(pressures)) if pressures
        else _missing("meanPressure", "noStrokes" if nothing_drawn else "noPressure"))

    # Hand use
    put(_missing("leftHandShare", "noStrokes") if nothing_drawn
        else _value("leftHandShare", sum(1 for line in drawn if line.get("hand") == LEFT_HAND) / len(drawn) * 100))

    # Colour: ink length per (coarsely quantised) colour of the stroke start
    bins: dict[tuple[int, int, int], dict[str, Any]] = {}
    warm_ink = cool_ink = 0.0
    for line, stat in zip(drawn, stats):
        color = line.get("startColor") or {}
        key = (color.get("r", 0) >> 5, color.get("g", 0) >> 5, color.get("b", 0) >> 5)
        entry = bins.setdefault(key, {"hex": rgb_to_hex(color), "ink": 0.0})
        entry["ink"] += stat["length"]
        hue, chromatic = hue_of(color)
        if chromatic:
            if is_warm_hue(hue):
                warm_ink += stat["length"]
            else:
                cool_ink += stat["length"]
    ranked = sorted(bins.values(), key=lambda b: b["ink"], reverse=True)
    palette = [{"hex": b["hex"], "share": b["ink"] / total_length} for b in ranked[:5]] if total_length > 0 else []
    # Strokes of zero length (a tap) have no ink, so shares of the ink cannot be told
    no_ink = "noStrokes" if nothing_drawn else "noInk"
    if warm_ink + cool_ink > 0:
        put(_value("warmShare", warm_ink / (warm_ink + cool_ink) * 100))
    else:
        put(_missing("warmShare", no_ink if total_length <= 0 else "noChromaticInk"))
    if total_length <= 0:
        put(_missing("distinctColours", no_ink))
    else:
        put(_value("distinctColours", sum(1 for b in ranked if b["ink"] / total_length >= 0.02)))

    # Tool behaviour: events exist from schema version 2 on (an empty list there means "none happened")
    if (metadata.get("schemaVersion") or 1) < 2:
        put(_missing("menuOpens", "noEvents"))
        put(_missing("toolChanges", "noEvents"))
    else:
        events = doc.get("events") or []
        put(_value("menuOpens", sum(1 for e in events if e.get("type") == EVENT_MENU_OPEN)))
        put(_value("toolChanges", sum(1 for e in events if e.get("type") in TOOL_EVENTS)))

    # Data quality flags (MET-08)
    flags: list[str] = []
    if (metadata.get("schemaVersion") or 1) < 2:
        flags.append("legacySchema")
    if (metadata.get("trackingDecimation") or 1) > 1:
        flags.append("thinnedTracking")
    if 0 < len(drawn) < 3:
        flags.append("fewStrokes")
    if not drawn and erased > 0:
        flags.append("allErased")
    if metadata.get("gameType") == 0 and not figure:
        flags.append("missingFigurePosition")
    if drawn and not times:
        flags.append("missingTimeData")
    if sittings:
        flags.append("multipleSittings")

    return {
        "drawingId": metadata.get("id"),
        "analyzerVersion": ANALYZER_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "metrics": [results[metric_id] for metric_id in catalog.METRIC_IDS],
        "palette": palette,
        "flags": flags,
    }


def metric_values(analysis: dict[str, Any]) -> dict[str, Optional[float]]:
    """{metric id: value or None} of an analysis."""

    return {m["id"]: m["value"] for m in analysis["metrics"]}
