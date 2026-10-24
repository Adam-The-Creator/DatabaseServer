"""
The template-based narrative summary (requirement REP-07): a few plain sentences that restate the metrics.
Deterministic text, no AI. The API returns structured items (id and parameters) and the web app writes the sentences
from its translations. The HTML reports are written here, so the English templates live here as well and have to read
the same as src/i18n/locales/en.json (keys narrative.*).
"""

from typing import Any, Optional

from analysis.common import IDLE_GAP_SECONDS, round_half_up
from analysis.metrics import metric_values

# {name} placeholders are filled from the item parameters
TEMPLATES = {
    "activityTimed": "{strokes} strokes were drawn with a total length of {length} m. The participant was drawing for "
                     "{active} s of the {span} s between the first and the last stroke.",
    "activity": "{strokes} strokes were drawn with a total length of {length} m.",
    "speed": "The mean drawing speed was {speed} m/s.",
    "space": "The drawing used a space of {volume} m³ with a height range of {height} m.",
    "hand": "{left} % of the strokes were drawn with the left hand.",
    "handDominant": "{left} % of the strokes were drawn with the left hand (dominant hand: {dominant}).",
    "colour": "{colours} distinct colours were used, and {warm} % of the coloured ink is warm.",
    "colourNeutral": "{colours} distinct colours were used.",
    "erase": "{erased} strokes were erased ({ratio} % of all strokes).",
    "tools": "The menu was opened {menu} times and the drawing tools were changed {tools} times.",
    "pauses": "Pauses of more than {gap} s between strokes added up to {seconds} s.",
    "quality": "{notes} data quality notes apply to this drawing, see the data quality section.",
}


def _round(value: float, digits: int = 0) -> float:
    rounded = round_half_up(value, digits)
    return int(rounded) if digits == 0 else rounded


def build_narrative(
        analysis: dict[str, Any], dominant_hand: Optional[str] = None, idle_gap_seconds: float = IDLE_GAP_SECONDS,
) -> list[dict[str, Any]]:
    """The summary sentences as [{id, params}]. A sentence is left out when the numbers it needs are missing."""

    v = metric_values(analysis)
    items: list[dict[str, Any]] = []

    def add(item_id: str, **params: Any) -> None:
        items.append({"id": item_id, "params": params})

    if v["strokesDrawn"] and v["totalLength"] is not None:
        if v["activeTime"] is not None and v["drawingSpan"] is not None:
            add("activityTimed", strokes=v["strokesDrawn"], length=_round(v["totalLength"], 1),
                active=_round(v["activeTime"]), span=_round(v["drawingSpan"]))
        else:
            add("activity", strokes=v["strokesDrawn"], length=_round(v["totalLength"], 1))
    if v["meanSpeed"] is not None:
        add("speed", speed=_round(v["meanSpeed"], 2))
    if v["boundingVolume"] is not None and v["heightRange"] is not None:
        add("space", volume=_round(v["boundingVolume"], 3), height=_round(v["heightRange"], 2))
    if v["leftHandShare"] is not None:
        if dominant_hand:
            add("handDominant", left=_round(v["leftHandShare"]), dominant=dominant_hand)
        else:
            add("hand", left=_round(v["leftHandShare"]))
    if v["distinctColours"] is not None:
        if v["warmShare"] is not None:
            add("colour", colours=v["distinctColours"], warm=_round(v["warmShare"]))
        else:
            add("colourNeutral", colours=v["distinctColours"])
    if v["strokesErased"]:
        add("erase", erased=v["strokesErased"], ratio=_round(v["eraseRatio"]))
    if v["menuOpens"] is not None and v["toolChanges"] is not None:
        add("tools", menu=v["menuOpens"], tools=v["toolChanges"])
    if v["idleTime"]:
        add("pauses", gap=_round(idle_gap_seconds, 1), seconds=_round(v["idleTime"]))
    if analysis["flags"]:
        add("quality", notes=len(analysis["flags"]))
    return items


def render_sentence(item: dict[str, Any]) -> str:
    return TEMPLATES[item["id"]].format(**item["params"])
