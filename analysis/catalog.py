"""
The metric catalogue (requirement MET-02): id, group, unit, decimals, and the texts of every metric.

The web app has the same texts in its translation file (src/i18n/locales/en.json, keys metrics.*). The wording has to
stay identical, because a report generated here must read like the screen the psychologist saw.
"""

import math
from typing import Optional

METRIC_GROUPS = {
    "counts": "Counts", "length": "Length", "time": "Time", "motion": "Motion", "space": "Space",
    "handUse": "Hand use", "colour": "Colour", "behaviour": "Behaviour",
}

# (id, group, unit, decimals, name, definition, note)
_METRICS = [
    ("strokesDrawn", "counts", "count", 0, "Strokes drawn",
     "Number of strokes that are still part of the drawing.",
     "A stroke is one continuous line, from pressing to releasing the trigger."),
    ("strokesErased", "counts", "count", 0, "Strokes erased",
     "Number of strokes that were drawn and then erased.",
     "Erased strokes stay in the data. They are analysed as behaviour, not as part of the drawing."),
    ("pointCount", "counts", "count", 0, "Points",
     "Number of sampled points of the drawn strokes.",
     "Depends on the drawing speed and the sampling rate, so it says little about detail by itself."),
    ("totalLength", "length", "m", 2, "Total length",
     "Summed length of all drawn strokes, in metres of the game world.",
     "It reflects how much was drawn, more is not better."),
    ("meanStrokeLength", "length", "m", 2, "Mean stroke length",
     "Total length divided by the number of drawn strokes.",
     "Short strokes can mean a careful, detailed way of drawing or hesitation. Look at the drawing as well."),
    ("drawingSpan", "time", "s", 0, "Time span",
     "Time from the start of the first stroke to the end of the last stroke.",
     "Includes the pauses between strokes, but not breaks of more than 30 minutes (the drawing was continued later)."),
    ("activeTime", "time", "s", 0, "Active drawing time",
     "Time during which at least one stroke was being drawn.",
     "The pauses between strokes are not included. With several authors, simultaneous drawing counts once."),
    ("idleTime", "time", "s", 0, "Pause time",
     "Total length of the pauses between strokes that are longer than the pause limit (5 s).",
     "Long pauses can mean thinking, hesitation or distraction. Pauses are only seen between strokes, so they say "
     "nothing about what the participant was doing. Breaks of more than 30 minutes are not pauses."),
    ("meanSpeed", "motion", "mps", 2, "Mean speed",
     "Total length divided by the active drawing time.",
     "Slow can mean careful or hesitant, fast can mean confident or hurried. It only describes."),
    ("directionalChange", "motion", "ratio", 3, "Directional change",
     "Mean turning angle between consecutive segments of the strokes, from 0 (straight) to 1 (turning back).",
     "Natural curves count too. High values come from many changes of direction, for example tremor, scribbling or "
     "tight curves. The definition is still under review."),
    ("meanPressure", "motion", "ratio", 2, "Mean trigger pressure",
     "Mean value of the controller trigger while drawing, from 0 to 1.",
     "Only drawings saved by newer versions of the game have it."),
    ("boundingVolume", "space", "m3", 3, "Used space",
     "Volume of the box that contains all drawn strokes.",
     "Shows how much space was used. One stray stroke makes the box larger (outlier trimming is planned)."),
    ("heightRange", "space", "m", 2, "Height range",
     "Difference between the highest and the lowest point of the drawing.",
     "Depends on how far the participant can reach."),
    ("figureDistance", "space", "m", 2, "Distance to the figure",
     "Mean distance of the drawn points from the figure shown in the scene.",
     "Needs the figure position, which is not stored with the drawings yet."),
    ("leftHandShare", "handUse", "percent", 0, "Left-hand share",
     "Share of the drawn strokes that were drawn with the left hand.",
     "Compare with the dominant hand of the participant."),
    ("warmShare", "colour", "percent", 0, "Warm colours",
     "Share of the coloured ink that is warm (red, orange, yellow, pink).",
     "Grey, black and white ink is not counted. The meaning of colours differs between people, so treat it as a "
     "description."),
    ("distinctColours", "colour", "count", 0, "Distinct colours",
     "Number of clearly different colours, each with at least 2 % of the ink.",
     "Similar shades are grouped together."),
    ("eraseRatio", "behaviour", "percent", 0, "Erased strokes",
     "Share of all strokes that were erased.",
     "Can reflect self-correction. Interpret it together with the drawing."),
    ("menuOpens", "behaviour", "count", 0, "Menu opens",
     "Number of times the participant opened the in-game menu.",
     "Only drawings saved by newer versions of the game record this."),
    ("toolChanges", "behaviour", "count", 0, "Tool changes",
     "Number of changes of the drawing mode, colour or width.",
     "Frequent changes can reflect exploring the tools or searching for the right one. Only newer drawings record "
     "this."),
]

METRICS = [
    {"id": m[0], "group": m[1], "unit": m[2], "decimals": m[3], "name": m[4], "definition": m[5], "note": m[6]}
    for m in _METRICS
]
METRIC_IDS = [m["id"] for m in METRICS]
_BY_ID = {m["id"]: m for m in METRICS}

REASONS = {
    "noStrokes": "No drawn strokes.",
    "noTimeData": "The drawing has no usable time information.",
    "noPressure": "Saved before the trigger pressure was recorded.",
    "noFigurePosition": "The figure position is not stored with this drawing yet.",
    "noChromaticInk": "Only grey, black or white ink was used.",
    "noInk": "The strokes have no length, so there is no ink to measure.",
    "noEvents": "Saved by a game version that did not record events.",
}

FLAGS = {
    "legacySchema": "Saved by an older version of the game (data version 1): some data is missing.",
    "thinnedTracking": "The body tracking was thinned out to fit the storage limit, so tracking-based results are "
                       "approximate.",
    "fewStrokes": "Very few strokes: the statistics are hardly meaningful.",
    "allErased": "All strokes were erased: nothing is left to analyse.",
    "missingFigurePosition": "The position of the figure is missing, so the distance to the figure cannot be computed.",
    "missingTimeData": "No time information: the time and speed results are unavailable.",
    "multipleSittings": "The drawing was continued after a break of more than 30 minutes: the time results only "
                        "count the time spent drawing.",
}

GAME_TYPES = {0: "Inner child", 1: "Word forest", 2: "Beach", 3: "Room", 4: "Bilateral drawing", 5: "Safe place"}

_UNIT_SYMBOLS = {"count": "", "m": " m", "m3": " m³", "s": "", "mps": " m/s", "percent": " %", "ratio": ""}


def metric(metric_id: str) -> dict:
    return _BY_ID[metric_id]


def game_type_name(game_type: int) -> str:
    return GAME_TYPES.get(game_type, f"Unknown ({game_type})")


def format_duration(total_seconds: float) -> str:
    """'45 s', '2 min 05 s', '1 h 05 min', '3 d 2 h': the two largest units that matter (same as the web app)."""

    seconds = int(math.floor(total_seconds + 0.5))  # half up, like the web app
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    if seconds < 86400:
        return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"
    return f"{seconds // 86400} d {seconds % 86400 // 3600} h"


def format_number(value: float, decimals: int) -> str:
    """Like the web app's Intl formatting: thousands separators, no trailing zeros."""

    rounded = math.floor(value * 10 ** decimals + 0.5) / 10 ** decimals  # half up, like the web app
    text = f"{rounded:,.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def format_metric(metric_id: str, value: Optional[float]) -> str:
    if value is None:
        return "–"
    definition = _BY_ID[metric_id]
    if definition["unit"] == "s":
        return format_duration(value)
    return f"{format_number(value, definition['decimals'])}{_UNIT_SYMBOLS[definition['unit']]}"
