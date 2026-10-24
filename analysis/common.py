"""
Helpers shared by the analysis modules.

A drawing is the plain dict stored in MongoDB (see models/mongodb/models.py). Nothing in the analysis package writes to a
database, and nothing assumes that the optional fields of schema version 2 are present: drawings of version 1 have no
`utcMs`, pressure, events or line stats (requirement MET-05).

Coordinates stay in game space (Unity: left-handed, Y up, metres). Only the exporters and the web viewer flip z.
"""

import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

Drawing = dict[str, Any]
Line = dict[str, Any]

LEFT_HAND = -1
RIGHT_HAND = 1
STATUS_ERASED = 0
STATUS_DRAWN = 1

# A pause longer than this between two strokes counts as idle time (adjustable per request).
IDLE_GAP_SECONDS = 5.0

# A gap longer than this is no pause but a break between two sittings: the drawing was continued later.
BREAK_SECONDS = 30 * 60.0
BREAK_DISPLAY_MS = 5000  # how long a break is drawn on the time axis

_LEGACY_STAMP = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{3})")


# --- Time ---

def legacy_ms(stamp: Optional[str]) -> Optional[int]:
    """
    Milliseconds of the game's legacy stamp "yyyy_MM_dd_HH_mm_ss_fff". The stamp has no time zone, so it is read as UTC
    and only differences between two stamps are meaningful (see legacy_offset_ms to compare it with real UTC times).
    """

    match = _LEGACY_STAMP.match(stamp or "")
    if not match:
        return None
    year, month, day, hour, minute, second, milli = (int(group) for group in match.groups())
    try:
        moment = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(moment.timestamp()) * 1000 + milli


def point_time(point: dict) -> Optional[int]:
    """The time of a point: the UTC time when it has one, the legacy stamp otherwise."""

    utc = point.get("utcMs")
    return utc if utc is not None else legacy_ms(point.get("timestamp"))


def stroke_times(line: Line) -> Optional[tuple[int, int]]:
    """(start, end) of a stroke in milliseconds, or None when its points carry no usable time."""

    points = line.get("points") or []
    if not points:
        return None
    start, end = point_time(points[0]), point_time(points[-1])
    if start is None or end is None:
        return None
    return (start, end) if end >= start else (end, start)


def legacy_offset_ms(doc: Drawing) -> Optional[int]:
    """
    UTC minus legacy stamp. Points carry both stamps in schema version 2, while the line history only has the legacy
    local-time stamp, so this offset puts the history on the same time axis as the points and the events.
    """

    for line in doc.get("lines") or []:
        for point in line.get("points") or []:
            utc = point.get("utcMs")
            if utc is not None:
                legacy = legacy_ms(point.get("timestamp"))
                if legacy is not None:
                    return utc - legacy
    return None


def history_time(stamp: Optional[str], offset: Optional[int]) -> Optional[int]:
    legacy = legacy_ms(stamp)
    if legacy is None:
        return None
    return legacy + (offset or 0)


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def idle_periods(intervals: Iterable[tuple[int, int]], gap_ms: float) -> list[tuple[int, int]]:
    """
    The pauses between the (merged) drawing intervals that are longer than gap_ms. A gap longer than BREAK_SECONDS is
    not a pause but a break between two sittings (see TimeAxis), so it is not listed.
    """

    merged = merge_intervals(intervals)
    return [(a[1], b[0]) for a, b in zip(merged, merged[1:]) if gap_ms < b[0] - a[1] <= BREAK_SECONDS * 1000]


def breaks_between(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """The breaks between sittings: gaps between the (merged) drawing intervals longer than BREAK_SECONDS."""

    merged = merge_intervals(intervals)
    return [(a[1], b[0]) for a, b in zip(merged, merged[1:]) if b[0] - a[1] > BREAK_SECONDS * 1000]


class TimeAxis:
    """
    The time axis every analysis answer uses (replay, timeline, tracking, geometry): milliseconds from the start of the
    drawing's activity. A drawing can be continued after a break of hours or days ("Save As New" of a loaded drawing),
    so breaks longer than BREAK_SECONDS are cut down to BREAK_DISPLAY_MS on this axis. Otherwise one sitting would be
    squeezed into a corner of a days-long axis.
    """

    def __init__(self, doc: "Drawing") -> None:
        intervals = merge_intervals(t for t in (stroke_times(line) for line in all_lines(doc)) if t)
        self.breaks = breaks_between(intervals)
        if intervals:
            self.origin = intervals[0][0]
        else:  # no strokes: the first event or tracked pose
            others = [e["utcMs"] for e in doc.get("events") or [] if e.get("utcMs") is not None]
            for behavior in doc.get("trackedBehaviors") or []:
                for channel in ("head", "rightHand", "leftHand", "player"):
                    others += [p["utcMs"] for p in behavior.get(channel) or [] if p.get("utcMs") is not None]
            self.origin = min(others) if others else 0

    def map(self, moment: int) -> int:
        """Absolute time to the axis. A time inside a break lands at the start of the break."""

        removed = 0
        for start, end in self.breaks:
            if moment >= end:
                removed += (end - start) - BREAK_DISPLAY_MS
            elif moment > start:
                return start - self.origin - removed
            else:
                break
        return moment - self.origin - removed

    def break_marks(self) -> list[dict[str, int]]:
        """Where the breaks are on the axis and how long they really were."""

        return [{"atMs": self.map(start), "realMs": end - start} for start, end in self.breaks]


# --- Strokes ---

def all_lines(doc: Drawing) -> list[Line]:
    return doc.get("lines") or []


def drawn_lines(doc: Drawing) -> list[Line]:
    return [line for line in all_lines(doc) if line.get("status") == STATUS_DRAWN]


def compute_line_stats(points: list[dict]) -> dict:
    """Length, bounds and duration of a stroke, derived the way the server derives its line stats on save."""

    if not points:
        return {"pointCount": 0, "length": 0.0, "durationMs": None,
                "boundsMinX": 0.0, "boundsMinY": 0.0, "boundsMinZ": 0.0,
                "boundsMaxX": 0.0, "boundsMaxY": 0.0, "boundsMaxZ": 0.0}
    first = points[0]
    min_x = max_x = first["x"]
    min_y = max_y = first["y"]
    min_z = max_z = first["z"]
    length = 0.0
    previous = first
    for p in points:
        length += math.hypot(p["x"] - previous["x"], p["y"] - previous["y"], p["z"] - previous["z"])
        min_x, max_x = min(min_x, p["x"]), max(max_x, p["x"])
        min_y, max_y = min(min_y, p["y"]), max(max_y, p["y"])
        min_z, max_z = min(min_z, p["z"]), max(max_z, p["z"])
        previous = p
    times = stroke_times({"points": points})
    return {"pointCount": len(points), "length": length,
            "durationMs": max(0, times[1] - times[0]) if times else None,
            "boundsMinX": min_x, "boundsMinY": min_y, "boundsMinZ": min_z,
            "boundsMaxX": max_x, "boundsMaxY": max_y, "boundsMaxZ": max_z}


def line_stats(line: Line) -> dict:
    """The stored line stats (schema version 2) when present, derived from the points otherwise (requirement MET-06)."""

    stats = line.get("stats")
    if stats and stats.get("pointCount") is not None:
        return stats
    return compute_line_stats(line.get("points") or [])


def time_base(doc: Drawing) -> str:
    """'utc' when the points carry real UTC times (schema 2), 'legacy' when only the local-time stamps exist."""

    for line in all_lines(doc):
        for point in line.get("points") or []:
            if point.get("utcMs") is not None:
                return "utc"
            return "legacy"
    return "utc"


# --- Colour ---

def rgb_to_hex(color: dict) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, round(color.get(key, 0)))) for key in ("r", "g", "b")))


def hue_of(color: dict) -> tuple[float, bool]:
    """Hue in degrees and whether the colour has enough saturation to count as coloured at all."""

    r, g, b = color.get("r", 0), color.get("g", 0), color.get("b", 0)
    high, low = max(r, g, b), min(r, g, b)
    delta = high - low
    if delta < 25:
        return 0.0, False
    if high == r:
        hue = ((g - b) / delta) % 6
    elif high == g:
        hue = (b - r) / delta + 2
    else:
        hue = (r - g) / delta + 4
    return (hue * 60 + 360) % 360, True


def is_warm_hue(hue: float) -> bool:
    """Red, orange and yellow, and the magentas next to red, count as warm."""

    return hue < 70 or hue >= 300


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# --- Misc ---

def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def round_half_up(value: float, digits: int = 0) -> float:
    """
    Rounds .5 up, like the web app does. Python's round() rounds half to even, so the same number would read
    differently on a screen and in a report.
    """

    factor = 10 ** digits
    return math.floor(value * factor + 0.5) / factor


def number(value: float, digits: int = 4) -> float:
    """Rounds a float for JSON and turns -0.0 into 0.0."""

    return round_half_up(value, digits) + 0.0
