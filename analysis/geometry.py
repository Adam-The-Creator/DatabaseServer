"""
Geometry for display and export: simplified strokes (requirements DRW-03, TEC-06) and the SVG snapshots used for
thumbnails and reports (SES-02, PLY-01, REP-02).
"""

import math
from typing import Any

from analysis.common import (
    Drawing, TimeAxis, all_lines, line_stats, number, point_time, rgb_to_hex, stroke_times, time_base,
)


def _distance_to_segment(p: tuple, a: tuple, b: tuple) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    length_sq = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2
    if length_sq == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1] + (p[2] - a[2]) * ab[2]) / length_sq))
    return math.dist(p, (a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t))


def simplify_indices(points: list[tuple], tolerance: float) -> list[int]:
    """Ramer-Douglas-Peucker: the indices of the points to keep, so no dropped point is further than tolerance."""

    count = len(points)
    if tolerance <= 0 or count <= 2:
        return list(range(count))
    keep = {0, count - 1}
    stack = [(0, count - 1)]
    while stack:  # iterative: a stroke can have thousands of points
        start, end = stack.pop()
        far_index, far_distance = -1, 0.0
        for i in range(start + 1, end):
            distance = _distance_to_segment(points[i], points[start], points[end])
            if distance > far_distance:
                far_index, far_distance = i, distance
        if far_distance > tolerance:
            keep.add(far_index)
            stack += [(start, far_index), (far_index, end)]
    return sorted(keep)


def stroke_geometry(doc: Drawing, tolerance: float = 0.0) -> dict[str, Any]:
    """
    The drawing as the viewer needs it: strokes with simplified points [x, y, z, t] (t in ms on the time axis of the
    drawing, or null without time data), the placed models and the few metadata fields. tolerance 0 keeps every point.
    """

    metadata = doc.get("metadata") or {}
    axis = TimeAxis(doc)
    strokes = []
    original = returned = 0
    duration = 0
    for line in all_lines(doc):
        points = line.get("points") or []
        coords = [(p["x"], p["y"], p["z"]) for p in points]
        keep = simplify_indices(coords, tolerance)
        rows = []
        for i in keep:
            t = point_time(points[i])
            rows.append([number(coords[i][0], 5), number(coords[i][1], 5), number(coords[i][2], 5),
                         None if t is None else axis.map(t)])
        stats = line_stats(line)
        times = stroke_times(line)
        if times:
            duration = max(duration, axis.map(times[1]))
        original += len(points)
        returned += len(rows)
        strokes.append({
            "id": line.get("id"),
            "authorId": line.get("userID"),
            "hand": line.get("hand"),
            "erased": line.get("status") != 1,
            "width": number((line.get("startWidth", 0.005) + line.get("endWidth", 0.005)) / 2, 5),
            "startColor": {k: (line.get("startColor") or {}).get(k, 0) for k in ("r", "g", "b")},
            "endColor": {k: (line.get("endColor") or line.get("startColor") or {}).get(k, 0) for k in ("r", "g", "b")},
            "length": number(stats["length"]),
            "pointCount": stats["pointCount"],
            "durationMs": None if times is None else times[1] - times[0],
            "points": rows,
        })

    return {
        "drawingId": metadata.get("id"),
        "tolerance": tolerance,
        "originMs": axis.origin,
        "timeBase": time_base(doc),
        "durationMs": duration,  # end of the last stroke on the time axis, the length of a replay
        "breaks": axis.break_marks(),  # breaks between sittings, cut down on the axis
        "pointCountOriginal": original,
        "pointCountReturned": returned,
        "hasTracking": any(
            any(b.get(c) for c in ("head", "rightHand", "leftHand", "player")) for b in doc.get("trackedBehaviors") or []
        ),
        "hasEvents": bool(doc.get("events")),
        "metadata": {key: metadata.get(key) for key in (
            "id", "owner", "collaborators", "date", "gameType", "sessionID", "showBoy", "showGirl",
            "sadChildCoordinates", "schemaVersion", "appVersion", "device", "trackingDecimation")},
        "placedModels": doc.get("placedModels") or [],
        "strokes": strokes,
    }


# --- SVG snapshots (front, top and side view of the drawn strokes) ---

def drawing_svg(doc: Drawing, view: str = "front", width: int = 240, height: int = 180,
                background: str = "#f1f5f9") -> str:
    """
    A static SVG snapshot of the drawn strokes from one side. Same projection as the web viewer (z is negated, so
    the drawing is not mirrored): front is x/y, top is x/z, side is z/y.
    """

    axes = {"front": (0, 1, True), "top": (0, 2, False), "side": (2, 1, True)}[view]
    ax, ay, flip_y = axes

    strokes = []
    for line in all_lines(doc):
        if line.get("status") != 1 or len(line.get("points") or []) < 2:
            continue
        pts = [(p["x"], p["y"], -p["z"]) for p in line["points"]]
        strokes.append((line, pts))

    pad = 12
    if not strokes:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
                f'height="{height}"><rect width="{width}" height="{height}" fill="{background}"/></svg>')

    lows = [min(p[axis] for _, pts in strokes for p in pts) for axis in range(3)]
    highs = [max(p[axis] for _, pts in strokes for p in pts) for axis in range(3)]
    span_x = max(highs[ax] - lows[ax], 0.05)
    span_y = max(highs[ay] - lows[ay], 0.05)
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
    offset_x = (width - span_x * scale) / 2
    offset_y = (height - span_y * scale) / 2

    def to_x(v: float) -> float:
        return offset_x + (v - lows[ax]) * scale

    def to_y(v: float) -> float:
        return offset_y + ((highs[ay] - v) if flip_y else (v - lows[ay])) * scale

    paths = []
    for line, pts in strokes:
        coordinates = " ".join(f"{to_x(p[ax]):.1f},{to_y(p[ay]):.1f}" for p in pts)
        stroke_width = min(4.0, max(1.2, ((line.get("startWidth", 0.005) + line.get("endWidth", 0.005)) / 2) * scale))
        paths.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{rgb_to_hex(line.get("startColor") or {})}" '
            f'stroke-width="{stroke_width:.1f}" stroke-linecap="round" stroke-linejoin="round"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}"><rect width="{width}" height="{height}" fill="{background}"/>{"".join(paths)}</svg>')
