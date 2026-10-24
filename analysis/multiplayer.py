"""
Analytics of drawings made by several authors (requirements MUL-01..03).

The author of a stroke is taken from the stroke itself (userID): the collaborator list of a drawing is not reliably
filled by the game.
"""

import math
from itertools import combinations
from typing import Any, Optional

from analysis.common import (
    Drawing, TimeAxis, all_lines, drawn_lines, line_stats, merge_intervals, number, stroke_times,
)

PROXIMITY_RADIUS = 0.05  # metres: strokes closer than this count as touching
TURN_GAP_SECONDS = 5.0  # strokes of one author closer in time than this belong to one turn


def _author_points(doc: Drawing) -> dict[str, list[tuple[float, float, float, str]]]:
    """The points of the drawn strokes per author, each tagged with its stroke."""

    points: dict[str, list[tuple[float, float, float, str]]] = {}
    for line in drawn_lines(doc):
        author = line.get("userID")
        for p in line.get("points") or []:
            points.setdefault(author, []).append((p["x"], p["y"], p["z"], line.get("id")))
    return points


def _cell(p: tuple, size: float) -> tuple[int, int, int]:
    return math.floor(p[0] / size), math.floor(p[1] / size), math.floor(p[2] / size)


def _has_neighbour(p: tuple, grid: dict[tuple[int, int, int], list], radius: float) -> bool:
    cx, cy, cz = _cell(p, radius)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if any(math.dist(p[:3], q[:3]) <= radius for q in grid.get((cx + dx, cy + dy, cz + dz), ())):
                    return True
    return False


def _near(source: list, target: list, radius: float) -> tuple[float, set[str]]:
    """Share of the source points within radius of any target point, and the source strokes that touch the target."""

    grid: dict[tuple[int, int, int], list] = {}
    for q in target:
        grid.setdefault(_cell(q, radius), []).append(q)
    touching: set[str] = set()
    near = 0
    for p in source:
        if _has_neighbour(p, grid, radius):
            near += 1
            touching.add(p[3])
    return (near / len(source) if source else 0.0), touching


def _min_distance(a: list, b: list) -> Optional[float]:
    """The smallest distance between two sets of points (on an even subsample of at most 400 points each)."""

    if not a or not b:
        return None
    sa = a[::max(1, len(a) // 400)]
    sb = b[::max(1, len(b) // 400)]
    return min(math.dist(p[:3], q[:3]) for p in sa for q in sb)


def _simultaneous_ms(intervals: list[tuple[int, int, str]]) -> int:
    """Time during which strokes of at least two different authors were being drawn (sweep over the interval ends)."""

    events = sorted([(start, 1, author) for start, _, author in intervals] +
                    [(end, -1, author) for _, end, author in intervals], key=lambda e: (e[0], e[1]))
    active: dict[str, int] = {}
    total = 0
    last = 0
    for moment, change, author in events:
        if sum(1 for count in active.values() if count > 0) >= 2:
            total += moment - last
        active[author] = active.get(author, 0) + change
        last = moment
    return total


def analyze_multiplayer(
        doc: Drawing, proximity: float = PROXIMITY_RADIUS, turn_gap_seconds: float = TURN_GAP_SECONDS,
) -> dict[str, Any]:
    axis = TimeAxis(doc)
    authors = list(dict.fromkeys(line.get("userID") for line in all_lines(doc)))
    drawn = drawn_lines(doc)

    # Contribution (MUL-01)
    total_ink = sum(line_stats(line)["length"] for line in drawn) or 1.0
    contributions = []
    for author in authors:
        own = [line for line in drawn if line.get("userID") == author]
        times = [t for t in (stroke_times(line) for line in own) if t]
        ink = sum(line_stats(line)["length"] for line in own)
        contributions.append({
            "playerId": author,
            "strokes": len(own),
            "erasedStrokes": sum(1 for line in all_lines(doc) if line.get("userID") == author) - len(own),
            "inkLength": number(ink),
            "inkShare": number(ink / total_ink),
            "drawingMs": sum(end - start for start, end in merge_intervals(times)) if times else None,
            "firstMs": axis.map(min(t[0] for t in times)) if times else None,
            "lastMs": axis.map(max(t[1] for t in times)) if times else None,
        })

    # Proximity and overlap between the strokes of different authors (MUL-02)
    points = _author_points(doc)
    pairs = []
    for a, b in combinations([author for author in authors if author in points], 2):
        a_near_b, a_touching = _near(points[a], points[b], proximity)
        b_near_a, b_touching = _near(points[b], points[a], proximity)
        closest = _min_distance(points[a], points[b])
        pairs.append({
            "a": a, "b": b,
            "aNearB": number(a_near_b), "bNearA": number(b_near_a),
            "touchingStrokesA": len(a_touching), "touchingStrokesB": len(b_touching),
            "minDistance": None if closest is None else number(closest),
        })

    # Turn-taking (MUL-03): consecutive strokes of one author close in time form a turn
    stroke_list = sorted(
        (times[0], times[1], line.get("userID"))
        for line in drawn for times in [stroke_times(line)] if times
    )
    absolute: list[dict[str, Any]] = []  # real times while merging, put on the time axis afterwards
    for start, end, author in stroke_list:
        last = absolute[-1] if absolute else None
        if last and last["playerId"] == author and start - last["end"] <= turn_gap_seconds * 1000:
            last["end"] = max(last["end"], end)
        else:
            absolute.append({"playerId": author, "start": start, "end": end})
    blocks = [{"playerId": b["playerId"], "startMs": axis.map(b["start"]), "endMs": axis.map(b["end"])} for b in absolute]
    active_ms = sum(end - start for start, end in merge_intervals((s, e) for s, e, _ in stroke_list))
    simultaneous = _simultaneous_ms(stroke_list)

    return {
        "drawingId": (doc.get("metadata") or {}).get("id"),
        "multiAuthor": len(authors) > 1,
        "authors": contributions,
        "proximity": {"radius": proximity, "pairs": pairs},
        "turns": {
            "gapSeconds": turn_gap_seconds,
            "blocks": blocks,
            "switches": sum(1 for a, b in zip(blocks, blocks[1:]) if a["playerId"] != b["playerId"]),
            "simultaneousMs": simultaneous,
            "simultaneousShare": number(simultaneous / active_ms) if active_ms else 0.0,
        },
    }
