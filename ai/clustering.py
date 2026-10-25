"""
Stroke clustering into object-like groups (requirement AI-11, no pretrained model): strokes that were drawn close to
each other in space and time are one group. The groups are the basis of the semantic labelling (AI-12) and the
description (AI-13) and can be highlighted in the viewer.

How: the strokes are resampled to a point every 2 cm, DBSCAN (scikit-learn) groups the points by their distance in
space and, when the drawing has time data, in time (a time of `timeEps` seconds counts like a distance of `spaceEps`
metres), and strokes that share a cluster are merged into one group. Strokes whose points are all noise stay ungrouped.
The result describes how the lines lie, it does not know what they depict.
"""

import math
from typing import Any

from analysis.common import (
    Drawing, TimeAxis, drawn_lines, line_stats, number, point_time, rgb_to_hex,
)

SPACING = 0.02          # metres between the resampled points
MIN_POINTS = 3          # DBSCAN: points needed to form a dense region
MAX_GROUPS = 60


def resample(points: list[dict], spacing: float = SPACING) -> list[int]:
    """Indices of the points to keep: the first, the last, and one each time the stroke has moved `spacing` further."""

    if not points:
        return []
    keep = [0]
    last = points[0]
    for i in range(1, len(points) - 1):
        p = points[i]
        if math.dist((p["x"], p["y"], p["z"]), (last["x"], last["y"], last["z"])) >= spacing:
            keep.append(i)
            last = p
    if len(points) > 1:
        keep.append(len(points) - 1)
    return keep


def _top_colours(stroke_ink: list[tuple[dict, float]], limit: int = 3) -> list[dict[str, Any]]:
    bins: dict[tuple[int, int, int], list] = {}
    total = sum(ink for _, ink in stroke_ink)
    for color, ink in stroke_ink:
        key = (color.get("r", 0) >> 5, color.get("g", 0) >> 5, color.get("b", 0) >> 5)
        entry = bins.setdefault(key, [rgb_to_hex(color), 0.0])
        entry[1] += ink
    ranked = sorted(bins.values(), key=lambda e: -e[1])[:limit]
    return [{"hex": hex_, "share": number(ink / total, 3) if total > 0 else 0.0} for hex_, ink in ranked]


def cluster_strokes(doc: Drawing, space_eps: float = 0.10, time_eps: float = 20.0) -> dict[str, Any]:
    import numpy as np
    from sklearn.cluster import DBSCAN

    lines = [line for line in drawn_lines(doc) if line.get("points")]
    axis = TimeAxis(doc)
    coords: list[list[float]] = []
    owner: list[int] = []
    times: list[float] = []
    timed = True
    for index, line in enumerate(lines):
        points = line["points"]
        for i in resample(points):
            p = points[i]
            moment = point_time(p)
            if moment is None:
                timed = False
            else:
                times.append(axis.map(moment) / 1000.0)
            coords.append([p["x"], p["y"], p["z"]])
            owner.append(index)

    labels = np.array([], dtype=int)
    if coords:
        matrix = np.array(coords, dtype=float)
        if timed:
            matrix = np.column_stack([matrix, np.array(times) / time_eps * space_eps])
        labels = DBSCAN(eps=space_eps, min_samples=MIN_POINTS, metric="chebyshev").fit(matrix).labels_

    # Strokes that share a dense region belong together (union-find over the strokes)
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_in_cluster: dict[int, int] = {}
    clustered = set()
    for label, stroke in zip(labels.tolist(), owner):
        if label < 0:
            continue
        clustered.add(stroke)
        if label in first_in_cluster:
            parent[find(stroke)] = find(first_in_cluster[label])
        else:
            first_in_cluster[label] = stroke

    members: dict[int, list[int]] = {}
    for stroke in sorted(clustered):
        members.setdefault(find(stroke), []).append(stroke)

    stats = [line_stats(line) for line in lines]
    total_ink = sum(s["length"] for s in stats)
    groups = []
    for stroke_indices in members.values():
        group_stats = [stats[i] for i in stroke_indices]
        ink = sum(s["length"] for s in group_stats)
        low = [min(s[f"boundsMin{axis_}"] for s in group_stats) for axis_ in "XYZ"]
        high = [max(s[f"boundsMax{axis_}"] for s in group_stats) for axis_ in "XYZ"]
        spans = [(point_time(lines[i]["points"][0]), point_time(lines[i]["points"][-1])) for i in stroke_indices]
        spans = [(axis.map(a), axis.map(b)) for a, b in spans if a is not None and b is not None]
        groups.append({
            "strokeIds": [lines[i].get("id") for i in stroke_indices],
            "strokeCount": len(stroke_indices),
            "authors": sorted({lines[i].get("userID") for i in stroke_indices if lines[i].get("userID")}),
            "length": number(ink),
            "inkShare": number(ink / total_ink, 3) if total_ink > 0 else 0.0,
            "boundsMin": [number(v, 3) for v in low],
            "boundsMax": [number(v, 3) for v in high],
            "size": [number(h - l, 3) for l, h in zip(low, high)],
            "centre": [number((l + h) / 2, 3) for l, h in zip(low, high)],
            "colours": _top_colours([(lines[i].get("startColor") or {}, stats[i]["length"]) for i in stroke_indices]),
            "startMs": min((a for a, _ in spans), default=None),
            "endMs": max((b for _, b in spans), default=None),
        })
    groups.sort(key=lambda g: (-g["length"], g["strokeIds"][0] or ""))
    truncated = len(groups) > MAX_GROUPS
    groups = groups[:MAX_GROUPS]
    for number_, group in enumerate(groups, 1):
        group["id"] = f"g{number_}"
    return {
        "strokeCount": len(lines),
        "groups": groups,
        "unclustered": [lines[i].get("id") for i in range(len(lines)) if i not in clustered],
        "timeUsed": timed and bool(coords),
        "truncated": truncated,
    }
