"""
The 3D density heatmap of a drawing (requirement DRW-06): how much ink lies in every voxel (cube) of space.
The web app draws the voxels in 3D and sums them up into the top, front and side projections.
"""

import math
from typing import Any

from analysis.common import Drawing, drawn_lines, number

MAX_CELLS = 40_000  # a finer grid than this is coarsened, so the response stays small


def _accumulate(doc: Drawing, voxel: float) -> tuple[dict[tuple[int, int, int], float], float]:
    """Ink length per voxel index (floor(position / voxel)). Every segment is sampled at least every half voxel."""

    cells: dict[tuple[int, int, int], float] = {}
    total = 0.0
    for line in drawn_lines(doc):
        points = line.get("points") or []
        for a, b in zip(points, points[1:]):
            length = math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))
            if length == 0:
                continue
            total += length
            samples = max(1, math.ceil(length / (voxel / 2)))
            share = length / samples
            for i in range(samples):
                f = (i + 0.5) / samples
                key = (
                    math.floor((a["x"] + (b["x"] - a["x"]) * f) / voxel),
                    math.floor((a["y"] + (b["y"] - a["y"]) * f) / voxel),
                    math.floor((a["z"] + (b["z"] - a["z"]) * f) / voxel),
                )
                cells[key] = cells.get(key, 0.0) + share
    return cells, total


def build_heatmap(doc: Drawing, voxel_size: float = 0.05) -> dict[str, Any]:
    """
    Sparse voxel grid: only the voxels with ink are returned, as [i, j, k, ink length in metres]. The voxel with index
    (i, j, k) spans origin + (i, j, k) * voxelSize, in game coordinates. The voxel size grows when it would give too
    many cells; the size that was used is in the answer.
    """

    voxel = max(voxel_size, 0.005)
    cells, total = _accumulate(doc, voxel)
    while len(cells) > MAX_CELLS:
        voxel *= 2
        cells, total = _accumulate(doc, voxel)

    metadata = doc.get("metadata") or {}
    figure = metadata.get("sadChildCoordinates")
    if not cells:
        return {"drawingId": metadata.get("id"), "voxelSize": voxel, "origin": [0, 0, 0], "dims": [0, 0, 0],
                "cells": [], "maxWeight": 0, "totalInk": 0, "figure": None}

    mins = [min(key[axis] for key in cells) for axis in range(3)]
    maxs = [max(key[axis] for key in cells) for axis in range(3)]
    rows = sorted(
        ([key[0] - mins[0], key[1] - mins[1], key[2] - mins[2], number(weight)] for key, weight in cells.items()),
        key=lambda row: -row[3],
    )
    return {
        "drawingId": metadata.get("id"),
        "voxelSize": number(voxel, 4),
        "origin": [number(mins[axis] * voxel, 4) for axis in range(3)],
        "dims": [maxs[axis] - mins[axis] + 1 for axis in range(3)],
        "cells": rows,
        "maxWeight": rows[0][3],
        "totalInk": number(total),
        "figure": {"x": figure["x"], "y": figure["y"], "z": figure["z"]} if figure else None,
    }
