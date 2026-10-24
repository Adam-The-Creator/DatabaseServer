"""
The tracked head and hand movement of a drawing (requirement DRW-08): paths, the amount of movement, where the head
was turned, and how much space was explored.

The game stores the poses of every player (head, both hands, body root). Tracking that was thinned out to fit the
storage limit (metadata.trackingDecimation > 1) is flagged as approximate.
"""

import math
from typing import Any, Optional

from analysis.common import Drawing, TimeAxis, legacy_ms, number

CHANNELS = ("head", "rightHand", "leftHand", "player")
YAW_BINS = 12
EXPLORED_CELL = 0.25  # metres: the floor is divided into cells of this size to measure the explored area


def pose_time(pose: dict) -> Optional[int]:
    utc = pose.get("utcMs")
    return utc if utc is not None else legacy_ms(pose.get("timestamp"))


def forward_vector(pose: dict) -> tuple[float, float, float]:
    """The direction the pose looks along (the rotation applied to Unity's forward axis (0, 0, 1))."""

    x, y, z, w = pose["rX"], pose["rY"], pose["rZ"], pose["rW"]
    return 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)


def _distance(poses: list[dict]) -> float:
    return sum(
        math.dist((a["pX"], a["pY"], a["pZ"]), (b["pX"], b["pY"], b["pZ"])) for a, b in zip(poses, poses[1:])
    )


def _path(poses: list[dict], axis: TimeAxis, max_points: int, with_forward: bool) -> list[list[Optional[float]]]:
    """
    [t, x, y, z] (and the forward vector for the head) of at most max_points evenly spread samples. t is on the time
    axis of the drawing, so it can be negative for samples before the first stroke.
    """

    if not poses:
        return []
    step = max(1, math.ceil(len(poses) / max_points))
    sampled = poses[::step]
    if sampled[-1] is not poses[-1]:
        sampled.append(poses[-1])
    path: list[list[Optional[float]]] = []
    for pose in sampled:
        t = pose_time(pose)
        row: list[Optional[float]] = [None if t is None else axis.map(t),
                                      number(pose["pX"]), number(pose["pY"]), number(pose["pZ"])]
        if with_forward:
            row += [number(c, 3) for c in forward_vector(pose)]
        path.append(row)
    return path


def _head_direction(head: list[dict]) -> Optional[dict[str, Any]]:
    if not head:
        return None
    sum_x = sum_y = sum_z = 0.0
    bins = [0] * YAW_BINS
    for pose in head:
        fx, fy, fz = forward_vector(pose)
        norm = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        sum_x, sum_y, sum_z = sum_x + fx / norm, sum_y + fy / norm, sum_z + fz / norm
        yaw = math.degrees(math.atan2(fx, fz)) % 360
        bins[int(yaw // (360 / YAW_BINS)) % YAW_BINS] += 1
    mean_norm = math.sqrt(sum_x ** 2 + sum_y ** 2 + sum_z ** 2) or 1.0
    mean = (sum_x / mean_norm, sum_y / mean_norm, sum_z / mean_norm)
    shares = [count / len(head) for count in bins]
    return {
        "meanYawDeg": number(math.degrees(math.atan2(mean[0], mean[2])) % 360, 1),
        "meanPitchDeg": number(math.degrees(math.asin(max(-1.0, min(1.0, mean[1])))), 1),
        "yawShares": [number(s, 3) for s in shares],  # shares of the samples per 30 degree sector, from yaw 0
        "yawCoverage": sum(1 for s in shares if s >= 0.05),  # sectors the head spent at least 5 % of the time in
    }


def _explored(head: list[dict]) -> Optional[dict[str, Any]]:
    if not head:
        return None
    cells = {(math.floor(p["pX"] / EXPLORED_CELL), math.floor(p["pZ"] / EXPLORED_CELL)) for p in head}
    heights = [p["pY"] for p in head]
    return {"areaM2": number(len(cells) * EXPLORED_CELL ** 2, 2),
            "headHeightMin": number(min(heights), 2), "headHeightMax": number(max(heights), 2)}


def summarize_tracking(doc: Drawing, max_points: int = 400) -> dict[str, Any]:
    metadata = doc.get("metadata") or {}
    behaviors = [b for b in doc.get("trackedBehaviors") or [] if any(b.get(c) for c in CHANNELS)]
    decimation = metadata.get("trackingDecimation") or 1
    axis = TimeAxis(doc)

    players = []
    for behavior in behaviors:
        head = behavior.get("head") or []
        times = [t for t in (pose_time(p) for p in head) if t is not None]
        players.append({
            "playerId": behavior.get("playerID"),
            "samples": {channel: len(behavior.get(channel) or []) for channel in CHANNELS},
            "durationMs": max(times) - min(times) if len(times) > 1 else None,
            "movement": {channel: number(_distance(behavior.get(channel) or []), 2) for channel in CHANNELS},
            "headDirection": _head_direction(head),
            "explored": _explored(head),
            "paths": {
                "head": _path(head, axis, max_points, True),
                "rightHand": _path(behavior.get("rightHand") or [], axis, max_points, False),
                "leftHand": _path(behavior.get("leftHand") or [], axis, max_points, False),
            },
        })

    return {
        "drawingId": metadata.get("id"),
        "available": bool(players),
        "originMs": axis.origin,
        "approximate": decimation > 1,
        "decimation": decimation if decimation > 1 else None,
        "players": players,
    }
