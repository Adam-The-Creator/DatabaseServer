"""Synthetic drawings with known answers for the analysis tests (requirement MET-04)."""

from datetime import datetime, timezone
from typing import Any, Optional

from analysis.common import compute_line_stats

T0 = 1_000_000  # start of the drawing in UTC milliseconds (an arbitrary but fixed value)
LOCAL_SHIFT_MS = 2 * 3600 * 1000  # the legacy stamps are local time: two hours ahead of UTC in these fixtures


def legacy_stamp(local_ms: int) -> str:
    moment = datetime.fromtimestamp(local_ms / 1000, tz=timezone.utc)
    return moment.strftime("%Y_%m_%d_%H_%M_%S_") + f"{local_ms % 1000:03d}"


def straight(length: float = 1.0, count: int = 11, y: float = 1.0, z: float = 0.0, x0: float = 0.0) -> list[tuple]:
    return [(x0 + i * length / (count - 1), y, z) for i in range(count)]


def make_line(
        line_id: str, coords: list[tuple], *, start_ms: int = T0, step_ms: int = 100, hand: int = 1, status: int = 1,
        color: tuple = (255, 0, 0), end_color: Optional[tuple] = None, author: str = "p1", utc: bool = True,
        pressure: Optional[float] = None, stats: bool = True, history: Optional[list] = None,
) -> dict[str, Any]:
    points = []
    for i, (x, y, z) in enumerate(coords):
        t = start_ms + i * step_ms
        point = {"x": x, "y": y, "z": z, "timestamp": legacy_stamp(t + LOCAL_SHIFT_MS)}
        if utc:
            point["utcMs"] = t
            if pressure is not None:
                point["pressure"] = pressure
        points.append(point)
    r, g, b = color
    er, eg, eb = end_color or color
    line: dict[str, Any] = {
        "id": line_id, "points": points, "startWidth": 0.005, "endWidth": 0.005,
        "startColor": {"r": r, "g": g, "b": b, "a": 1.0}, "endColor": {"r": er, "g": eg, "b": eb, "a": 1.0},
        "hand": hand, "userID": author, "status": status,
        "history": history if history is not None else [{
            "eventType": 1, "invoker": author, "hand": hand, "timestamp": points[0]["timestamp"]}],
    }
    if stats:
        line["stats"] = compute_line_stats(points)
    return line


def make_drawing(
        lines: list[dict], *, drawing_id: str = "d1", owner: str = "p1", game_type: int = 2, schema: Optional[int] = 2,
        events: Optional[list] = None, tracked: Optional[list] = None, placed: Optional[list] = None,
        decimation: Optional[int] = None, figure: Optional[dict] = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "id": drawing_id, "owner": owner, "collaborators": [], "version": "1.0", "date": "2026.10.01-10:00",
        "gameType": game_type, "sessionID": "s1", "showBoy": True, "showGirl": True, "sadChildCoordinates": figure,
    }
    if schema is not None:
        metadata.update({"schemaVersion": schema, "appVersion": "test", "device": "test"})
    if decimation:
        metadata["trackingDecimation"] = decimation
    return {"metadata": metadata, "lines": lines, "events": events if events is not None else [],
            "trackedBehaviors": tracked or [], "placedModels": placed or []}


def pose(x: float, y: float, z: float, t: int, q: tuple = (0.0, 0.0, 0.0, 1.0)) -> dict[str, Any]:
    return {"pX": x, "pY": y, "pZ": z, "rX": q[0], "rY": q[1], "rZ": q[2], "rW": q[3], "utcMs": t}
