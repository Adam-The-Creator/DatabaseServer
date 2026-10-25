"""
Canonical views of a drawing as PNG images (requirements AI-12, AI-13): front, top, side and isometric, of the whole
drawing and of a group of strokes. The strokes are drawn in their own colours on a white background, which is what the
image models were trained on. These are exactly the images the models see, and the web app can show them (AI-04,
evidence). Standard library and numpy only: no imaging library.
"""

import math
import struct
import zlib
from typing import Optional

import numpy as np

from analysis.common import Drawing, drawn_lines

VIEWS = ("front", "top", "side", "iso")
SIZE = 336
PAD = 18
ISO_YAW = math.radians(35)
ISO_PITCH = math.radians(25)


def _project(points: np.ndarray, view: str) -> np.ndarray:
    """Game space (x right, y up, z forward) to image space (u right, v up)."""

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if view == "front":
        return np.column_stack([x, y])
    if view == "top":
        return np.column_stack([x, z])
    if view == "side":
        return np.column_stack([z, y])
    if view == "iso":
        across = x * math.cos(ISO_YAW) - z * math.sin(ISO_YAW)
        depth = x * math.sin(ISO_YAW) + z * math.cos(ISO_YAW)
        return np.column_stack([across, y * math.cos(ISO_PITCH) - depth * math.sin(ISO_PITCH)])
    raise ValueError("Unknown view")


def _disk(radius: int) -> list[tuple[int, int]]:
    return [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius + radius]


def render_view(doc: Drawing, view: str, stroke_ids: Optional[set] = None, size: int = SIZE) -> bytes:
    """A PNG of the drawn strokes (or only the strokes with the given ids) seen from one side."""

    strokes = []
    for line in drawn_lines(doc):
        points = line.get("points") or []
        if not points or (stroke_ids is not None and line.get("id") not in stroke_ids):
            continue
        xyz = np.array([[p["x"], p["y"], p["z"]] for p in points], dtype=float)
        start, end = line.get("startColor") or {}, line.get("endColor") or line.get("startColor") or {}
        strokes.append((_project(xyz, view),
                        np.array([start.get("r", 0), start.get("g", 0), start.get("b", 0)], dtype=float),
                        np.array([end.get("r", 0), end.get("g", 0), end.get("b", 0)], dtype=float),
                        (line.get("startWidth", 0.005) + line.get("endWidth", 0.005)) / 2))

    image = np.full((size, size, 3), 255, dtype=np.uint8)
    if strokes:
        stacked = np.vstack([s[0] for s in strokes])
        low, high = stacked.min(axis=0), stacked.max(axis=0)
        span = max(float(high[0] - low[0]), float(high[1] - low[1]), 0.05)
        scale = (size - 2 * PAD) / span
        centre = (low + high) / 2
        for uv, start, end, width in strokes:
            px = (uv[:, 0] - centre[0]) * scale + size / 2
            py = size / 2 - (uv[:, 1] - centre[1]) * scale  # image y points down
            along = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
            total = float(along[-1])
            if total > 0:
                steps = np.arange(0.0, total, 0.7)
                sx, sy = np.interp(steps, along, px), np.interp(steps, along, py)
                fraction = steps / total
            else:  # a tap: one point
                steps, sx, sy, fraction = np.array([0.0]), px[:1], py[:1], np.zeros(1)
            colours = (start[None, :] * (1 - fraction[:, None]) + end[None, :] * fraction[:, None]).clip(0, 255).astype(np.uint8)
            radius = int(min(5, max(2, round(width * scale / 2))))
            xs, ys = np.round(sx).astype(int), np.round(sy).astype(int)
            for dx, dy in _disk(radius):
                x, y = xs + dx, ys + dy
                inside = (x >= 0) & (x < size) & (y >= 0) & (y < size)
                image[y[inside], x[inside]] = colours[inside]
    return encode_png(image)


def encode_png(image: np.ndarray) -> bytes:
    """An 8-bit RGB PNG of an (height, width, 3) array."""

    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
