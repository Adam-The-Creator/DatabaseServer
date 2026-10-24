"""
Exports (requirements EXP-01, EXP-03): the metrics of a session as CSV, and the strokes of a drawing as glTF, OBJ or
PLY. All of them describe the strokes as lines. The width of the strokes is not exported.

The 3D files use a right-handed, Y-up system like most 3D tools, so the game's z axis is negated (otherwise the
drawing would be mirrored).
"""

import base64
import csv
import io
import json
import struct
from typing import Any, Optional

from analysis import catalog
from analysis.common import Drawing, all_lines, drawn_lines, round_half_up, srgb_to_linear

CSV_BASE_COLUMNS = [
    "drawingId", "drawingName", "sessionId", "sessionName", "gameType", "date",
    "participantId", "participantName", "age", "gender", "dominantHand",
]
CSV_DIALECTS = ("r", "excel")


# --- CSV ---

def _cell(value: Any, dialect: str) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text.replace(".", ",") if dialect == "excel" else text
    return str(value)


def session_csv(rows: list[dict[str, Any]], dialect: str = "r") -> bytes:
    """
    One row per drawing: the drawing and participant fields, then every metric (empty when it has no value).
    Dialect "r": comma separated with a decimal point, UTF-8 (R, SPSS, pandas). Dialect "excel": semicolon separated with
    a decimal comma and a byte order mark, which a Hungarian or German Excel opens correctly.
    """

    if dialect not in CSV_DIALECTS:
        raise ValueError(f"Unknown CSV dialect: {dialect}")
    out = io.StringIO()
    writer = csv.writer(out, delimiter=";" if dialect == "excel" else ",", lineterminator="\r\n")
    writer.writerow(CSV_BASE_COLUMNS + catalog.METRIC_IDS)
    for row in rows:
        writer.writerow([_cell(row.get(column), dialect) for column in CSV_BASE_COLUMNS + catalog.METRIC_IDS])
    return out.getvalue().encode("utf-8-sig" if dialect == "excel" else "utf-8")


# --- 3D files ---

def _strokes(doc: Drawing, include_erased: bool) -> list[dict[str, Any]]:
    """The strokes with at least two points: positions in the export's coordinate system and the colour per point."""

    result = []
    for line in (all_lines(doc) if include_erased else drawn_lines(doc)):
        points = line.get("points") or []
        if len(points) < 2:
            continue
        start, end = line.get("startColor") or {}, line.get("endColor") or line.get("startColor") or {}
        last = len(points) - 1
        positions, colors = [], []
        for i, p in enumerate(points):
            t = i / last
            positions.append((p["x"], p["y"], -p["z"]))
            colors.append(tuple(
                (start.get(k, 0) + t * (end.get(k, 0) - start.get(k, 0))) / 255 for k in ("r", "g", "b")
            ))
        result.append({"id": line.get("id"), "positions": positions, "colors": colors})
    return result


def to_gltf(doc: Drawing, include_erased: bool = False) -> bytes:
    """A glTF 2.0 file (JSON with the data embedded): one line strip per stroke, with vertex colours."""

    blob = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    primitives: list[dict[str, Any]] = []

    def add_attribute(values: list[tuple[float, float, float]], with_bounds: bool) -> int:
        data = b"".join(struct.pack("<3f", *v) for v in values)
        buffer_views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": 34962})
        blob.extend(data)
        accessor: dict[str, Any] = {"bufferView": len(buffer_views) - 1, "componentType": 5126,
                                    "count": len(values), "type": "VEC3"}
        if with_bounds:  # glTF requires the bounds of the positions
            accessor["min"] = [min(v[axis] for v in values) for axis in range(3)]
            accessor["max"] = [max(v[axis] for v in values) for axis in range(3)]
        accessors.append(accessor)
        return len(accessors) - 1

    for stroke in _strokes(doc, include_erased):
        linear = [tuple(srgb_to_linear(c) for c in color) for color in stroke["colors"]]  # glTF colours are linear
        primitives.append({
            "attributes": {"POSITION": add_attribute(stroke["positions"], True), "COLOR_0": add_attribute(linear, False)},
            "mode": 3,  # LINE_STRIP
            "material": 0,
        })

    metadata = doc.get("metadata") or {}
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "VR Drawing 3D analysis"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": metadata.get("id", "drawing")}],
        "meshes": [{"name": "strokes", "primitives": primitives}],
        "materials": [{"name": "stroke", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0}}],
        "extras": {"drawingId": metadata.get("id"),
                   "note": "Strokes as line strips with vertex colours. The width is not exported. "
                           "Right-handed, Y up: the game's z axis is negated."},
    }
    if primitives:
        gltf["buffers"] = [{"byteLength": len(blob),
                            "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}]
        gltf["bufferViews"] = buffer_views
        gltf["accessors"] = accessors
    else:
        gltf["meshes"] = []
        gltf["nodes"] = [{"name": metadata.get("id", "drawing")}]
    return json.dumps(gltf, separators=(",", ":")).encode("utf-8")


def to_obj(doc: Drawing, include_erased: bool = False) -> bytes:
    """A Wavefront OBJ file: vertices with the colour extension (v x y z r g b) and one polyline (l) per stroke."""

    out = ["# VR Drawing 3D: strokes as polylines, width not exported, z negated (right-handed, Y up)"]
    index = 1
    for n, stroke in enumerate(_strokes(doc, include_erased), start=1):
        out.append(f"o stroke_{n}")
        for (x, y, z), (r, g, b) in zip(stroke["positions"], stroke["colors"]):
            out.append(f"v {x:.5f} {y:.5f} {z:.5f} {r:.4f} {g:.4f} {b:.4f}")
        out.append("l " + " ".join(str(index + i) for i in range(len(stroke["positions"]))))
        index += len(stroke["positions"])
    return ("\n".join(out) + "\n").encode("utf-8")


def to_ply(doc: Drawing, include_erased: bool = False) -> bytes:
    """An ASCII PLY file: vertices with colours and an edge between consecutive points of a stroke."""

    vertices: list[str] = []
    edges: list[str] = []
    for stroke in _strokes(doc, include_erased):
        first = len(vertices)
        for (x, y, z), color in zip(stroke["positions"], stroke["colors"]):
            r, g, b = (int(round_half_up(c * 255)) for c in color)
            vertices.append(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}")
        edges += [f"{first + i} {first + i + 1}" for i in range(len(stroke["positions"]) - 1)]
    header = [
        "ply", "format ascii 1.0", "comment VR Drawing 3D: strokes as edges, width not exported, z negated",
        f"element vertex {len(vertices)}", "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element edge {len(edges)}", "property int vertex1", "property int vertex2", "end_header",
    ]
    return ("\n".join(header + vertices + edges) + "\n").encode("utf-8")


EXPORTERS = {"gltf": (to_gltf, "model/gltf+json", ".gltf"), "obj": (to_obj, "text/plain", ".obj"),
             "ply": (to_ply, "text/plain", ".ply")}


def export_drawing(doc: Drawing, fmt: str, include_erased: bool = False) -> tuple[bytes, str, str]:
    """(file bytes, media type, file extension)."""

    if fmt not in EXPORTERS:
        raise ValueError(f"Unknown format: {fmt}")
    function, media_type, extension = EXPORTERS[fmt]
    return function(doc, include_erased), media_type, extension
