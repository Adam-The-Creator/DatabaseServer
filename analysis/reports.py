"""
The HTML reports (requirements REP-01..08). A report is ONE self-contained file: inline styles, inline SVG snapshots
and a small inline 3D viewer, no external resource (no CDN), so it opens offline and prints well.
All text that comes from the data (names, descriptions, ...) is HTML-escaped (REP-05).
"""

import html
import json
from datetime import datetime, timezone
from typing import Any, Optional

from analysis import catalog
from analysis.common import Drawing, all_lines, drawn_lines, rgb_to_hex
from analysis.geometry import drawing_svg
from analysis.narrative import render_sentence

# Bump when the layout or the wording of a report changes: stored reports of other versions count as out of date.
TEMPLATE_VERSION = "1"

DISCLAIMER = "The numbers describe the drawing. They support the psychologist's judgement and are not a diagnosis."

_STYLES = """
body{font-family:system-ui,sans-serif;margin:0;background:#f8fafc;color:#0f172a}
main{max-width:880px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:16px;margin:0 0 12px}
.note{color:#64748b;font-size:13px}
section{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #f1f5f9;vertical-align:top}
th{color:#475569;font-weight:600}.def{color:#64748b;font-size:12px}.num{text-align:right;white-space:nowrap}
.swatch{display:inline-block;width:14px;height:14px;border-radius:3px;border:1px solid #cbd5e1;vertical-align:-2px;margin-right:6px}
.views{display:flex;flex-wrap:wrap;gap:12px}.views figure{margin:0}figcaption{font-size:12px;color:#64748b}
.viewer canvas{width:100%;max-width:760px;background:#f1f5f9;border-radius:8px;cursor:grab;touch-action:none}
.spark{vertical-align:middle}
@media print{body{background:#fff}section{break-inside:avoid}.viewer{display:none}}
"""

# A minimal 3D viewer: orbit by dragging, zoom with the wheel. Reads its strokes from the JSON block next to it.
_VIEWER_SCRIPT = """
(function () {
  var canvas = document.getElementById('viewer-canvas');
  if (!canvas) return;
  var strokes = JSON.parse(document.getElementById('viewer-data').textContent);
  var ctx = canvas.getContext('2d');
  var min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  strokes.forEach(function (s) { s.p.forEach(function (p) {
    for (var i = 0; i < 3; i++) { min[i] = Math.min(min[i], p[i]); max[i] = Math.max(max[i], p[i]); } }); });
  if (!isFinite(min[0])) return;
  var c = [0, 1, 2].map(function (i) { return (min[i] + max[i]) / 2; });
  var size = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 0.2);
  var yaw = 0.6, pitch = 0.25, zoom = 1, drag = null;
  function project(p) {
    var x = p[0] - c[0], y = p[1] - c[1], z = -(p[2] - c[2]);
    var cx = Math.cos(yaw), sx = Math.sin(yaw), x1 = x * cx + z * sx, z1 = -x * sx + z * cx;
    var cy = Math.cos(pitch), sy = Math.sin(pitch), y1 = y * cy - z1 * sy, z2 = y * sy + z1 * cy;
    var d = size * 2.4, f = canvas.height * 0.9 * zoom, k = f / (d - z2);
    return [canvas.width / 2 + x1 * k, canvas.height / 2 - y1 * k, k];
  }
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    strokes.forEach(function (s) {
      ctx.beginPath();
      s.p.forEach(function (p, i) { var q = project(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
      ctx.strokeStyle = s.c; ctx.lineWidth = Math.max(1.2, s.w * project(s.p[0])[2]); ctx.stroke();
    });
  }
  canvas.addEventListener('pointerdown', function (e) { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', function () { drag = null; });
  canvas.addEventListener('pointermove', function (e) {
    if (!drag) return;
    yaw += (e.clientX - drag[0]) * 0.008; pitch = Math.max(-1.5, Math.min(1.5, pitch + (e.clientY - drag[1]) * 0.008));
    drag = [e.clientX, e.clientY]; draw();
  });
  canvas.addEventListener('wheel', function (e) { e.preventDefault(); zoom = Math.max(0.3, Math.min(6, zoom * (e.deltaY < 0 ? 1.1 : 0.9))); draw(); }, { passive: false });
  draw();
})();
"""


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _page(title: str, body: str, script: str = "") -> str:
    scripts = f"<script>{script}</script>" if script else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>'
            f'<style>{_STYLES}</style></head><body><main>{body}</main>{scripts}</body></html>')


def _header(title: str, analyzer_version: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f"<h1>{esc(title)}</h1><p class=\"note\">Generated {esc(now)} · analyzer {esc(analyzer_version)} · "
            f"report template {esc(TEMPLATE_VERSION)}</p>")


def _table(rows: list[tuple[str, str]]) -> str:
    return "<table>" + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows) + "</table>"


def _metrics_table(analysis: dict[str, Any]) -> str:
    rows = []
    for metric in analysis["metrics"]:
        definition = catalog.metric(metric["id"])
        if metric["value"] is not None:
            shown = esc(catalog.format_metric(metric["id"], metric["value"]))
        else:
            shown = esc(f"n/a ({catalog.REASONS.get(metric.get('unavailableReason', ''), 'no value')})")
        rows.append(f"<tr><td>{esc(definition['name'])}<div class=\"def\">{esc(definition['definition'])} "
                    f"{esc(definition['note'])}</div></td><td class=\"num\">{shown}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _palette(analysis: dict[str, Any]) -> str:
    if not analysis["palette"]:
        return '<p class="note">None</p>'
    return "<table>" + "".join(
        f'<tr><td><span class="swatch" style="background:{esc(c["hex"])}"></span>{esc(c["hex"])}</td>'
        f'<td class="num">{round(c["share"] * 100)} %</td></tr>' for c in analysis["palette"]) + "</table>"


def _flags(analysis: dict[str, Any]) -> str:
    if not analysis["flags"]:
        return '<p class="note">No data quality issues found.</p>'
    return "<ul>" + "".join(f"<li>{esc(catalog.FLAGS[flag])}</li>" for flag in analysis["flags"]) + "</ul>"


def _viewer(doc: Drawing) -> str:
    """The interactive 3D view: the drawn strokes, lightly simplified, embedded as JSON."""

    strokes = []
    for line in drawn_lines(doc):
        points = line.get("points") or []
        if len(points) < 2:
            continue
        step = max(1, len(points) // 150)
        sampled = points[::step] + ([points[-1]] if (len(points) - 1) % step else [])
        strokes.append({"c": rgb_to_hex(line.get("startColor") or {}),
                        "w": (line.get("startWidth", 0.005) + line.get("endWidth", 0.005)) / 2,
                        "p": [[round(p["x"], 4), round(p["y"], 4), round(p["z"], 4)] for p in sampled]})
    data = json.dumps(strokes, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")
    return ('<section class="viewer"><h2>Interactive 3D view</h2><canvas id="viewer-canvas" width="760" height="460"></canvas>'
            '<p class="note">Drag to rotate, scroll to zoom.</p>'
            f'<script id="viewer-data" type="application/json">{data}</script></section>')


# --- Drawing report ---

def drawing_report(
        doc: Drawing, analysis: dict[str, Any], narrative: list[dict[str, Any]], title: str,
        player: Optional[dict[str, Any]], session: Optional[dict[str, Any]],
) -> str:
    metadata = doc.get("metadata") or {}
    player = player or {}
    figures = "".join(
        f'<figure>{drawing_svg(doc, view)}<figcaption>{esc(view.capitalize())} view</figcaption></figure>'
        for view in ("front", "top", "side"))
    info = _table([
        ("Participant", player.get("name") or f"#{str(metadata.get('owner', ''))[:8]}"),
        ("Age", player.get("age") if player.get("age") is not None else "None"),
        ("Gender", player.get("gender") or "None"),
        ("Dominant hand", player.get("dominantHand") or "None"),
        ("Session", (session or {}).get("name") or metadata.get("sessionID")),
        ("Game type", catalog.game_type_name(metadata.get("gameType", -1))),
        ("Date", metadata.get("date")),
        ("Strokes", f"{len(drawn_lines(doc))} drawn, {len(all_lines(doc)) - len(drawn_lines(doc))} erased"),
    ])
    sentences = "".join(f"<p>{esc(render_sentence(item))}</p>" for item in narrative) or '<p class="note">None</p>'
    body = (
        _header(title, analysis["analyzerVersion"]) +
        f"<section><h2>Overview</h2>{info}</section>"
        f'<section><h2>Snapshots</h2><div class="views">{figures}</div></section>' +
        _viewer(doc) +
        f"<section><h2>Summary</h2>{sentences}</section>"
        f"<section><h2>Metrics</h2>{_metrics_table(analysis)}</section>"
        f"<section><h2>Colours</h2>{_palette(analysis)}</section>"
        f"<section><h2>Data quality</h2>{_flags(analysis)}</section>"
        f'<p class="note">{esc(DISCLAIMER)}</p>'
    )
    return _page(title, body, _VIEWER_SCRIPT)


# --- Session report ---

def session_report(
        session: dict[str, Any], stats: dict[str, Any], players: dict[str, dict[str, Any]],
        drawings: list[dict[str, Any]], analyzer_version: str,
) -> str:
    """drawings: [{name, playerId, gameType, date, svg}] of the session, for the gallery."""

    title = session.get("name") or "Session"

    def person(player_id: str) -> str:
        return (players.get(player_id) or {}).get("name") or f"#{player_id[:8]}"

    info = _table([
        ("Session", title),
        ("Description", session.get("description") or "None"),
        ("Started", session.get("startDate")),
        ("Ended", session.get("endDate") or "Open"),
        ("Drawings", stats["drawingCount"]),
    ])
    participants = "<table><tr><th>Participant</th><th class=\"num\">Drawings</th><th class=\"num\">Strokes</th>" \
                   "<th class=\"num\">Total length</th><th class=\"num\">Active time</th></tr>" + "".join(
        f"<tr><td>{esc(person(row['playerId']))}</td><td class=\"num\">{row['drawings']}</td>"
        f"<td class=\"num\">{esc(catalog.format_metric('strokesDrawn', row['metrics'].get('strokesDrawn')))}</td>"
        f"<td class=\"num\">{esc(catalog.format_metric('totalLength', row['metrics'].get('totalLength')))}</td>"
        f"<td class=\"num\">{esc(catalog.format_metric('activeTime', row['metrics'].get('activeTime')))}</td></tr>"
        for row in stats["perParticipant"]) + "</table>"
    distribution = "<table><tr><th>Metric</th><th class=\"num\">Min</th><th class=\"num\">Median</th>" \
                   "<th class=\"num\">Max</th></tr>" + "".join(
        f"<tr><td>{esc(catalog.metric(d['id'])['name'])}</td>"
        f"<td class=\"num\">{esc(catalog.format_metric(d['id'], d['min']))}</td>"
        f"<td class=\"num\">{esc(catalog.format_metric(d['id'], d['median']))}</td>"
        f"<td class=\"num\">{esc(catalog.format_metric(d['id'], d['max']))}</td></tr>"
        for d in stats["distributions"]) + "</table>"
    gallery = '<div class="views">' + "".join(
        f"<figure>{d['svg']}<figcaption>{esc(d['name'])}, {esc(person(d['playerId']))}, "
        f"{esc(catalog.game_type_name(d['gameType']))}</figcaption></figure>" for d in drawings) + "</div>"
    body = (
        _header(title, analyzer_version) +
        f"<section><h2>Overview</h2>{info}</section>"
        f"<section><h2>Participants</h2>{participants}</section>"
        f"<section><h2>Drawings</h2>{gallery}</section>"
        f"<section><h2>Distribution of the metrics</h2>{distribution}</section>"
        f'<p class="note">{esc(DISCLAIMER)}</p>'
    )
    return _page(title, body)


# --- Participant history report (REP-08) ---

TREND_METRICS = ["strokesDrawn", "totalLength", "activeTime", "meanSpeed", "boundingVolume", "leftHandShare",
                 "warmShare", "eraseRatio"]


def _sparkline(values: list[float]) -> str:
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    width, height, pad = 90, 24, 3
    step = (width - 2 * pad) / max(1, len(values) - 1)
    points = " ".join(f"{pad + i * step:.1f},{height - pad - (v - low) / span * (height - 2 * pad):.1f}"
                      for i, v in enumerate(values))
    return (f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Trend"><polyline points="{points}" fill="none" stroke="#4f46e5" stroke-width="2"/></svg>')


def participant_report(
        player: dict[str, Any], history: list[dict[str, Any]], analyzer_version: str,
) -> str:
    """history: [{name, sessionName, gameType, date, metrics: {id: value or None}}] sorted by date."""

    title = f"{player.get('name') or 'Participant'}: history"
    info = _table([
        ("Name", player.get("name") or "None"), ("Age", player.get("age") if player.get("age") is not None else "None"),
        ("Gender", player.get("gender") or "None"), ("Dominant hand", player.get("dominantHand") or "None"),
        ("Drawings", len(history)),
    ])

    sections = []
    for game_type in sorted({h["gameType"] for h in history}):
        rows = [h for h in history if h["gameType"] == game_type]
        table = "<table><tr><th>Date</th><th>Drawing</th><th>Session</th><th class=\"num\">Strokes</th>" \
                "<th class=\"num\">Total length</th><th class=\"num\">Active time</th></tr>" + "".join(
            f"<tr><td>{esc(r['date'])}</td><td>{esc(r['name'])}</td><td>{esc(r['sessionName'])}</td>"
            f"<td class=\"num\">{esc(catalog.format_metric('strokesDrawn', r['metrics'].get('strokesDrawn')))}</td>"
            f"<td class=\"num\">{esc(catalog.format_metric('totalLength', r['metrics'].get('totalLength')))}</td>"
            f"<td class=\"num\">{esc(catalog.format_metric('activeTime', r['metrics'].get('activeTime')))}</td></tr>"
            for r in rows) + "</table>"
        trend = ""
        if len(rows) >= 2:
            lines = []
            for metric_id in TREND_METRICS:
                values = [r["metrics"].get(metric_id) for r in rows if r["metrics"].get(metric_id) is not None]
                if len(values) >= 2:
                    first, last = values[0], values[-1]
                    change = f"{(last - first) / first * 100:+.0f} %" if first else "n/a"
                    lines.append(
                        f"<tr><td>{esc(catalog.metric(metric_id)['name'])}</td>"
                        f"<td class=\"num\">{esc(catalog.format_metric(metric_id, first))}</td>"
                        f"<td class=\"num\">{esc(catalog.format_metric(metric_id, last))}</td>"
                        f"<td class=\"num\">{esc(change)}</td><td>{_sparkline(values)}</td></tr>")
            trend = ("<h2 style=\"margin-top:16px\">Change from the first to the latest drawing</h2>"
                     "<table><tr><th>Metric</th><th class=\"num\">First</th><th class=\"num\">Latest</th>"
                     "<th class=\"num\">Change</th><th>Trend</th></tr>" + "".join(lines) + "</table>")
        sections.append(f"<section><h2>{esc(catalog.game_type_name(game_type))}: {len(rows)} drawings</h2>"
                        f"{table}{trend}</section>")

    body = (_header(title, analyzer_version) + f"<section><h2>Participant</h2>{info}</section>" +
            "".join(sections) + f'<p class="note">{esc(DISCLAIMER)}</p>')
    return _page(title, body)
