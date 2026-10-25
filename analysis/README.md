# Analysis module

The server side of the DataAnalysis web application (its requirements are in `DataAnalysis/requirements.md`).
It is **additive**: it only reads the existing SQLite tables and MongoDB drawings and never changes either. What it
stores (reports, thumbnails, stored results) are files in one folder.

```text
analysis/
  common.py      shared helpers, the time axis (TimeAxis), rounding
  catalog.py     the metric catalogue: ids, units and the texts shown to psychologists
  metrics.py     the metrics, the colour palette and the data quality flags of one drawing
  narrative.py   the template-based summary sentences (no AI)
  timeline.py    strokes, erasing, menu use, tool changes and pauses on one time axis
  tracking.py    head and hand paths, movement, head direction, explored space
  heatmap.py     3D ink density in voxels
  multiplayer.py contribution, proximity and turn-taking of the authors
  cohort.py      session statistics and percentiles against a group
  geometry.py    simplified strokes for the viewer, SVG snapshots
  exports.py     CSV (two dialects), glTF, OBJ, PLY
  reports.py     self-contained HTML reports (with a small inline 3D viewer)
  store.py       the file store, with validated file names only
  service.py     reads the databases and calls the functions above
routers/analysis.py   the HTTP endpoints
tests/                unit tests (see below)
```

Everything that calculates is a pure function over the drawing dict as MongoDB stores it, so it can be tested
without a database. Only `service.py` touches the databases.

## Endpoints (all under `/analysis`)

Every endpoint needs a Psychologist token, except the geometry and the thumbnail of a drawing, which its owner may
read too. All IDs in paths are validated (`[A-Za-z0-9_-]`, up to 64 characters). Browse them with the server's
`/docs` page.

| Endpoint | What it returns |
| -------- | --------------- |
| `GET /overview` | The dashboard numbers in one call |
| `GET /sessions` | Sessions with their number of drawings, participants and drawings per game type |
| `GET /drawings?limit=` | Every drawing with participant and session names, newest first |
| `GET /over-time` | Drawings per day |
| `GET /sessions/{id}/stats` | Distribution of every metric over a session, and the mean per participant |
| `GET /sessions/{id}/export.csv?dialect=r\|excel` | One row per drawing, for SPSS, R or Excel |
| `GET /drawings/{id}/metrics` | Metrics, palette, data quality flags and the narrative |
| `GET /drawings/{id}/geometry?tolerance=` | Strokes for the viewer, simplified by a tolerance in metres (0 keeps every point) |
| `GET /drawings/{id}/thumbnail` | Front view as an SVG image |
| `GET /drawings/{id}/timeline?idleGap=` | Behaviour timeline (the pause limit in seconds) |
| `GET /drawings/{id}/tracking?maxPoints=` | Tracked head and hand movement |
| `GET /drawings/{id}/heatmap?voxelSize=` | Sparse voxel grid of ink length |
| `GET /drawings/{id}/multiplayer?proximity=` | Authors: contribution, closeness, turns |
| `GET /drawings/{id}/cohort?sessionId=&gameType=&ageMin=&ageMax=` | Percentile of every metric among a group |
| `GET /drawings/{id}/export?format=gltf\|obj\|ply&includeErased=` | The strokes as a 3D file |
| `GET /participants/{id}/history` | A participant's drawings over time with their metrics |
| `GET/POST /reports`, `GET /reports/{id}/html`, `DELETE /reports/{id}` | Stored reports |

## Things worth knowing

- **Time axis.** Every time in an answer is milliseconds from the start of the drawing's activity. A drawing can be
  continued after a break of hours or days (a loaded drawing saved again), so a gap of more than 30 minutes between
  strokes is a *break between sittings*: it is cut down to 5 seconds on the axis, left out of the time span and of the
  pause time, and the drawing gets the data quality flag `multipleSittings`. The line history only carries the legacy
  local-time stamp, so it is aligned to the UTC axis with the offset between a point's two stamps.
- **Old drawings.** Schema version 1 drawings (no UTC times, pressure, events or stored line stats) work. A metric that
  cannot be computed has no value and a reason, never a made-up number.
- **Reports and staleness.** A report remembers a *content fingerprint* of what it was made from (computed by MongoDB
  from a projection, so the document is not transferred). While the fingerprint, the analyzer version, the report
  template version and the state of the AI results (`aiSignature`: which AI functions are on and when the stored heavy
  results were made) are unchanged, the stored report is reused. Otherwise it is listed as out of date. The AI sections of
  a calculated kind (indicators, comparison) reflect the other drawings at the time of the report: regenerate to refresh.
- **Rounding.** Numbers round half up, like the web app. Python's `round()` rounds half to even, which made the same
  number read differently in a report and on screen.
- **The store.** `analysis_store/` next to the working directory, or the folder in `ANALYSIS_STORE_PATH`
  (environment variable, or `config.py`). It is ignored by git.
- **Not done:** background jobs (report generation takes well under a second at the current data size) and server side
  pagination of the lists.
- **No new dependency.** The calculations use the standard library only. (The AI module in `ai/`, which builds on these
  results, uses numpy and scikit-learn. Reports ask it for their AI sections and are made without them if it fails.)

## Tests

```text
python -m unittest discover -s tests -t .
```

The tests use synthetic drawings with known answers (a straight line, a square, overlapping and idle strokes, legacy
drawings, a tap without length, a drawing continued two days later, escaping in the reports, ...). The matching
TypeScript stand-ins of the web app's demo mode were compared with this module on the same drawings, with identical
results.
