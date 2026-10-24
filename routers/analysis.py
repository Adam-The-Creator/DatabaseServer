"""
Analysis API of the DataAnalysis web application: overview and statistics, drawing metrics, geometry for the viewer,
behaviour timeline, tracked movement, heatmap, multiplayer analytics, cohort comparison, participant history, exports
and reports.

An additive module: it only reads the existing tables and drawings and writes nothing but its file store
(analysis/store.py). Access: every endpoint needs a Psychologist token. The only exceptions are the geometry and the
thumbnail of a drawing, which the owner of that drawing may read too (the player view).
"""

import sqlite3
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from analysis import service
from analysis.cohort import compare_with_cohort, session_stats
from analysis.common import IDLE_GAP_SECONDS
from analysis.exports import CSV_DIALECTS, export_drawing, session_csv
from analysis.geometry import stroke_geometry
from analysis.heatmap import build_heatmap
from analysis.metrics import metric_values
from analysis.multiplayer import PROXIMITY_RADIUS, analyze_multiplayer
from analysis.store import AnalysisStore, ID_PATTERN, REPORT_ID_PATTERN
from analysis.timeline import build_timeline
from analysis.tracking import summarize_tracking
from database import get_db
from security import CurrentPlayer, get_current_player, require_psychologist

router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"],
    dependencies=[Depends(get_current_player)],
)

# Path and query IDs are validated before they reach a query or a file name (requirement TEC-09)
_ID = ID_PATTERN.pattern
DrawingId = Annotated[str, Path(pattern=_ID, description="Drawing ID")]
SessionId = Annotated[str, Path(pattern=_ID, description="Session ID")]
PlayerId = Annotated[str, Path(pattern=_ID, description="Player ID")]
ReportId = Annotated[str, Path(pattern=REPORT_ID_PATTERN.pattern, description="Report ID")]

psychologist = [Depends(require_psychologist)]


def get_store() -> AnalysisStore:
    return AnalysisStore()


def owner_or_psychologist(drawing_id: DrawingId, current: CurrentPlayer = Depends(get_current_player)) -> CurrentPlayer:
    """The owner of a drawing may read its geometry and thumbnail, a Psychologist may read every drawing's."""

    if not current.is_psychologist and service.drawing_owner(drawing_id) != current.id:
        raise HTTPException(status_code=403, detail="You can only access your own drawings.")
    return current


def _drawing_or_404(drawing_id: str) -> dict[str, Any]:
    document = service.load_drawing(drawing_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Drawing not found")
    return document


def _download(content: bytes, media_type: str, filename: str) -> Response:
    return Response(content=content, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# --- Overview ---

@router.get("/overview", summary="Dashboard numbers in one call", dependencies=psychologist)
def get_overview(db: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
    return service.overview(db)


@router.get("/sessions", summary="Sessions with their drawing and participant counts", dependencies=psychologist)
def get_session_summaries(db: sqlite3.Connection = Depends(get_db)) -> list[dict[str, Any]]:
    return service.session_summaries(db)


@router.get("/drawings", summary="All drawings with participant and session names, newest first", dependencies=psychologist)
def get_drawing_list(limit: Optional[int] = Query(None, ge=1, le=2000), db: sqlite3.Connection = Depends(get_db)) -> list[dict[str, Any]]:
    return service.drawing_list(db, limit)


@router.get("/over-time", summary="Number of drawings per day", dependencies=psychologist)
def get_over_time(db: sqlite3.Connection = Depends(get_db)) -> list[dict[str, Any]]:
    return service.drawings_over_time(db)


# --- Sessions ---

@router.get("/sessions/{session_id}/stats", summary="Distribution of every metric over a session", dependencies=psychologist)
def get_session_stats(session_id: SessionId, db: sqlite3.Connection = Depends(get_db),
                      store: AnalysisStore = Depends(get_store)) -> dict[str, Any]:
    if service.session_row(db, session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    entries = []
    for meta in service.drawing_metas(db, session_id=session_id):
        found = service.drawing_analysis(meta["id"], store)
        if found:
            entries.append((meta["playerID"], found[0]))
    return session_stats(session_id, entries)


@router.get("/sessions/{session_id}/export.csv", summary="Metrics of a session as CSV (one row per drawing)",
            dependencies=psychologist)
def export_session_csv(session_id: SessionId, dialect: Literal["r", "excel"] = "r",
                       db: sqlite3.Connection = Depends(get_db), store: AnalysisStore = Depends(get_store)) -> Response:
    if dialect not in CSV_DIALECTS or service.session_row(db, session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    content = session_csv(service.session_csv_rows(db, store, session_id), dialect)
    return _download(content, "text/csv; charset=utf-8", f"session-{session_id}-{dialect}.csv")


# --- Drawings ---

@router.get("/drawings/{drawing_id}/metrics", summary="Metrics, palette, flags and summary of a drawing",
            dependencies=psychologist)
def get_metrics(drawing_id: DrawingId, db: sqlite3.Connection = Depends(get_db),
                store: AnalysisStore = Depends(get_store)) -> dict[str, Any]:
    found = service.drawing_analysis(drawing_id, store)
    if found is None:
        raise HTTPException(status_code=404, detail="Drawing not found")
    analysis = found[0]
    return {**analysis, "narrative": service.drawing_narrative(db, drawing_id, analysis)}


@router.get("/drawings/{drawing_id}/geometry", summary="Strokes for the viewer, simplified by a tolerance in metres",
            dependencies=[Depends(owner_or_psychologist)])
def get_geometry(drawing_id: DrawingId, tolerance: float = Query(0.0, ge=0.0, le=0.2)) -> dict[str, Any]:
    return stroke_geometry(_drawing_or_404(drawing_id), tolerance)


@router.get("/drawings/{drawing_id}/thumbnail", summary="Front view of a drawing as an SVG image",
            dependencies=[Depends(owner_or_psychologist)])
def get_thumbnail(drawing_id: DrawingId, store: AnalysisStore = Depends(get_store)) -> Response:
    svg = service.drawing_thumbnail(drawing_id, store)
    if svg is None:
        raise HTTPException(status_code=404, detail="Drawing not found")
    return Response(content=svg, media_type="image/svg+xml")


@router.get("/drawings/{drawing_id}/timeline", summary="Behaviour timeline: strokes, erasing, menu, tools, pauses",
            dependencies=psychologist)
def get_timeline(drawing_id: DrawingId, idleGap: float = Query(IDLE_GAP_SECONDS, ge=0.5, le=600)) -> dict[str, Any]:
    return build_timeline(_drawing_or_404(drawing_id), idleGap)


@router.get("/drawings/{drawing_id}/tracking", summary="Tracked head and hand movement", dependencies=psychologist)
def get_tracking(drawing_id: DrawingId, maxPoints: int = Query(400, ge=20, le=5000)) -> dict[str, Any]:
    return summarize_tracking(_drawing_or_404(drawing_id), maxPoints)


@router.get("/drawings/{drawing_id}/heatmap", summary="3D ink density in voxels", dependencies=psychologist)
def get_heatmap(drawing_id: DrawingId, voxelSize: float = Query(0.05, ge=0.01, le=0.5)) -> dict[str, Any]:
    return build_heatmap(_drawing_or_404(drawing_id), voxelSize)


@router.get("/drawings/{drawing_id}/multiplayer", summary="Contribution, proximity and turn-taking of the authors",
            dependencies=psychologist)
def get_multiplayer(drawing_id: DrawingId, proximity: float = Query(PROXIMITY_RADIUS, ge=0.005, le=0.5)) -> dict[str, Any]:
    return analyze_multiplayer(_drawing_or_404(drawing_id), proximity)


@router.get("/drawings/{drawing_id}/cohort", summary="Percentile of the drawing's metrics among a group of drawings",
            dependencies=psychologist)
def get_cohort(
        drawing_id: DrawingId,
        sessionId: Optional[str] = Query(None, pattern=_ID),
        gameType: Optional[int] = Query(None, ge=0, le=99),
        ageMin: Optional[int] = Query(None, ge=0, le=120),
        ageMax: Optional[int] = Query(None, ge=0, le=120),
        db: sqlite3.Connection = Depends(get_db), store: AnalysisStore = Depends(get_store),
) -> dict[str, Any]:
    found = service.drawing_analysis(drawing_id, store)
    if found is None:
        raise HTTPException(status_code=404, detail="Drawing not found")
    ids = service.cohort_drawing_ids(db, drawing_id, sessionId, gameType, ageMin, ageMax)
    result = compare_with_cohort(metric_values(found[0]), service.metrics_of(ids, store))
    return {"drawingId": drawing_id, "filters": {"sessionId": sessionId, "gameType": gameType, "ageMin": ageMin,
                                                 "ageMax": ageMax}, **result}


@router.get("/drawings/{drawing_id}/export", summary="Strokes as a glTF, OBJ or PLY file", dependencies=psychologist)
def export_geometry(drawing_id: DrawingId, format: Literal["gltf", "obj", "ply"] = "gltf",
                    includeErased: bool = False) -> Response:
    content, media_type, extension = export_drawing(_drawing_or_404(drawing_id), format, includeErased)
    return _download(content, media_type, f"drawing-{drawing_id}{extension}")


# --- Participants ---

@router.get("/participants/{player_id}/history", summary="A participant's drawings over time with their metrics",
            dependencies=psychologist)
def get_participant_history(player_id: PlayerId, db: sqlite3.Connection = Depends(get_db),
                            store: AnalysisStore = Depends(get_store)) -> dict[str, Any]:
    history = service.participant_history(db, store, player_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return history


# --- Reports ---

class ReportRequest(BaseModel):
    kind: Literal["drawing", "session", "participant"]
    targetId: str = Field(pattern=_ID)
    regenerate: bool = False


@router.get("/reports", summary="Stored reports, newest first", dependencies=psychologist)
def list_reports(db: sqlite3.Connection = Depends(get_db), store: AnalysisStore = Depends(get_store)) -> list[dict[str, Any]]:
    return [service.report_info(db, meta) for meta in store.list_reports()]


@router.post("/reports", summary="Generate a report, or reuse the stored one while it is up to date",
             dependencies=psychologist)
def create_report(request: ReportRequest, db: sqlite3.Connection = Depends(get_db),
                  store: AnalysisStore = Depends(get_store)) -> dict[str, Any]:
    created = service.create_report(db, store, request.kind, request.targetId, request.regenerate)
    if created is None:
        raise HTTPException(status_code=404, detail="The report's subject was not found")
    report, reused = created
    return {"report": report, "reused": reused}


@router.get("/reports/{report_id}/html", summary="A stored report (self-contained HTML)", dependencies=psychologist)
def get_report_html(report_id: ReportId, store: AnalysisStore = Depends(get_store)) -> HTMLResponse:
    html = store.read_report_html(report_id)
    if html is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return HTMLResponse(content=html)


@router.delete("/reports/{report_id}", summary="Delete a stored report", dependencies=psychologist)
def delete_report(report_id: ReportId, store: AnalysisStore = Depends(get_store)) -> dict[str, str]:
    if not store.delete_report(report_id):
        raise HTTPException(status_code=404, detail="Report not found")
    return {"message": "Report deleted successfully."}
