"""
AI decision support for the DataAnalysis web application (requirements AI-01..AI-17): atypical and similar drawings, stroke
groups, fuzzy indicators, semantic labels, descriptions and conclusions by local models, session insights, the model list
and the background jobs.

Every endpoint needs a Psychologist token (the admin token is refused by the general rule, a player never reaches
AI output, ACC-02). Every answer that contains AI output is an "AI envelope" (ai/envelope.py): labelled as AI-generated,
with the model, parameters, confidence and evidence. The functions can be switched off in config.py (AI-03): a
switched-off function answers with an envelope that says so.
"""

import sqlite3
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from pydantic import BaseModel, ConfigDict

from ai import fuzzy, registry, semantic, service, settings
from ai.jobs import manager
from ai.store import AiStore
from analysis import service as data
from analysis.store import ID_PATTERN
from database import get_db
from security import get_current_player, require_psychologist

router = APIRouter(
    prefix="/ai",
    tags=["AI"],
    dependencies=[Depends(get_current_player), Depends(require_psychologist)],
)

_ID = ID_PATTERN.pattern
DrawingId = Annotated[str, Path(pattern=_ID, description="Drawing ID")]
SessionId = Annotated[str, Path(pattern=_ID, description="Session ID")]
ModelId = Annotated[str, Path(pattern=r"^[a-z0-9-]{1,32}$", description="Model ID from GET /ai/models")]
JobId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$", description="Job ID")]
HeavyFeature = Literal["labels", "description", "summary"]
Scope = Literal["gameType", "session", "all"]


def get_context(db: sqlite3.Connection = Depends(get_db)) -> service.Context:
    return service.Context(db)


def _found(value: Optional[Any], what: str = "Drawing") -> Any:
    if value is None:
        raise HTTPException(status_code=404, detail=f"{what} not found")
    return value


# --- Settings, models, rule base ---

@router.get("/status", summary="Which AI functions are switched on, evaluated and available")
def get_status() -> dict[str, Any]:
    return service.status()


@router.get("/models", summary="The libraries, rule base and models the AI functions use, with version, licence and status")
def get_models() -> list[dict[str, Any]]:
    registry.clear_cache()
    return registry.models()


@router.post("/models/{model_id}/download", summary="Download a model (an explicit action, runs as a background job)")
def download_model(model_id: ModelId) -> dict[str, Any]:
    job = service.start_download(model_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Model not found")
    registry.clear_cache()
    return job.to_dict()


@router.get("/rules", summary="The fuzzy rule base, as it is read from its file")
def get_rules() -> dict[str, Any]:
    try:
        return {"file": fuzzy.rule_base_path().name, "ruleBase": fuzzy.load_rule_base(), "problems": []}
    except fuzzy.RuleBaseError as error:
        return {"file": fuzzy.rule_base_path().name, "ruleBase": None, "problems": error.problems}


@router.get("/vocabulary", summary="The vocabulary of the semantic labelling")
def get_vocabulary() -> dict[str, Any]:
    try:
        return {"file": semantic.vocabulary_path().name, "vocabulary": semantic.load_vocabulary(), "problem": None}
    except ValueError as error:
        return {"file": semantic.vocabulary_path().name, "vocabulary": None, "problem": str(error)}


# --- Drawing level, computed on request ---

@router.get("/drawings/{drawing_id}/cohort", summary="AI-10: is the drawing atypical, which drawings are similar, a 2D map")
def get_cohort(drawing_id: DrawingId, scope: Scope = "gameType", ageMin: Optional[int] = Query(None, ge=0, le=120),
               ageMax: Optional[int] = Query(None, ge=0, le=120), ctx: service.Context = Depends(get_context)) -> dict[str, Any]:
    return _found(service.drawing_cohort(ctx, drawing_id, scope, ageMin, ageMax))


@router.get("/drawings/{drawing_id}/groups", summary="AI-11: strokes clustered into object-like groups")
def get_groups(drawing_id: DrawingId, spaceEps: float = Query(0.10, ge=0.02, le=1.0), timeEps: float = Query(20.0, ge=1, le=600),
               ctx: service.Context = Depends(get_context)) -> dict[str, Any]:
    return _found(service.drawing_groups(ctx, drawing_id, spaceEps, timeEps))


@router.get("/drawings/{drawing_id}/indicators", summary="AI-14: fuzzy logic indicators")
def get_indicators(drawing_id: DrawingId, scope: Scope = "gameType", ageMin: Optional[int] = Query(None, ge=0, le=120),
                   ageMax: Optional[int] = Query(None, ge=0, le=120), ctx: service.Context = Depends(get_context)) -> dict[str, Any]:
    return _found(service.drawing_indicators(ctx, drawing_id, scope, ageMin, ageMax))


# --- Drawing level, heavy: a stored result and a background job ---

class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regenerate: bool = False


@router.get("/drawings/{drawing_id}/{feature}", summary="AI-12/13/16: the stored result, whether it is current, and the latest job")
def get_heavy(drawing_id: DrawingId, feature: HeavyFeature, ctx: service.Context = Depends(get_context)) -> dict[str, Any]:
    return _found(service.heavy_state(ctx, feature, drawing_id))


@router.post("/drawings/{drawing_id}/{feature}", summary="Start the function as a background job (reuses a current result)")
def start_heavy(drawing_id: DrawingId, feature: HeavyFeature, request: StartRequest = StartRequest(),
                ctx: service.Context = Depends(get_context)) -> dict[str, Any]:
    return _found(service.start_heavy(ctx, feature, drawing_id, request.regenerate))


@router.delete("/drawings/{drawing_id}/{feature}", summary="Discard a stored AI result")
def delete_heavy(drawing_id: DrawingId, feature: HeavyFeature) -> dict[str, str]:
    if not AiStore().delete(feature, drawing_id):
        raise HTTPException(status_code=404, detail="No stored result")
    return {"message": "AI result discarded."}


@router.get("/drawings/{drawing_id}/views/{view}.png", summary="The rendered view the models are shown (evidence)",
            response_class=Response)
def get_view(drawing_id: DrawingId, view: Literal["front", "top", "side", "iso"], groupId: Optional[str] = Query(None, pattern=r"^g\d{1,3}$"),
             ctx: service.Context = Depends(get_context)) -> Response:
    if not settings.module_available("numpy"):
        raise HTTPException(status_code=503, detail="numpy is not installed on the server.")
    from ai.render import render_view

    document = _found(data.load_drawing(drawing_id))
    stroke_ids = None
    if groupId:
        groups = service.drawing_groups(ctx, drawing_id)
        group = next((g for g in (groups or {}).get("result", {}).get("groups", []) if g["id"] == groupId), None)
        stroke_ids = set(_found(group, "Group")["strokeIds"])
    return Response(content=render_view(document, view, stroke_ids), media_type="image/png")


# --- Session level ---

@router.get("/sessions/{session_id}/cohort", summary="AI-10: atypical drawings and the 2D map of a session")
def get_session_cohort(session_id: SessionId, gameType: Optional[int] = Query(None, ge=0, le=99),
                       ctx: service.Context = Depends(get_context)) -> dict[str, Any]:
    return _found(service.session_cohort(ctx, session_id, gameType), "Session")


@router.get("/sessions/{session_id}/insights", summary="AI-15: what recurs across the drawings of a session")
def get_session_insights(session_id: SessionId, ctx: service.Context = Depends(get_context)) -> dict[str, Any]:
    return _found(service.session_insights(ctx, session_id), "Session")


# --- Jobs ---

@router.get("/jobs", summary="The background jobs, newest first")
def list_jobs() -> list[dict[str, Any]]:
    return [job.to_dict() for job in manager.list()]


@router.get("/jobs/{job_id}", summary="Status and progress of a job")
def get_job(job_id: JobId) -> dict[str, Any]:
    return _found(manager.get(job_id), "Job").to_dict()


@router.post("/jobs/{job_id}/cancel", summary="Cancel a job (it stops at its next step)")
def cancel_job(job_id: JobId) -> dict[str, Any]:
    return _found(manager.cancel(job_id), "Job").to_dict()
