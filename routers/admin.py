"""
The admin dashboard of the web app (requirements ADM-01..ADM-13): usage numbers, questionnaire results and the feedback
messages. Every endpoint needs a token with the Admin role. The senders are pseudonym codes, never accounts (SC-09).
"""

import sqlite3
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Depends, Path, Query, Response
from pydantic import BaseModel, ConfigDict

import config
from database import get_db
from feedback import results
from feedback.store import FeedbackError
from routers.feedback import as_http
from security import CurrentPlayer, require_admin

router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(require_admin)],
)

MessageId = Annotated[str, Path(pattern=r"^[0-9a-f-]{36}$", description="Message ID")]
Date = Annotated[Optional[str], Query(pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")]
Dialect = Annotated[Literal["r", "excel"], Query(description="r: commas, decimal point. excel: semicolons, decimal comma.")]


class ReadMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read: bool


def _filters(target, role, ageBand, version, dateFrom, dateTo, counting) -> results.Filters:
    return results.Filters(target=target, role=role, age_band=ageBand, version=version, date_from=dateFrom,
                           date_to=dateTo, counting=counting)


def _download(content: bytes, filename: str) -> Response:
    return Response(content=content, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# --- API Endpoints ---
#
# - Usage numbers: accounts per role, accounts that used the game, new accounts per week, submissions
# - Results of the questionnaires (SUS, UEQ, smiley), per target and role, with a trend
# - The feedback messages: list with filters, mark as read or unread, delete
# - CSV exports of the questionnaires and of the messages

@router.get("/usage", summary="Usage numbers in one call")
def get_usage(db: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
    return results.usage(db)


@router.get("/questionnaires/results", summary="Results of the questionnaires")
def get_questionnaire_results(
        target: Optional[str] = Query(None, max_length=16),
        role: Optional[int] = Query(None, ge=0, le=1),
        ageBand: Optional[str] = Query(None, max_length=16),
        version: Optional[int] = Query(None, ge=1, le=1000),
        dateFrom: Date = None,
        dateTo: Date = None,
        counting: str = Query("latest", max_length=8, description="latest: one submission per account, all: every one"),
        db: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    try:
        return results.questionnaire_results(db, _filters(target, role, ageBand, version, dateFrom, dateTo, counting))
    except FeedbackError as error:
        raise as_http(error)


@router.get("/messages", summary="The feedback messages, newest first")
def get_messages(
        target: Optional[str] = Query(None, max_length=16),
        category: Optional[str] = Query(None, max_length=16),
        role: Optional[int] = Query(None, ge=0, le=1),
        ageBand: Optional[str] = Query(None, max_length=16),
        dateFrom: Date = None,
        dateTo: Date = None,
        read: Optional[str] = Query(None, max_length=8, description="read or unread"),
        q: Optional[str] = Query(None, max_length=100, description="Text search"),
        limit: int = Query(25, ge=1, le=config.FEEDBACK_PAGE_SIZE_MAX),
        offset: int = Query(0, ge=0, le=100_000),
        db: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    try:
        return results.list_messages(
            db, limit=limit, offset=offset, target=target, category=category, role=role, age_band=ageBand,
            date_from=dateFrom, date_to=dateTo, read=read, q=q)
    except FeedbackError as error:
        raise as_http(error)


# PUT, not PATCH: the CORS setting of the server only allows GET, POST, PUT, DELETE
@router.put("/messages/{message_id}/read", summary="Mark a message read or unread")
def put_message_read(message_id: MessageId, body: ReadMessage, db: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
    try:
        return results.set_read(db, message_id, body.read)
    except FeedbackError as error:
        raise as_http(error)


@router.delete("/messages/{message_id}", status_code=204, summary="Delete a message")
def delete_message(
        message_id: MessageId, current: CurrentPlayer = Depends(require_admin), db: sqlite3.Connection = Depends(get_db),
) -> Response:
    try:
        results.delete_message(db, message_id, current.id)
    except FeedbackError as error:
        raise as_http(error)
    return Response(status_code=204)


@router.get("/export/questionnaires", summary="CSV of the answers and scores of one questionnaire")
def export_questionnaires(
        instrument: Literal["SUS", "UEQ", "SMILEY"] = Query(...),
        dialect: Dialect = "r",
        target: Optional[str] = Query(None, max_length=16),
        role: Optional[int] = Query(None, ge=0, le=1),
        ageBand: Optional[str] = Query(None, max_length=16),
        version: Optional[int] = Query(None, ge=1, le=1000),
        dateFrom: Date = None,
        dateTo: Date = None,
        db: sqlite3.Connection = Depends(get_db),
) -> Response:
    try:
        content = results.questionnaires_csv(
            db, instrument, _filters(target, role, ageBand, version, dateFrom, dateTo, "all"), dialect)
    except FeedbackError as error:
        raise as_http(error)
    return _download(content, f"questionnaire-{instrument.lower()}-{dialect}.csv")


@router.get("/export/messages", summary="CSV of the feedback messages")
def export_messages(
        dialect: Dialect = "r",
        target: Optional[str] = Query(None, max_length=16),
        category: Optional[str] = Query(None, max_length=16),
        role: Optional[int] = Query(None, ge=0, le=1),
        ageBand: Optional[str] = Query(None, max_length=16),
        dateFrom: Date = None,
        dateTo: Date = None,
        read: Optional[str] = Query(None, max_length=8),
        q: Optional[str] = Query(None, max_length=100),
        db: sqlite3.Connection = Depends(get_db),
) -> Response:
    try:
        content = results.messages_csv(
            db, dialect, target=target, category=category, role=role, age_band=ageBand, date_from=dateFrom,
            date_to=dateTo, read=read, q=q)
    except FeedbackError as error:
        raise as_http(error)
    return _download(content, f"messages-{dialect}.csv")
