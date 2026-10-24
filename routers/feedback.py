"""
Feedback and questionnaires of the players and psychologists (requirements FBK-01..FBK-20, SEC-01, SEC-04..SEC-08).

Every endpoint needs a token. The sender is always the owner of the token, never somebody named in the body or the path
(requirement SEC-06), and an account only ever reads its own submissions. The admin reads everything pseudonymously in
routers/admin.py. The rules (limits, validation, scoring) are in feedback/.
"""

import sqlite3
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

import config
from database import get_db
from feedback import instruments, store
from security import CurrentPlayer, get_current_player

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
    dependencies=[Depends(get_current_player)],
)


# --- DTOs (unknown fields are refused, requirement SEC-04) ---

class QuestionnaireMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clientId: str = Field(max_length=64, description="A UUID the client made up; a repeated one is the same submission")
    instrument: str = Field(max_length=16)
    version: int
    target: str = Field(max_length=16)
    answers: dict[str, Any] = Field(description="Item ID -> number")


class FeedbackMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clientId: str = Field(max_length=64)
    target: str = Field(max_length=16)
    category: Optional[str] = Field(None, max_length=16)
    text: str = Field(max_length=2 * config.FEEDBACK_MESSAGE_MAX_CHARS)


def as_http(error: store.FeedbackError) -> HTTPException:
    headers = {"Retry-After": str(error.retry_after)} if error.retry_after else None
    return HTTPException(status_code=error.status, detail=error.detail, headers=headers)


# --- API Endpoints ---
#
# - Get the questionnaires (item IDs, scale, version) to build the forms from
# - Send a questionnaire (SUS, UEQ, or the smiley rating for children)
# - Send a feedback message
# - Get the caller's own submissions

@router.get("/instruments", summary="The questionnaires: version, scale and item IDs")
def get_instruments() -> list[dict[str, Any]]:
    return instruments.definitions()


@router.post("/questionnaires", status_code=201, summary="Send a filled-in questionnaire")
def send_questionnaire(
        message: QuestionnaireMessage,
        current: CurrentPlayer = Depends(get_current_player),
        db: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    try:
        return store.submit_questionnaire(
            db, current.id, message.clientId, message.instrument, message.version, message.target, message.answers)
    except store.FeedbackError as error:
        raise as_http(error)


@router.post("/messages", status_code=201, summary="Send a feedback message")
def send_message(
        message: FeedbackMessage,
        current: CurrentPlayer = Depends(get_current_player),
        db: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    try:
        return store.submit_message(db, current.id, message.clientId, message.target, message.category, message.text)
    except store.FeedbackError as error:
        raise as_http(error)


@router.get("/mine", summary="The caller's own submissions, newest first")
def get_my_submissions(
        limit: int = Query(25, ge=1, le=config.FEEDBACK_PAGE_SIZE_MAX),
        offset: int = Query(0, ge=0, le=100_000),
        current: CurrentPlayer = Depends(get_current_player),
        db: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    return store.my_submissions(db, current.id, limit, offset)
