"""
Storage and rules of the submissions of the players and psychologists (requirements FBK-01..FBK-20, SEC-01, SEC-04,
SEC-06..SEC-08, TEC-14).

The three tables are new (SC-08). They have no foreign keys on purpose: the existing tables and their delete rules stay
as they are. Every check that decides whether a submission is accepted (idempotency, the send limits, the caps) runs
in the same write transaction as the insert, so parallel requests cannot all pass the limit.
"""

import json
import secrets
import sqlite3
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import config
import throttle
from feedback import instruments
from security import Role

# The alphabet of the pseudonym codes has no characters that look alike (0/O, 1/I)
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_DAY_MS = 24 * 3600 * 1000


class FeedbackError(Exception):
    """A refused submission: the HTTP status and the message for the user, and how long to wait if that helps."""

    def __init__(self, status: int, detail: str, retry_after: Optional[int] = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.retry_after = retry_after


def now_ms() -> int:
    return int(time.time() * 1000)


def iso(ms: int) -> str:
    """UTC time as ISO 8601, e.g. 2026-10-24T09:30:12Z."""

    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_schema(cursor: sqlite3.Cursor) -> None:
    """Creates the tables of this module (idempotent, like the other tables of database.py)."""

    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS QuestionnaireSubmissions (
                ID TEXT PRIMARY KEY,
                ClientID TEXT NOT NULL UNIQUE,
                PlayerID TEXT NOT NULL,
                Role INTEGER,
                AgeBand TEXT NOT NULL,
                Target TEXT NOT NULL,
                Instrument TEXT NOT NULL,
                InstrumentVersion INTEGER NOT NULL,
                Answers TEXT NOT NULL,
                Scores TEXT NOT NULL,
                CreatedMs INTEGER NOT NULL
            )
        """
    )
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS FeedbackMessages (
                ID TEXT PRIMARY KEY,
                ClientID TEXT NOT NULL UNIQUE,
                PlayerID TEXT NOT NULL,
                Role INTEGER,
                AgeBand TEXT NOT NULL,
                Target TEXT NOT NULL,
                Category TEXT,
                Text TEXT NOT NULL,
                CreatedMs INTEGER NOT NULL,
                IsRead INTEGER NOT NULL DEFAULT 0
            )
        """
    )
    # The pseudonym of an account for the admin (requirement ADM-12, SC-09)
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS FeedbackProfiles (
                PlayerID TEXT PRIMARY KEY,
                Code TEXT NOT NULL UNIQUE
            )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_qsub_player ON QuestionnaireSubmissions(PlayerID, Instrument, Target, CreatedMs);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_qsub_created ON QuestionnaireSubmissions(CreatedMs);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fmsg_player ON FeedbackMessages(PlayerID, CreatedMs);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fmsg_created ON FeedbackMessages(CreatedMs);")


# --- Who is sending ---

@dataclass(frozen=True)
class Sender:
    player_id: str
    role: int
    age_band: str


def sender_of(db: sqlite3.Connection, player_id: str) -> Sender:
    """The role and the age band come from the database, not from the token, so they are always current."""

    row = db.execute(
        "SELECT p.Role, i.Age FROM Players p LEFT JOIN PlayerInfo i ON i.ID = p.PlayerInfoID WHERE p.ID = ?",
        (player_id,),
    ).fetchone()
    if row is None:
        raise FeedbackError(404, "Account not found.")
    role, age = row
    if role not in (Role.PLAYER, Role.PSYCHOLOGIST):
        raise FeedbackError(403, "Only players and psychologists can send feedback.")
    return Sender(player_id, int(role), instruments.age_band(age))


def _check_target(sender: Sender, target: str, allowed: tuple[str, ...]) -> None:
    if target not in allowed:
        raise FeedbackError(400, "This cannot be rated here.")
    # Players rate the game, psychologists the game and the web app (requirement FBK-02)
    if sender.role == Role.PLAYER and target != "game":
        raise FeedbackError(403, "Players can only rate the game.")


# --- Text of a message ---

_ALLOWED_CONTROLS = {"\n", "\t"}


def clean_text(text: Any) -> str:
    """
    Plain text only: normalised, line breaks unified, and refused when it has control characters (a null byte, ...)
    or characters that change how text is displayed (the bidirectional overrides). The text is never interpreted
    as HTML or markdown anywhere (requirement SEC-05).
    """

    if not isinstance(text, str):
        raise FeedbackError(422, "The message is missing.")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not text:
        raise FeedbackError(422, "The message is empty.")
    if len(text) > config.FEEDBACK_MESSAGE_MAX_CHARS:
        raise FeedbackError(422, f"The message is too long (at most {config.FEEDBACK_MESSAGE_MAX_CHARS} characters).")
    for char in text:
        category = unicodedata.category(char)
        if (category == "Cc" and char not in _ALLOWED_CONTROLS) or char in "‪‫‬‭‮⁦⁧⁨⁩":
            raise FeedbackError(422, "The message contains characters that are not allowed.")
    return text


def _client_id(value: Any) -> str:
    """The ID the client made up for the submission (requirement SEC-07): a UUID."""

    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        raise FeedbackError(422, "The submission ID is invalid.")


# --- Pseudonym ---

def pseudonym(db: sqlite3.Connection, player_id: str) -> str:
    """The stable code of an account for the admin. Created the first time the account sends something."""

    row = db.execute("SELECT Code FROM FeedbackProfiles WHERE PlayerID = ?", (player_id,)).fetchone()
    if row:
        return row[0]
    for _ in range(20):
        code = "P-" + "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))
        try:
            db.execute("INSERT INTO FeedbackProfiles (PlayerID, Code) VALUES (?, ?)", (player_id, code))
            return code
        except sqlite3.IntegrityError:
            continue  # the code is taken (a PlayerID cannot be, the write lock is held)
    raise FeedbackError(500, "Could not create a code.")


# --- Submissions ---

def _wait_error(message: str, wait_ms: int) -> FeedbackError:
    seconds = max(1, -(-wait_ms // 1000))
    return FeedbackError(429, message.format(wait=throttle.human_wait(seconds)), retry_after=seconds)


def submit_questionnaire(
        db: sqlite3.Connection, player_id: str, client_id: Any, instrument_id: str, version: int, target: str,
        answers: Any, at_ms: Optional[int] = None,
) -> dict[str, Any]:
    """Validates, scores and stores a questionnaire. Returns {id, createdAt}; a repeated client ID returns the first."""

    client_id = _client_id(client_id)
    try:
        instrument = instruments.get_instrument(instrument_id, version)
        values = instruments.validate_answers(instrument, answers)
    except instruments.InvalidAnswers as error:
        raise FeedbackError(422, str(error))
    scores = instruments.score(instrument, values)
    at = at_ms if at_ms is not None else now_ms()

    db.execute("BEGIN IMMEDIATE")  # one writer at a time: the checks below and the insert are one step
    try:
        sender = sender_of(db, player_id)
        _check_target(sender, target, instrument.targets)

        existing = db.execute("SELECT ID, PlayerID, CreatedMs FROM QuestionnaireSubmissions WHERE ClientID = ?", (client_id,)).fetchone()
        if existing:
            if existing[1] != player_id:
                raise FeedbackError(409, "The submission ID is already used.")
            db.rollback()
            return {"id": existing[0], "createdAt": iso(existing[2])}

        interval_ms = config.FEEDBACK_QUESTIONNAIRE_INTERVAL_SECONDS * 1000
        last = db.execute(
            "SELECT MAX(CreatedMs) FROM QuestionnaireSubmissions WHERE PlayerID = ? AND Instrument = ? AND Target = ?",
            (player_id, instrument.id, target),
        ).fetchone()[0]
        if last is not None and at - last < interval_ms:
            raise _wait_error("You can send this questionnaire again in {wait}.", last + interval_ms - at)

        stored = db.execute("SELECT COUNT(*) FROM QuestionnaireSubmissions WHERE PlayerID = ?", (player_id,)).fetchone()[0]
        if stored >= config.FEEDBACK_MAX_QUESTIONNAIRES_PER_ACCOUNT:
            raise FeedbackError(409, "You have reached the maximum number of stored questionnaires.")

        pseudonym(db, player_id)
        submission_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO QuestionnaireSubmissions (ID, ClientID, PlayerID, Role, AgeBand, Target, Instrument, "
            "InstrumentVersion, Answers, Scores, CreatedMs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (submission_id, client_id, player_id, sender.role, sender.age_band, target, instrument.id, instrument.version,
             json.dumps(values), json.dumps(scores), at),
        )
        db.commit()
        return {"id": submission_id, "createdAt": iso(at)}
    except BaseException:
        db.rollback()
        raise


def submit_message(
        db: sqlite3.Connection, player_id: str, client_id: Any, target: str, category: Optional[str], text: Any,
        at_ms: Optional[int] = None,
) -> dict[str, Any]:
    """Validates and stores a feedback message. Returns {id, createdAt}; a repeated client ID returns the first."""

    client_id = _client_id(client_id)
    text = clean_text(text)
    if category is not None and category not in instruments.CATEGORIES:
        raise FeedbackError(422, "Unknown category.")
    at = at_ms if at_ms is not None else now_ms()

    db.execute("BEGIN IMMEDIATE")
    try:
        sender = sender_of(db, player_id)
        _check_target(sender, target, instruments.TARGETS)

        existing = db.execute("SELECT ID, PlayerID, CreatedMs FROM FeedbackMessages WHERE ClientID = ?", (client_id,)).fetchone()
        if existing:
            if existing[1] != player_id:
                raise FeedbackError(409, "The submission ID is already used.")
            db.rollback()
            return {"id": existing[0], "createdAt": iso(existing[2])}

        interval_ms = config.FEEDBACK_MESSAGE_INTERVAL_SECONDS * 1000
        last = db.execute("SELECT MAX(CreatedMs) FROM FeedbackMessages WHERE PlayerID = ?", (player_id,)).fetchone()[0]
        if last is not None and at - last < interval_ms:
            raise _wait_error("Please wait {wait} before you send another message.", last + interval_ms - at)

        today, oldest = db.execute(
            "SELECT COUNT(*), MIN(CreatedMs) FROM FeedbackMessages WHERE PlayerID = ? AND CreatedMs > ?",
            (player_id, at - _DAY_MS),
        ).fetchone()
        if today >= config.FEEDBACK_MESSAGES_PER_DAY:
            raise _wait_error("You have reached the daily limit of messages. You can send the next one in {wait}.",
                              oldest + _DAY_MS - at)

        stored = db.execute("SELECT COUNT(*) FROM FeedbackMessages WHERE PlayerID = ?", (player_id,)).fetchone()[0]
        if stored >= config.FEEDBACK_MAX_MESSAGES_PER_ACCOUNT:
            raise FeedbackError(409, "You have reached the maximum number of stored messages.")

        pseudonym(db, player_id)
        message_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO FeedbackMessages (ID, ClientID, PlayerID, Role, AgeBand, Target, Category, Text, CreatedMs, IsRead) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (message_id, client_id, player_id, sender.role, sender.age_band, target, category, text, at),
        )
        db.commit()
        return {"id": message_id, "createdAt": iso(at)}
    except BaseException:
        db.rollback()
        raise


def my_submissions(db: sqlite3.Connection, player_id: str, limit: int, offset: int) -> dict[str, Any]:
    """The account's own submissions, newest first (requirement FBK-04). No scores: only what the user sent."""

    limit = max(1, min(limit, config.FEEDBACK_PAGE_SIZE_MAX))
    total = sum(
        db.execute(f"SELECT COUNT(*) FROM {table} WHERE PlayerID = ?", (player_id,)).fetchone()[0]
        for table in ("QuestionnaireSubmissions", "FeedbackMessages")
    )
    rows = db.execute(
        """
        SELECT 'questionnaire', ID, CreatedMs, Target, Instrument, NULL, NULL FROM QuestionnaireSubmissions WHERE PlayerID = ?
        UNION ALL
        SELECT 'message', ID, CreatedMs, Target, NULL, Category, Text FROM FeedbackMessages WHERE PlayerID = ?
        ORDER BY 3 DESC LIMIT ? OFFSET ?
        """,
        (player_id, player_id, limit, max(0, offset)),
    ).fetchall()
    items = []
    for kind, item_id, created, target, instrument, category, text in rows:
        item: dict[str, Any] = {"kind": kind, "id": item_id, "createdAt": iso(created), "target": target}
        if kind == "questionnaire":
            item["instrument"] = instrument
        else:
            item["category"] = category
            item["text"] = text
        items.append(item)
    return {"total": total, "items": items}
