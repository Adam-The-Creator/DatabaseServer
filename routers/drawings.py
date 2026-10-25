import math
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import bson
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, ValidationError, field_validator

import database
from database import get_active_session_id, get_db, execute_with_retry
from models.mongodb.models import (
    Drawing, GameType, Line, LineStats, TrackedBehavior, Collaborator, PlacedModel, Metadata, DrawingEvent,
    SadChildCoordinate
)
from models.sqlite.models import DrawingMeta
from security import CurrentPlayer, get_current_player, require_psychologist
from utils import get_date

# Create the router instance
router = APIRouter(
    prefix="/drawings",
    tags=["Drawings"],
    dependencies=[Depends(get_current_player)],
)

# --- DTOs ---

class DrawingSaveMessage(BaseModel):
    name: Optional[str] = None
    owner: str
    collaborators: Optional[List[Collaborator]] = None
    version: Optional[str] = "1.0"
    gameType: GameType
    sessionID: str = ""  # empty: the drawing is saved into the active session
    lines: List[Line]
    trackedBehaviors: Optional[List[TrackedBehavior]] = None
    placedModels: Optional[List[PlacedModel]] = None
    events: Optional[List[DrawingEvent]] = None
    appVersion: Optional[str] = None
    device: Optional[str] = None
    sadChildCoordinates: Optional[SadChildCoordinate] = None

    @field_validator("sadChildCoordinates", mode="wrap")
    @classmethod
    def _figure_or_nothing(cls, value, handler):
        """It is optional metadata: a malformed or non-finite position is dropped instead of refusing the drawing."""

        try:
            figure = handler(value)
        except ValidationError:
            return None
        if figure is not None and not all(math.isfinite(number) for number in figure.model_dump().values() if number is not None):
            return None
        return figure


class DrawingUpdateMessage(BaseModel):
    name: Optional[str] = None
    lines: Optional[List[Line]] = None
    trackedBehaviors: Optional[List[TrackedBehavior]] = None
    placedModels: Optional[List[PlacedModel]] = None
    events: Optional[List[DrawingEvent]] = None
    collaborators: Optional[List[Collaborator]] = None


# --- API Endpoints ---
#
# - Save drawing (in SQLite DB and Mongo DB)
# - Update drawing (in SQLite DB and Mongo DB)
# - Import drawing (in SQLite DB and Mongo DB)
# - Get list of drawing IDs (from SQLite DB)
# - Get drawing data (JSON format) by drawing ID (Drawing from Mongo DB)
# - Get drawing meta (DrawingMeta from SQLite DB)
# - Delete drawing (from SQLite DB and Mongo DB)
#
# Access: every endpoint requires a valid token (router-level dependency).
# - Reading a drawing (or its meta) by ID is open to any logged-in player, so a guest can load the host's
#   drawing after joining a room. Listing all drawing IDs is Psychologist-only.
# - Save: the owner in the message has to be the caller.
# - Update: owner, listed collaborator, or anyone while a room is active for that drawing.
# - Import: the owner has to be the caller (Psychologists may import for anyone).
# - Delete: owner only.


def _find_owner(drawing_id: str, cursor: sqlite3.Cursor) -> Optional[str]:
    """Owner of an existing drawing: the SQLite link row first, then the MongoDB document (either may be missing)."""

    cursor.execute("SELECT PlayerID FROM DrawingMeta WHERE ID = ?", (drawing_id,))
    row = cursor.fetchone()
    if row:
        return row[0]

    document = database.mongo_collection.find_one({"metadata.id": drawing_id}, {"metadata.owner": 1})
    return document["metadata"]["owner"] if document else None


# Version of the stored document layout. 2 = utcMs/pressure on points, utcMs on poses, events, line stats and the
# metadata additions (schemaVersion, appVersion, device, trackingDecimation). Everything added is optional, so
# drawings of version 1 (the field is absent) still load.
SCHEMA_VERSION = 2

# MongoDB rejects documents above 16 MiB. The limit used here leaves room for _id and BSON overhead.
MAX_DOCUMENT_BYTES = 15 * 1024 * 1024


def _legacy_timestamp_ms(timestamp: Optional[str]) -> Optional[int]:
    """Milliseconds of the game's legacy stamp 'yyyy_MM_dd_HH_mm_ss_fff'. It has no time zone, so only the difference
    between two stamps is meaningful."""

    try:
        moment = datetime.strptime(timestamp[:19], "%Y_%m_%d_%H_%M_%S").replace(tzinfo=timezone.utc)
        return int(moment.timestamp()) * 1000 + int(timestamp[20:23])
    except (TypeError, ValueError):
        return None


def _line_stats(line: Line) -> LineStats:
    """Length, bounds and duration of a line, derived from its points."""

    points = line.points
    if not points:
        return LineStats(pointCount=0, length=0.0)

    first = points[0]
    last = points[-1]
    min_x = max_x = first.x
    min_y = max_y = first.y
    min_z = max_z = first.z
    length = 0.0
    previous = first
    for point in points:
        length += math.hypot(point.x - previous.x, point.y - previous.y, point.z - previous.z)
        min_x = min(min_x, point.x)
        max_x = max(max_x, point.x)
        min_y = min(min_y, point.y)
        max_y = max(max_y, point.y)
        min_z = min(min_z, point.z)
        max_z = max(max_z, point.z)
        previous = point

    start_utc = end_utc = duration = None
    if first.utcMs is not None and last.utcMs is not None:
        start_utc, end_utc = first.utcMs, last.utcMs
        duration = max(0, end_utc - start_utc)
    else:
        start_ms = _legacy_timestamp_ms(first.timestamp)
        end_ms = _legacy_timestamp_ms(last.timestamp)
        if start_ms is not None and end_ms is not None:
            duration = max(0, end_ms - start_ms)

    return LineStats(
        pointCount=len(points),
        length=length,
        durationMs=duration,
        startUtcMs=start_utc,
        endUtcMs=end_utc,
        boundsMinX=min_x, boundsMinY=min_y, boundsMinZ=min_z,
        boundsMaxX=max_x, boundsMaxY=max_y, boundsMaxZ=max_z,
    )


def _bson_size(value: dict) -> int:
    return len(bson.encode(value))


def _fit_tracking(tracked: List[dict], other_bytes: int) -> tuple[List[dict], int]:
    """
    Keeps the tracked poses inside the space MongoDB leaves next to the rest of the document (other_bytes).
    Only the tracking is ever reduced: when it does not fit, every Nth sample is kept (the same N for all players
    and channels, so the lists stay aligned). Returns the poses and N (1 = untouched).

    The alternative would be a rejected save, which loses the whole drawing.
    """

    budget = MAX_DOCUMENT_BYTES - other_bytes
    factor = 1
    for _ in range(10):
        candidate = tracked if factor == 1 else [
            {key: (value[::factor] if isinstance(value, list) else value) for key, value in behavior.items()}
            for behavior in tracked
        ]
        size = _bson_size({"trackedBehaviors": candidate})
        if size <= budget:
            return candidate, factor
        if budget <= 0:
            break
        factor = max(factor + 1, math.ceil(factor * size / budget))

    raise HTTPException(status_code=413, detail="The drawing is too large to be stored (MongoDB document size limit).")


def _prepare_document(drawing: Drawing) -> dict:
    """Derives the line stats and makes the drawing fit MongoDB's document size limit. Returns the document to store."""

    for line in drawing.lines:
        line.stats = _line_stats(line)

    document = drawing.model_dump()
    other_bytes = _bson_size({key: value for key, value in document.items() if key != "trackedBehaviors"})
    tracked, factor = _fit_tracking(document.get("trackedBehaviors") or [], other_bytes)
    document["trackedBehaviors"] = tracked
    document["metadata"]["trackingDecimation"] = factor if factor > 1 else None
    return document


@router.post("/", response_model=Drawing, summary="Save new drawing (MongoDB) and link metadata (SQLite)")
def save_drawing(message: DrawingSaveMessage, db: sqlite3.Connection = Depends(get_db),
                 current: CurrentPlayer = Depends(get_current_player)) -> Drawing:
    """Saves a new drawing using DrawingSaveMessage, generating IDs and resolving missing Session data."""

    if message.owner != current.id:
        raise HTTPException(status_code=403, detail="A drawing can only be saved for the logged-in player.")

    cursor = db.cursor()

    # 1. Check if the Session is closed and fetch missing metadata. Without a session the active one is used,
    #    so saving a drawing is always possible.
    session_id = message.sessionID or get_active_session_id(db)
    cursor.execute("SELECT EndDate, ShowBoy, ShowGirl FROM Sessions WHERE ID = ?", (session_id,))
    session_row = cursor.fetchone()

    if session_row is None:
        raise HTTPException(status_code=400, detail="Provided SessionID does not exist.")
    if session_row[0] is not None:
        raise HTTPException(status_code=403, detail="Cannot save drawing: The session is closed.")

    # 2. Extract SQLite data to populate MongoDB Metadata
    show_boy = bool(session_row[1])
    show_girl = bool(session_row[2])

    # 3. Generate IDs and names
    drawing_id = str(uuid.uuid4())
    drawing_name = message.name if message.name else f"Drawing_{drawing_id}"

    # 4. Construct the full Drawing document
    metadata = Metadata(
        id=drawing_id,
        owner=message.owner,
        collaborators=message.collaborators or [],
        version=message.version,
        date=get_date(),
        gameType=message.gameType,
        sessionID=session_id,
        showBoy=show_boy,
        showGirl=show_girl,
        sadChildCoordinates=message.sadChildCoordinates,
        schemaVersion=SCHEMA_VERSION,
        appVersion=message.appVersion,
        device=message.device
    )

    drawing = Drawing(
        metadata=metadata,
        lines=message.lines,
        trackedBehaviors=message.trackedBehaviors or [],
        placedModels=message.placedModels or [],
        events=message.events or []
    )

    # Line stats are derived and the document is kept below MongoDB's size limit (may raise 413).
    document = _prepare_document(drawing)
    drawing = Drawing.model_validate(document)

    # 5. Write the SQLite linking row FIRST, but do not commit yet - it stays
    #    inside an open transaction. If anything below fails, including the
    #    process crashing before reaching db.commit(), SQLite discards this row
    #    automatically (on rollback, or implicitly when the connection closes
    #    without a commit). That guarantees it never ends up with metadata
    #    pointing at a drawing that was never actually saved to MongoDB.
    try:
        virtual_path = f"VRDrawing3D/Drawings/{metadata.owner}/{drawing_name}/{drawing_name}.json"
        execute_with_retry(
            cursor,
            """
            INSERT INTO DrawingMeta (ID, PlayerID, Name, Path, GameType, SessionID)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metadata.id,
                metadata.owner,
                drawing_name,
                virtual_path,
                int(metadata.gameType),
                metadata.sessionID
            ))
    except sqlite3.IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"SQLite Integrity Error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save to SQLite: {str(e)}")

    # 6. Only now write the full drawing document to MongoDB, and only commit the
    #    pending SQLite row once this has actually succeeded - that way the two
    #    stores can never end up disagreeing about whether the drawing exists.
    try:
        database.mongo_collection.insert_one(document)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save drawing to MongoDB: {str(e)}")

    db.commit()
    return drawing


@router.put("/{drawing_id}", response_model=Drawing, summary="Update an existing drawing")
def update_drawing(drawing_id: str, update_msg: DrawingUpdateMessage, db: sqlite3.Connection = Depends(get_db),
                   current: CurrentPlayer = Depends(get_current_player)):
    """Updates specific properties of an existing drawing in MongoDB"""

    existing = database.mongo_collection.find_one({"metadata.id": drawing_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Drawing not found")

    # The game does not record collaborators yet (the list it sends is always empty), so a guest who joined
    # through a room would never be listed. An active room means the owner opened the drawing for
    # collaboration, so it is treated as the collaboration grant for as long as the room exists.
    metadata = existing.get("metadata", {})
    is_owner = metadata.get("owner") == current.id
    is_collaborator = any(c.get("id") == current.id for c in metadata.get("collaborators") or [])
    if not is_owner and not is_collaborator:
        room_cursor = db.cursor()
        room_cursor.execute("SELECT 1 FROM ActiveRooms WHERE DrawingID = ? LIMIT 1", (drawing_id,))
        if room_cursor.fetchone() is None:
            raise HTTPException(status_code=403, detail="Only the owner or a collaborator can update this drawing.")

    updates = {}

    # Process Mongo Diffs
    if update_msg.lines is not None:
        for line in update_msg.lines:
            line.stats = _line_stats(line)
        updates["lines"] = [line.model_dump() for line in update_msg.lines]
    if update_msg.trackedBehaviors is not None:
        updates["trackedBehaviors"] = [tb.model_dump() for tb in update_msg.trackedBehaviors]
    if update_msg.placedModels is not None:
        updates["placedModels"] = [pm.model_dump() for pm in update_msg.placedModels]
    if update_msg.events is not None:
        updates["events"] = [event.model_dump() for event in update_msg.events]
    if update_msg.collaborators is not None:
        updates["metadata.collaborators"] = [c.model_dump() for c in update_msg.collaborators]

    # The stored document is the existing one with the updates applied: keep it below MongoDB's size limit by
    # thinning the tracking if needed (the existing tracking counts, too, when only the lines grew).
    projected = {key: value for key, value in existing.items() if key not in ("_id", "trackedBehaviors")}
    projected.update({key: value for key, value in updates.items() if "." not in key and key != "trackedBehaviors"})
    tracked, factor = _fit_tracking(
        updates["trackedBehaviors"] if "trackedBehaviors" in updates else existing.get("trackedBehaviors") or [],
        _bson_size(projected)
    )
    if "trackedBehaviors" in updates or factor > 1:
        updates["trackedBehaviors"] = tracked
        updates["metadata.trackingDecimation"] = factor if factor > 1 else None

    if updates:
        database.mongo_collection.update_one({"metadata.id": drawing_id}, {"$set": updates})

    # Process SQLite Diffs. Note: MongoDB has already been updated above by this
    # point (its document has no separate "name" field to keep in sync, so there's
    # no cross-store field to roll back here) - if this SQLite write fails, the
    # content update in Mongo still stands; we report the rename failure
    # explicitly rather than silently leaving the caller unsure what succeeded.
    if update_msg.name is not None:
        try:
            cursor = db.cursor()
            execute_with_retry(cursor, "UPDATE DrawingMeta SET Name = ? WHERE ID = ?", (update_msg.name, drawing_id))
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Drawing content was updated, but renaming failed in SQLite: {str(e)}"
            )

    updated_data = database.mongo_collection.find_one({"metadata.id": drawing_id})
    updated_data.pop("_id", None)
    return updated_data


@router.post("/import", response_model=Drawing, summary="Import a fully formed drawing directly")
def import_drawing(drawing: Drawing, db: sqlite3.Connection = Depends(get_db),
                   current: CurrentPlayer = Depends(get_current_player)) -> Drawing:
    """Import an already established drawing (JSON format) directly into the databases"""

    cursor = db.cursor()

    # The import replaces an existing drawing with the same ID, so neither the new owner nor the
    # current owner of that ID may be somebody else (unless the caller is a Psychologist).
    if not current.is_psychologist:
        existing_owner = _find_owner(drawing.metadata.id, cursor)
        if drawing.metadata.owner != current.id or (existing_owner is not None and existing_owner != current.id):
            raise HTTPException(status_code=403, detail="A drawing can only be imported for the logged-in player.")

    # Line stats are derived and the document is kept below MongoDB's size limit (may raise 413).
    document = _prepare_document(drawing)

    drawing_name = f"Drawing_{drawing.metadata.id}"  # TODO: Templates should be defined in the config
    virtual_path = f"VRDrawing3D/Drawings/{drawing.metadata.owner}/{drawing_name}/{drawing_name}.json"

    # Same ordering as save_drawing: write SQLite first (uncommitted), then Mongo,
    # then commit - so a failure at either step can't leave the two stores
    # disagreeing about whether the drawing exists.
    try:
        execute_with_retry(
            cursor,
            """
            REPLACE INTO DrawingMeta (ID, PlayerID, Name, Path, GameType, SessionID)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                drawing.metadata.id,
                drawing.metadata.owner,
                drawing_name,
                virtual_path,
                int(drawing.metadata.gameType),
                drawing.metadata.sessionID
            ))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to import drawing to SQLite: {str(e)}")

    try:
        database.mongo_collection.update_one(
            {"metadata.id": drawing.metadata.id},
            {"$set": document},
            upsert=True
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to import drawing to MongoDB: {str(e)}")

    db.commit()
    return Drawing.model_validate(document)


@router.get("/", response_model=List[str], summary="Get list of drawing IDs", dependencies=[Depends(require_psychologist)])
def get_drawing_ids(db: sqlite3.Connection = Depends(get_db)) -> List[str]:
    """Get list of drawing IDs (from SQLite DB)"""
    try:
        cursor = db.cursor()
        cursor.execute("SELECT ID FROM DrawingMeta")
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch drawing IDs: {str(e)}")


@router.get("/{drawing_id}", response_model=Drawing, summary="Get a full 3D drawing by ID")
def get_drawing(drawing_id: str):
    """Return a 3D drawing by ID directly from MongoDB."""

    drawing_data = database.mongo_collection.find_one({"metadata.id": drawing_id})
    if drawing_data is not None:
        drawing_data.pop("_id", None)
        return drawing_data

    raise HTTPException(status_code=404, detail="Drawing not found in MongoDB")


@router.get("/{drawing_id}/meta", response_model=DrawingMeta, summary="Get drawing metadata by ID")
def get_drawing_meta(drawing_id: str, db: sqlite3.Connection = Depends(get_db)) -> DrawingMeta:
    """Get drawing meta (DrawingMeta from SQLite DB)"""

    try:
        cursor = db.cursor()
        cursor.execute(
            "SELECT ID, PlayerID, Name, Path, GameType, SessionID FROM DrawingMeta WHERE ID = ?",
            (drawing_id,)
        )
        row = cursor.fetchone()

        if row:
            return DrawingMeta(
                id=row[0],
                playerID=row[1],
                name=row[2],
                path=row[3],
                gameType=row[4],
                sessionID=row[5] if row[5] else ""
            )

        raise HTTPException(status_code=404, detail="Drawing metadata not found in SQLite DB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch drawing metadata: {str(e)}")


@router.delete("/{drawing_id}", summary="Delete a drawing from MongoDB and SQLite")
def delete_drawing(drawing_id: str, db: sqlite3.Connection = Depends(get_db),
                   current: CurrentPlayer = Depends(get_current_player)) -> dict[str, str | bool]:
    """Delete drawing"""

    cursor = db.cursor()

    # A drawing that exists in neither store has no owner and falls through to the 404 below.
    owner = _find_owner(drawing_id, cursor)
    if owner is not None and owner != current.id:
        raise HTTPException(status_code=403, detail="Only the owner can delete this drawing.")

    # 1. Delete the SQLite linking row FIRST. ActiveRooms.DrawingID has a FOREIGN
    #    KEY pointing at DrawingMeta.ID, so if a room is still linked to this
    #    drawing, SQLite refuses the delete with an IntegrityError. We handle that
    #    specifically instead of letting it fall through as a generic 500 - and
    #    because MongoDB hasn't been touched yet at this point, failing here
    #    leaves both stores exactly as they were (no orphaned/half-deleted state).
    try:
        execute_with_retry(cursor, "DELETE FROM DrawingMeta WHERE ID = ?", (drawing_id,))
        sqlite_rows_affected = cursor.rowcount
    except sqlite3.IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Cannot delete drawing: it still has an active multiplayer room linked to it. Delete that room first."
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete from SQLite: {str(e)}")

    # 2. Only now remove the drawing's content from MongoDB, and only commit the
    #    pending SQLite delete once this has actually succeeded. If the Mongo
    #    delete fails, rolling back restores the SQLite row too, so the two
    #    stores can't end up disagreeing about whether the drawing still exists.
    try:
        mongo_result = database.mongo_collection.delete_one({"metadata.id": drawing_id})
        mongo_success = mongo_result.deleted_count > 0
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete from MongoDB: {str(e)}")

    db.commit()
    sqlite_success = sqlite_rows_affected > 0

    if not mongo_success and not sqlite_success:
        raise HTTPException(status_code=404, detail="Drawing not found in any database.")

    return {
        "message": "Drawing deleted successfully.",
        "mongo_deleted": mongo_success,
        "sqlite_deleted": sqlite_success
    }