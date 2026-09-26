import sqlite3
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from database import mongo_collection, get_db
from models.mongodb.models import Drawing, GameType, Line, TrackedBehavior, Collaborator, PlacedModel, Metadata
from models.sqlite.models import DrawingMeta
from utils import get_date

# Create the router instance
router = APIRouter(
    prefix="/drawings",
    tags=["Drawings"]
)

# --- DTOs ---

class DrawingSaveMessage(BaseModel):
    name: Optional[str] = None
    owner: str
    collaborators: Optional[List[Collaborator]] = None
    version: Optional[str] = "1.0"
    gameType: GameType
    sessionID: str
    lines: List[Line]
    trackedBehaviors: Optional[List[TrackedBehavior]] = None
    placedModels: Optional[List[PlacedModel]] = None


class DrawingUpdateMessage(BaseModel):
    name: Optional[str] = None
    lines: Optional[List[Line]] = None
    trackedBehaviors: Optional[List[TrackedBehavior]] = None
    placedModels: Optional[List[PlacedModel]] = None
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


@router.post("/", response_model=Drawing, summary="Save new drawing (MongoDB) and link metadata (SQLite)")
def save_drawing(message: DrawingSaveMessage, db: sqlite3.Connection = Depends(get_db)) -> Drawing:
    """Saves a new drawing using DrawingSaveMessage, generating IDs and resolving missing Session data."""

    cursor = db.cursor()

    # 1. Check if the Session is closed and fetch missing metadata
    cursor.execute("SELECT EndDate, ShowBoy, ShowGirl FROM Sessions WHERE ID = ?", (message.sessionID,))
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
        sessionID=message.sessionID,
        showBoy=show_boy,
        showGirl=show_girl,
        sadChildCoordinates=None
    )

    drawing = Drawing(
        metadata=metadata,
        lines=message.lines,
        trackedBehaviors=message.trackedBehaviors or [],
        placedModels=message.placedModels or []
    )

    # 5. Save complex data to MongoDB
    mongo_collection.insert_one(drawing.model_dump())

    # 6. Save linking metadata to SQLite
    try:
        virtual_path = f"VRDrawing3D/Drawings/{metadata.owner}/{drawing_name}/{drawing_name}.json"
        cursor.execute(
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
        db.commit()
    except sqlite3.IntegrityError as e:
        # Rollback Mongo insertion if SQLite fails
        mongo_collection.delete_one({"metadata.id": metadata.id})
        raise HTTPException(status_code=400, detail=f"SQLite Integrity Error: {str(e)}")
    except Exception as e:
        mongo_collection.delete_one({"metadata.id": metadata.id})
        raise HTTPException(status_code=500, detail=f"Failed to save to SQLite: {str(e)}")

    return drawing


@router.put("/{drawing_id}", response_model=Drawing, summary="Update an existing drawing")
def update_drawing(drawing_id: str, update_msg: DrawingUpdateMessage, db: sqlite3.Connection = Depends(get_db)):
    """Updates specific properties of an existing drawing in MongoDB"""

    existing = mongo_collection.find_one({"metadata.id": drawing_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Drawing not found")

    updates = {}

    # Process Mongo Diffs
    if update_msg.lines is not None:
        updates["lines"] = [line.model_dump() for line in update_msg.lines]
    if update_msg.trackedBehaviors is not None:
        updates["trackedBehaviors"] = [tb.model_dump() for tb in update_msg.trackedBehaviors]
    if update_msg.placedModels is not None:
        updates["placedModels"] = [pm.model_dump() for pm in update_msg.placedModels]
    if update_msg.collaborators is not None:
        updates["metadata.collaborators"] = [c.model_dump() for c in update_msg.collaborators]

    if updates:
        mongo_collection.update_one({"metadata.id": drawing_id}, {"$set": updates})

    # Process SQLite Diffs
    if update_msg.name is not None:
        try:
            cursor = db.cursor()
            cursor.execute("UPDATE DrawingMeta SET Name = ? WHERE ID = ?", (update_msg.name, drawing_id))
            db.commit()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update SQLite name: {str(e)}")

    updated_data = mongo_collection.find_one({"metadata.id": drawing_id})
    updated_data.pop("_id", None)
    return updated_data


@router.post("/import", response_model=Drawing, summary="Import a fully formed drawing directly")
def import_drawing(drawing: Drawing, db: sqlite3.Connection = Depends(get_db)) -> Drawing:
    """Import an already established drawing (JSON format) directly into the databases"""

    drawing_dict = drawing.model_dump()
    mongo_collection.update_one(
        {"metadata.id": drawing.metadata.id},
        {"$set": drawing_dict},
        upsert=True
    )

    try:
        drawing_name = f"Drawing_{drawing.metadata.id}"  # TODO: Templates should be defined in the config
        virtual_path = f"VRDrawing3D/Drawings/{drawing.metadata.owner}/{drawing_name}/{drawing_name}.json"

        cursor = db.cursor()
        cursor.execute(
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
        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import drawing to SQLite: {str(e)}")

    return drawing


@router.get("/", response_model=List[str], summary="Get list of drawing IDs")
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

    drawing_data = mongo_collection.find_one({"metadata.id": drawing_id})
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
def delete_drawing(drawing_id: str, db: sqlite3.Connection = Depends(get_db)) -> dict[str, str | bool]:
    """Delete drawing"""

    # Delete from MongoDB
    mongo_result = mongo_collection.delete_one({"metadata.id": drawing_id})
    mongo_success = mongo_result.deleted_count > 0

    # Delete from SQLite
    try:
        cursor = db.cursor()
        cursor.execute("DELETE FROM DrawingMeta WHERE ID = ?", (drawing_id,))
        sqlite_rows_affected = cursor.rowcount
        db.commit()
        sqlite_success = sqlite_rows_affected > 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete from SQLite: {str(e)}")

    if not mongo_success and not sqlite_success:
        raise HTTPException(status_code=404, detail="Drawing not found in any database.")

    return {
        "message": "Drawing deleted successfully.",
        "mongo_deleted": mongo_success,
        "sqlite_deleted": sqlite_success
    }