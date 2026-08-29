import sqlite3
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database import sqlite_cursor, sqlite_conn, mongo_collection
from models.mongodb.models import Drawing
from models.sqlite.models import DrawingMeta, Session
from utils import get_date

# Create the router instance
router = APIRouter(
    prefix="/sessions",
    tags=["Sessions"]
)


# --- DTOs ---

class SessionCreateMessage(BaseModel):
    name: str
    description: Optional[str] = None
    showBoy: int = 1
    showGirl: int = 1
    multiplayer: int = 0


class SessionUpdateMessage(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    showBoy: Optional[int] = None
    showGirl: Optional[int] = None
    multiplayer: Optional[int] = None


# --- API Endpoints ---
#
# - Get list of drawing meta for session (DrawingMeta from SQLite DB)
# - Get drawing data (JSON format) by drawing ID (Drawing from Mongo DB)
# - Get session info for session (Session from SQLite DB)
# - Get list of player IDs those are participated on that session (from SQLite DB)
# - Get list of session IDs (from SQLite DB)
# - Get latest session ID (from SQLite DB)
# - Create new session (in SQLite DB)
# - Update session (in SQLite DB)
# - Close session (set endDate) (in SQLite DB)
# - Delete session (in SQLite DB)


@router.get("/{session_id}/drawings", response_model=List[DrawingMeta],
            summary="Get drawing metadata for a specific session")
def get_drawings(session_id: str) -> List[DrawingMeta]:
    """Get list of drawing meta for session (DrawingMeta from SQLite DB)"""

    drawings = []
    try:
        sqlite_cursor.execute(
            "SELECT ID, PlayerID, Name, Path, GameType, SessionID FROM DrawingMeta WHERE SessionID = ? ORDER BY Name ASC",
            (session_id,)
        )
        rows = sqlite_cursor.fetchall()
        for row in rows:
            drawings.append(DrawingMeta(
                id=row[0],
                playerID=row[1],
                name=row[2],
                path=row[3],
                gameType=row[4],
                sessionID=row[5] if row[5] else ""
            ))
        return drawings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch session drawings: {str(e)}")


@router.get("/{session_id}/drawings/{drawing_id}", response_model=Drawing, summary="Get full drawing data by ID")
def get_drawing_data(session_id: str, drawing_id: str):
    """Get drawing data (JSON format) by drawing ID (Drawing from Mongo DB)"""

    drawing_data = mongo_collection.find_one({"metadata.id": drawing_id, "metadata.sessionID": session_id})

    if drawing_data is not None:
        drawing_data.pop("_id", None)
        return drawing_data

    raise HTTPException(status_code=404, detail="Drawing not found for this session in MongoDB")


@router.get("/{session_id}", response_model=Session, summary="Get session info for session")
def get_session_info(session_id: str) -> Session:
    """Get session info for session (Session from SQLite DB)"""

    sqlite_cursor.execute(
        "SELECT ID, Name, Description, StartDate, EndDate, ShowBoy, ShowGirl, Multiplayer FROM Sessions WHERE ID = ?",
        (session_id,)
    )
    row = sqlite_cursor.fetchone()
    if row:
        return Session(
            id=row[0],
            name=row[1],
            description=row[2],
            startDate=row[3],
            endDate=row[4],
            showBoy=row[5] if row[5] is not None else 1,  # Fallback, if NULL
            showGirl=row[6] if row[6] is not None else 1,  # Fallback, if NULL
            multiplayer=row[7] if row[7] is not None else 0
        )
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/{session_id}/players", response_model=List[str], summary="Get list of player IDs participated")
def get_participating_players(session_id: str) -> List[str]:
    """Get list of player IDs those are participated on that session (from SQLite DB)"""

    try:
        # Fetch distinct player IDs associated with drawings in this session
        sqlite_cursor.execute("SELECT DISTINCT PlayerID FROM DrawingMeta WHERE SessionID = ?", (session_id,))
        rows = sqlite_cursor.fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch players: {str(e)}")


@router.get("/", response_model=List[str], summary="Get list of session IDs")
def get_session_ids() -> List[str]:
    """Get list of session IDs (from SQLite DB)"""

    try:
        sqlite_cursor.execute("SELECT ID FROM Sessions ORDER BY StartDate ASC")
        rows = sqlite_cursor.fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch session IDs: {str(e)}")


@router.get("/latest/id", response_model=str, summary="Get latest session ID")
def get_latest_session_id() -> str:
    """Get latest session ID (from SQLite DB)"""
    try:
        # SQLite can order the ISO-like date string (yyyy.MM.dd-HH:mm) lexicographically
        sqlite_cursor.execute("SELECT ID FROM Sessions ORDER BY StartDate DESC LIMIT 1")
        row = sqlite_cursor.fetchone()

        if row:
            return row[0]

        raise HTTPException(status_code=404, detail="No sessions found in the database")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch the latest session ID: {str(e)}")


@router.post("/", response_model=Session, summary="Create new session")
def create_session(session_data: SessionCreateMessage) -> Session:
    """Create new session (in SQLite DB)"""

    session_id = str(uuid.uuid4())
    start_date = get_date()

    try:
        sqlite_cursor.execute(
            """
            INSERT INTO Sessions (ID, Name, Description, StartDate, EndDate, ShowBoy, ShowGirl, Multiplayer) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, session_data.name, session_data.description,
                start_date, None, session_data.showBoy,
                session_data.showGirl, session_data.multiplayer
            )
        )
        sqlite_conn.commit()

        return Session(
            id=session_id,
            name=session_data.name,
            description=session_data.description,
            startDate=start_date,
            endDate=None,
            showBoy=session_data.showBoy,
            showGirl=session_data.showGirl,
            multiplayer=session_data.multiplayer
        )
    except Exception as e:
        sqlite_conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.put("/{session_id}", response_model=Session, summary="Update session")
def update_session(session_id: str, update_data: SessionUpdateMessage) -> Session:
    """Update session (in SQLite DB)"""

    # 1. Verify existence
    sqlite_cursor.execute("SELECT ID FROM Sessions WHERE ID = ?", (session_id,))
    if not sqlite_cursor.fetchone():
        raise HTTPException(status_code=404, detail="Session not found")

    # 2. Formulate dynamic SQL query for diffs
    updates = []
    params = []

    if update_data.name is not None:
        updates.append("Name = ?")
        params.append(update_data.name)
    if update_data.description is not None:
        updates.append("Description = ?")
        params.append(update_data.description)
    if update_data.showBoy is not None:
        updates.append("ShowBoy = ?")
        params.append(update_data.showBoy)
    if update_data.showGirl is not None:
        updates.append("ShowGirl = ?")
        params.append(update_data.showGirl)
    if update_data.multiplayer is not None:
        updates.append("Multiplayer = ?")
        params.append(update_data.multiplayer)

    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    params.append(session_id)
    query = f"UPDATE Sessions SET {', '.join(updates)} WHERE ID = ?"

    # 3. Apply updates
    try:
        sqlite_cursor.execute(query, tuple(params))
        sqlite_conn.commit()

        # Return the modified object
        sqlite_cursor.execute(
            "SELECT ID, Name, Description, StartDate, EndDate, ShowBoy, ShowGirl, Multiplayer FROM Sessions WHERE ID = ?",
            (session_id,)
        )
        row = sqlite_cursor.fetchone()
        return Session(
            id=row[0], name=row[1], description=row[2], startDate=row[3],
            endDate=row[4], showBoy=row[5], showGirl=row[6], multiplayer=row[7]
        )
    except Exception as e:
        sqlite_conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}")


@router.put("/{session_id}/close", summary="Close session")
def close_session(session_id: str) -> dict[str, str]:
    """Close session (set endDate) (in SQLite DB)"""

    end_date = get_date()

    # We only update if it exists and hasn't been closed already
    sqlite_cursor.execute(
        "UPDATE Sessions SET EndDate = ? WHERE ID = ? AND EndDate IS NULL",
        (end_date, session_id)
    )

    if sqlite_cursor.rowcount == 0:
        raise HTTPException(status_code=400, detail="Session not found or already closed")

    sqlite_conn.commit()
    return {"message": "Session closed successfully.", "endDate": end_date}


@router.delete("/{session_id}", summary="Delete session")
def delete_session(session_id: str) -> dict[str, str]:
    """Delete session (in SQLite DB)"""

    try:
        sqlite_cursor.execute("DELETE FROM Sessions WHERE ID = ?", (session_id,))
        if sqlite_cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Session not found")

        sqlite_conn.commit()
        return {"message": "Session deleted successfully."}
    except sqlite3.IntegrityError:
        # Triggers if the Session is tied to existing Drawings via Foreign Keys
        raise HTTPException(status_code=400, detail="Cannot delete a session that has associated drawings.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")