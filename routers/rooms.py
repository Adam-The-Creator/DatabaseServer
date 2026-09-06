import uuid
import random
import string
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import sqlite_conn, sqlite_cursor
from models.sqlite.models import ActiveRoom


# Create the router instance
router = APIRouter(
    prefix="/rooms",
    tags=["Rooms"],
)

# --- DTOs ---

class RoomCreateRequest(BaseModel):
    name: str
    roomAddress: str
    sessionID: str
    drawingID: str
    hostID: str

class RoomJoinResponse(BaseModel):
    roomCode: str
    roomAddress: str
    sessionID: str
    drawingID: str
    gameType: int

# --- API Endpoints ---
#
# - Create new room (in SQLite DB)
# - Delete room (in SQLite DB)
# - Validate room code


@router.post("/", response_model=ActiveRoom, summary="Create new room")
def create_room(req: RoomCreateRequest) -> ActiveRoom:
    """Create new room (in SQLite DB) generating a UUID and 6-char shortcode."""
    room_id = str(uuid.uuid4())
    room_code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    try:
        sqlite_cursor.execute(
            "INSERT INTO ActiveRooms (ID, RoomCode, Name, RoomAddress, SessionID, DrawingID, HostID) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (room_id, room_code, req.name, req.roomAddress, req.sessionID, req.drawingID, req.hostID)
        )
        sqlite_conn.commit()
        return ActiveRoom(
            id=room_id,
            name=req.name,
            roomCode=room_code,
            roomAddress=req.roomAddress,
            sessionID=req.sessionID,
            drawingID=req.drawingID,
            hostID=req.hostID
        )
    except Exception as e:
        sqlite_conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create room: {str(e)}")


@router.get("/{room_code}", response_model=RoomJoinResponse, summary="Validate room code")
def join_room(room_code: str) -> RoomJoinResponse:
    """Validate room code and retrieve linked address/session/drawing data."""
    sqlite_cursor.execute(
        """
        SELECT r.RoomCode, r.RoomAddress, r.SessionID, r.DrawingID, d.GameType 
        FROM ActiveRooms r
        JOIN DrawingMeta d ON r.DrawingID = d.ID
        WHERE r.RoomCode = ?
        """, (room_code.upper(),)
    )
    row = sqlite_cursor.fetchone()

    if row:
        return RoomJoinResponse(roomCode=row[0], roomAddress=row[1], sessionID=row[2], drawingID=row[3], gameType=row[4])

    raise HTTPException(status_code=404, detail="Room not found or expired")


@router.delete("/{room_code}", summary="Delete room")
def delete_room(room_code: str):
    """Delete room (in SQLite DB)"""
    sqlite_cursor.execute("DELETE FROM ActiveRooms WHERE RoomCode = ?", (room_code.upper(),))
    if sqlite_cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Room not found")

    sqlite_conn.commit()
    return {"message": "Room deleted successfully."}