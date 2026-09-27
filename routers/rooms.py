import sqlite3
import uuid
import random
import string
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from database import get_db, execute_with_retry
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
# - Validate room code
# - Delete room (in SQLite DB)
# TODO: Add more GET endpoints for more validation
#       - Get room ID
#       - Get room name
#       - Get room address
#       - Get sessionID for room
#       - Get drawingID for room
#       - Get hostID for room
#       - Get gameType for room


@router.post("/", response_model=ActiveRoom, summary="Create new room")
def create_room(req: RoomCreateRequest, db: sqlite3.Connection = Depends(get_db)) -> ActiveRoom:
    """Create new room (in SQLite DB) generating a UUID and 6-char shortcode."""
    room_id = str(uuid.uuid4())
    room_code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    try:
        cursor = db.cursor()
        execute_with_retry(
            cursor,
            "INSERT INTO ActiveRooms (ID, RoomCode, Name, RoomAddress, SessionID, DrawingID, HostID) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (room_id, room_code, req.name, req.roomAddress, req.sessionID, req.drawingID, req.hostID)
        )
        db.commit()
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
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create room: {str(e)}")


@router.get("/{room_code}", response_model=RoomJoinResponse, summary="Validate room code")
def join_room(room_code: str, db: sqlite3.Connection = Depends(get_db)) -> RoomJoinResponse:
    """Validate room code and retrieve linked address/session/drawing data."""
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT r.RoomCode, r.RoomAddress, r.SessionID, r.DrawingID, d.GameType 
        FROM ActiveRooms r
        JOIN DrawingMeta d ON r.DrawingID = d.ID
        WHERE r.RoomCode = ?
        """, (room_code.upper(),)
    )
    row = cursor.fetchone()

    if row:
        return RoomJoinResponse(roomCode=row[0], roomAddress=row[1], sessionID=row[2], drawingID=row[3], gameType=row[4])

    raise HTTPException(status_code=404, detail="Room not found or expired")


@router.delete("/{room_code}", summary="Delete room")
def delete_room(room_code: str, db: sqlite3.Connection = Depends(get_db)):
    """Delete room (in SQLite DB)"""
    cursor = db.cursor()
    execute_with_retry(cursor, "DELETE FROM ActiveRooms WHERE RoomCode = ?", (room_code.upper(),))
    if cursor.rowcount == 0:
        db.rollback()
        raise HTTPException(status_code=404, detail="Room not found")

    db.commit()
    return {"message": "Room deleted successfully."}


@router.get("/code/{room_code}/id", response_model=str, summary="Get room ID by room code")
def get_room_id_by_code(room_code: str, db: sqlite3.Connection = Depends(get_db)) -> str:
    """Get room ID"""
    cursor = db.cursor()
    cursor.execute("SELECT ID FROM ActiveRooms WHERE RoomCode = ?", (room_code.upper(),))
    row = cursor.fetchone()
    if row:
        return row[0]
    raise HTTPException(status_code=404, detail="Room not found")


@router.get("/{room_id}/name", response_model=str, summary="Get room name by ID")
def get_room_name(room_id: str, db: sqlite3.Connection = Depends(get_db)) -> str:
    """Get room name"""
    cursor = db.cursor()
    cursor.execute("SELECT Name FROM ActiveRooms WHERE ID = ?", (room_id,))
    row = cursor.fetchone()
    if row:
        return row[0]
    raise HTTPException(status_code=404, detail="Room not found")


@router.get("/{room_id}/address", response_model=str, summary="Get room address by ID")
def get_room_address(room_id: str, db: sqlite3.Connection = Depends(get_db)) -> str:
    """Get room address"""
    cursor = db.cursor()
    cursor.execute("SELECT RoomAddress FROM ActiveRooms WHERE ID = ?", (room_id,))
    row = cursor.fetchone()
    if row:
        return row[0]
    raise HTTPException(status_code=404, detail="Room not found")


@router.get("/{room_id}/session", response_model=str, summary="Get sessionID for room by ID")
def get_room_session_id(room_id: str, db: sqlite3.Connection = Depends(get_db)) -> str:
    """Get sessionID for room"""
    cursor = db.cursor()
    cursor.execute("SELECT SessionID FROM ActiveRooms WHERE ID = ?", (room_id,))
    row = cursor.fetchone()
    if row:
        return row[0]
    raise HTTPException(status_code=404, detail="Room not found")


@router.get("/{room_id}/drawing", response_model=str, summary="Get drawingID for room by ID")
def get_room_drawing_id(room_id: str, db: sqlite3.Connection = Depends(get_db)) -> str:
    """Get drawingID for room"""
    cursor = db.cursor()
    cursor.execute("SELECT DrawingID FROM ActiveRooms WHERE ID = ?", (room_id,))
    row = cursor.fetchone()
    if row:
        return row[0]
    raise HTTPException(status_code=404, detail="Room not found")


@router.get("/{room_id}/host", response_model=str, summary="Get hostID for room by ID")
def get_room_host_id(room_id: str, db: sqlite3.Connection = Depends(get_db)) -> str:
    """Get hostID for room"""
    cursor = db.cursor()
    cursor.execute("SELECT HostID FROM ActiveRooms WHERE ID = ?", (room_id,))
    row = cursor.fetchone()
    if row:
        return row[0]
    raise HTTPException(status_code=404, detail="Room not found")


@router.get("/{room_id}/gametype", response_model=int, summary="Get gameType for room by ID")
def get_room_gametype(room_id: str, db: sqlite3.Connection = Depends(get_db)) -> int:
    """Get gameType for room"""
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT d.GameType 
        FROM ActiveRooms r
        JOIN DrawingMeta d ON r.DrawingID = d.ID
        WHERE r.ID = ?
        """, (room_id,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    raise HTTPException(status_code=404, detail="Room not found")