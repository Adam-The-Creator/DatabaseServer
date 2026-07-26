from typing import List, Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from database import sqlite_conn, sqlite_cursor, mongo_collection
from models.mongodb.models import Drawing
from models.sqlite.models import DrawingMeta
from models.sqlite.models import PlayerInfo

# Create the router instance
router = APIRouter(
    prefix="/players",
    tags=["Players"],
)


# --- DTOs ---

class PlayerInfoUpdateMessage(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    dominantHand: Optional[str] = None


# --- API Endpoints ---
#
# - Get list of drawing meta for player (DrawingMeta from SQLite DB)
# - Get drawing data (JSON format) by drawing ID (Drawing from Mongo DB)
# - Get name of player (from SQLite)
# - Get gender of player (from SQLite)
# - Get age of player (from SQLite)
# - Get dominant hand of player (from SQLite)
# - Get the date of latest login (from SQLite)
# - Get the date of account registration (from SQLite)
# - Get the role of player (from SQLite)
# - Get player info of player (PlayerInfo from SQLite)
# - Get list of player IDs (from SQLite)
# - Update name of player (in SQLite)
# - Update gender of player (in SQLite)
# - Update age of player (in SQLite)
# - Update dominant hand of player (in SQLite)
# - Update player info of player (PlayerInfo in SQLite)


@router.get("/{user_id}/drawings", response_model=List[DrawingMeta], summary="Get drawing metadata for a specific player")
def get_drawings(user_id: str) -> List[DrawingMeta]:
    """Get list of drawing meta for player (DrawingMeta from SQLite DB)"""

    drawings = []
    try:
        sqlite_cursor.execute(
            "SELECT ID, PlayerID, Name, Path, GameType, SessionID FROM DrawingMeta WHERE PlayerID = ? ORDER BY Name ASC",
            (user_id,)
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
        raise HTTPException(status_code=500, detail=f"Failed to fetch user drawings: {str(e)}")


@router.get("/{user_id}/drawings/{drawing_id}", response_model=Drawing, summary="Get full drawing data by ID")
def get_drawing_data(user_id: str, drawing_id: str):
    """Get drawing data (JSON format) by drawing ID (Drawing from Mongo DB)"""

    # Ensuring we query the drawing that specifically belongs to the user
    drawing_data = mongo_collection.find_one({"metadata.id": drawing_id, "metadata.owner": user_id})

    if drawing_data is not None:
        drawing_data.pop("_id", None)
        return drawing_data

    raise HTTPException(status_code=404, detail="Drawing not found for this player in MongoDB")


@router.get("/{user_id}/name", summary="Get name of player")
def get_name(user_id: str) -> str:
    """Get name of player (from SQLite)"""

    sqlite_cursor.execute(
        "SELECT pi.Name FROM PlayerInfo pi JOIN Players p ON pi.ID = p.PlayerInfoID WHERE p.ID = ?",
        (user_id,)
    )
    row = sqlite_cursor.fetchone()
    if row:
        return row[0] if row[0] is not None else ""
    raise HTTPException(status_code=404, detail="Player not found")


@router.get("/{user_id}/gender", summary="Get gender of player")
def get_gender(user_id: str) -> str:
    """Get gender of player (from SQLite)"""

    sqlite_cursor.execute(
        "SELECT pi.Gender FROM PlayerInfo pi JOIN Players p ON pi.ID = p.PlayerInfoID WHERE p.ID = ?",
        (user_id,)
    )
    row = sqlite_cursor.fetchone()
    if row:
        return row[0] if row[0] is not None else ""
    raise HTTPException(status_code=404, detail="Player not found")


@router.get("/{user_id}/age", summary="Get age of player")
def get_age(user_id: str) -> int:
    """Get age of player (from SQLite)"""

    sqlite_cursor.execute(
        "SELECT pi.Age FROM PlayerInfo pi JOIN Players p ON pi.ID = p.PlayerInfoID WHERE p.ID = ?",
        (user_id,)
    )
    row = sqlite_cursor.fetchone()
    if row:
        return row[0] if row[0] is not None else 0
    raise HTTPException(status_code=404, detail="Player not found")


@router.get("/{user_id}/dominant-hand", summary="Get dominant hand of player")
def get_dominant_hand(user_id: str) -> str:
    """Get dominant hand of player (from SQLite)"""

    sqlite_cursor.execute(
        "SELECT pi.DominantHand FROM PlayerInfo pi JOIN Players p ON pi.ID = p.PlayerInfoID WHERE p.ID = ?",
        (user_id,)
    )
    row = sqlite_cursor.fetchone()
    if row:
        return row[0] if row[0] is not None else ""
    raise HTTPException(status_code=404, detail="Player not found")


@router.get("/{user_id}/latest-login", summary="Get the date of latest login")
def get_latest_login(user_id: str) -> str:
    """Get the date of latest login (from SQLite)"""

    sqlite_cursor.execute("SELECT SignedIn FROM Players WHERE ID = ?", (user_id,))
    row = sqlite_cursor.fetchone()
    if row:
        return row[0] if row[0] is not None else ""
    raise HTTPException(status_code=404, detail="Player not found")


@router.get("/{user_id}/registration-date", summary="Get the date of account registration")
def get_registration_date(user_id: str) -> str:
    """Get the date of account registration (from SQLite)"""

    sqlite_cursor.execute("SELECT Created FROM Players WHERE ID = ?", (user_id,))
    row = sqlite_cursor.fetchone()
    if row:
        return row[0] if row[0] is not None else ""
    raise HTTPException(status_code=404, detail="Player not found")


@router.get("/{user_id}/role", summary="Get the role of player")
def get_role(user_id: str) -> int:
    """Get the role of player (from SQLite)"""

    sqlite_cursor.execute("SELECT Role FROM Players WHERE ID = ?", (user_id,))
    row = sqlite_cursor.fetchone()
    if row:
        return row[0] if row[0] is not None else -1
    raise HTTPException(status_code=404, detail="Player not found")


@router.get("/{user_id}/info", response_model=PlayerInfo, summary="Get player info of player")
def get_player_info(user_id: str) -> PlayerInfo:
    """Get player info of player (PlayerInfo from SQLite)"""

    sqlite_cursor.execute(
        "SELECT pi.ID, pi.Name, pi.Gender, pi.Age, pi.DominantHand FROM PlayerInfo pi JOIN Players p ON pi.ID = p.PlayerInfoID WHERE p.ID = ?",
        (user_id,)
    )
    row = sqlite_cursor.fetchone()
    if row:
        return PlayerInfo(
            id=row[0],
            name=row[1],
            gender=row[2],
            age=row[3],
            dominantHand=row[4]
        )
    raise HTTPException(status_code=404, detail="Player not found")


@router.get("/", response_model=List[str], summary="Get list of player IDs")
def get_player_ids() -> List[str]:
    """Get list of player IDs (from SQLite)"""

    try:
        sqlite_cursor.execute("SELECT ID FROM Players")
        rows = sqlite_cursor.fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch player IDs: {str(e)}")


@router.put("/{user_id}/name", summary="Update name of player")
def update_player_name(user_id: str, name: str = Body(..., embed=True)):
    """Update name of player (in SQLite)"""

    sqlite_cursor.execute(
        "UPDATE PlayerInfo SET Name = ? WHERE ID = (SELECT PlayerInfoID FROM Players WHERE ID = ?)",
        (name, user_id)
    )
    if sqlite_cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    sqlite_conn.commit()
    return {"message": "Player name updated successfully."}


@router.put("/{user_id}/gender", summary="Update gender of player")
def update_player_gender(user_id: str, gender: str = Body(..., embed=True)):
    """Update gender of player (in SQLite)"""

    sqlite_cursor.execute(
        "UPDATE PlayerInfo SET Gender = ? WHERE ID = (SELECT PlayerInfoID FROM Players WHERE ID = ?)",
        (gender, user_id)
    )
    if sqlite_cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    sqlite_conn.commit()
    return {"message": "Player gender updated successfully."}


@router.put("/{user_id}/age", summary="Update age of player")
def update_player_age(user_id: str, age: int = Body(..., embed=True)):
    """Update age of player (in SQLite)"""

    sqlite_cursor.execute(
        "UPDATE PlayerInfo SET Age = ? WHERE ID = (SELECT PlayerInfoID FROM Players WHERE ID = ?)",
        (age, user_id)
    )
    if sqlite_cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    sqlite_conn.commit()
    return {"message": "Player age updated successfully."}


@router.put("/{user_id}/dominant-hand", summary="Update dominant hand of player")
def update_player_dominant_hand(user_id: str, dominant_hand: str = Body(..., embed=True)):
    """Update dominant hand of player (in SQLite)"""

    sqlite_cursor.execute(
        "UPDATE PlayerInfo SET DominantHand = ? WHERE ID = (SELECT PlayerInfoID FROM Players WHERE ID = ?)",
        (dominant_hand, user_id)
    )
    if sqlite_cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    sqlite_conn.commit()
    return {"message": "Player dominant hand updated successfully."}


@router.put("/{user_id}/info", summary="Update player info of player")
def update_player_info(user_id: str, info: PlayerInfoUpdateMessage):
    """Update player info of player (PlayerInfo in SQLite)"""

    sqlite_cursor.execute(
        """
        UPDATE PlayerInfo 
        SET Name = ?, Gender = ?, Age = ?, DominantHand = ? 
        WHERE ID = (SELECT PlayerInfoID FROM Players WHERE ID = ?)
        """,
        (info.name, info.gender, info.age, info.dominantHand, user_id)
    )
    if sqlite_cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    sqlite_conn.commit()
    return {"message": "Player info updated successfully."}