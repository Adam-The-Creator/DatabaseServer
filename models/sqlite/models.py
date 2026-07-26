from pydantic import BaseModel
from typing import Optional

class PlayerInfo(BaseModel):
    id: str
    name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    dominantHand: Optional[str] = None

class Password(BaseModel):
    id: str
    salt: str
    password: str

class Player(BaseModel):
    id: str
    username: str
    passwordID: str
    playerInfoID: str
    signedIn: Optional[str] = None
    created: str
    role: int

# TODO: Clarify the role of Session table
#       and add more tables if necessary.
#       HOW SHOULD THIS TABLE BE DEFINED?
#       Original role:
#       - Represents psychological assessments
#       with some information of the VR Drawing
#       Space / VR Drawing mode.
class Session(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    startDate: str
    endDate: Optional[str] = None
    showBoy: int
    showGirl: int
    multiplayer: int

class DrawingMeta(BaseModel):
    id: str
    playerID: str
    name: str
    path: str
    gameType: int
    sessionID: str