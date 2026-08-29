from enum import IntEnum
from pydantic import BaseModel
from typing import List, Optional


class Collaborator(BaseModel):
    id: str
    joined: str


class GameType(IntEnum):
    INNER_CHILD = 0
    WORD_FOREST = 1
    BEACH = 2
    ROOM = 3
    BILATERAL_DRAWING = 4
    SAFE_PLACE = 5


class SadChildCoordinate(BaseModel):
    x: float
    y: float
    z: float
    forwardX: Optional[float]
    forwardY: Optional[float]
    forwardZ: Optional[float]


class Point(BaseModel):
    x: float
    y: float
    z: float
    timestamp: str


class LineColor(BaseModel):
    r: int
    g: int
    b: int
    a: float


class Hand(IntEnum):
    LEFT = -1
    RIGHT = 1


class LineEventType(IntEnum):
    ERASE = 0
    DRAW = 1
    UNDO = 2


class Status(IntEnum):
    ERASED = 0
    DRAWN = 1


class Pose(BaseModel):
    pX: float
    pY: float
    pZ: float
    rX: float
    rY: float
    rZ: float
    rW: float
    timestamp: str


class PlacedModel(BaseModel):
    modelName: str
    positionX: float
    positionY: float
    positionZ: float
    rotationX: float
    rotationY: float
    rotationZ: float
    rotationW: float
    scaleX: float
    scaleY: float
    scaleZ: float
    # timestamp: str  # TODO


class LineEvent(BaseModel):
    eventType: LineEventType
    invoker: str
    timestamp: str
    hand: Hand


class Metadata(BaseModel):
    id: str
    owner: str
    collaborators: Optional[List[Collaborator]]
    version: str
    date: str
    gameType: GameType
    sessionID: str
    showBoy: bool
    showGirl: bool
    sadChildCoordinates: Optional[SadChildCoordinate]


class Line(BaseModel):
    id: str
    points: List[Point]
    startWidth: float
    endWidth: float
    startColor: LineColor
    endColor: LineColor
    hand: Hand
    userID: str
    history: List[LineEvent]
    status: Status


class TrackedBehavior(BaseModel):
    playerID: str
    head: List[Pose]
    rightHand: List[Pose]
    leftHand: List[Pose]
    player: List[Pose]


class Drawing(BaseModel):
    metadata: Metadata
    lines: List[Line]
    trackedBehaviors: Optional[List[TrackedBehavior]]
    placedModels: Optional[List[PlacedModel]]