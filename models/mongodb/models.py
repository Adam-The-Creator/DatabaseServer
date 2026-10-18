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
    timestamp: str  # legacy local-time string "yyyy_MM_dd_HH_mm_ss_fff", kept for older drawings and tools
    # Added with schema version 2: absent in drawings that were saved before.
    utcMs: Optional[int] = None  # UTC time in Unix milliseconds
    pressure: Optional[float] = None  # trigger value (0..1) while the point was drawn


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
    # Tracked poses (schema version 2) only carry utcMs, the string costs more than the pose itself.
    timestamp: Optional[str] = None
    utcMs: Optional[int] = None  # UTC time in Unix milliseconds


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


class DrawingEventType(IntEnum):
    MENU_OPEN = 0
    MENU_CLOSE = 1
    MODE_CHANGE = 2  # value: the new drawing mode (None, Draw, Erase, Grab)
    COLOR_CHANGE = 3  # value: "start=#RRGGBBAA end=#RRGGBBAA", recorded once the colour settled
    WIDTH_CHANGE = 4  # value: "start=0.0030 end=0.0030", recorded once the width settled


class DrawingEvent(BaseModel):
    """A change of the player's tools or menus. Strokes and erasures are not repeated here (see Line.history)."""
    type: DrawingEventType
    playerID: str
    hand: Hand
    utcMs: int
    value: Optional[str] = None


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
    # Added with schema version 2: absent in drawings that were saved before.
    schemaVersion: Optional[int] = None
    appVersion: Optional[str] = None  # version of the game that saved the drawing
    device: Optional[str] = None  # device model of the machine that saved the drawing
    # N > 1: only every Nth tracked sample was kept so that the document fits MongoDB's size limit.
    trackingDecimation: Optional[int] = None


class LineStats(BaseModel):
    """Derived by the server on every save/update, so analysis does not have to walk all points."""
    pointCount: int
    length: float  # length of the polyline in metres (world space)
    durationMs: Optional[int] = None
    startUtcMs: Optional[int] = None  # only when the points carry utcMs
    endUtcMs: Optional[int] = None
    boundsMinX: float = 0.0
    boundsMinY: float = 0.0
    boundsMinZ: float = 0.0
    boundsMaxX: float = 0.0
    boundsMaxY: float = 0.0
    boundsMaxZ: float = 0.0


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
    stats: Optional[LineStats] = None


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
    events: Optional[List[DrawingEvent]] = None