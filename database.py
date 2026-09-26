import sqlite3
import time

from typing import Union, Any, Mapping
from pymongo import MongoClient
from pymongo.synchronous.database import Database

import config


mongo_client: Union[MongoClient, None] = None
mongo_collection: Union[Database[Mapping[str, Any] | Any], None] = None


def connect_mongo() -> None:
    """
    Establishes the MongoDB connection and populates the module-level
    `mongo_client` / `mongo_collection` globals.
    """
    global mongo_client, mongo_collection

    mongo_client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=3000)
    mongo_client.admin.command('ping')
    mongo_db = mongo_client[config.MONGO_DB_NAME]
    mongo_collection = mongo_db[config.COLLECTION_NAME]


def close_mongo() -> None:
    """Cleanly closes the MongoDB connection pool. Called on app shutdown."""
    global mongo_client, mongo_collection

    if mongo_client is not None:
        mongo_client.close()
    mongo_client = None
    mongo_collection = None


def get_sqlite_connection() -> sqlite3.Connection:
    """
    Opens a brand-new SQLite connection.
    """
    conn = sqlite3.connect(
        getattr(config, "SQLITE_DB_PATH", "vr_drawings.db"),
        check_same_thread=False,
    )
    conn.execute("PRAGMA foreign_keys = ON;")

    # WAL mode lets readers run concurrently with a writer instead of blocking on
    # every write (the default "rollback journal" mode locks the whole DB file for
    # any write). This is a per-database-file setting (stored in the file header),
    # so it only needs to be set once, but re-issuing it on every connection is
    # cheap and keeps things correct if the DB file is ever recreated.
    conn.execute("PRAGMA journal_mode = WAL;")

    # NORMAL is the recommended synchronous level when using WAL: still safe against
    # app/OS crashes, just relaxes fsync frequency compared to the default FULL,
    # which is what gives WAL most of its speed benefit.
    conn.execute("PRAGMA synchronous = NORMAL;")

    # If two requests DO try to write at the same moment (WAL allows one writer at
    # a time), SQLite's default behavior is to raise "database is locked" instantly.
    # busy_timeout makes it retry internally for up to 5s before giving up, so a
    # burst of simultaneous game requests degrades to "slightly slower" instead of
    # "random 500s".
    conn.execute("PRAGMA busy_timeout = 5000;")

    return conn


def get_db():
    """
    FastAPI dependency. Yields a fresh SQLite connection scoped to a single request,
    and always closes it afterward (even if the request raises an exception).

    Usage in a router:

        @router.get("/{id}")
        def handler(id: str, db: sqlite3.Connection = Depends(get_db)):
            cursor = db.cursor()
            cursor.execute(...)
    """
    conn = get_sqlite_connection()
    try:
        yield conn
    finally:
        conn.close()


def execute_with_retry(cursor: sqlite3.Cursor, query: str, params: tuple = (),
                        *, max_attempts: int = 3, base_delay: float = 0.1) -> sqlite3.Cursor:
    """
    Runs a single write statement (INSERT/UPDATE/DELETE/REPLACE) and retries it if
    SQLite reports the database as locked or busy.

    `busy_timeout` (set on every connection in get_sqlite_connection) already makes
    SQLite wait internally before raising this error, so this is a second line of
    defense for the rarer case where that internal wait is still exceeded (e.g. a
    long-running WAL checkpoint holding the writer lock). Retries with a short
    exponential backoff instead of failing the request outright.

    Returns the same cursor, so callers can keep using .rowcount / .lastrowid as
    before - only the `cursor.execute(...)` call itself needs to be replaced with
    `execute_with_retry(cursor, ...)` at write call sites.
    """
    attempt = 0
    while True:
        try:
            cursor.execute(query, params)
            return cursor
        except sqlite3.OperationalError as e:
            attempt += 1
            message = str(e).lower()
            if "locked" not in message and "busy" not in message:
                raise
            if attempt >= max_attempts:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))


def initialize_sqlite_tables(cursor: sqlite3.Cursor) -> None:

    # TODO: SQLite DB shall be updated based on diffs (ALTER TABLE/...)

    # 1. PlayerInfo Table
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS PlayerInfo (
                ID TEXT PRIMARY KEY,
                Name TEXT,
                Gender TEXT,
                Age INTEGER,
                DominantHand TEXT
            )
        """
    )

    # 2. Passwords Table
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS Passwords (
                ID TEXT PRIMARY KEY,
                Salt TEXT,
                Password TEXT
            )
        """
    )

    # 3. Players Table
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS Players (
                ID TEXT PRIMARY KEY,
                Username TEXT UNIQUE,
                PasswordID TEXT UNIQUE,
                PlayerInfoID TEXT UNIQUE,
                SignedIn TEXT,
                Created TEXT,
                Role INTEGER,
                FOREIGN KEY (PasswordID) REFERENCES Passwords(ID),
                FOREIGN KEY (PlayerInfoID) REFERENCES PlayerInfo(ID)
            )
        """
    )

    # 4. Sessions Table
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS Sessions (
                ID TEXT PRIMARY KEY,
                Name TEXT,
                Description TEXT,
                StartDate TEXT,
                EndDate TEXT,
                ShowBoy INTEGER,
                ShowGirl INTEGER,
                Multiplayer INTEGER
            )
        """
    )

    # TODO: PREVIOUSLY IT WAS Drawings THAT SHOULD ALSO BE HANDLED TO PRESERVE BACKWARD COMPATIBILITY
    # 5. Drawings Table
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS DrawingMeta (
                ID TEXT PRIMARY KEY,
                PlayerID TEXT,
                Name TEXT,
                Path TEXT,
                GameType INTEGER,
                SessionID TEXT,
                FOREIGN KEY (PlayerID) REFERENCES Players(ID),
                FOREIGN KEY (SessionID) REFERENCES Sessions(ID)
            )
        """
    )

    # 6. ActiveRooms Table
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS ActiveRooms (
                ID TEXT PRIMARY KEY,
                RoomCode TEXT UNIQUE,
                Name TEXT,
                RoomAddress TEXT,
                SessionID TEXT,
                DrawingID TEXT,
                HostID TEXT,
                FOREIGN KEY (SessionID) REFERENCES Sessions(ID),
                FOREIGN KEY (DrawingID) REFERENCES DrawingMeta(ID),
                FOREIGN KEY (HostID) REFERENCES Players(ID)
            )
        """
    )

    # 7. Indexes on foreign-key / frequently-filtered columns.
    # UNIQUE columns (Players.Username, ActiveRooms.RoomCode, ...) already get an
    # implicit index from their UNIQUE constraint, so those aren't repeated here.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_drawingmeta_sessionid ON DrawingMeta(SessionID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_drawingmeta_playerid ON DrawingMeta(PlayerID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_activerooms_sessionid ON ActiveRooms(SessionID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_activerooms_drawingid ON ActiveRooms(DrawingID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_activerooms_hostid ON ActiveRooms(HostID);")


def verify_sqlite_schema() -> None:
    """
    Opens a one-off connection, ensures all tables/indexes exist, then closes it.
    Called once from the FastAPI lifespan handler at startup.
    """
    conn = get_sqlite_connection()
    try:
        initialize_sqlite_tables(conn.cursor())
        conn.commit()
    finally:
        conn.close()