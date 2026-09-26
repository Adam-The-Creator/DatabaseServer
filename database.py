import sqlite3

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

import config


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


# --- Initialize MongoDB connection ---
try:
    mongo_client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=3000)
    mongo_client.admin.command('ping')
    mongo_db = mongo_client[config.MONGO_DB_NAME]
    mongo_collection = mongo_db[config.COLLECTION_NAME]
    print("✅ Successfully connected to MongoDB!")
except ServerSelectionTimeoutError:
    print("❌ ERROR: Connection to MongoDB timed out!")
    print(f"Make sure that MongoDB runs on {config.MONGO_URI} address!")
    exit(1)

# --- Initialize SQLite schema (one-off connection, used only at startup) ---
try:
    _startup_conn = get_sqlite_connection()
    initialize_sqlite_tables(_startup_conn.cursor())
    _startup_conn.commit()
    _startup_conn.close()
    print("✅ Successfully connected to SQLite and verified relational schemas!")
except Exception as e:
    print(f"❌ ERROR: Failed to setup SQLite: {e}")
    exit(1)