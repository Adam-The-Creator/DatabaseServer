import sqlite3

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

import config


def initialize_sqlite_tables():

    # TODO: SQLite DB shall be updated based on diffs (ALTER TABLE/...)

    # 1. PlayerInfo Table
    sqlite_cursor.execute(
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
    sqlite_cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS Passwords (
                ID TEXT PRIMARY KEY,
                Salt TEXT,
                Password TEXT
            )
        """
    )

    # 3. Players Table
    sqlite_cursor.execute(
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
    sqlite_cursor.execute(
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
    sqlite_cursor.execute(
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

# --- Initialize SQLite connection and schemas ---
try:
    sqlite_conn = sqlite3.connect(getattr(config, "SQLITE_DB_PATH", "vr_drawings.db"), check_same_thread=False)
    sqlite_cursor = sqlite_conn.cursor()
    sqlite_cursor.execute("PRAGMA foreign_keys = ON;")

    initialize_sqlite_tables()

    sqlite_conn.commit()
    print("✅ Successfully connected to SQLite and verified relational schemas!")
except Exception as e:
    print(f"❌ ERROR: Failed to setup SQLite: {e}")
    exit(1)
