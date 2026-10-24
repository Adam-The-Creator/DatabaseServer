"""
Creation of the admin account (requirement ADM-02). It happens on the server PC only: `server.py --create-admin NAME`
(also in the packaged exe). The sign-up endpoint cannot create an admin (ADM-03).

The command can be run again: for an existing admin it sets a new password. An account of another role is never
converted, so a player or psychologist cannot be turned into an admin by mistake.
"""

import sqlite3
import uuid

import config
from database import execute_with_retry
from security import Role
from utils import encrypt, generate_salt, get_date

MAX_USERNAME_LENGTH = 64


class AdminAccountError(Exception):
    """The reason the account was not created, in words for the person at the console."""


def create_admin(db: sqlite3.Connection, username: str, password: str) -> str:
    """Creates the admin account, or sets a new password for an existing admin. Returns "created" or "updated". Commits."""

    username = username.strip()
    if not username or len(username) > MAX_USERNAME_LENGTH:
        raise AdminAccountError(f"The username must be 1 to {MAX_USERNAME_LENGTH} characters long.")
    if len(password) < config.ADMIN_MIN_PASSWORD_LENGTH:
        raise AdminAccountError(f"The password must be at least {config.ADMIN_MIN_PASSWORD_LENGTH} characters long.")

    cursor = db.cursor()
    cursor.execute("SELECT PasswordID, Role FROM Players WHERE Username = ?", (username,))
    existing = cursor.fetchone()
    salt = generate_salt()
    password_hash = encrypt(password + salt)

    try:
        if existing:
            password_id, role = existing
            if role != Role.ADMIN:
                raise AdminAccountError("An account with this username exists and is not an admin. It is not converted.")
            execute_with_retry(cursor, "UPDATE Passwords SET Salt = ?, Password = ? WHERE ID = ?",
                               (salt, password_hash, password_id))
            db.commit()
            return "updated"

        password_id, player_info_id, player_id = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
        execute_with_retry(cursor, "INSERT INTO Passwords (ID, Salt, Password) VALUES (?, ?, ?)",
                           (password_id, salt, password_hash))
        execute_with_retry(cursor, "INSERT INTO PlayerInfo (ID, Name, Gender, Age, DominantHand) VALUES (?, ?, ?, ?, ?)",
                           (player_info_id, username, None, 0, None))
        execute_with_retry(
            cursor,
            "INSERT INTO Players (ID, Username, PasswordId, PlayerInfoID, SignedIn, Created, Role) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (player_id, username, password_id, player_info_id, None, get_date(), int(Role.ADMIN)))
        db.commit()
        return "created"
    except BaseException:
        db.rollback()
        raise
