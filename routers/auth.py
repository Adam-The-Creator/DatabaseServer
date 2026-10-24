import hmac
import math
import sqlite3
import uuid
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
import throttle
from database import get_db, execute_with_retry
from security import Role, create_access_token
from utils import get_date, encrypt, generate_salt

# Create the router instance
router = APIRouter(
    prefix="/auth",
    tags=["Auth"],
)


# --- DTOs ---

class LoginMessage(BaseModel):
    username: str
    password: str


class SignUpMessage(BaseModel):
    username: str
    password: str
    role: int = 0


# --- API Endpoints ---
#
# - Login (returns the JWT access token used by every other endpoint)
# - Sign up
#
# These two are the only public endpoints (besides the health check at "/").

@router.post("/signup", summary="Register a new player")
def sign_up(sign_up_data: SignUpMessage, db: sqlite3.Connection = Depends(get_db)):
    """Signup"""

    if not sign_up_data.username or not sign_up_data.password:
        raise HTTPException(status_code=400, detail="Sign up message is invalid.")

    # Only these two roles can be registered. The admin account is created on the server (requirement ADM-03).
    if sign_up_data.role not in (Role.PLAYER, Role.PSYCHOLOGIST):
        raise HTTPException(status_code=400, detail="Only the Player and Psychologist roles can be registered.")

    cursor = db.cursor()

    # 1. Check if username exists
    cursor.execute("SELECT COUNT(*) FROM Players WHERE Username = ?", (sign_up_data.username,))
    if cursor.fetchone()[0] > 0:
        raise HTTPException(status_code=409, detail="Username already exists.")

    # 2. Generate IDs and cryptographic data
    player_id = str(uuid.uuid4())
    password_id = str(uuid.uuid4())
    player_info_id = str(uuid.uuid4())

    salt = generate_salt()
    password_hash = encrypt(sign_up_data.password + salt)
    created_date = get_date()

    try:
        # 3. Insert into Passwords table
        execute_with_retry(
            cursor,
            "INSERT INTO Passwords (ID, Salt, Password) VALUES (?, ?, ?)",
            (password_id, salt, password_hash)
        )

        # 4. Insert into PlayerInfo table (with default empty values as in C#)
        execute_with_retry(
            cursor,
            "INSERT INTO PlayerInfo (ID, Name, Gender, Age, DominantHand) VALUES (?, ?, ?, ?, ?)",
            (player_info_id, None, None, 0, None)
        )

        # 5. Insert into Players table
        execute_with_retry(
            cursor,
            "INSERT INTO Players (ID, Username, PasswordId, PlayerInfoID, SignedIn, Created, Role) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (player_id, sign_up_data.username, password_id, player_info_id, None, created_date, sign_up_data.role)
        )

        db.commit()
        return {"message": "User signed up successfully.", "playerId": player_id}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database transaction failed: {str(e)}")


def _wrong_credentials(username: str, ip: str) -> HTTPException:
    """The answer for an unknown user and for a wrong password alike, so usernames cannot be enumerated."""

    throttle.login_guard.failure(username, ip)
    return HTTPException(status_code=401, detail="Invalid username or password.")


@router.post("/login", summary="Authenticate a player")
def login(login_data: LoginMessage, request: Request, db: sqlite3.Connection = Depends(get_db)):
    """Login"""

    if not login_data.username or not login_data.password:
        raise HTTPException(status_code=400, detail="Login message is invalid.")

    # Repeated failures for the same username from the same address are blocked for a while (requirement SEC-03)
    ip = throttle.client_ip(request.scope)
    blocked = throttle.login_guard.blocked_for(login_data.username, ip)
    if blocked:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed logins. Try again in {throttle.human_wait(blocked)}.",
            headers={"Retry-After": str(math.ceil(blocked))},
        )

    cursor = db.cursor()

    # 1. Find user by Username
    cursor.execute("SELECT ID, PasswordID, Role FROM Players WHERE Username = ?", (login_data.username,))
    user_row = cursor.fetchone()

    if not user_row:
        raise _wrong_credentials(login_data.username, ip)

    player_id, password_data_id, role = user_row[0], user_row[1], user_row[2]

    # 2. Retrieve Password Data (Salt and Hash)
    cursor.execute("SELECT Salt, Password FROM Passwords WHERE ID = ?", (password_data_id,))
    password_row = cursor.fetchone()

    if not password_row:
        raise HTTPException(status_code=500, detail="Password data not found for user, database inconsistency.")

    salt, stored_password_hash = password_row[0], password_row[1]

    # 3. Validate Password Hash (constant-time comparison)
    expected_hash = encrypt(login_data.password + salt)
    if hmac.compare_digest((stored_password_hash or "").encode("utf-8"), expected_hash.encode("utf-8")):
        throttle.login_guard.success(login_data.username, ip)

        # 4. Success! Update the SignedIn timestamp
        current_time = get_date()
        try:
            execute_with_retry(
                cursor,
                "UPDATE Players SET SignedIn = ? WHERE ID = ?",
                (current_time, player_id)
            )
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to update login time: {str(e)}")

        access_token, expires_in = create_access_token(player_id, role)
        return {
            "message": "Login successful.",
            "playerId": player_id,
            "accessToken": access_token,
            "tokenType": "bearer",
            "expiresIn": expires_in,
        }
    else:
        raise _wrong_credentials(login_data.username, ip)