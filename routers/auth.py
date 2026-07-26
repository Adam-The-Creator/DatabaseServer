import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import sqlite_conn, sqlite_cursor
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
# - Login
# - Sign up


@router.post("/signup", summary="Register a new player")
def sign_up(sign_up_data: SignUpMessage):
    """Signup"""

    if not sign_up_data.username or not sign_up_data.password:
        raise HTTPException(status_code=400, detail="Sign up message is invalid.")

    # 1. Check if username exists
    sqlite_cursor.execute("SELECT COUNT(*) FROM Players WHERE Username = ?", (sign_up_data.username,))
    if sqlite_cursor.fetchone()[0] > 0:
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
        sqlite_cursor.execute(
            "INSERT INTO Passwords (ID, Salt, Password) VALUES (?, ?, ?)",
            (password_id, salt, password_hash)
        )

        # 4. Insert into PlayerInfo table (with default empty values as in C#)
        sqlite_cursor.execute(
            "INSERT INTO PlayerInfo (ID, Name, Gender, Age, DominantHand) VALUES (?, ?, ?, ?, ?)",
            (player_info_id, None, None, 0, None)
        )

        # 5. Insert into Players table
        sqlite_cursor.execute(
            "INSERT INTO Players (ID, Username, PasswordId, PlayerInfoID, SignedIn, Created, Role) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (player_id, sign_up_data.username, password_id, player_info_id, None, created_date, sign_up_data.role)
        )

        sqlite_conn.commit()
        return {"message": "User signed up successfully.", "playerId": player_id}

    except Exception as e:
        sqlite_conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database transaction failed: {str(e)}")


@router.post("/login", summary="Authenticate a player")
def login(login_data: LoginMessage):
    """Login"""

    if not login_data.username or not login_data.password:
        raise HTTPException(status_code=400, detail="Login message is invalid.")

    # 1. Find user by Username
    sqlite_cursor.execute("SELECT ID, PasswordID FROM Players WHERE Username = ?", (login_data.username,))
    user_row = sqlite_cursor.fetchone()

    if not user_row:
        raise HTTPException(status_code=401, detail="User not found or Invalid credentials.")

    player_id, password_data_id = user_row[0], user_row[1]

    # 2. Retrieve Password Data (Salt and Hash)
    sqlite_cursor.execute("SELECT Salt, Password FROM Passwords WHERE ID = ?", (password_data_id,))
    password_row = sqlite_cursor.fetchone()

    if not password_row:
        raise HTTPException(status_code=500, detail="Password data not found for user, database inconsistency.")

    salt, stored_password_hash = password_row[0], password_row[1]

    # 3. Validate Password Hash
    if stored_password_hash == encrypt(login_data.password + salt):
        # 4. Success! Update the SignedIn timestamp
        current_time = get_date()
        try:
            sqlite_cursor.execute(
                "UPDATE Players SET SignedIn = ? WHERE ID = ?",
                (current_time, player_id)
            )
            sqlite_conn.commit()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update login time: {str(e)}")

        return {"message": "Login successful.", "playerId": player_id}
    else:
        raise HTTPException(status_code=401, detail="Invalid password.")