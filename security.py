import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Optional

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import config


class Role(IntEnum):
    """Mirrors Assets/Database/DatabaseManagement/SQLiteDB/Role.cs in the game."""

    NONE = -1
    PLAYER = 0
    PSYCHOLOGIST = 1


# --- Signing secret ---

_MIN_SECRET_LENGTH = 32


def _load_secret() -> tuple[str, bool]:
    """
    Returns (secret, is_ephemeral). The secret is read from the environment so that it is never committed.
    Without it, a random one is generated for this process only: every token issued becomes invalid when
    the server restarts, which is acceptable for local mode but should be avoided for a remote server.
    """
    secret = os.environ.get(config.JWT_SECRET_ENV_VAR)
    if secret:
        if len(secret) < _MIN_SECRET_LENGTH:
            raise RuntimeError(
                f"{config.JWT_SECRET_ENV_VAR} must be at least {_MIN_SECRET_LENGTH} characters long."
            )
        return secret, False
    return secrets.token_urlsafe(48), True


_SECRET, SECRET_IS_EPHEMERAL = _load_secret()


# --- Tokens ---

@dataclass(frozen=True)
class CurrentPlayer:
    """The authenticated caller, taken from the verified token (never from the request body or path)."""

    id: str
    role: int

    @property
    def is_psychologist(self) -> bool:
        return self.role == Role.PSYCHOLOGIST


def create_access_token(player_id: str, role: Optional[int]) -> tuple[str, int]:
    """Creates a signed token for the player. Returns (token, expires_in_seconds)."""

    now = datetime.now(timezone.utc)
    expires_in = int(timedelta(hours=config.JWT_EXPIRE_HOURS).total_seconds())
    payload = {
        "sub": player_id,
        "role": int(role) if role is not None else int(Role.NONE),
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }
    return jwt.encode(payload, _SECRET, algorithm=config.JWT_ALGORITHM), expires_in


# --- FastAPI dependencies ---

# auto_error is off so that a missing header produces the same 401 (with WWW-Authenticate) as a bad token.
_bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def get_current_player(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> CurrentPlayer:
    """Requires a valid, unexpired bearer token. Raises 401 otherwise."""

    if credentials is None:
        raise _unauthorized("Not authenticated.")

    try:
        claims = jwt.decode(
            credentials.credentials,
            _SECRET,
            algorithms=[config.JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token has expired.")
    except jwt.InvalidTokenError:
        raise _unauthorized("Invalid token.")

    return CurrentPlayer(id=claims["sub"], role=int(claims.get("role", Role.NONE)))


def require_psychologist(current: CurrentPlayer = Depends(get_current_player)) -> CurrentPlayer:
    """Requires the Psychologist role. Raises 403 otherwise."""

    if not current.is_psychologist:
        raise HTTPException(status_code=403, detail="Psychologist role required.")
    return current


# The two dependencies below read the `user_id` path parameter, so they only fit routes declared with {user_id}.

def self_or_psychologist(user_id: str, current: CurrentPlayer = Depends(get_current_player)) -> CurrentPlayer:
    """The caller may only access their own data, unless they are a Psychologist."""

    if current.id != user_id and not current.is_psychologist:
        raise HTTPException(status_code=403, detail="You can only access your own data.")
    return current


def self_only(user_id: str, current: CurrentPlayer = Depends(get_current_player)) -> CurrentPlayer:
    """The caller may only modify their own data."""

    if current.id != user_id:
        raise HTTPException(status_code=403, detail="You can only modify your own data.")
    return current
