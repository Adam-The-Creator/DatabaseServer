import argparse
import getpass
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import ServerSelectionTimeoutError

import admin_account
import config
import database
import security
from routers import drawings, players, auth, sessions, rooms
from routers import admin as admin_api
from routers import analysis as analysis_api  # aliased: "analysis" is also the name of the package it uses
from routers import feedback as feedback_api
from throttle import ProtectionMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    # Connections are established here (ASGI startup) and so the connections can be closed gracefully on shutdown.
    try:
        database.connect_mongo()
        print("✅ Successfully connected to MongoDB!")
    except ServerSelectionTimeoutError:
        print("❌ ERROR: Connection to MongoDB timed out!")
        print(f"Make sure that MongoDB runs on {config.MONGO_URI} address!")
        raise
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to MongoDB: {e}")
        raise

    try:
        database.verify_sqlite_schema()
        print("✅ Successfully connected to SQLite and verified relational schemas!")
    except Exception as e:
        print(f"❌ ERROR: Failed to setup SQLite: {e}")
        raise

    if security.SECRET_IS_EPHEMERAL:
        print(f"⚠️ {config.JWT_SECRET_ENV_VAR} is not set: using a random secret for this run, "
              "so issued tokens become invalid when the server restarts.")

    yield  # <-- the app serves requests while suspended here

    # --- Shutdown ---
    database.close_mongo()
    print("🛑 MongoDB connection closed. Server shutting down.")


app = FastAPI(title="VR Drawing 3D REST API (Hybrid Mongo/SQLite)", lifespan=lifespan)

@app.get("/", summary="Health check endpoint")
def health_check():
    """Basic root endpoint to verify the server is online."""
    return {"status": "online", "message": "VR Drawing Database Server is running"}

# Throttling per client address, request size cap and security headers (requirements SEC-02, SEC-04, SEC-09). Added
# before the CORS middleware, which is then outside of it: a refused request still gets the CORS headers, otherwise the
# browser would show a CORS error instead of the reason.
app.add_middleware(ProtectionMiddleware)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS,
    allow_headers=config.CORS_ALLOW_HEADERS,
)

# Register the router with the main application
app.include_router(auth.router)
app.include_router(players.router)
app.include_router(drawings.router)
app.include_router(sessions.router)
app.include_router(rooms.router)
app.include_router(analysis_api.router)
app.include_router(feedback_api.router)
app.include_router(admin_api.router)


def create_admin_account(username: str) -> int:
    """
    `--create-admin NAME`: creates the admin account of the web app (or sets a new password for it) and exits. The
    password comes from the environment variable named in config.py, or is asked for. It is never printed or stored.
    """

    password = os.environ.get(config.ADMIN_PASSWORD_ENV_VAR)
    if not password:
        password = getpass.getpass(f"Password for the admin '{username}': ")
        if getpass.getpass("Password again: ") != password:
            print("The two passwords are not the same.")
            return 1

    database.verify_sqlite_schema()  # the tables exist, even on a server that has never been started
    conn = database.get_sqlite_connection()
    try:
        outcome = admin_account.create_admin(conn, username, password)
    except admin_account.AdminAccountError as error:
        print(f"Not done: {error}")
        return 1
    finally:
        conn.close()
    print(f"The admin account '{username}' was {outcome}.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start VR Drawing 3D Database server")
    parser.add_argument("--host", type=str, default=config.HOST, help="IP address of the host server")
    parser.add_argument("--port", type=int, default=config.PORT, help="Port of the host server")
    parser.add_argument("--persistentDataPath", type=str, default=config.PERSISTENT_DATA_PATH, help="Persistent Data Path of the game")
    parser.add_argument("--create-admin", metavar="USERNAME", default=None,
                        help="Create the admin account of the web app (or set a new password for it), then exit")

    args = parser.parse_args()

    if args.create_admin is not None:
        sys.exit(create_admin_account(args.create_admin))

    print(f"🚀 Start the server: http://{args.host}:{args.port}")
    print(f"🔎 Test: http://{args.host}:{args.port}/docs")

    # The app object (not the "server:app" import string) is passed so that this also works when the server is
    # packaged into an exe (PyInstaller), where the script is not importable as the module "server".
    uvicorn.run(app, host=args.host, port=args.port, reload=False)