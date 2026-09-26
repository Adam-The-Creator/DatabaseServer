import argparse
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import ServerSelectionTimeoutError

import config
import database
from routers import drawings, players, auth, sessions, rooms


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

    yield  # <-- the app serves requests while suspended here

    # --- Shutdown ---
    database.close_mongo()
    print("🛑 MongoDB connection closed. Server shutting down.")


app = FastAPI(title="VR Drawing 3D REST API (Hybrid Mongo/SQLite)", lifespan=lifespan)

@app.get("/", summary="Health check endpoint")
def health_check():
    """Basic root endpoint to verify the server is online."""
    return {"status": "online", "message": "VR Drawing Database Server is running"}

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start VR Drawing 3D Database server")
    parser.add_argument("--host", type=str, default=config.HOST, help="IP address of the host server")
    parser.add_argument("--port", type=int, default=config.PORT, help="Port of the host server")
    parser.add_argument("--persistentDataPath", type=str, default=config.PERSISTENT_DATA_PATH, help="Persistent Data Path of the game")

    args = parser.parse_args()

    print(f"🚀 Start the server: http://{args.host}:{args.port}")
    print(f"🔎 Test: http://{args.host}:{args.port}/docs")

    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)