import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
import database  # Import the database module to trigger the initialization prints and schemas
from routers import drawings, players, auth, sessions

app = FastAPI(title="VR Drawing 3D REST API (Hybrid Mongo/SQLite)")


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start VR Drawing 3D Database server")
    parser.add_argument("--host", type=str, default=config.HOST, help="IP address of the host server")
    parser.add_argument("--port", type=int, default=config.PORT, help="Port of the host server")
    parser.add_argument("--persistentDataPath", type=str, default=config.PERSISTENT_DATA_PATH, help="Persistent Data Path of the game")

    args = parser.parse_args()

    print(f"🚀 Start the server: http://{args.host}:{args.port}")
    print(f"🔎 Test: http://{args.host}:{args.port}/docs")

    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)