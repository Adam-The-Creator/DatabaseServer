# Local Mongo DB uri
MONGO_URI = "mongodb://localhost:27017/"

# Database name
MONGO_DB_NAME = "vr_drawings_db"

# Name of the collection
COLLECTION_NAME = "drawings"

# SQLite configuration
SQLITE_DB_PATH = "vr_drawings.db"
SQLITE_DB_NAME = SQLITE_DB_PATH.split("/")[-1]

PERSISTENT_DATA_PATH = ""

# Database server configuration
HOST = "127.0.0.1"
PORT = 8000


# --- CORS Configuration ---
# List of origins (frontend URLs) allowed to communicate with this API.
# Use ["*"] to allow any origin (only for local testing, NEVER for production).
CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",        # Example local React/Vue frontend port
    "http://127.0.0.1:3000",
]

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["GET", "POST", "OPTIONS"] # Restrict to needed methods, or ["*"] for all
CORS_ALLOW_HEADERS = ["*"] # Or specific headers like ["Content-Type", "Authorization"]