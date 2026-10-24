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

# Folder of the analysis file store: stored reports, thumbnails and analysis results (see analysis/store.py).
# The environment variable ANALYSIS_STORE_PATH overrides it. Relative paths are relative to the working directory.
ANALYSIS_STORE_PATH = "analysis_store"

# Database server configuration
HOST = "127.0.0.1"
PORT = 8000


# --- Authentication (JWT) ---
# The signing secret is deliberately NOT stored here (this file is committed). Set the environment variable
# named below to a random string of at least 32 characters. If it is not set, a random secret is generated
# on every start, so all issued tokens become invalid when the server restarts.
JWT_SECRET_ENV_VAR = "JWT_SECRET"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 12


# --- Request protection (requirements SEC-02, SEC-03, SEC-04) ---
# Every limit below is a token bucket: it holds up to CAPACITY requests, and gets REFILL_PER_SECOND back every second.
# Normal use of the game and of the web app stays far below them.
THROTTLE_IP_CAPACITY = 300
THROTTLE_IP_REFILL_PER_SECOND = 20
THROTTLE_ACCOUNT_CAPACITY = 300
THROTTLE_ACCOUNT_REFILL_PER_SECOND = 20
# Login and sign-up only (on top of the limits above). Generous, because a whole group may sit behind one address.
THROTTLE_AUTH_CAPACITY = 60
THROTTLE_AUTH_REFILL_PER_SECOND = 1

# Repeated failed logins for the same username from the same address are blocked for a while.
LOGIN_MAX_FAILURES = 5
LOGIN_BLOCK_SECONDS = 15 * 60

# Request bodies bigger than this are refused (413) before they are read.
AUTH_MAX_BODY_BYTES = 4 * 1024
FEEDBACK_MAX_BODY_BYTES = 16 * 1024

# Addresses of reverse proxies / tunnels (ngrok runs on this PC) whose X-Forwarded-For header is believed. For anything
# else the header is ignored, otherwise a client could invent its own address to get round the limits.
TRUSTED_PROXIES = ["127.0.0.1", "::1"]


# --- Feedback and questionnaires (requirements FBK-*, SEC-01, SEC-08) ---
# A questionnaire (same instrument and target) can be sent again after this long.
FEEDBACK_QUESTIONNAIRE_INTERVAL_SECONDS = 60 * 60
# Messages: at least this long between two, and at most this many in 24 hours.
FEEDBACK_MESSAGE_INTERVAL_SECONDS = 60
FEEDBACK_MESSAGES_PER_DAY = 20
# How many submissions one account can have stored.
FEEDBACK_MAX_MESSAGES_PER_ACCOUNT = 500
FEEDBACK_MAX_QUESTIONNAIRES_PER_ACCOUNT = 200
FEEDBACK_MESSAGE_MAX_CHARS = 2000
FEEDBACK_PAGE_SIZE_MAX = 100
# Results of fewer submissions than this are not shown ("not enough data"), which also protects anonymity.
FEEDBACK_MIN_GROUP_SIZE = 3

# Minimum length of the password of the admin account (see server.py --create-admin). The command reads the password
# from the environment variable below when it is set (nothing is stored here), otherwise it asks for it.
ADMIN_MIN_PASSWORD_LENGTH = 10
ADMIN_PASSWORD_ENV_VAR = "ADMIN_PASSWORD"


# --- CORS Configuration ---
# List of origins (frontend URLs) allowed to communicate with this API.
# Use ["*"] to allow any origin (only for local testing, NEVER for production).
CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",        # Example local React/Vue frontend port
    "http://127.0.0.1:3000",
]

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["GET", "POST", "OPTIONS", "PUT", "DELETE"] # Restrict to needed methods, or ["*"] for all
CORS_ALLOW_HEADERS = ["*"] # Or specific headers like ["Content-Type", "Authorization"]