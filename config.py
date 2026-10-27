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

# Folder of the built web app (DataAnalysis: `pnpm build`, it has an index.html). The server then serves it at /app, so the
# psychologists open the app at the same address as the game (the ngrok URL), without CORS (requirements TEC-08, SEC-09).
# None = the server does not serve the web app. The environment variable named below overrides it (the launcher sets it).
# Relative paths are relative to the working directory.
WEBAPP_PATH = "../DataAnalysis/dist/"
WEBAPP_PATH_ENV_VAR = "WEBAPP_PATH"

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


# --- AI based analysis and decision support (requirements AI-01..AI-17) ---
# All AI runs on this machine (AI-01): there is no cloud call, and the language/vision models are only ever asked
# through a local Ollama server. AI output is decision support for the psychologist, never a diagnosis (AI-05).
# A switch is read when the server starts, so changing one needs a restart (AI-03).
AI_ENABLED = True  # the global switch: False turns every AI function off, the rest of the app is not affected

# One switch per function. The names are used by the web app.
AI_FEATURES = {
    "cohort": True,       # AI-10 atypical drawings, similar drawings, 2D map (scikit-learn, no model weights)
    "groups": True,       # AI-11 stroke clustering into object-like groups (scikit-learn, no model weights)
    "indicators": True,   # AI-14 fuzzy logic indicators from the rule base in ai/rules/
    "insights": True,     # AI-15 session level insights
    "labels": True,       # AI-12 zero-shot labelling by a CLIP model (needs PyTorch + open_clip + the weights)
    "description": True,  # AI-13 description by a vision-language model (needs Ollama + a vision model)
    "summary": True,      # AI-16 conclusion written by a language model (needs Ollama + a text model)
}

# AI-07: a function counts as evaluated once it was tried on a sample set of drawings together with a psychologist
# and the agreement and failure modes were written down (see ai/README.md). Until then the web app shows an extra
# "not yet evaluated" note. Set a function to True here only after that trial.
AI_EVALUATED = {name: False for name in AI_FEATURES}

# A group of fewer drawings than this gets no atypical-drawing detection, map or similar drawings (AI-10).
AI_MIN_COHORT = 8
# Fuzzy indicators use the cohort percentiles when the cohort has at least this many drawings, fixed ranges otherwise.
AI_MIN_FUZZY_COHORT = 8
# The second opinion of the isolation forest needs a few more drawings than the robust z-scores.
AI_MIN_ISOLATION_FOREST = 12

# Background jobs of the heavy functions (AI-02, TEC-05): how many run at once, how many finished ones are remembered.
AI_MAX_CONCURRENT_JOBS = 1
AI_JOB_HISTORY = 100

# A local Ollama server (https://ollama.com) serves the language and vision models. Only a loopback address is accepted.
OLLAMA_URL = "http://127.0.0.1:11434"
AI_TEXT_MODEL = "qwen2.5:7b-instruct"   # AI-16
AI_VISION_MODEL = "qwen2.5vl:7b"        # AI-13
AI_OLLAMA_TIMEOUT_SECONDS = 180
# Asking whether Ollama is there and which models it has must be quick: a closed local port takes about 2 s to be
# refused on Windows, which would stall every status call, so such a question gives up after this long.
AI_OLLAMA_PROBE_SECONDS = 1.0
# CLIP model of the zero-shot labelling (AI-12), as OpenCLIP names them
AI_CLIP_MODEL = "ViT-B-32"
AI_CLIP_PRETRAINED = "laion2b_s34b_b79k"
# Where the fuzzy rule base and the label vocabulary are read from. None = the files that come with the server.
AI_RULES_PATH = None
AI_VOCABULARY_PATH = None


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