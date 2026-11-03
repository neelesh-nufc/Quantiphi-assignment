from pathlib import Path

# Project root (the folder containing app.py)
BASE_DIR = Path(__file__).resolve().parent.parent

USERS_FILE = BASE_DIR / "config" / "users.yaml"
PDF_DIR = BASE_DIR / "data" / "pdfs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "earnings_calls"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# company id (from the filename) -> display name
COMPANY_NAMES = {
    "tcs": "TCS",
    "infosys": "Infosys",
    "hdfcbank": "HDFC Bank",
    "reliance": "Reliance Industries",
}

TOP_K = 5   # number of chunks retrieved per question

GROQ_MODEL= "openai/gpt-oss-120b"
MAX_HISTORY_MESSAGES = 6  

from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables (GROQ_API_KEY)
