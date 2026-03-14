import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
DB_DIR = CHROMA_DB_DIR

COLLECTION_NAME = "course_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
DEFAULT_RETRIEVAL_K = 4
RETRIEVAL_FETCH_K_FACTOR = 4

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "220"))
