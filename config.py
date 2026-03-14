import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
DB_DIR = CHROMA_DB_DIR

COLLECTION_NAME = "course_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")