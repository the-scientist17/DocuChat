import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma_db"))
COLLECTION_NAME = "docuchat"

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-6-luna")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ~1000 characters is roughly a couple of paragraphs, small enough to stay on topic
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 4))
