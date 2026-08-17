"""
Seed script - creates mock_data.db with 50 rows of fake ML pipeline errors.
Run once before starting the MCP server
"""
import sqlite3
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

def _resolve_path(filename: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    
    for base in [cwd, script_dir, script_dir.parent]:
        if (base / filename).exists():
            return base / filename
    return cwd / filename  # Fallback

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(_resolve_path("chroma_db"))).strip())
MOCK_DB_PATH = Path(os.getenv("MOCK_DB_PATH", str(_resolve_path("mock_data.db"))).strip())

STAGES = [
    "data_ingestion", "preprocessing", "feature_extraction", "embedding",
    "vector_store_write", "retrieval", "llm_synthesis", "post_processing",
]
LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LEVEL_WEIGHTS = [5, 20, 30, 35, 10]

MESSAGES = {
    "data_ingestion": [
        "HTTP 429 Too Many Requests — backing off for 30s",
        "Connection timeout after 15s fetching {url}",
        "SSL certificate verification failed for {url}",
        "Empty response body received from {url}",
    ],
    "preprocessing": [
        "BeautifulSoup found no content container for {url}",
        "Extracted text length 12 chars — below minimum threshold",
        "HTML parser encountered malformed tag",
    ],
    "feature_extraction": [
        "SentenceSplitter produced 0 nodes for document {doc_id}",
        "Chunk size 1024 exceeds model context window — truncating",
    ],
    "embedding": [
        "OllamaEmbedding timeout after 30s on chunk {chunk_id}",
        "Embedding dimension mismatch: expected 768, got 512",
        "nomic-embed-text model not found locally — pull required",
    ],
    "vector_store_write": [
        "ChromaDB write conflict on collection 'fastapi_tiangolo_com'",
        "Duplicate document ID detected — skipping upsert",
    ],
    "retrieval": [
        "Top-k=5 returned 0 results for query '{query}'",
        "Similarity score 0.12 below threshold 0.3 — no results returned",
    ],
    "llm_synthesis": [
        "qwen2.5:3b took 47s — exceeds 15s SLA",
        "LLM returned empty string for query '{query}'",
        "Response truncated at 512 tokens — max_tokens limit reached",
    ],
    "post_processing": [
        "JSON serialisation failed: circular reference in response",
        "Pydantic validation error: 'score' field expected float, got None",
    ],
}

def random_message(stage: str) -> str:
    template = random.choice(MESSAGES[stage])
    return template.format(
        url=f"https://fastapi.tiangolo.com/tutorial/page-{random.randint(1,50)}/",
        doc_id=f"doc_{random.randint(1000, 9999)}",
        chunk_id=f"chunk_{random.randint(1000, 9999)}",
        query=random.choice([
            "how to use dependency injection",
            "create a PUT endpoint",
            "background tasks",
            "OAuth2 with JWT",
        ]),
    )

def seed(db_path: Path = MOCK_DB_PATH, n_rows: int = 50) -> None:
    if db_path.exists():
        print(f"'{db_path}' already exists — dropping and recreating.")
        db_path.unlink()

    con = sqlite3.connect(str(db_path))
    con.execute("""
        CREATE TABLE error_logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level     TEXT NOT NULL,
            stage     TEXT NOT NULL,
            message   TEXT NOT NULL
        )
    """)

    base_dt = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0) - timedelta(days=9)
    random.seed(42)
    rows = []

    for _ in range(n_rows):
        offset_seconds = random.randint(0, 10 * 24 * 3600)
        ts = (base_dt + timedelta(seconds=offset_seconds)).strftime("%Y-%m-%d %H:%M:%S")
        level = random.choices(LEVELS, weights=LEVEL_WEIGHTS, k=1)[0]
        stage = random.choice(STAGES)
        message = random_message(stage)
        rows.append((ts, level, stage, message))

    con.executemany(
        "INSERT INTO error_logs (timestamp, level, stage, message) VALUES (?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()

    print(f"Seeded {n_rows} rows into '{db_path}'.")
    print("\nSample dates to try with query_error_logs:")
    dates = Counter(r[0][:10] for r in rows)
    for date, count in sorted(dates.most_common(5)):
        print(f"  {date}  ({count} rows)")

if __name__ == "__main__":
    seed()