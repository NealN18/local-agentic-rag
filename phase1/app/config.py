from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    """All values can be overridden with environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    chroma_dir: Path = Path("./chroma_db")
    chroma_collection: str = "fastapi_tiangolo_com"

    # Models
    embed_model: str = "nomic-embed-text"
    llm_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"

    # Retrieval / API
    similarity_top_k: int = 5
    llm_request_timeout: float = 60.0
    http_timeout: float = 20.0

    # Scraping / ingestion
    user_agent: str = "Phase1RAGBot/1.0 (local documentation ingestion)"
    crawl_delay: float = 0.7
    max_pages: int = 100
    max_depth: int = 2
    min_doc_length: int = 50
    respect_robots: bool = True
    use_js: bool = False

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Comma-separated path filters.
    # Example:
    # ALLOWED_PATHS_CSV=/tutorial/,/advanced/
    # DENIED_PATHS_CSV=/release-notes/,/blog/
    allowed_paths_csv: str = ""
    denied_paths_csv: str = "/release-notes/,/blog/,/search"

    # API security / CORS
    api_key: str | None = None
    cors_origins_csv: str = ""

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    @model_validator(mode="after")
    def _strip_string_values(self) -> "Settings":
        """
        Protect against accidental trailing spaces in env vars.
        Example: CHROMA_DIR="./chroma_db " becomes "./chroma_db".
        """
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)

            if isinstance(value, str):
                setattr(self, field_name, value.strip())
            elif isinstance(value, Path):
                setattr(self, field_name, Path(str(value).strip()))

        return self

    @property
    def allowed_paths(self) -> list[str]:
        return split_csv(self.allowed_paths_csv)

    @property
    def denied_paths(self) -> list[str]:
        return split_csv(self.denied_paths_csv)

    @property
    def cors_origins(self) -> list[str]:
        return split_csv(self.cors_origins_csv)


def collection_name_from_url(url: str) -> str:
    """
    Convert a URL into a safe Chroma collection name.

    Example:
        https://fastapi.tiangolo.com -> fastapi_tiangolo_com
    """
    parsed = urlparse(url.strip())
    raw = (parsed.netloc or parsed.path or "docs").lower()
    raw = raw.split(":")[0]

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_")
    slug = slug[:63]

    if len(slug) < 3:
        slug = f"{slug}_db"

    return slug


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())

    scheme = parsed.scheme.lower() or "http"

    netloc = parsed.netloc.lower()
    if "@" in netloc:
        netloc = netloc.split("@")[-1]
    if ":" in netloc:
        netloc = netloc.split(":")[0]

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, "", "", ""))


def url_origin(url: str) -> str:
    parsed = urlparse(url.strip())
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def path_matches(path: str, prefix: str) -> bool:
    """
    Match URL paths safely against allow/deny prefixes.

    Examples:
        path="/tutorial", prefix="/tutorial/"  -> True
        path="/tutorial/first-steps", prefix="/tutorial/" -> True
        path="/tutorial-extra", prefix="/tutorial/" -> False
    """
    prefix = prefix.strip()

    if not prefix:
        return False

    if prefix == "/":
        return True

    prefix = prefix.rstrip("/")

    if not prefix:
        return True

    return path == prefix or path.startswith(prefix + "/")


def path_allowed(
    url: str,
    allowed_paths: list[str],
    denied_paths: list[str],
    ) -> bool:

    path = urlparse(url).path or "/"

    if any(path_matches(path, denied) for denied in denied_paths if denied):
        return False

    if allowed_paths:
        return any(path_matches(path, allowed) for allowed in allowed_paths if allowed)

    return True


@lru_cache
def get_settings() -> Settings:
    return Settings()