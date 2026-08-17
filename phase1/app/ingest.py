from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import chromadb
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import (
    Settings,
    collection_name_from_url,
    get_settings,
    normalize_url,
    path_allowed,
    url_origin,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = Path(".ingest_checkpoint.json")

DEFAULT_NOISE_TAGS = [
    "nav",
    "header",
    "footer",
    "aside",
    "script",
    "style",
    "button",
    "form",
    "iframe",
    "noscript",
    "svg",
    "[class*='sidebar']",
    "[class*='cookie']",
    "[class*='banner']",
    "[class*='ad-']",
    "[id*='ad-']",
    "[class*='popup']",
]

DEFAULT_CONTENT_SELECTORS = [
    "article",
    "main",
    "[role='main']",
    ".content",
    ".main-content",
    ".documentation",
    ".doc-content",
    ".markdown-body",
    ".rst-content",
    "#content",
    "#main",
    "body",
]

SKIP_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".css",
    ".js",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".mp4",
    ".webm",
    ".mp3",
    ".wav",
)


def save_checkpoint(
    discovered: list[str],
    scraped: list[str],
    queue: list | None = None,
    ) -> None:
    payload = {
        "discovered": discovered,
        "scraped": scraped,
        "queue": queue or [],
    }
    CHECKPOINT_FILE.write_text(json.dumps(payload, indent=2))


def load_checkpoint() -> tuple[list[str], list[str], list]:
    if not CHECKPOINT_FILE.exists():
        return [], [], []

    data = json.loads(CHECKPOINT_FILE.read_text())
    return (
        data.get("discovered", []),
        data.get("scraped", []),
        data.get("queue", []),
    )


class RobotsChecker:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache: dict[str, RobotFileParser] = {}
        self.session = build_http_session(settings)

    def load_parser(self, origin: str) -> RobotFileParser:
        parser = RobotFileParser()
        robots_url = f"{origin}/robots.txt"

        try:
            response = self.session.get(
                robots_url,
                timeout=self.settings.http_timeout,
            )

            if response.status_code == 200:
                parser.parse(response.text.splitlines())
                logger.info(
                    f"Loaded robots.txt for {origin} "
                    f"(entries={len(parser.entries)}, "
                    f"allow_all={parser.allow_all}, "
                    f"disallow_all={parser.disallow_all})"
                )

            elif response.status_code == 404:
                logger.info(
                    f"No robots.txt found for {origin}. Allowing by default."
                )
                parser.allow_all = True

            else:
                logger.warning(
                    f"robots.txt for {origin} returned HTTP {response.status_code}. "
                    "Allowing by default for local development. "
                    "For stricter behavior, change this fallback."
                )
                parser.allow_all = True

        except Exception as exc:
            logger.warning(
                f"Could not fetch robots.txt for {origin}: {exc}. "
                "Allowing by default for local development."
            )
            parser.allow_all = True

        return parser

    def can_fetch(self, url: str) -> bool:
        if not self.settings.respect_robots:
            return True

        origin = url_origin(url)

        parser = self.cache.get(origin)
        if parser is None:
            parser = self.load_parser(origin)
            self.cache[origin] = parser

        try:
            allowed = parser.can_fetch(self.settings.user_agent, url)

            if not allowed:
                logger.debug(
                    f"robots.txt disallows {url} for user agent "
                    f"'{self.settings.user_agent}'"
                )

            return allowed

        except Exception as exc:
            logger.warning(
                f"robots.txt check failed for {url}: {exc}. Allowing by default."
            )
            return True


def build_http_session(settings: Settings) -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": settings.user_agent,
            "Accept": "text/html,application/xhtml+xml",
        }
    )

    return session


class StaticFetcher:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = build_http_session(settings)

    def __call__(self, url: str) -> str:
        response = self.session.get(url, timeout=self.settings.http_timeout)
        response.raise_for_status()
        return response.text


class PlaywrightFetcher:

    def __init__(self, settings: Settings):
        self.settings = settings
        self._playwright = None
        self._browser = None
        self._context = None

    def __enter__(self) -> "PlaywrightFetcher":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Install it with:\n"
                "pip install playwright\n"
                "playwright install chromium"
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            user_agent=self.settings.user_agent
        )

        return self

    def __call__(self, url: str) -> str:
        if self._context is None:
            raise RuntimeError("PlaywrightFetcher must be used as a context manager.")

        page = self._context.new_page()

        try:
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=int(self.settings.http_timeout * 1000),
            )
            page.wait_for_timeout(250)
            return page.content()
        finally:
            page.close()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._context:
            self._context.close()

        if self._browser:
            self._browser.close()

        if self._playwright:
            self._playwright.stop()

        return False


def extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return " ".join(soup.title.string.split())

    h1 = soup.find("h1")
    if h1:
        return " ".join(h1.get_text().split())

    return ""


def extract_content(
    html: str,
    url: str,
    content_selectors: list[str],
    noise_tags: list[str],
    ) -> tuple[str, str]:

    soup = BeautifulSoup(html, "html.parser")

    container: Tag | None = None
    for selector in content_selectors:
        container = soup.select_one(selector)
        if container is not None:
            logger.debug(f"Content matched selector '{selector}' for {url}")
            break

    if container is None:
        raise ValueError(f"No content container found for {url}")

    for selector in noise_tags:
        for tag in container.select(selector):
            tag.decompose()

    code_blocks: list[tuple[str, str]] = []

    for pre_tag in container.find_all("pre"):
        code_tag = pre_tag.find("code")
        code_text = code_tag.get_text() if code_tag else pre_tag.get_text()

        language = ""
        if code_tag:
            classes = code_tag.get("class") or []
            for css_class in classes:
                if css_class.startswith("language-") or css_class.startswith("lang-"):
                    language = css_class.split("-", 1)[1]
                    break

        code_text = code_text.strip()

        fenced = f"```{language}\n{code_text}\n```"

        token = f"RAGCODEBLOCK{len(code_blocks)}END"
        code_blocks.append((token, fenced))

        pre_tag.replace_with(NavigableString(token))

    title = extract_title(soup)

    try:
        import html2text

        handler = html2text.HTML2Text()
        handler.ignore_links = False
        handler.ignore_images = True
        handler.body_width = 0
        handler.protect_links = True
        handler.unicode_snob = True

        text = handler.handle(str(container))
    except Exception as exc:
        logger.warning(
            f"html2text failed for {url}: {exc}. Falling back to plain text."
        )
        text = container.get_text(separator="\n", strip=True)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    for token, fenced in code_blocks:
        text = text.replace(token, f"\n\n{fenced}\n\n")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return title, text


def discover_urls(
    base_url: str,
    fetch_fn,
    settings: Settings,
    robots: RobotsChecker,
    resume: bool = False,
    ) -> list[str]:

    start_url = normalize_url(base_url)
    origin = url_origin(start_url)

    prior_discovered, prior_scraped, prior_queue = (
        load_checkpoint() if resume else ([], [], [])
    )

    visited: set[str] = set(prior_discovered)
    discovered: list[str] = list(prior_discovered)

    queue: deque[tuple[str, int]] = deque()
    queued: set[str] = set()

    if prior_queue:
        for item in prior_queue:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                item_url, item_depth = item
            else:
                item_url, item_depth = item, 0

            item_url = normalize_url(str(item_url))
            queue.append((item_url, int(item_depth)))
            queued.add(item_url)
    elif not prior_discovered:
        queue.append((start_url, 0))
        queued.add(start_url)

    logger.info(
        f"Crawling from {start_url} "
        f"(max_pages={settings.max_pages}, max_depth={settings.max_depth})"
    )

    while queue and len(discovered) < settings.max_pages:
        url, depth = queue.popleft()
        queued.discard(url)

        if url in visited:
            continue

        url_allowed = path_allowed(
            url,
            settings.allowed_paths,
            settings.denied_paths,
        )

        is_start_url = (url == start_url)

        if not url_allowed and not is_start_url:
            logger.debug(f"Skipping disallowed path: {url}")
            continue

        if not robots.can_fetch(url):
            logger.info(f"Skipping {url} because robots.txt disallows it.")
            continue

        visited.add(url)

        try:
            html = fetch_fn(url)
        except Exception as exc:
            logger.warning(f"Skipping {url} during crawl: {exc}")
            time.sleep(settings.crawl_delay)
            continue

        if url_allowed:
            discovered.append(url)
            logger.info(f"Discovered {url} (depth={depth})")
        else:
            logger.info(
                f"Fetched {url} for link discovery only "
                "(not added to ingestion because it is not in allowed paths)."
            )

        if depth < settings.max_depth:
            soup = BeautifulSoup(html, "html.parser")

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()

                if not href:
                    continue

                if href.startswith(("mailto:", "javascript:", "tel:")):
                    continue

                absolute = urljoin(url, href)
                clean = normalize_url(absolute)
                parsed = urlparse(clean)

                if parsed.scheme not in {"http", "https"}:
                    continue

                if url_origin(clean) != origin:
                    continue

                if parsed.path.lower().endswith(SKIP_EXTENSIONS):
                    continue

                if clean in visited or clean in queued:
                    continue

                if not path_allowed(
                    clean,
                    settings.allowed_paths,
                    settings.denied_paths,
                ):
                    continue

                queued.add(clean)
                queue.append((clean, depth + 1))

        time.sleep(settings.crawl_delay)

        if len(discovered) % 10 == 0:
            save_checkpoint(
                discovered=discovered,
                scraped=prior_scraped,
                queue=[list(item) for item in queue],
            )

    save_checkpoint(
        discovered=discovered,
        scraped=prior_scraped,
        queue=[list(item) for item in queue],
    )

    logger.info(f"Crawl complete. {len(discovered)} pages discovered.")
    return discovered


def scrape_documents(
    urls: list[str],
    fetch_fn,
    settings: Settings,
    robots: RobotsChecker,
    content_selectors: list[str],
    noise_tags: list[str],
    already_scraped: set[str] | None = None,
    ) -> list[Document]:

    documents: list[Document] = []
    scraped_so_far: list[str] = list(already_scraped or [])
    skipped_checkpoint = 0

    for raw_url in urls:
        url = normalize_url(raw_url)

        if already_scraped and url in already_scraped:
            skipped_checkpoint += 1
            continue

        if not path_allowed(url, settings.allowed_paths, settings.denied_paths):
            logger.debug(f"Skipping disallowed path during scrape: {url}")
            continue

        if not robots.can_fetch(url):
            logger.info(f"Skipping {url} during scrape because robots.txt disallows it.")
            continue

        try:
            html = fetch_fn(url)
            title, text = extract_content(
                html=html,
                url=url,
                content_selectors=content_selectors,
                noise_tags=noise_tags,
            )

            if len(text) < settings.min_doc_length:
                logger.warning(
                    f"Skipping {url} — content too short ({len(text)} chars)."
                )
                continue

            doc_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            metadata = {
                "source": url,
                "title": title,
                "doc_id": doc_id,
                "content_hash": content_hash,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "embed_model": settings.embed_model,
            }

            documents.append(
                Document(
                    id_=doc_id,
                    text=text,
                    metadata=metadata,
                )
            )

            scraped_so_far.append(url)
            logger.info(f"Scraped {url} ({len(text):,} chars)")

            save_checkpoint(
                discovered=urls,
                scraped=scraped_so_far,
                queue=[],
            )

        except requests.exceptions.RequestException as exc:
            logger.error(f"Network error scraping {url}: {exc}")
        except Exception as exc:
            logger.exception(f"Unexpected error scraping {url}: {exc}")

        time.sleep(settings.crawl_delay)

    if skipped_checkpoint:
        logger.info(
            f"Skipped {skipped_checkpoint} already-scraped URLs from checkpoint."
        )

    logger.info(f"Successfully scraped {len(documents)} / {len(urls)} pages.")
    return documents


def chunk_documents(
    documents: list[Document],
    settings: Settings,
    ) -> list:
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    for index, node in enumerate(nodes):
        node.metadata["chunk_index"] = index

    logger.info(
        f"Split {len(documents)} documents into {len(nodes)} nodes "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})."
    )

    return nodes


def store_in_chromadb(
    nodes: list,
    collection_name: str,
    settings: Settings,
    mode: str,
    ) -> None:
    if not nodes:
        logger.info("No nodes to store.")
        return

    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))

    if mode == "replace":
        try:
            chroma_client.delete_collection(collection_name)
            logger.info(f"Dropped existing collection '{collection_name}'.")
        except Exception:
            logger.info(f"No existing collection '{collection_name}' to drop.")

    collection_metadata = {
        "embed_model": settings.embed_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata=collection_metadata,
    )

    if mode == "refresh":
        sources = sorted(
            {
                node.metadata.get("source")
                for node in nodes
                if node.metadata.get("source")
            }
        )

        for source in sources:
            collection.delete(where={"source": source})

        logger.info(
            f"Refresh mode: deleted existing chunks for {len(sources)} source URL(s)."
        )

    elif mode == "append":
        existing = collection.get(include=["metadatas"])
        existing_sources: set[str] = {
            metadata.get("source", "")
            for metadata in (existing.get("metadatas") or [])
            if metadata and metadata.get("source")
        }

        before = len(nodes)
        nodes = [
            node
            for node in nodes
            if node.metadata.get("source") not in existing_sources
        ]

        skipped = before - len(nodes)
        if skipped:
            logger.info(
                f"Append mode: skipped {skipped} node(s) from already-indexed sources."
            )

    if not nodes:
        logger.info("Collection is already up to date.")
        return

    logger.info(
        f"Embedding {len(nodes)} node(s) with '{settings.embed_model}' "
        f"into ChromaDB collection '{collection_name}'."
    )

    embed_model = OllamaEmbedding(
        model_name=settings.embed_model,
        base_url=settings.ollama_base_url,
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    count = collection.count()
    logger.info(f"Collection '{collection_name}' now holds {count} vectors total.")


def run_ingestion(
    fetch_fn,
    args: argparse.Namespace,
    settings: Settings,
    collection_name: str,
    ) -> None:
    robots = RobotsChecker(settings)

    content_selectors = (args.content_selectors or []) + DEFAULT_CONTENT_SELECTORS
    noise_tags = DEFAULT_NOISE_TAGS + (args.noise_tags or [])

    if args.crawl:
        urls = discover_urls(
            base_url=args.base_url,
            fetch_fn=fetch_fn,
            settings=settings,
            robots=robots,
            resume=args.resume,
        )
    elif args.urls_file:
        raw_urls = args.urls_file.read_text().splitlines()
        urls = [
            normalize_url(line.strip())
            for line in raw_urls
            if line.strip() and not line.strip().startswith("#")
        ]
        logger.info(f"Loaded {len(urls)} URLs from {args.urls_file}.")
    elif args.pages:
        base = args.base_url.rstrip("/")
        urls = []

        for page_path in args.pages:
            if not page_path.startswith("/"):
                page_path = f"/{page_path}"

            urls.append(normalize_url(f"{base}{page_path}"))

        logger.info(f"Using {len(urls)} explicit page paths.")
    else:
        urls = [normalize_url(args.base_url)]
        logger.info("No --crawl / --urls-file / --pages specified. Ingesting root URL only.")

    _, prior_scraped, _ = load_checkpoint() if args.resume else ([], [], [])
    already_scraped = set(prior_scraped)

    documents = scrape_documents(
        urls=urls,
        fetch_fn=fetch_fn,
        settings=settings,
        robots=robots,
        content_selectors=content_selectors,
        noise_tags=noise_tags,
        already_scraped=already_scraped,
    )

    if not documents:
        logger.error("No documents were scraped. Exiting.")
        return

    nodes = chunk_documents(documents, settings)

    store_in_chromadb(
        nodes=nodes,
        collection_name=collection_name,
        settings=settings,
        mode=args.collection_mode,
    )

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Ingestion complete. Checkpoint file removed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape a documentation site and ingest it into a local ChromaDB.\n\n"
            "Examples:\n"
            "python -m app.ingest https://fastapi.tiangolo.com\n"
            "python -m app.ingest https://fastapi.tiangolo.com --pages /tutorial/first-steps/\n"
            "python -m app.ingest https://fastapi.tiangolo.com --crawl --max-pages 30 --max-depth 1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "base_url",
        metavar="URL",
        help="Root URL of the documentation site to ingest.",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--crawl",
        action="store_true",
        help="Auto-discover pages by crawling from the base URL.",
    )
    source.add_argument(
        "--urls-file",
        type=Path,
        metavar="FILE",
        help="Path to a plain-text file of URLs to scrape, one per line.",
    )
    source.add_argument(
        "--pages",
        nargs="+",
        metavar="PATH",
        help=(
            "Explicit URL paths to append to the base URL. "
            "Example: --pages /tutorial/first-steps/ /tutorial/body/"
        ),
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to crawl. Defaults to settings.MAX_PAGES.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum crawl depth. Defaults to settings.MAX_DEPTH.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Seconds to wait between HTTP requests.",
    )
    parser.add_argument(
        "--js",
        action="store_true",
        help="Use Playwright for JavaScript-rendered pages.",
    )
    parser.add_argument(
        "--content-selectors",
        nargs="+",
        metavar="SELECTOR",
        default=[],
        help="CSS selectors for main content. Tried before built-in defaults.",
    )
    parser.add_argument(
        "--noise-tags",
        nargs="+",
        metavar="SELECTOR",
        default=[],
        help="Extra CSS selectors to strip as noise.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=None,
        help="Skip pages whose extracted text is shorter than this.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="SentenceSplitter chunk size.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="SentenceSplitter chunk overlap.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=None,
        help="ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        metavar="NAME",
        help=(
            "ChromaDB collection name. "
            "Defaults to CHROMA_COLLECTION from .env, or a slug from the URL hostname."
        ),
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Ollama embedding model.",
    )
    parser.add_argument(
        "--collection-mode",
        choices=["append", "replace", "refresh"],
        default="append",
        help=(
            "append: add only new sources. "
            "replace: drop collection and rebuild. "
            "refresh: replace only sources present in this ingestion batch."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint file if present.",
    )
    parser.add_argument(
        "--allow-paths",
        nargs="+",
        default=None,
        help="Only crawl/scrape paths starting with these prefixes.",
    )
    parser.add_argument(
        "--deny-paths",
        nargs="+",
        default=None,
        help="Never crawl/scrape paths starting with these prefixes.",
    )
    parser.add_argument(
        "--no-robots",
        action="store_true",
        help="Disable robots.txt checking. Use carefully and responsibly.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    overrides: dict = {}

    if args.max_pages is not None:
        overrides["max_pages"] = args.max_pages

    if args.max_depth is not None:
        overrides["max_depth"] = args.max_depth

    if args.delay is not None:
        overrides["crawl_delay"] = args.delay

    if args.js:
        overrides["use_js"] = True

    if args.min_length is not None:
        overrides["min_doc_length"] = args.min_length

    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size

    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap

    if args.chroma_dir is not None:
        overrides["chroma_dir"] = args.chroma_dir

    if args.embed_model is not None:
        overrides["embed_model"] = args.embed_model

    if args.allow_paths is not None:
        overrides["allowed_paths_csv"] = ",".join(args.allow_paths)

    if args.deny_paths is not None:
        overrides["denied_paths_csv"] = ",".join(args.deny_paths)

    if args.no_robots:
        overrides["respect_robots"] = False

    settings = get_settings().model_copy(update=overrides)

    base_url = args.base_url.rstrip("/")
    derived_collection = collection_name_from_url(base_url)

    collection_name = (
        args.collection
        or settings.chroma_collection
        or derived_collection
    )

    logger.info(f"Target URL      : {base_url}")
    logger.info(f"Collection      : {collection_name}")
    logger.info(f"Collection mode : {args.collection_mode}")
    logger.info(f"Chroma dir      : {settings.chroma_dir}")
    logger.info(f"Embed model     : {settings.embed_model}")
    logger.info(f"robots.txt      : {'enabled' if settings.respect_robots else 'disabled'}")

    if collection_name != derived_collection:
        logger.info(
            f"Derived collection name would be '{derived_collection}'. "
            f"Using '{collection_name}' instead."
        )

    logger.info(
        "If you query this collection with the API, make sure "
        f"CHROMA_COLLECTION={collection_name} is set in your .env."
    )

    if args.collection_mode == "replace" and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Removed stale checkpoint file because --collection-mode=replace.")

    if settings.use_js:
        with PlaywrightFetcher(settings) as fetch_fn:
            run_ingestion(
                fetch_fn=fetch_fn,
                args=args,
                settings=settings,
                collection_name=collection_name,
            )
    else:
        fetch_fn = StaticFetcher(settings)
        run_ingestion(
            fetch_fn=fetch_fn,
            args=args,
            settings=settings,
            collection_name=collection_name,
        )


if __name__ == "__main__":
    main()