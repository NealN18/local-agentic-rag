import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone

import chromadb
import trafilatura
from trafilatura import extract, extract_metadata
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import get_settings
from app.scraper.policies import load_policies, get_policy_for_url, is_path_allowed
from app.scraper.discovery import discover_urls
from app.scraper.fetchers import build_session, check_robots, fetch_static, fetch_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STATE_FILE = Path("scraper_state.json")

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def extract_universal(html: str, url: str) -> tuple[str, dict]:
    text = extract(
        html, 
        url=url, 
        output_format="markdown", 
        include_tables=True, 
        include_links=True,
        favor_precision=False 
    ) or ""
    
    metadata_obj = None
    try:
        try:
            metadata_obj = extract_metadata(html, url=url)
        except TypeError:
            metadata_obj = extract_metadata(html)
    except Exception as e:
        logger.warning(f"Metadata extraction failed for {url}: {e}")
        
    clean_meta = {
        "source": url,
        "title": getattr(metadata_obj, "title", "") or "",
        "author": getattr(metadata_obj, "author", "") or "",
        "date": getattr(metadata_obj, "date", "") or "",
        "hostname": urlparse(url).netloc
    }
    
    clean_meta = {k: v for k, v in clean_meta.items() if v}
    if "source" not in clean_meta:
        clean_meta["source"] = url
        
    return text, clean_meta

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Universal Compliant Scraper")
    parser.add_argument("base_url", help="Root URL to ingest")
    parser.add_argument("--collection", required=True, help="ChromaDB collection name")
    parser.add_argument("--mode", choices=["append", "refresh", "replace"], default="refresh")
    args = parser.parse_args()

    settings = get_settings()
    policies = load_policies()
    state = load_state()
    
    session = build_session(get_policy_for_url(args.base_url, policies))
    
    logger.info(f"🔍 Discovering URLs for {args.base_url}...")
    urls = discover_urls(args.base_url, session, timeout=20)
    logger.info(f"Found {len(urls)} candidate URLs.")
    
    documents = []
    for url in urls:
        parsed = urlparse(url)
        policy = get_policy_for_url(url, policies)
        
        if not is_path_allowed(url, policy):
            logger.debug(f"Denied by policy: {url}")
            continue
            
        if not check_robots(parsed, policy, session):
            logger.info(f"Blocked by robots.txt: {url}")
            continue
            
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        if args.mode != "replace" and url_hash in state:
            logger.debug(f"Skipping unchanged (cached): {url}")
            continue
            
        logger.info(f"Fetching: {url}")
        html = fetch_static(url, policy, session) if policy.fetcher == "static" else fetch_playwright(url, policy)
        
        if not html:
            state[url_hash] = {"status": "failed", "url": url}
            continue
            
        text, metadata = extract_universal(html, url)
        if not text or len(text) < 50:
            logger.warning(f"Content too short/empty: {url}")
            continue
            
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        state[url_hash] = {"status": "success", "hash": content_hash, "url": url}
        
        documents.append(Document(
            id_=url_hash, text=text, 
            metadata=metadata
        ))

    save_state(state)
    if not documents:
        logger.error("No documents to ingest. (If using 'refresh', they may all be cached. Delete scraper_state.json to force re-fetch).")
        return

    logger.info(f"Chunking {len(documents)} documents...")
    splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if args.mode == "replace":
        try: chroma_client.delete_collection(args.collection)
        except: pass
        
    collection = chroma_client.get_or_create_collection(args.collection)
    
    if args.mode == "refresh":
        sources = [n.metadata["source"] for n in nodes]
        for source in sources:
            collection.delete(where={"source": source})

    logger.info(f"Embedding {len(nodes)} nodes into '{args.collection}'...")
    embed_model = OllamaEmbedding(model_name=settings.embed_model, base_url=settings.ollama_base_url)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model, show_progress=True)
    logger.info(f"Ingestion complete. Collection holds {collection.count()} vectors.")

if __name__ == "__main__":
    main()