import ast
import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError as ChromaNotFoundError
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP-server] %(levelname)s - %(message)s",
    stream=sys.stderr,  
)
logger = logging.getLogger(__name__)

def _resolve_path(filename: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    for base in [cwd, script_dir, script_dir.parent]:
        if (base / filename).exists():
            return base / filename
    return cwd / filename  

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(_resolve_path("chroma_db"))).strip())
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "fastapi_tiangolo_com").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
MOCK_DB_PATH = Path(os.getenv("MOCK_DB_PATH", str(_resolve_path("mock_data.db"))).strip())

DEFAULT_TOP_K = 5
TOOL_TIMEOUT = 15.0

retriever_state = {
    "index": None,
    "retriever": None,
    "initialized": False,
    "error": None,
}

def build_retriever(top_k: int = DEFAULT_TOP_K) -> tuple[VectorStoreIndex, VectorIndexRetriever]:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION)
    except (ValueError, ChromaNotFoundError) as exc:
        raise RuntimeError(
            f"Collection '{CHROMA_COLLECTION}' not found in '{CHROMA_DIR}'. Run ingest.py first."
        ) from exc

    logger.info(f"Loaded collection '{CHROMA_COLLECTION}' ({chroma_collection.count()} vectors).")
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return index, VectorIndexRetriever(index=index, similarity_top_k=top_k)

def get_retriever(top_k: int = DEFAULT_TOP_K) -> tuple[VectorStoreIndex, VectorIndexRetriever]:
    if retriever_state["error"]:
        raise RuntimeError(retriever_state["error"])

    if not retriever_state["initialized"]:
        try:
            index, retriever = build_retriever(top_k)
            retriever_state["index"] = index
            retriever_state["retriever"] = retriever
            retriever_state["initialized"] = True
        except Exception as exc:
            retriever_state["error"] = str(exc)
            raise

    if top_k != DEFAULT_TOP_K:
        return retriever_state["index"], VectorIndexRetriever(
            index=retriever_state["index"], similarity_top_k=top_k
        )
    return retriever_state["index"], retriever_state["retriever"]

server = Server("rag-mcp-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_documentation",
            description="Search the local documentation vector store and return context chunks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language question."},
                    "top_k": {"type": "integer", "default": DEFAULT_TOP_K, "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="validate_python_syntax",
            description="Check if Python code is syntactically valid. Returns JSON with 'valid' boolean.",
            inputSchema={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source code."}},
                "required": ["code"],
            },
        ),
        Tool(
            name="query_error_logs",
            description="Query ML pipeline error logs by date (YYYY-MM-DD). Returns JSON array.",
            inputSchema={
                "type": "object",
                "properties": {"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}},
                "required": ["date"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "search_documentation":
            return await asyncio.wait_for(tool_search_documentation(arguments), timeout=TOOL_TIMEOUT)
        elif name == "validate_python_syntax":
            return tool_validate_python_syntax(arguments)
        elif name == "query_error_logs":
            return tool_query_error_logs(arguments)
        else:
            raise ValueError(f"Unknown tool: {name!r}")
    except asyncio.TimeoutError:
        return [TextContent(type="text", text=json.dumps({"error": f"Timeout after {TOOL_TIMEOUT}s"}))]
    except Exception as exc:
        logger.exception(f"Tool {name} failed")
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

def extract_python_code(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].strip() == "```": lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()

async def tool_search_documentation(args: dict) -> list[TextContent]:
    query = args["query"]
    top_k = max(1, min(20, int(args.get("top_k", DEFAULT_TOP_K))))
    min_score = args.get("min_score")
    
    _, retriever = get_retriever(top_k)
    nodes = await retriever.aretrieve(query)

    if min_score is not None:
        nodes = [n for n in nodes if n.score is not None and n.score >= min_score]

    chunks = [{
        "rank": i, "source": n.metadata.get("source"), "title": n.metadata.get("title"),
        "score": float(n.score) if n.score else None, "text": n.get_content().strip()
    } for i, n in enumerate(nodes, 1)]

    return [TextContent(type="text", text=json.dumps({"query": query, "count": len(chunks), "chunks": chunks}, indent=2))]

def tool_validate_python_syntax(args: dict) -> list[TextContent]:
    code = extract_python_code(args.get("code", ""))
    try:
        ast.parse(code)
        payload = {"valid": True, "error": None}
    except SyntaxError as exc:
        payload = {"valid": False, "error": {"lineno": exc.lineno, "msg": exc.msg, "text": exc.text.rstrip() if exc.text else None}}
    return [TextContent(type="text", text=json.dumps(payload, indent=2))]

def tool_query_error_logs(args: dict) -> list[TextContent]:
    date = args.get("date", "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return [TextContent(type="text", text=json.dumps({"error": "Invalid date. Use YYYY-MM-DD."}))]
    if not MOCK_DB_PATH.exists():
        return [TextContent(type="text", text=json.dumps({"error": f"DB not found at {MOCK_DB_PATH}"}))]

    con = sqlite3.connect(str(MOCK_DB_PATH))
    cur = con.execute("SELECT timestamp, level, stage, message FROM error_logs WHERE timestamp LIKE ? ORDER BY timestamp DESC LIMIT 10", (f"{date}%",))
    logs = [{"timestamp": ts, "level": lv, "stage": st, "message": msg} for ts, lv, st, msg in cur.fetchall()]
    con.close()
    return [TextContent(type="text", text=json.dumps({"date": date, "count": len(logs), "logs": logs}, indent=2))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())