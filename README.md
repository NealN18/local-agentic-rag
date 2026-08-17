# Local Agentic RAG Platform

A 4-phase local Retrieval-Augmented Generation (RAG) and Agentic platform. This project attempts to demonstrate how to build a pipeline from scratch: starting with basic document ingestion, evolving into an MCP-powered self-correcting LangGraph agent, adding enterprise observability, and finishing with a universal, compliant web scraper.

> **A Note on this Repository:**  
> This project was built as an experimental, proof-of-concept foundation to explore the modern local AI stack (LlamaIndex, LangGraph, MCP, and Trafilatura). It is designed to be a starting point rather than a finalised. I highly encourage anyone to expand upon it, and add production-grade authentication, scale the vector databases, or integrate new tooling.

---
## Architecture Overview

### Phase 1: Foundation & Local RAG
* **Ingestion (`app/ingest.py`)**: Depth-limited HTML crawler with `robots.txt` compliance and checkpoint/resume capabilities.
* **Vector Store**: Local ChromaDB persistence.
* **API (`app/api.py`)**: FastAPI endpoint exposing LlamaIndex's `RetrieverQueryEngine` with strict `min_score` filtering and anti-hallucination prompts.

### Phase 2: MCP Server & LangGraph Agent
* **MCP Server (`server.py`)**: Exposes tools (Documentation Search, Python Syntax Validation, SQLite Log Querying) via the Model Context Protocol over stdio.
* **Agentic Loop (`agent.py`)**: A stateful LangGraph ReAct agent capable of self-correction. If the agent writes broken Python, it routes itself through a syntax-validation loop until the code is fixed or the retry limit is reached.

### Phase 3: Observability & Conditional Routing
* **Tracing**: Full integration with **Langfuse** to visually trace LLM thoughts, tool calls, and routing decisions.
* **Explicit Routing**: Graph edges explicitly handle edge cases, such as routing to a `no_context` fallback node when vector searches return 0 relevant chunks, preventing LLM hallucinations.

### Phase 4: Universal Compliant Scraper
* ** Discovery (`app/scraper/discovery.py`)**: Prioritizes machine-readable `llms.txt` and `sitemap.xml` files before falling back to a lightweight BFS HTML spider.
* **Universal Extraction (`app/scraper/pipeline.py`)**: Uses `trafilatura` to automatically strip boilerplate (navbars, ads, footers) from *any* website without writing custom CSS selectors.
* **Domain Policies (`policies.yaml`)**: Enforces per-domain rate limits, allow/deny paths, and strict CAPTCHA-detection aborts to ensure compliant scraping.

---

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) running locally
* Docker (Optional, for local Langfuse observability)

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Create and activate a virtual environment
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Mac/Linux: source .venv/bin/activate

# Pull Ollama models
ollama pull nomic-embed-text
ollama pull qwen2.5:7b  # Or qwen2.5:3b for lower RAM usage

# Config .env
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (Required for JS-rendered scraping & CAPTCHA detection)
playwright install chromium
```

---

## Usage Guide

### Phase 1: Ingest & Query API
```bash
# Ingest documentation using the Phase 1 custom crawler
python -m app.ingest https://fastapi.tiangolo.com --crawl --max-pages 30 --collection fastapi_docs --collection-mode replace

# Start the FastAPI RAG server
uvicorn app.api:app --reload
```
*Test it:* Open `http://localhost:8000/docs` to interact with the Swagger UI, or send `POST` requests to `/query`.

### Phase 2 & 3: Run the Agentic Assistant
Ensure your `.env` file points to the correct `CHROMA_COLLECTION` (e.g., `fastapi_docs`).

```bash
# Ask a documentation question
python -m app.agent "How do I define a path parameter in FastAPI?"

# Test the self-correcting syntax loop
python -m app.agent --code "def my_endpoint(app): return {"

# Start the interactive REPL (with memory)
python -m app.agent -i
```
*Note: If you have Langfuse configured in your `.env`, check your Langfuse dashboard to see the visual trace of the agent's decision-making process*

### Phase 4: Universal Web Scraping
Scrape almost any authorized website on the internet. The scraper will automatically find sitemaps, respect `robots.txt`, and extract clean Markdown.

```bash
# Scrape a tech blog
python -m app.scraper.pipeline https://blog.langchain.dev/ --collection langchain_blog --mode replace

# Scrape Wikipedia
python -m app.scraper.pipeline https://en.wikipedia.org/wiki/Artificial_intelligence --collection wiki_ai --mode replace
```
*After scraping, update your `.env` to point `CHROMA_COLLECTION` to your new collection (e.g., `langchain_blog`) and query it using the Phase 2/3 agent*

---


## Contributing & Expanding

This codebase is designed to be modular. Some next steps for expansion might include:
* Implementing **Parent-Child (Auto-Merging) Chunking** for better code-block retrieval.
* Adding **Ragas** or **DeepEval** for automated RAG evaluation scoring.
* Building a **Streamlit** or **Gradio** frontend for the interactive REPL.
* Implementing **OAuth2** or **JWT** authentication for the FastAPI endpoints.
