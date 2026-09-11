# Advanced Multimodal & Agentic RAG

> An Advanced Multimodal and Agentic Retrieval-Augmented Generation Framework for Evidence-Grounded Multi-Document Question Answering.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- An OpenAI or Google Gemini API key

### 1. Clone & Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# Set OPENAI_API_KEY or GOOGLE_API_KEY
```

### 2. Start Infrastructure

```bash
# Start PostgreSQL; the default app vector store is local ChromaDB
docker-compose up -d

# Optional: start Qdrant when VECTOR_DB_PROVIDER=qdrant
docker-compose --profile qdrant up -d
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the UI

Navigate to **http://localhost:8000** in your browser.

---

## 📂 Project Structure

```
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── config/               # Pydantic Settings configuration
│   ├── api/                  # API endpoints (health, query)
│   ├── ingestion/            # Document upload & ingestion pipeline
│   ├── processing/           # Document processors (PDF, TXT, MD)
│   │   └── chunking/        # Chunking strategies (fixed, sentence)
│   ├── embeddings/           # Embedding providers (SentenceTransformers, OpenAI)
│   ├── retrieval/            # Retrieval pipeline
│   │   └── vector_store/    # Vector DB backends (Qdrant, ChromaDB)
│   ├── generation/           # LLM integration & answer generation
│   │   └── llm/             # LLM providers (OpenAI, Gemini)
│   ├── database/             # SQLAlchemy models & repositories
│   └── schemas/              # Pydantic request/response models
├── frontend/                 # Web UI (HTML/CSS/JS)
├── tests/                    # Test suite
├── docker-compose.yml        # PostgreSQL + optional Qdrant service
└── requirements.txt          # Python dependencies
```

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/documents/upload` | Upload a document |
| `GET` | `/api/v1/documents/` | List all documents |
| `GET` | `/api/v1/documents/{id}` | Get document details |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document |
| `POST` | `/api/v1/query` | Ask a question |

## 🏗️ Architecture

```
User Query → Embedding → Vector Search → Top-K Chunks → LLM + Context → Grounded Answer + Citations
```

The default vector backend is ChromaDB. Qdrant is optional; set `VECTOR_DB_PROVIDER=qdrant`
and start the `qdrant` Compose profile before switching.

Production requires a non-placeholder `AUTH_SECRET`. Usage limits are disabled by default;
set `USAGE_MAX_TOKENS` or `USAGE_MAX_COST_USD` to a positive value to enforce limits over
`USAGE_WINDOW_DAYS`.

## 📋 Development Phases

- [x] **Phase 1** — Baseline RAG (PDF → chunks → embeddings → vector search → LLM)
- [x] **Phase 2** — Hybrid Search (BM25 + reranking + query rewriting)
- [x] **Phase 3** — Multimodal (OCR + VLM for images/charts/diagrams)
- [x] **Phase 4** — Structured Data (CSV/Excel via Pandas Agent)
- [x] **Phase 5** — Knowledge Graph (Neo4j + multi-hop reasoning)
- [x] **Phase 6** — Agentic RAG (planner + self-correction + tools)
- [x] **Phase 7** — Reliability (citations + contradiction detection)
- [x] **Phase 8** — Evaluation (RAGAS benchmarks + experiment comparison)

## 📄 License

This project is for educational and research purposes.
