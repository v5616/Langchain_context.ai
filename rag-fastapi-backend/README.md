# Context RAG API

FastAPI backend for the Context document Q&A app. It accepts PDFs, creates a FAISS index with Gemini embeddings, and answers questions using retrieved document chunks.

## Setup

```bash
cd rag-fastapi-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add a Gemini API key to `.env`, then start the server:

```bash
uvicorn main:app --reload --port 8000
```

The API docs are available at `http://localhost:8000/docs`.

## Endpoints

- `GET /api/health` reports whether a document is indexed.
- `POST /api/upload` accepts a PDF in the `file` multipart field.
- `POST /api/chat` accepts `{ "question": "..." }` and returns an answer with source pages.
