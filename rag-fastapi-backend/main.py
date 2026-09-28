import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).with_name(".env"))

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app = FastAPI(title="Context RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

vector_store: FAISS | None = None
rag_chain: Any = None
indexed_document_names: list[str] = []


class ChatQuery(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class HealthResponse(BaseModel):
    status: str
    indexed_document: str | None


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ready" if rag_chain is not None else "waiting_for_document",
        indexed_document=", ".join(indexed_document_names) or None,
    )


@app.post("/api/upload")
async def upload_document(files: list[UploadFile] = File(...)) -> dict[str, object]:
    global indexed_document_names, rag_chain, vector_store

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Upload up to 10 PDFs at a time.")
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY is not configured on the backend.",
        )

    stage = "saving the PDF"
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks: list[Document] = []
        filenames: list[str] = []

        for file in files:
            filename = Path(file.filename or "document.pdf").name
            if Path(filename).suffix.lower() != ".pdf":
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} is not a PDF. Only PDF files are supported.",
                )

            stage = f"reading {filename}"
            contents = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(contents) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{filename} exceeds the 20MB per-file limit.",
                )

            temporary_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
                    temporary_file.write(contents)
                    temporary_path = temporary_file.name

                documents = PyPDFLoader(temporary_path).load()
                if not documents:
                    raise HTTPException(
                        status_code=422,
                        detail=f"{filename} does not contain readable pages.",
                    )
                for document in documents:
                    document.metadata["source"] = filename
                chunks.extend(splitter.split_documents(documents))
                filenames.append(filename)
            finally:
                if temporary_path:
                    Path(temporary_path).unlink(missing_ok=True)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="The selected PDFs contain no extractable text.",
            )

        embedding_model = os.getenv(
            "GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"
        )
        if embedding_model == "models/text-embedding-004":
            embedding_model = "models/gemini-embedding-001"

        stage = "creating document embeddings"
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        new_vector_store = FAISS.from_documents(chunks, embeddings)

        chat_model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-3.6-flash")
        if chat_model in {"gemini-2.0-flash", "models/gemini-2.0-flash"}:
            chat_model = "gemini-3.6-flash"

        stage = "configuring the chat model"
        llm = ChatGoogleGenerativeAI(model=chat_model, temperature=0)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer using only the provided document context. "
                    "If the answer is not present, say: I cannot find the answer in the uploaded document.\n\n"
                    "<context>\n{context}\n</context>",
                ),
                ("human", "{input}"),
            ]
        )
        answer_chain = create_stuff_documents_chain(llm, prompt)
        new_rag_chain = create_retrieval_chain(
            new_vector_store.as_retriever(search_kwargs={"k": 4}), answer_chain
        )

        vector_store = new_vector_store
        rag_chain = new_rag_chain
        indexed_document_names = filenames
        return {
            "message": "Documents uploaded, embedded, and indexed successfully.",
            "documents": filenames,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Failed while {stage}: {error}"
        ) from error
    finally:
        for file in files:
            await file.close()


@app.post("/api/chat")
async def chat_with_document(query: ChatQuery) -> dict[str, object]:
    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No document has been indexed yet. Upload a PDF first.",
        )

    try:
        response = rag_chain.invoke({"input": query.question})
        sources = _source_metadata(response.get("context", []))
        return {"answer": _answer_text(response["answer"]), "sources": sources}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Could not answer question: {error}") from error


def _source_metadata(documents: list[Document]) -> list[str]:
    sources: list[str] = []
    seen: set[tuple[str, int | None]] = set()
    for document in documents:
        page = document.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else None
        source = (str(document.metadata.get("source", "uploaded document")), page_number)
        if source in seen:
            continue
        seen.add(source)
        page_label = f" · p.{source[1]}" if source[1] is not None else ""
        sources.append(f"{Path(source[0]).name}{page_label}")
    return sources


def _answer_text(answer: object) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        return "".join(
            item.get("text", "") for item in answer if isinstance(item, dict)
        )
    return str(answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
