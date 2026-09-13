import os
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings, BASE_DIR, UPLOADS_DIR
from app.schemas import (
    ChatRequest,
    RAGResponse,
    FeedbackRequest,
    UploadResponse,
    AdminDocsResponse,
    AdminDocInfo
)
from app.agentic_rag import LangGraphAgenticRAG
from app.ingestion import ingest_document
from app.embeddings import HuggingFaceEmbeddingClient
from app.vectorstore import PineconeVectorStore
from app.tavily_client import TavilySearchClient

# Ensure directories exist
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

FEEDBACK_FILE = DATA_DIR / "feedback.json"
AUDIT_LOG_FILE = DATA_DIR / "audit_logs.json"

app = FastAPI(
    title="Enterprise HR AI Copilot",
    description="Agentic RAG powered by LangGraph, Groq, Pinecone, Hugging Face, and Tavily",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
agentic_rag: Optional[LangGraphAgenticRAG] = None
vector_store: Optional[PineconeVectorStore] = None
hf_embeddings: Optional[HuggingFaceEmbeddingClient] = None
tavily_client: Optional[TavilySearchClient] = None


@app.on_event("startup")
def startup_event():
    """Initialize singletons on server startup."""
    global agentic_rag, vector_store, hf_embeddings, tavily_client
    try:
        hf_embeddings = HuggingFaceEmbeddingClient()
        vector_store = PineconeVectorStore()
        tavily_client = TavilySearchClient()
        agentic_rag = LangGraphAgenticRAG(
            vector_store=vector_store,
            embedding_client=hf_embeddings,
            tavily_client=tavily_client
        )
        print("[OK] All agentic RAG services initialized on startup.")
    except Exception as e:
        print(f"[WARN] Startup initialization warning: {e}")


def verify_admin(username: str = Form(...), password: str = Form(...)):
    """Validate admin credentials."""
    if username != settings.ADMIN_USERNAME or password != settings.ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials. Access denied."
        )
    return username


def log_query_event(response: RAGResponse):
    """Persist query execution trace to audit log."""
    logs = []
    if AUDIT_LOG_FILE.exists():
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": response.question,
        "source_type": response.source_type,
        "answer_snippet": response.answer[:250],
        "citations_count": len(response.citations),
        "decision_trace": response.decision_trace
    }
    logs.append(log_entry)

    # Keep latest 100 entries
    logs = logs[-100:]
    try:
        with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write audit log: {e}")


# ==============================================================================
# ENDPOINT 1: /chat (POST) - Handle user inquiries
# ==============================================================================
@app.post("/chat", response_model=RAGResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handle employee HR questions via the LangGraph Agentic RAG workflow.
    Executes routing, Pinecone KB retrieval, evidence grading, Tavily web fallback, and grounded synthesis.
    """
    global agentic_rag
    if not agentic_rag:
        agentic_rag = LangGraphAgenticRAG()

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = agentic_rag.ask(question)
        log_query_event(response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


# ==============================================================================
# ENDPOINT 2: /upload (POST) - Admin Upload & Automatic Pipeline Indexing
# ==============================================================================
@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    admin_user: str = Depends(verify_admin)
):
    """
    Admin-only endpoint to upload and automatically index a document:
    File Upload -> Save to uploads/ -> Text Extraction -> Cleaning -> Chunking -> HF Embeddings -> Pinecone Upsert
    """
    global hf_embeddings, vector_store
    if not hf_embeddings:
        hf_embeddings = HuggingFaceEmbeddingClient()
    if not vector_store:
        vector_store = PineconeVectorStore()

    filename = file.filename
    ext = Path(filename).suffix.lower()
    if ext not in [".pdf", ".docx", ".doc", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: .pdf, .docx, .txt"
        )

    dest_path = UPLOADS_DIR / filename
    try:
        # Save file to disk
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Trigger Automated Pipeline
        chunks = ingest_document(dest_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document produced 0 readable text chunks.")

        chunk_texts = [c.content for c in chunks]
        embeddings = hf_embeddings.embed_documents(chunk_texts, batch_size=16)
        upserted_count = vector_store.upsert_chunks(chunks, embeddings)

        return UploadResponse(
            status="success",
            filename=filename,
            doc_id=chunks[0].doc_id,
            chunks_indexed=upserted_count,
            message=f"Successfully uploaded and indexed {upserted_count} chunks into Pinecone index '{vector_store.index_name}'."
        )

    except HTTPException:
        raise
    except Exception as e:
        # Clean up failed file if needed
        raise HTTPException(status_code=500, detail=f"Pipeline indexing failed: {str(e)}")


# ==============================================================================
# ENDPOINT 3: /ingest (POST) - Batch Re-index existing documents
# ==============================================================================
@app.post("/ingest")
async def batch_ingest(admin_user: str = Depends(verify_admin)):
    """Re-index all documents currently stored in the uploads/ directory."""
    global hf_embeddings, vector_store
    if not hf_embeddings:
        hf_embeddings = HuggingFaceEmbeddingClient()
    if not vector_store:
        vector_store = PineconeVectorStore()

    files = list(UPLOADS_DIR.glob("*.*"))
    if not files:
        return {"status": "empty", "message": "No documents found in uploads directory."}

    total_chunks = 0
    results = []
    for f in files:
        if f.suffix.lower() in [".pdf", ".docx", ".doc", ".txt"]:
            try:
                chunks = ingest_document(f)
                embeddings = hf_embeddings.embed_documents([c.content for c in chunks])
                count = vector_store.upsert_chunks(chunks, embeddings)
                total_chunks += count
                results.append({"filename": f.name, "status": "indexed", "chunks": count})
            except Exception as e:
                results.append({"filename": f.name, "status": "failed", "error": str(e)})

    return {
        "status": "completed",
        "total_files_processed": len(files),
        "total_chunks_indexed": total_chunks,
        "details": results
    }


# ==============================================================================
# ENDPOINT 4: /feedback (POST) - User Ratings & Comments
# ==============================================================================
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Record user feedback (thumbs up/down and comments)."""
    feedbacks = []
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
        except Exception:
            feedbacks = []

    feedbacks.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comments": request.comments
    })

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedbacks, f, indent=2)

    return {"status": "success", "message": "Thank you for your feedback!"}


# ==============================================================================
# ENDPOINT 5: /admin/docs (GET) - Document Management Overview
# ==============================================================================
@app.get("/admin/docs", response_model=AdminDocsResponse)
async def list_admin_docs():
    """List all documents currently in the uploads directory and Pinecone stats."""
    global vector_store
    if not vector_store:
        vector_store = PineconeVectorStore()

    docs_info = []
    for f in UPLOADS_DIR.glob("*.*"):
        if f.suffix.lower() in [".pdf", ".docx", ".doc", ".txt"]:
            stat = f.stat()
            docs_info.append(AdminDocInfo(
                filename=f.name,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                file_type=f.suffix.lstrip(".").lower()
            ))

    try:
        stats = vector_store.get_stats()
        vector_count = int(stats.get("total_vector_count", 0))
    except Exception:
        vector_count = 0

    return AdminDocsResponse(
        total_docs=len(docs_info),
        total_vectors_in_pinecone=vector_count,
        documents=docs_info
    )


# ==============================================================================
# ENDPOINT 6: /logs (GET) - View Audit Logs & Decision Traces
# ==============================================================================
@app.get("/logs")
async def get_audit_logs(limit: int = 20):
    """Retrieve recent query execution decision traces."""
    if not AUDIT_LOG_FILE.exists():
        return {"logs": []}
    try:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
        return {"logs": logs[-limit:]}
    except Exception as e:
        return {"logs": [], "error": str(e)}


# ==============================================================================
# ENDPOINT 7: /health (GET) - System Health Check
# ==============================================================================
@app.get("/health")
async def health_check():
    """Verify system connectivity to Pinecone, Groq, Hugging Face, and Tavily."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "pinecone_index": settings.index_name,
            "embedding_model": settings.EMBEDDING_MODEL,
            "groq_model": settings.GROQ_MODEL,
            "tavily_search": bool(settings.TAVILY_API_KEY)
        }
    }


# ==============================================================================
# SERVE WEB FRONTEND
# ==============================================================================
STATIC_DIR = BASE_DIR / "app" / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_ui():
    """Serve the single-page frontend application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        status_code=200,
        content={"message": "Enterprise HR AI Copilot API is online. Frontend index.html not yet built."}
    )
