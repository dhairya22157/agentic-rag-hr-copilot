import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from app.schemas import DocumentChunk, ChunkMetadata
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_document_pages
from app.config import settings


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash prefix to establish a deterministic doc_id."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while block := f.read(8192):
            sha256.update(block)
    return sha256.hexdigest()[:12]


def enrich_chunks(
    raw_chunks: List[Dict[str, Any]], 
    file_path: Path
) -> List[DocumentChunk]:
    """Transform raw text chunks into fully validated DocumentChunk objects."""
    doc_id = compute_file_hash(file_path)
    file_size = file_path.stat().st_size
    file_type = file_path.suffix.lstrip(".").lower()
    now_iso = datetime.now(timezone.utc).isoformat()
    
    enriched = []
    for item in raw_chunks:
        content = item["content"]
        page_num = item["page_number"]
        chunk_idx = item["chunk_index"]
        total_pages = item["total_pages"]
        
        # Deterministic unique chunk ID
        chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx:03d}"
        
        metadata = ChunkMetadata(
            source=file_path.name,
            file_type=file_type,
            file_size_bytes=file_size,
            page_number=page_num,
            total_pages=total_pages,
            chunk_index=chunk_idx,
            char_count=len(content),
            word_count=len(content.split()),
            estimated_tokens=max(1, len(content) // 4),
            created_at=now_iso
        )
        
        enriched.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                metadata=metadata
            )
        )
        
    return enriched


def ingest_document(
    file_path: str | Path,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> List[DocumentChunk]:
    """
    End-to-End Ingestion Pipeline:
    File -> Multi-Format Loader -> Text Sanitization -> Recursive Chunking -> Metadata Validation
    
    Returns:
        List[DocumentChunk]: Validated chunks with rich metadata.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Target document not found: {path}")
        
    # 1. Page-aware loading
    pages = load_document(path)
    
    # 2. Cleaning & Chunking
    raw_chunks = chunk_document_pages(
        pages_data=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # 3. Metadata Enrichment
    return enrich_chunks(raw_chunks=raw_chunks, file_path=path)
