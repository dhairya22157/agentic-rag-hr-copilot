from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to an individual document chunk."""
    source: str = Field(..., description="File name of the source document")
    file_type: str = Field(..., description="Extension type: pdf, docx, txt")
    file_size_bytes: int = Field(..., description="Size of original file in bytes")
    page_number: int = Field(..., description="Page number of the chunk (1-indexed)")
    total_pages: int = Field(..., description="Total pages in the source document")
    chunk_index: int = Field(..., description="Global chunk index in document")
    char_count: int = Field(..., description="Character count of content")
    word_count: int = Field(..., description="Word count of content")
    estimated_tokens: int = Field(..., description="Approximate token count (char // 4)")
    created_at: str = Field(..., description="ISO-8601 ingestion timestamp")


class DocumentChunk(BaseModel):
    """A clean, validated text chunk ready for vector embedding and retrieval."""
    chunk_id: str = Field(..., description="Deterministic unique identifier for vector indexing")
    doc_id: str = Field(..., description="Deterministic hash ID for the parent document")
    content: str = Field(..., description="Cleaned text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Structured metadata dictionary")


class Citation(BaseModel):
    """Source reference for evidence grounding in generated answers."""
    source: str = Field(..., description="Filename or web URL containing the cited information")
    page_number: Optional[int] = Field(default=None, description="Page number where the information appears (for documents)")
    chunk_id: Optional[str] = Field(default=None, description="Chunk ID for provenance tracking")
    snippet: str = Field(..., description="Short snippet supporting the answer")


class RAGQuery(BaseModel):
    """Input query to the RAG pipeline."""
    question: str = Field(..., description="User's natural language question")
    top_k: int = Field(default=4, description="Number of context chunks to retrieve")
    filter_source: Optional[str] = Field(default=None, description="Optional filter by source filename")


class EvidenceGrade(BaseModel):
    """Evaluation result for an individual retrieved chunk."""
    chunk_id: str
    is_relevant: bool
    reason: str


class RAGResponse(BaseModel):
    """Standardized response from RAG, Advanced RAG, and LangGraph Agentic pipelines."""
    question: str = Field(..., description="Original user question")
    rewritten_query: Optional[str] = Field(default=None, description="Search-optimized reformulated query")
    answer: str = Field(..., description="Synthesized, grounded answer")
    citations: List[Citation] = Field(default_factory=list, description="Direct source citations")
    retrieved_chunks_count: int = Field(default=0, description="Total chunks fetched from vector store or web")
    relevant_chunks_count: int = Field(default=0, description="Chunks retained after relevance grading")
    is_grounded: bool = Field(default=True, description="Whether answer is strictly supported by evidence")
    model_used: str = Field(default="", description="LLM used for synthesis")
    source_type: str = Field(default="kb", description="Origin of evidence: 'kb', 'web', or 'fallback'")
    decision_trace: List[str] = Field(default_factory=list, description="Audit trail of agent routing & grading steps")


class AgentState(TypedDict):
    """State graph object flowing through the LangGraph Agentic RAG workflow."""
    question: str
    original_question: str
    router_decision: str
    documents: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    kb_evidence_grade: str
    web_evidence_grade: str
    retry_count: int
    max_retries: int
    answer: str
    citations: List[Citation]
    source_type: str
    decision_trace: List[str]
