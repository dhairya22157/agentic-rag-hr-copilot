from typing import List, Dict, Any, Optional
from pinecone import Pinecone

from app.config import settings
from app.schemas import DocumentChunk


class PineconeVectorStore:
    """
    Pinecone Vector Store wrapper for indexing and similarity search.
    Stores document chunk content and provenance directly in vector metadata.
    """

    def __init__(
        self,
        api_key: str = settings.PINECONE_API_KEY,
        index_name: str = settings.index_name,
        namespace: str = settings.PINECONE_NAMESPACE
    ):
        if not api_key:
            raise ValueError("PINECONE_API_KEY is missing. Please check your .env file.")
            
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace or None
        self.index = self.pc.Index(self.index_name)

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 50
    ) -> int:
        """
        Upsert document chunks and vectors to Pinecone with chunk text in metadata.
        
        Returns:
            int: Total vectors upserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        records = []
        for chunk, vector in zip(chunks, embeddings):
            records.append({
                "id": chunk.chunk_id,
                "values": vector,
                "metadata": {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.metadata.source,
                    "file_type": chunk.metadata.file_type,
                    "page_number": chunk.metadata.page_number,
                    "total_pages": chunk.metadata.total_pages,
                    "chunk_index": chunk.metadata.chunk_index,
                    "char_count": chunk.metadata.char_count,
                    "word_count": chunk.metadata.word_count,
                    "estimated_tokens": chunk.metadata.estimated_tokens,
                    "content": chunk.content  # Stored directly for instant retrieval
                }
            })

        total_upserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            total_upserted += len(batch)

        return total_upserted

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone for top_k most similar chunks.
        
        Returns:
            List[Dict[str, Any]]: Matches with id, score, and metadata.
        """
        query_params: Dict[str, Any] = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True
        }
        if self.namespace:
            query_params["namespace"] = self.namespace
        if filter_dict:
            query_params["filter"] = filter_dict

        results = self.index.query(**query_params)
        
        matches = []
        for match in results.get("matches", []):
            matches.append({
                "chunk_id": match.get("id"),
                "score": float(match.get("score", 0.0)),
                "content": match.get("metadata", {}).get("content", ""),
                "source": match.get("metadata", {}).get("source", ""),
                "page_number": int(match.get("metadata", {}).get("page_number", 1)),
                "metadata": match.get("metadata", {})
            })
            
        return matches

    def get_stats(self) -> Dict[str, Any]:
        """Fetch index statistics."""
        return self.index.describe_index_stats()
