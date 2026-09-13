from typing import List, Optional
from groq import Groq

from app.config import settings
from app.schemas import RAGResponse, Citation
from app.embeddings import HuggingFaceEmbeddingClient
from app.vectorstore import PineconeVectorStore


class GroqLLMClient:
    """
    Groq High-Speed LLM Client for ultra-fast generation and synthesis.
    """

    def __init__(
        self,
        api_key: str = settings.GROQ_API_KEY,
        model_name: str = settings.GROQ_MODEL
    ):
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Please check your .env file.")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> str:
        """Call Groq chat completion endpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()


class StandardRAG:
    """
    Standard Production RAG Pipeline:
    Query -> Embed -> Vector Similarity Search -> Context Augmentation -> LLM Synthesis.
    """

    SYSTEM_PROMPT = """You are the Enterprise HR AI Copilot for ABC Company.
Your goal is to answer employee HR inquiries accurately, professionally, and strictly based on the provided policy documents.

Guidelines:
1. Answer ONLY using the facts present in the Provided Context below.
2. If the context does not contain enough information to answer the question, clearly state: "I cannot find this information in the current HR policies." Do not speculate or invent policies.
3. Include inline citations to the source and page number in the format: [Source: <filename>, Page: <page_number>].
4. Be concise, respectful, and crystal clear.
"""

    def __init__(
        self,
        vector_store: Optional[PineconeVectorStore] = None,
        embedding_client: Optional[HuggingFaceEmbeddingClient] = None,
        llm_client: Optional[GroqLLMClient] = None
    ):
        self.vector_store = vector_store or PineconeVectorStore()
        self.embedding_client = embedding_client or HuggingFaceEmbeddingClient()
        self.llm_client = llm_client or GroqLLMClient()

    def ask(self, question: str, top_k: int = 4) -> RAGResponse:
        """Execute standard RAG question answering."""
        # 1. Embed query
        query_vector = self.embedding_client.embed_query(question)

        # 2. Retrieve top-k chunks from Pinecone
        matches = self.vector_store.similarity_search(query_vector, top_k=top_k)

        # 3. Assemble context
        context_blocks = []
        citations: List[Citation] = []

        for idx, match in enumerate(matches, 1):
            source = match.get("source", "Unknown")
            page_num = match.get("page_number", 1)
            chunk_id = match.get("chunk_id", "")
            content = match.get("content", "")

            context_blocks.append(
                f"--- Context Block {idx} [Source: {source}, Page: {page_num}, ID: {chunk_id}] ---\n{content}"
            )
            citations.append(
                Citation(
                    source=source,
                    page_number=page_num,
                    chunk_id=chunk_id,
                    snippet=content[:200]
                )
            )

        context_str = "\n\n".join(context_blocks)

        prompt = f"""Provided Context:
{context_str}

User Question:
{question}

Provide a direct, policy-compliant answer with citations:"""

        # 4. Synthesize with Groq LLM
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.1
        )

        return RAGResponse(
            question=question,
            rewritten_query=None,
            answer=answer,
            citations=citations,
            retrieved_chunks_count=len(matches),
            relevant_chunks_count=len(matches),
            is_grounded=True,
            model_used=self.llm_client.model_name
        )
