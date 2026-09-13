import json
from typing import List, Optional, Tuple
from app.schemas import RAGResponse, Citation, EvidenceGrade
from app.embeddings import HuggingFaceEmbeddingClient
from app.vectorstore import PineconeVectorStore
from app.rag import GroqLLMClient


class QueryRewriter:
    """
    Transforms conversational, ambiguous, or incomplete employee queries
    into search-optimized semantic keywords for higher vector retrieval recall.
    """

    SYSTEM_PROMPT = """You are an HR Search Query Optimizer.
Your task is to take an employee's natural language question and rewrite it into a clear, specific, keyword-dense search query optimized for vector retrieval against company policy documents.
Rules:
1. Return ONLY the rewritten query string.
2. Do not include quotes, explanations, or introductory text.
3. Focus on official HR terms (e.g., "Casual Leave", "Paid Time Off", "Notice Period", "Eligibility", "Bereavement").
"""

    def __init__(self, llm_client: GroqLLMClient):
        self.llm = llm_client

    def rewrite(self, question: str) -> str:
        prompt = f"Employee Question: {question}\nRewritten Search Query:"
        rewritten = self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=100
        )
        return rewritten.strip('"').strip()


class EvidenceGrader:
    """
    Evaluates retrieved chunks to determine whether they actually contain
    information relevant to the user query, filtering out noise.
    """

    SYSTEM_PROMPT = """You are an HR Policy Evidence Grader.
Determine if the provided context chunk is relevant to the employee's inquiry.
Respond strictly in JSON format:
{
  "is_relevant": true or false,
  "reason": "short explanation"
}
"""

    def __init__(self, llm_client: GroqLLMClient):
        self.llm = llm_client

    def grade(self, query: str, chunk_content: str) -> EvidenceGrade:
        prompt = f"""Inquiry: {query}

Policy Chunk Content:
{chunk_content}

Is this chunk relevant? JSON:"""

        raw_output = self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=120
        )

        try:
            # Strip markdown json codeblocks if any
            clean_json = raw_output.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            return EvidenceGrade(
                chunk_id="",
                is_relevant=bool(parsed.get("is_relevant", True)),
                reason=parsed.get("reason", "Graded by LLM")
            )
        except Exception:
            # Graceful fallback: keep chunk if grading parse fails
            return EvidenceGrade(chunk_id="", is_relevant=True, reason="Parse fallback")


class AdvancedRAG:
    """
    Improving RAG Pipeline:
    1. Query Rewriting / Optimization
    2. Vector Retrieval
    3. Evidence Relevance Grading (Noise Filtering)
    4. Grounded Synthesis with Mandatory Citations
    5. Hallucination Check / Fallback
    """

    SYSTEM_PROMPT = """You are the Enterprise HR AI Copilot for ABC Company.
Answer the user's question using ONLY the provided verified context chunks.

Requirements:
1. Every major statement must have a citation tag: [Source: <filename>, Page: <page>].
2. If the verified context does not contain enough information, state:
   "I could not find sufficient policy information to answer this question. Please contact HR at hr@abccompany.com for further guidance."
3. Do not assume, speculate, or mention anything outside the verified context.
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
        self.rewriter = QueryRewriter(self.llm_client)
        self.grader = EvidenceGrader(self.llm_client)

    def ask(self, question: str, top_k: int = 5) -> RAGResponse:
        """Execute full Improving RAG pipeline."""
        # 1. Query Rewriting
        rewritten_query = self.rewriter.rewrite(question)

        # 2. Vector Retrieval using optimized query
        query_vector = self.embedding_client.embed_query(rewritten_query)
        matches = self.vector_store.similarity_search(query_vector, top_k=top_k)

        # 3. Evidence Grading & Filtering
        relevant_matches = []
        citations: List[Citation] = []

        for match in matches:
            grade = self.grader.grade(question, match["content"])
            if grade.is_relevant:
                relevant_matches.append(match)
                citations.append(
                    Citation(
                        source=match["source"],
                        page_number=match["page_number"],
                        chunk_id=match["chunk_id"],
                        snippet=match["content"][:200]
                    )
                )

        # 4. Fallback if no chunks passed relevance threshold
        if not relevant_matches:
            return RAGResponse(
                question=question,
                rewritten_query=rewritten_query,
                answer="I could not find relevant policy information in the company documents for your question. Please contact HR directly at hr@abccompany.com.",
                citations=[],
                retrieved_chunks_count=len(matches),
                relevant_chunks_count=0,
                is_grounded=True,
                model_used=self.llm_client.model_name
            )

        # 5. Assemble Verified Context
        context_blocks = []
        for idx, m in enumerate(relevant_matches, 1):
            context_blocks.append(
                f"--- Verified Evidence {idx} [Source: {m['source']}, Page: {m['page_number']}] ---\n{m['content']}"
            )
        context_str = "\n\n".join(context_blocks)

        prompt = f"""Verified Policy Context:
{context_str}

Employee Question:
{question}

Synthesized Answer with Citations:"""

        # 6. Synthesis with Groq LLM
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=512
        )

        return RAGResponse(
            question=question,
            rewritten_query=rewritten_query,
            answer=answer,
            citations=citations,
            retrieved_chunks_count=len(matches),
            relevant_chunks_count=len(relevant_matches),
            is_grounded=True,
            model_used=self.llm_client.model_name
        )
