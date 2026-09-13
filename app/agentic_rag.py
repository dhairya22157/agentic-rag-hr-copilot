import json
from typing import List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END

from app.config import settings
from app.schemas import AgentState, RAGResponse, Citation
from app.embeddings import HuggingFaceEmbeddingClient
from app.vectorstore import PineconeVectorStore
from app.rag import GroqLLMClient
from app.tavily_client import TavilySearchClient


class LangGraphAgenticRAG:
    """
    LangGraph Agentic RAG Workflow:
    1. Router Agent (LLM)
    2. Retrieve from Pinecone (KB)
    3. Grade KB Evidence (LLM) -> Yes: Step 4 | No: Step 5
    4. Generate Answer from KB (LLM) -> Step 9
    5. Web Search (Tavily)
    6. Grade Web Evidence (LLM) -> Yes: Step 7 | No: Step 8
    7. Generate Answer from Web (LLM) -> Step 9
    8. Query Rewrite & Retry (max N retries) -> Loop to Step 2 or Step 9
    9. Final Answer with Sources & Transparent Decision Trace
    """

    def __init__(
        self,
        vector_store: Optional[PineconeVectorStore] = None,
        embedding_client: Optional[HuggingFaceEmbeddingClient] = None,
        llm_client: Optional[GroqLLMClient] = None,
        tavily_client: Optional[TavilySearchClient] = None,
        max_retries: int = 2
    ):
        self.vector_store = vector_store or PineconeVectorStore()
        self.embedding_client = embedding_client or HuggingFaceEmbeddingClient()
        self.llm_client = llm_client or GroqLLMClient()
        self.tavily_client = tavily_client or TavilySearchClient()
        self.max_retries = max_retries
        self.graph = self._build_graph()

    # ==========================================================================
    # NODE 1: Router Agent (LLM)
    # ==========================================================================
    def router_agent(self, state: AgentState) -> Dict[str, Any]:
        """Classify user intent: Internal HR Knowledge Base ('kb') vs External Web ('web')."""
        question = state["question"]
        system_prompt = """You are an expert HR Inquiry Router.
Classify the user query into either 'kb' or 'web':
- 'kb': Questions about company internal policies, leave entitlement, employee rules, hours, probation, internal benefits.
- 'web': General external labor law questions, other companies, broad industry benchmarks, or external current news.

Respond ONLY with valid JSON:
{"decision": "kb" or "web", "reason": "brief reason"}
"""
        prompt = f"Question: {question}\nJSON classification:"
        trace = list(state.get("decision_trace", []))
        
        try:
            raw = self.llm_client.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.0, max_tokens=100)
            clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            decision = parsed.get("decision", "kb").lower()
            if decision not in ("kb", "web"):
                decision = "kb"
            reason = parsed.get("reason", "")
        except Exception:
            decision = "kb"
            reason = "Default to internal knowledge base"

        step_log = f"[Step 1: Router Agent] Decided: '{decision.upper()}' (Reason: {reason})"
        trace.append(step_log)

        return {
            "router_decision": decision,
            "decision_trace": trace
        }

    # ==========================================================================
    # NODE 2: Retrieve from Pinecone (KB)
    # ==========================================================================
    def retrieve_kb(self, state: AgentState) -> Dict[str, Any]:
        """Perform semantic search against Pinecone index."""
        question = state["question"]
        trace = list(state.get("decision_trace", []))

        query_vector = self.embedding_client.embed_query(question)
        documents = self.vector_store.similarity_search(query_vector, top_k=4)

        doc_summary = ", ".join([f"{d.get('source')} (P.{d.get('page_number')})" for d in documents]) or "None"
        trace.append(f"[Step 2: Retrieve from KB] Retrieved {len(documents)} chunks from Pinecone: [{doc_summary}]")

        return {
            "documents": documents,
            "decision_trace": trace
        }

    # ==========================================================================
    # NODE 3: Grade KB Evidence (LLM)
    # ==========================================================================
    def grade_kb_evidence(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate if retrieved KB documents are sufficient to answer question."""
        question = state["question"]
        documents = state.get("documents", [])
        trace = list(state.get("decision_trace", []))

        if not documents:
            trace.append("[Step 3: Grade KB Evidence] No documents found. Grade: 'not_enough'")
            return {"kb_evidence_grade": "not_enough", "decision_trace": trace}

        context_preview = "\n\n".join([f"- (Page {d.get('page_number')}): {d.get('content', '')[:300]}" for d in documents])

        system_prompt = """You are a strict HR Knowledge Evaluator.
Determine if the provided context contains sufficient factual information to answer the user's question completely and accurately.
Respond ONLY in JSON:
{"grade": "enough" or "not_enough", "reason": "brief reason"}
"""
        prompt = f"Question: {question}\n\nContext:\n{context_preview}\n\nJSON Evaluation:"

        try:
            raw = self.llm_client.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.0, max_tokens=100)
            clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            grade = parsed.get("grade", "not_enough").lower()
            reason = parsed.get("reason", "")
            if grade not in ("enough", "not_enough"):
                grade = "not_enough"
        except Exception:
            grade = "enough" if len(documents) > 0 else "not_enough"
            reason = "Fallback evaluator"

        trace.append(f"[Step 3: Grade KB Evidence] Grade: '{grade.upper()}' (Reason: {reason})")

        return {
            "kb_evidence_grade": grade,
            "decision_trace": trace
        }

    # ==========================================================================
    # NODE 4: Generate Answer from KB (LLM)
    # ==========================================================================
    def generate_from_kb(self, state: AgentState) -> Dict[str, Any]:
        """Synthesize answer using verified company HR context with citations."""
        question = state["question"]
        documents = state.get("documents", [])
        trace = list(state.get("decision_trace", []))

        context_str = "\n\n".join([
            f"--- Document [Source: {d.get('source')}, Page: {d.get('page_number')}] ---\n{d.get('content')}"
            for d in documents
        ])

        system_prompt = """You are the Enterprise HR AI Copilot for ABC Company.
Answer the user's question using ONLY the provided HR Policy Context.
Include inline citations like [Source: <filename>, Page: <page_number>].
Be concise, clear, and professional.
"""
        prompt = f"HR Policy Context:\n{context_str}\n\nUser Question:\n{question}\n\nAnswer with Citations:"

        answer = self.llm_client.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.1, max_tokens=512)

        citations = []
        for d in documents:
            citations.append(Citation(
                source=d.get("source", "Internal Policy"),
                page_number=d.get("page_number"),
                chunk_id=d.get("chunk_id"),
                snippet=d.get("content", "")[:200]
            ))

        trace.append("[Step 4: Generate from KB] Successfully synthesized answer from company HR policy evidence")

        return {
            "answer": answer,
            "citations": citations,
            "source_type": "kb",
            "decision_trace": trace
        }

    # ==========================================================================
    # NODE 5: Web Search (Tavily)
    # ==========================================================================
    def web_search_tavily(self, state: AgentState) -> Dict[str, Any]:
        """Fallback to external search using Tavily API."""
        question = state["question"]
        trace = list(state.get("decision_trace", []))

        trace.append(f"[Step 5: Web Search (Tavily)] Searching external web for query: '{question}'")
        web_results = self.tavily_client.search(query=question, max_results=4)

        urls_found = ", ".join([w.get("url", "") for w in web_results[:2]]) or "None"
        trace.append(f"[Step 5: Web Search (Tavily)] Found {len(web_results)} external sources: [{urls_found}]")

        return {
            "web_results": web_results,
            "decision_trace": trace
        }

    # ==========================================================================
    # NODE 6: Grade Web Evidence (LLM)
    # ==========================================================================
    def grade_web_evidence(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate if web search results are sufficient to answer question."""
        question = state["question"]
        web_results = state.get("web_results", [])
        trace = list(state.get("decision_trace", []))

        if not web_results:
            trace.append("[Step 6: Grade Web Evidence] No web results found. Grade: 'not_enough'")
            return {"web_evidence_grade": "not_enough", "decision_trace": trace}

        web_preview = "\n\n".join([f"- {w.get('title')}: {w.get('content')[:250]}" for w in web_results])

        system_prompt = """You are a strict Web Evidence Evaluator.
Determine if the provided web search results contain sufficient facts to answer the user inquiry.
Respond ONLY in JSON:
{"grade": "enough" or "not_enough", "reason": "brief reason"}
"""
        prompt = f"Inquiry: {question}\n\nWeb Results:\n{web_preview}\n\nJSON Evaluation:"

        try:
            raw = self.llm_client.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.0, max_tokens=100)
            clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            grade = parsed.get("grade", "not_enough").lower()
            reason = parsed.get("reason", "")
            if grade not in ("enough", "not_enough"):
                grade = "not_enough"
        except Exception:
            grade = "enough" if len(web_results) > 0 else "not_enough"
            reason = "Fallback web evaluator"

        trace.append(f"[Step 6: Grade Web Evidence] Grade: '{grade.upper()}' (Reason: {reason})")

        return {
            "web_evidence_grade": grade,
            "decision_trace": trace
        }

    # ==========================================================================
    # NODE 7: Generate Answer from Web (LLM)
    # ==========================================================================
    def generate_from_web(self, state: AgentState) -> Dict[str, Any]:
        """Synthesize answer using external web search evidence with URL citations."""
        question = state["question"]
        web_results = state.get("web_results", [])
        trace = list(state.get("decision_trace", []))

        context_str = "\n\n".join([
            f"--- Web Source: {w.get('title')} ({w.get('url')}) ---\n{w.get('content')}"
            for w in web_results
        ])

        system_prompt = """You are an HR Information Specialist.
The company's internal knowledge base did not have this policy, so answer using the provided external web sources.
Requirements:
1. Clearly state that this information comes from external sources/general statutory laws, not the company's internal handbook.
2. Include source citations with URLs: [Source: <title> (<url>)].
3. Be professional and accurate.
"""
        prompt = f"External Web Context:\n{context_str}\n\nUser Question:\n{question}\n\nAnswer with Citations:"

        answer = self.llm_client.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.1, max_tokens=512)

        citations = []
        for w in web_results:
            citations.append(Citation(
                source=f"{w.get('title')} ({w.get('url')})",
                page_number=None,
                chunk_id=None,
                snippet=w.get("content", "")[:200]
            ))

        trace.append("[Step 7: Generate from Web] Synthesized answer from external web evidence")

        return {
            "answer": answer,
            "citations": citations,
            "source_type": "web",
            "decision_trace": trace
        }

    # ==========================================================================
    # NODE 8: Query Rewrite & Retry
    # ==========================================================================
    def rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """Rewrite search terms and manage retry loop counter."""
        current_question = state["question"]
        retry_count = state.get("retry_count", 0) + 1
        max_retries = state.get("max_retries", self.max_retries)
        trace = list(state.get("decision_trace", []))

        system_prompt = """You are an HR Search Query Optimizer.
Rewrite the query to improve retrieval. Focus on synonyms, formal HR policy terminology, or alternate keywords.
Return ONLY the reformulated query string.
"""
        prompt = f"Original Query: {current_question}\nRewritten Query:"
        new_query = self.llm_client.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.2, max_tokens=60).strip('"').strip()

        trace.append(f"[Step 8: Query Rewrite & Retry] Query reformulate: '{current_question}' -> '{new_query}' (Attempt {retry_count}/{max_retries})")

        return {
            "question": new_query,
            "retry_count": retry_count,
            "decision_trace": trace
        }

    # ==========================================================================
    # NODE 9: Final Answer with Sources
    # ==========================================================================
    def finalize_answer(self, state: AgentState) -> Dict[str, Any]:
        """Package final answer, citations, and execution trace."""
        trace = list(state.get("decision_trace", []))
        answer = state.get("answer")

        # Fallback if no answer was generated after max retries
        if not answer:
            answer = (
                "I apologize, but I could not find verified company policy or external statutory information "
                "to answer your question after multiple search attempts. Please reach out directly to the HR Department "
                "at hr@abccompany.com."
            )
            state["source_type"] = "fallback"
            trace.append("[Step 9: Final Answer] Max retries exhausted without sufficient evidence; provided HR contact fallback.")
        else:
            trace.append("[Step 9: Final Answer] Final response ready with verified source citations.")

        return {
            "answer": answer,
            "decision_trace": trace
        }

    # ==========================================================================
    # CONDITIONAL EDGES
    # ==========================================================================
    def route_after_router(self, state: AgentState) -> Literal["retrieve_kb", "web_search_tavily"]:
        """Edge 1: Router decision."""
        if state.get("router_decision") == "web":
            return "web_search_tavily"
        return "retrieve_kb"

    def route_after_kb_grade(self, state: AgentState) -> Literal["generate_from_kb", "web_search_tavily"]:
        """Edge 2: KB evidence grade."""
        if state.get("kb_evidence_grade") == "enough":
            return "generate_from_kb"
        return "web_search_tavily"

    def route_after_web_grade(self, state: AgentState) -> Literal["generate_from_web", "rewrite_query"]:
        """Edge 3: Web evidence grade."""
        if state.get("web_evidence_grade") == "enough":
            return "generate_from_web"
        return "rewrite_query"

    def route_after_rewrite(self, state: AgentState) -> Literal["retrieve_kb", "finalize_answer"]:
        """Edge 4: Retry loop check."""
        if state.get("retry_count", 0) <= state.get("max_retries", self.max_retries):
            return "retrieve_kb"
        return "finalize_answer"

    # ==========================================================================
    # COMPOSE STATE GRAPH
    # ==========================================================================
    def _build_graph(self):
        builder = StateGraph(AgentState)

        # Add Nodes
        builder.add_node("router_agent", self.router_agent)
        builder.add_node("retrieve_kb", self.retrieve_kb)
        builder.add_node("grade_kb_evidence", self.grade_kb_evidence)
        builder.add_node("generate_from_kb", self.generate_from_kb)
        builder.add_node("web_search_tavily", self.web_search_tavily)
        builder.add_node("grade_web_evidence", self.grade_web_evidence)
        builder.add_node("generate_from_web", self.generate_from_web)
        builder.add_node("rewrite_query", self.rewrite_query)
        builder.add_node("finalize_answer", self.finalize_answer)

        # Entry point
        builder.set_entry_point("router_agent")

        # Add Edges
        builder.add_conditional_edges(
            "router_agent",
            self.route_after_router,
            {
                "retrieve_kb": "retrieve_kb",
                "web_search_tavily": "web_search_tavily"
            }
        )

        builder.add_edge("retrieve_kb", "grade_kb_evidence")

        builder.add_conditional_edges(
            "grade_kb_evidence",
            self.route_after_kb_grade,
            {
                "generate_from_kb": "generate_from_kb",
                "web_search_tavily": "web_search_tavily"
            }
        )

        builder.add_edge("generate_from_kb", "finalize_answer")

        builder.add_edge("web_search_tavily", "grade_web_evidence")

        builder.add_conditional_edges(
            "grade_web_evidence",
            self.route_after_web_grade,
            {
                "generate_from_web": "generate_from_web",
                "rewrite_query": "rewrite_query"
            }
        )

        builder.add_edge("generate_from_web", "finalize_answer")

        builder.add_conditional_edges(
            "rewrite_query",
            self.route_after_rewrite,
            {
                "retrieve_kb": "retrieve_kb",
                "finalize_answer": "finalize_answer"
            }
        )

        builder.add_edge("finalize_answer", END)

        return builder.compile()

    # ==========================================================================
    # PUBLIC EXECUTION API
    # ==========================================================================
    def ask(self, question: str) -> RAGResponse:
        """Run complete LangGraph Agentic RAG workflow."""
        initial_state: AgentState = {
            "question": question,
            "original_question": question,
            "router_decision": "",
            "documents": [],
            "web_results": [],
            "kb_evidence_grade": "",
            "web_evidence_grade": "",
            "retry_count": 0,
            "max_retries": self.max_retries,
            "answer": "",
            "citations": [],
            "source_type": "kb",
            "decision_trace": [f"[Workflow Initiated] Inquiry: '{question}'"]
        }

        final_state = self.graph.invoke(initial_state)

        return RAGResponse(
            question=final_state["original_question"],
            rewritten_query=final_state["question"] if final_state["question"] != final_state["original_question"] else None,
            answer=final_state["answer"],
            citations=final_state.get("citations", []),
            retrieved_chunks_count=len(final_state.get("documents", [])) + len(final_state.get("web_results", [])),
            relevant_chunks_count=len(final_state.get("citations", [])),
            is_grounded=True,
            model_used=self.llm_client.model_name,
            source_type=final_state.get("source_type", "kb"),
            decision_trace=final_state.get("decision_trace", [])
        )
