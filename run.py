import sys
from app.agentic_rag import LangGraphAgenticRAG

def print_trace(trace):
    print("\n[Decision Trace]:")
    for step in trace:
        print(f"  -> {step}")

def main():
    print("=" * 65)
    print("  LangGraph Agentic HR Copilot (Groq + Pinecone + Tavily Web Search)")
    print("=" * 65)
    print("Initializing LangGraph Agentic Workflow...")
    
    try:
        agent = LangGraphAgenticRAG()
        print("[OK] Agentic Workflow Ready!\n")
    except Exception as e:
        print(f"[ERR] Initialization failed: {e}")
        return

    # If query passed as CLI argument: python run.py "question"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Question: {query}\n")
        resp = agent.ask(query)
        print_trace(resp.decision_trace)
        print("\n" + "=" * 65)
        print(f"SOURCE TYPE: {resp.source_type.upper()}")
        print("=" * 65)
        print("Answer:\n")
        print(resp.answer)
        if resp.citations:
            print("\nCitations:")
            for c in resp.citations[:4]:
                if c.page_number:
                    print(f"  * [Document: {c.source}, Page: {c.page_number}]")
                else:
                    print(f"  * [Web: {c.source}]")
        return

    print("Type your question below (or type 'exit' to quit).")
    print("Try asking about internal policies OR external statutory laws!\n")
    
    while True:
        try:
            query = input("Ask Copilot > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            print("\nExecuting LangGraph Agentic Workflow...")
            resp = agent.ask(query)
            
            print_trace(resp.decision_trace)
            
            print("\n" + "=" * 65)
            print(f"SOURCE: {resp.source_type.upper()} | MODEL: {resp.model_used}")
            print("=" * 65)
            print(f"Answer:\n\n{resp.answer}\n")
            
            if resp.citations:
                print("Verified Sources / Citations:")
                seen = set()
                for c in resp.citations[:4]:
                    key = (c.source, c.page_number)
                    if key not in seen:
                        seen.add(key)
                        if c.page_number:
                            print(f"  * [File: {c.source}, Page: {c.page_number}]")
                        else:
                            print(f"  * [Web Source: {c.source}]")
            print("=" * 65 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")

if __name__ == "__main__":
    main()
