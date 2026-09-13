import sys
from app.advanced_rag import AdvancedRAG

def main():
    print("=" * 60)
    print("  Enterprise HR AI Copilot (Groq LLM + Pinecone + Hugging Face)")
    print("=" * 60)
    print("Initializing Advanced RAG Pipeline (connecting to Pinecone & Groq)...")
    
    try:
        rag = AdvancedRAG()
        print("[OK] System Ready!\n")
    except Exception as e:
        print(f"[ERR] Initialization failed: {e}")
        return

    # If query passed as CLI argument: python run.py "question"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Question: {query}\n")
        resp = rag.ask(query)
        print("Answer:")
        print(resp.answer)
        if resp.citations:
            print("\nCitations:")
            for c in resp.citations:
                print(f"  * [Source: {c.source}, Page: {c.page_number}]")
        return

    print("Type your HR question below (or type 'exit' to quit).\n")
    while True:
        try:
            query = input("Ask HR Question > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            print("\nSearching policy documents & synthesizing...")
            resp = rag.ask(query)
            
            print("\n" + "-" * 50)
            print(f"Rewritten Query : {resp.rewritten_query}")
            print(f"Evidence Used   : {resp.relevant_chunks_count} relevant chunks out of {resp.retrieved_chunks_count}")
            print("-" * 50)
            print(f"\nAnswer:\n{resp.answer}\n")
            
            if resp.citations:
                print("Verified Citations:")
                seen = set()
                for c in resp.citations:
                    citation_key = (c.source, c.page_number)
                    if citation_key not in seen:
                        seen.add(citation_key)
                        print(f"  * [Source: {c.source}, Page: {c.page_number}]")
            print("=" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")

if __name__ == "__main__":
    main()
