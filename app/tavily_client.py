from typing import List, Dict, Any
from tavily import TavilyClient
from app.config import settings


class TavilySearchClient:
    """
    Tavily Web Search Client for real-time external knowledge fallback.
    """

    def __init__(self, api_key: str = settings.TAVILY_API_KEY):
        if not api_key:
            raise ValueError("TAVILY_API_KEY is missing. Please check your .env file.")
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """
        Execute web search using Tavily.
        
        Returns:
            List[Dict[str, Any]]: List of results with title, url, content, and score.
        """
        try:
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False
            )
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", "Web Source"),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": float(item.get("score", 0.0))
                })
            return results
        except Exception as e:
            print(f"[Tavily Search Warning] Error searching for '{query}': {e}")
            return []
