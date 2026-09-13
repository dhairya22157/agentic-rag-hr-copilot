import json
import time
import urllib.request
import urllib.error
from typing import List, Union

from app.config import settings


class HuggingFaceEmbeddingClient:
    """
    Embedding Client using Hugging Face Inference Router API.
    Generates dense embeddings compatible with Pinecone (1024 dimensions with BAAI/bge-large-en-v1.5).
    """

    def __init__(
        self,
        api_token: str = settings.HUGGINGFACEHUB_API_TOKEN,
        model_name: str = settings.EMBEDDING_MODEL,
        dimension: int = settings.EMBEDDING_DIMENSION
    ):
        if not api_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is missing. Please check your .env file.")
            
        self.api_token = api_token
        self.model_name = model_name
        self.dimension = dimension
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"

    def _call_api(self, texts: List[str], retries: int = 3) -> List[List[float]]:
        """Call the HF Router API with retry logic for model loading / rate limits."""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "User-Agent": "rag-copilot/1.0"
        }
        payload = json.dumps({"inputs": texts}).encode("utf-8")

        for attempt in range(retries):
            req = urllib.request.Request(self.api_url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    
                    # Some endpoints wrap single input or batch differently
                    if isinstance(data, list):
                        # Ensure list of lists of floats
                        if len(data) > 0 and isinstance(data[0], list):
                            return data
                        elif len(data) > 0 and isinstance(data[0], (int, float)):
                            return [data]
                    raise ValueError(f"Unexpected embedding response format: {type(data)}")

            except urllib.error.HTTPError as e:
                err_content = e.read().decode("utf-8", errors="replace")
                
                # Check for 503 (model warming up / loading)
                if e.code == 503:
                    time.sleep(4)
                    continue
                elif e.code == 429:
                    time.sleep(5)
                    continue
                else:
                    raise RuntimeError(f"Hugging Face API error ({e.code}): {err_content}")
            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"Failed to generate embeddings: {e}")
                time.sleep(2)

        raise TimeoutError("Hugging Face inference service timed out after maximum retries.")

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single query string."""
        if not text.strip():
            raise ValueError("Query text cannot be empty.")
        embeddings = self._call_api([text])
        return embeddings[0]

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Generate embedding vectors for a list of document strings with batching.
        """
        if not texts:
            return []
            
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._call_api(batch)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
