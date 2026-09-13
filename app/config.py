import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Groq LLM Configuration
    GROQ_API_KEY: str = Field(default="", description="Groq API key for LLM inference")
    GROQ_MODEL: str = Field(default="qwen/qwen3.8-27b", description="Groq model name")

    # Hugging Face Embeddings Configuration
    HUGGINGFACEHUB_API_TOKEN: str = Field(default="", description="Hugging Face API token")
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Hugging Face model for embeddings (1024-dim matches Pinecone index)"
    )
    EMBEDDING_DIMENSION: int = Field(default=1024, description="Embedding vector dimension")

    # Pinecone Vector Store Configuration
    PINECONE_API_KEY: str = Field(default="", description="Pinecone API key")
    index_name: str = Field(default="ragbot", description="Pinecone index name")
    PINECONE_NAMESPACE: str = Field(default="", description="Optional Pinecone namespace")

    # Tavily Web Search (fallback)
    TAVILY_API_KEY: str = Field(default="", description="Tavily API key")

    # Chunking defaults
    CHUNK_SIZE: int = Field(default=700, description="Default character chunk size")
    CHUNK_OVERLAP: int = Field(default=120, description="Default character chunk overlap")


settings = Settings()
