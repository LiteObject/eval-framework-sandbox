"""Application configuration helpers for the evaluation sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    """Runtime configuration for the sandbox."""

    documents_path: Path = Path(
        os.getenv("DOCUMENTS_PATH", "data/documents/sample_docs")
    )
    embeddings_cache_path: Path = Path(
        os.getenv("EMBEDDINGS_CACHE_PATH", "results/embeddings.pkl")
    )
    top_k: int = int(os.getenv("TOP_K", "3"))
    use_gpu: bool = os.getenv("USE_GPU", "false").lower() == "true"

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    azure_openai_endpoint: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str | None = os.getenv("AZURE_OPENAI_API_KEY")
    langchain_api_key: str | None = os.getenv("LANGCHAIN_API_KEY")
    deepeval_api_key: str | None = os.getenv("DEEPEVAL_API_KEY")
    langchain_openai_model: str = os.getenv("LANGCHAIN_OPENAI_MODEL", "gpt-3.5-turbo")
    langchain_use_ollama: bool = (
        os.getenv("LANGCHAIN_USE_OLLAMA", "false").lower() == "true"
    )
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_base_url: str | None = os.getenv("OLLAMA_BASE_URL")


settings = Settings()
