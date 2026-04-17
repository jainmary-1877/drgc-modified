"""
Configuration management for the Text-to-SQL Agent.
Loads environment variables and provides typed configuration access.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model_reasoning: str = Field(default="qwen2.5:14b", env="OLLAMA_MODEL_REASONING")
    ollama_model_fast: str = Field(default="qwen2.5:14b", env="OLLAMA_MODEL_FAST")
    ollama_model_sql: str = Field(default="sqlcoder:15b", env="OLLAMA_MODEL_SQL")
    ollama_temperature: float = Field(default=0.0, env="OLLAMA_TEMPERATURE")

    # Database Configuration
    database_uri: str = Field(..., env="DATABASE_URI")

    # Vector Store Configuration (ChromaDB)
    vector_store_path: str = Field(default="./data/vector_store", env="VECTOR_STORE_PATH")
    chroma_collection_name: str = Field(default="sql_examples", env="CHROMA_COLLECTION_NAME")

 
    embedding_model: str = Field(env="EMBEDDING_MODEL")

    # Caching Configuration
    enable_semantic_cache: bool = Field(default=True, env="ENABLE_SEMANTIC_CACHE")
    cache_similarity_threshold: float = Field(default=0.95, env="CACHE_SIMILARITY_THRESHOLD")

    # Agent Configuration
    max_iterations: int = Field(default=3, env="MAX_ITERATIONS")
    enable_self_correction: bool = Field(default=True, env="ENABLE_SELF_CORRECTION")
    enable_dynamic_few_shot: bool = Field(default=True, env="ENABLE_DYNAMIC_FEW_SHOT")
    few_shot_examples_count: int = Field(default=3, env="FEW_SHOT_EXAMPLES_COUNT")
    query_timeout_seconds: int = Field(default=30, env="QUERY_TIMEOUT_SECONDS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()