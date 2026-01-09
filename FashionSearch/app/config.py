"""
Configuration settings for FashionSearch application.
Load settings from environment variables with defaults.
"""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Milvus Configuration
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "fashion_search"
    
    # Model Configuration
    model_name: str = "google/siglip2-so400m-patch16-naflex"
    embedding_dim: int = 1152
    
    # Search Configuration
    default_top_k: int = 10
    
    # Rocchio Algorithm Parameters
    rocchio_alpha: float = 1.0
    rocchio_beta: float = 0.75
    rocchio_gamma: float = 0.25
    
    # Data Path
    data_dir: str = "./data"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Create a global settings instance for easy access
settings = get_settings()
