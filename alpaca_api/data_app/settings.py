import os
from pydantic import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings that can be loaded from environment variables."""

    # Alpaca API credentials
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_paper: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"

    # Application environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = environment == "development"

    # AWS settings (optional, for future features)
    aws_access_key: str = os.getenv("AWS_ACCESS_KEY", "")
    aws_secret_key: str = os.getenv("AWS_SECRET_KEY", "")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")

    # Data cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching to avoid
    repeatedly parsing environment variables.
    """
    return Settings()