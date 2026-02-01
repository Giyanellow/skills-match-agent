"""
Application configuration management using Pydantic Settings.

This module provides environment-based configuration for the resume analysis application,
including API keys for LLM providers and environment settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    API keys and configuration are loaded with the following priority:
    1. Environment variables
    2. .env file in project root
    3. Default values (where applicable)

    Attributes:
        app_name: Application name for logging/identification
        environment: Deployment environment (development/staging/production)
        debug: Enable debug mode for verbose logging
        ANTHROPIC_API_KEY: API key for Anthropic/Claude models
        OPENAI_API_KEY: API key for OpenAI/GPT models
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    app_name: str = "Resume Analyzer Agent"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # LLM Provider API Keys
    # Default empty string prevents build-time errors when Settings() is instantiated
    ANTHROPIC_API_KEY: str = Field(alias="ANTHROPIC_API_KEY", default="")
    OPENAI_API_KEY: str = Field(alias="OPENAI_API_KEY", default="")

    @property
    def is_production(self) -> bool:
        """
        Check if running in production environment.

        Returns:
            True if environment is "production"
        """
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings instance.

    Uses LRU cache to ensure settings are loaded only once per application lifetime.
    This pattern is recommended by Pydantic Settings for performance.

    Returns:
        Singleton Settings instance

    Example:
        ```python
        from src.config.config import get_settings

        settings = get_settings()
        if settings.ANTHROPIC_API_KEY:
            # Use Anthropic API
            pass
        ```
    """
    return Settings()
