from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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

    # LLM Provider Keys
    # Field has default value of "" because linter recognized return Settings()
    # in get_settings as required on build time
    ANTHROPIC_API_KEY: str = Field(alias="ANTHROPIC_API_KEY", default="")
    OPENAI_API_KEY: str = Field(alias="OPENAI_API_KEY", default="")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
