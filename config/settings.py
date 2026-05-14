"""
ATLAS Configuration Management
================================
Loads all settings from environment variables / config/keys.env.
Uses pydantic-settings for type-safe config.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the config/keys.env path relative to this file
_ENV_FILE = Path(__file__).resolve().parent / "keys.env"


class ATLASSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Environment ──────────────────────────────────────────
    environment: str = Field("development", alias="ENVIRONMENT")
    log_level:   str = Field("INFO",        alias="LOG_LEVEL")

    # ── LLM ──────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")

    # ── Market Data ──────────────────────────────────────────
    polygon_api_key: str = Field(..., alias="POLYGON_API_KEY")

    # ── Exchanges ────────────────────────────────────────────
    binance_api_key:    str = Field(..., alias="BINANCE_API_KEY")
    binance_secret_key: str = Field(..., alias="BINANCE_SECRET_KEY")

    alpaca_api_key:    str = Field(..., alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., alias="ALPACA_SECRET_KEY")
    alpaca_base_url:   str = Field("https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL")

    # ── Database ─────────────────────────────────────────────
    timescaledb_host:     str       = Field("localhost",  alias="TIMESCALEDB_HOST")
    timescaledb_port:     int       = Field(5432,         alias="TIMESCALEDB_PORT")
    timescaledb_db:       str       = Field("atlas",      alias="TIMESCALEDB_DB")
    timescaledb_user:     str       = Field("atlas_user", alias="TIMESCALEDB_USER")
    timescaledb_password: SecretStr = Field(...,          alias="TIMESCALEDB_PASSWORD")

    # ── Redis ────────────────────────────────────────────────
    redis_host:     str       = Field("localhost", alias="REDIS_HOST")
    redis_port:     int       = Field(6379,        alias="REDIS_PORT")
    redis_password: SecretStr = Field(...,         alias="REDIS_PASSWORD")

    # ── Computed helpers ──────────────────────────────────────
    @property
    def db_dsn(self) -> str:
        """asyncpg / psycopg2 connection string."""
        pw = self.timescaledb_password.get_secret_value()
        return (
            f"postgresql://{self.timescaledb_user}:{pw}"
            f"@{self.timescaledb_host}:{self.timescaledb_port}/{self.timescaledb_db}"
        )

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> ATLASSettings:
    """Return a cached singleton of ATLASSettings."""
    return ATLASSettings()


# Convenience alias
settings = get_settings()
