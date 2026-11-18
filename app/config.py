from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Reads values from environment variables (and .env locally).
    Field name DATABASE_URL <-> env var DATABASE_URL (case-insensitive)."""

    database_url: str
    redis_url: str
    upload_dir: str = "/app/uploads"
    openai_api_key: str = ""

    # JWT auth
    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 7 days

    # Optional: tracing for the LangChain/LangGraph pipelines. Empty keys = tracing disabled.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# One shared instance, imported everywhere: `from app.config import settings`
settings = Settings()
