from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Reads values from environment variables (and .env locally).
    Field name DATABASE_URL <-> env var DATABASE_URL (case-insensitive)."""

    database_url: str
    redis_url: str
    rabbitmq_url: str
    upload_dir: str = "/app/uploads"
    openai_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# One shared instance, imported everywhere: `from app.config import settings`
settings = Settings()
