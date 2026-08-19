"""환경변수와 .env 파일에서 Giftie 실행 설정을 읽습니다."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션, 인증, 모델 및 타임아웃 설정."""
    app_name: str = "giftie-ai-service"
    api_key: str = "local-development-key"
    model_backend: str = "mock"
    model_id: str = "Qwen/Qwen3-4B"
    local_model_id: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    preload_model: bool = False
    max_new_tokens: int = 600
    temperature: float = 0.2
    request_timeout_seconds: int = 45
    # auto는 TAVILY_API_KEY가 있으면 Tavily를, 없으면 안전한 fallback을 사용합니다.
    product_search_provider: str = "auto"
    tavily_api_key: str | None = None
    product_search_timeout_seconds: float = 8.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """프로세스 동안 재사용할 설정 객체를 최초 한 번만 생성합니다."""
    return Settings()


# 다른 모듈은 이 singleton을 가져와 동일한 설정을 사용합니다.
settings = get_settings()
