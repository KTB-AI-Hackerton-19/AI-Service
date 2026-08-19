"""환경변수와 .env 파일에서 Giftie 실행 설정을 읽습니다."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션, 인증, 모델 및 타임아웃 설정."""
    app_name: str = "giftie-ai-service"
    api_key: str = "local-development-key"
    # mock | vllm | mlx | transformers
    # vllm 이 서버 기본 경로다. mlx/transformers 는 Mac 로컬 개발용이며,
    # 이 두 값에서는 이미지 분석이 mock 으로 떨어진다(VLM 을 못 돌리기 때문).
    model_backend: str = "mock"
    model_id: str = "Qwen/Qwen3-4B"
    local_model_id: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    preload_model: bool = False
    max_new_tokens: int = 600
    # 추천·메시지 생성용 샘플링. Gemma 공식 권장값이며 문장 다양성이 품질인 영역이다.
    # (이미지 추출은 아래 vision_temperature 로 따로 둔다)
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    request_timeout_seconds: int = 45

    # ------------------------------------------------------------------ 공용 vLLM 엔진
    # 추천과 이미지 분석이 같은 모델(Gemma4-12B-QAT + MTP)을 같은 vLLM 서버에서 쓴다.
    # GPU 한 장에 모델을 두 벌 올리지 않으므로 메모리와 기동 시간이 모두 절약된다.
    # model_backend 를 "vllm" 으로 두면 두 기능 모두 이 엔진을 사용한다.
    # FastAPI 가 8000 을 쓰므로 vLLM 은 8001 로 띄운다(-p 8001:8000).
    vllm_base_url: str = "http://localhost:8001"
    vllm_model: str = "gemma4-12b-qat"
    vllm_api_key: str = "EMPTY"
    vllm_timeout_seconds: float = 90.0

    # 이미지 분석 전용 생성 파라미터. 엔진은 같지만 추출은 창의성이 필요 없어 temperature 0 이다.
    vision_max_new_tokens: int = 900
    vision_temperature: float = 0.0

    # 이미지 다운로드·정규화
    image_max_bytes: int = 12 * 1024 * 1024
    image_max_edge: int = 1280  # 장변 리사이즈. 벤치 이미지(720x1280)에는 무변환
    image_fetch_timeout_seconds: float = 15.0
    # 기본값은 사설·루프백 주소 차단(SSRF 방어)이다.
    # 로컬에서 이미지를 직접 띄워 종단 테스트할 때만 켜고, 운영에서는 절대 켜지 않는다.
    allow_private_image_hosts: bool = False

    # GiftData.gift_price 는 필수이고 0을 못 받는다. 이미지에서 금액을 못 읽었을 때
    # True 면 502 로 실패시키고, False 면 카테고리별 추정가로 채우고 이름에 "(금액 미상)"을 붙인다.
    strict_price: bool = False

    # ------------------------------------------------------------------ 캘린더(MCP)
    calendar_mcp_url: str = "http://localhost:8300/mcp"
    calendar_mcp_timeout_seconds: float = 30.0
    # 데모용 단일 계정 토큰. 비어 있으면 실제 등록 없이 초안 JSON 만 만든다.
    google_access_token: str = ""
    # 기본값 false: 캘린더 등록은 사용자가 확인 화면에서 승인한 뒤 /confirm 에서만 일어난다.
    # 승인 UI 가 없는 개발 단계에서 흐름을 확인할 때만 true 로 둔다.
    calendar_auto_register: bool = False
    google_calendar_id: str = "primary"
    calendar_default_lead_days: int = 30  # target_date 가 없을 때 답례일까지의 기본 간격
    notification_lead_days: int = 7  # 답례일 며칠 전에 알릴지
    default_timezone: str = "Asia/Seoul"

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
