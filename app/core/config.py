"""환경변수와 .env 파일에서 Giftie 실행 설정을 읽습니다."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션, 인증, 모델 및 타임아웃 설정."""
    app_name: str = "giftie-ai-service"
    api_key: str = "local-development-key"
    # mock | bedrock | vllm | mlx | transformers
    # bedrock 은 GPU 없이 쓰는 관리형 경로, vllm 은 자체 GPU 경로다.
    # 둘 다 추천과 이미지 분석에 같은 모델을 쓴다. mlx/transformers 는 Mac 로컬
    # 개발용이며, 이 두 값에서는 이미지 분석이 mock 으로 떨어진다(VLM 을 못 돌리기 때문).
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
    # --------------------------------------------------------------- 공용 Bedrock 엔진
    # 추천과 이미지 분석이 Amazon Bedrock 의 같은 Claude 모델을 쓴다. GPU 가 필요 없고
    # 모델 적재 시간도 없으므로 model_backend="bedrock" 이면 두 기능 모두 이 경로다.
    #
    # 계정마다 열려 있는 호출 경로가 다르다.
    #   invoke: 레거시 bedrock-runtime(InvokeModel). 추론 프로파일 ID 를 쓴다.
    #           예) us.anthropic.claude-haiku-4-5-20251001-v1:0
    #   mantle: Messages 엔드포인트(bedrock-mantle.{region}.api.aws). anthropic. 접두사 ID.
    #           예) anthropic.claude-haiku-4-5
    # 잘못 고르면 모든 모델이 403 이 된다. 403 이면 이 값을 가장 먼저 의심할 것.
    bedrock_api_style: str = "invoke"
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    # max_new_tokens(600) 는 Gemma 기준입니다. Claude 는 스키마를 프롬프트로 받는 만큼
    # 출력이 길어 600 에서는 JSON 이 잘립니다(실측). 그래서 별도 예산을 둡니다.
    bedrock_max_tokens: int = 2_048
    bedrock_max_retries: int = 2
    bedrock_timeout_seconds: float = 90.0
    # 인증은 아래 둘 중 하나만 쓴다. 함께 지정하면 SDK 가 거부한다.
    #   1) Bedrock API 키(Bearer 토큰). SDK 가 쓰는 환경변수 이름으로 .env 에 적어도 인식한다.
    #   2) 미지정 시 표준 AWS credential chain(환경변수 / ~/.aws / IAM 역할)의 SigV4.
    bedrock_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"),
    )
    bedrock_aws_profile: str | None = None
    # 사설 엔드포인트(VPC/PrivateLink)나 로컬 검증 스텁을 쓸 때만 지정한다.
    bedrock_base_url: str | None = None

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

    # ------------------------------------------------------------------ 상품 검색(Tavily)
    # 추천 카테고리와 가격 범위가 정해진 뒤, 실제로 살 수 있는 상품을 찾아 링크를 붙인다.
    # 검색 여부를 모델이 판단하지 않고 파이프라인이 결정론적으로 호출한다.
    # 12B 급 모델의 tool calling 신뢰성에 기대지 않고, 호출 횟수가 고정이라 지연도 예측 가능하다.
    tavily_api_key: str = ""
    tavily_enabled: bool = True
    tavily_url: str = "https://api.tavily.com/search"
    tavily_timeout_seconds: float = 15.0
    tavily_search_depth: str = "basic"  # basic | advanced (advanced 는 크레딧 2배)
    tavily_max_results: int = 8
    # 검색 스니펫의 숫자는 같은 브랜드 다른 옵션의 가격일 수 있어 믿을 수 없다.
    # Extract 로 상품 페이지 본문의 "판매가 N원" 을 읽어 실제 가격을 확정한다.
    # 유효한 URL 4개까지 1~2초면 끝나지만, 접근이 막힌 URL 이 섞이면 재시도로 길어져 제한 시간을 둔다.
    tavily_extract_url: str = "https://api.tavily.com/extract"
    tavily_extract_depth: str = "advanced"  # basic 은 국내 쇼핑몰 상당수를 못 읽는다(실측)
    # 최종 노출은 3개이므로 확정 대상을 그보다 조금만 넉넉히 잡습니다.
    # 대상이 많을수록 접근 안 되는 URL 이 섞일 확률이 올라가고 그만큼 느려집니다.
    tavily_extract_limit: int = 8
    tavily_extract_timeout_seconds: float = 8.0
    # 한 묶음에 몰아 보내면 접근이 막힌 URL 하나가 나머지 결과까지 함께 잃게 만든다(실측).
    tavily_extract_batch_size: int = 3
    # 신뢰할 수 있는 국내 거래 플랫폼만 검색한다. 블로그·카페의 광고성 글을 걸러 내기 위함이다.
    # 주의: Tavily 는 country 파라미터를 include_domains 와 함께 쓰면 결과가 0건이 된다(실측).
    product_search_domains: list[str] = [
        "coupang.com",
        "gift.kakao.com",
        "shopping.naver.com",
        "ssg.com",
        "gmarket.co.kr",
        "11st.co.kr",
        "lotteon.com",
        "kurly.com",
        "oliveyoung.co.kr",
    ]
    # 가격을 확정하기 전에 모아 두는 후보 수. 최종 개수보다 넉넉해야 예산에 맞는 것이 남는다.
    product_candidate_limit: int = 8
    product_suggestion_limit: int = 3

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
