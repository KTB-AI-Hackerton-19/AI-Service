"""Amazon Bedrock 위의 Claude 클라이언트 생성과 오류 해석을 한 곳에 모읍니다.

추천과 이미지 분석이 같은 계정·같은 모델을 쓰므로, vLLM 설정을 두 기능이
공유하는 것과 같은 방식으로 여기서 클라이언트를 만들어 나눠 씁니다.

Bedrock 은 계정마다 열려 있는 호출 경로가 다릅니다. ``BEDROCK_API_STYLE`` 로
고르며, 잘못 고르면 모든 모델이 403 으로 막히므로 오류 메시지에 서버가 준
원문을 그대로 실어 원인을 바로 알 수 있게 합니다.
"""

import json
import logging
from threading import Lock
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

API_STYLES = frozenset({"invoke", "mantle"})

_sync_client: Any = None
_async_client: Any = None
_lock = Lock()


class BedrockClientError(RuntimeError):
    """Bedrock 클라이언트를 만들 수 없을 때 발생합니다."""


def _client_kwargs() -> dict[str, Any]:
    """인증 방식과 접속 설정을 조립합니다.

    ``BEDROCK_API_KEY`` 가 있으면 Bearer 토큰으로, 없으면 표준 AWS credential
    chain 의 SigV4 로 인증합니다. SDK 가 둘을 동시에 받지 않으므로 함께 지정하면
    먼저 막습니다.

    Raises:
        BedrockClientError: 설정이 서로 충돌할 때.
    """
    if settings.bedrock_api_style not in API_STYLES:
        raise BedrockClientError(
            f"지원하지 않는 BEDROCK_API_STYLE 입니다: {settings.bedrock_api_style}"
        )
    if settings.bedrock_api_key and settings.bedrock_aws_profile:
        raise BedrockClientError(
            "BEDROCK_API_KEY 와 BEDROCK_AWS_PROFILE 은 함께 쓸 수 없습니다."
        )
    kwargs: dict[str, Any] = {
        "aws_region": settings.bedrock_region,
        "timeout": settings.bedrock_timeout_seconds,
        "max_retries": settings.bedrock_max_retries,
    }
    if settings.bedrock_api_key:
        kwargs["api_key"] = settings.bedrock_api_key
    else:
        kwargs["aws_profile"] = settings.bedrock_aws_profile
    if settings.bedrock_base_url:
        kwargs["base_url"] = settings.bedrock_base_url
    return kwargs


def _build(is_async: bool) -> Any:
    """설정된 호출 경로에 맞는 클라이언트 클래스를 골라 생성합니다."""
    if settings.bedrock_api_style not in API_STYLES:
        raise BedrockClientError(
            f"지원하지 않는 BEDROCK_API_STYLE 입니다: {settings.bedrock_api_style}"
        )
    try:
        import anthropic
    except ImportError as exc:
        raise BedrockClientError(
            "anthropic[bedrock] 패키지가 설치되어 있지 않습니다."
        ) from exc

    names = {
        ("invoke", False): "AnthropicBedrock",
        ("invoke", True): "AsyncAnthropicBedrock",
        ("mantle", False): "AnthropicBedrockMantle",
        ("mantle", True): "AsyncAnthropicBedrockMantle",
    }
    client_class = getattr(anthropic, names[(settings.bedrock_api_style, is_async)])
    try:
        return client_class(**_client_kwargs())
    except BedrockClientError:
        raise
    except Exception as exc:
        raise BedrockClientError(f"Bedrock 클라이언트를 만들지 못했습니다: {exc}") from exc


def get_client() -> Any:
    """추천 경로가 쓰는 동기 클라이언트를 프로세스당 한 번만 만듭니다."""
    global _sync_client
    if _sync_client is None:
        with _lock:
            if _sync_client is None:
                _sync_client = _build(is_async=False)
    return _sync_client


def get_async_client() -> Any:
    """이미지 분석 경로가 쓰는 비동기 클라이언트를 프로세스당 한 번만 만듭니다."""
    global _async_client
    if _async_client is None:
        with _lock:
            if _async_client is None:
                _async_client = _build(is_async=True)
    return _async_client


def reset_clients() -> None:
    """설정을 바꿔 다시 만들어야 할 때 캐시를 비웁니다. 테스트에서 씁니다."""
    global _sync_client, _async_client
    with _lock:
        _sync_client = None
        _async_client = None


def upstream_message(exc: Exception) -> str:
    """Bedrock 이 돌려준 오류 본문에서 사람이 읽을 메시지만 뽑아냅니다.

    권한 부족인지, 모델 미활성인지, 사용 사례 양식 미제출인지는 이 메시지에만
    담깁니다. 감춰 두면 원인 파악이 불가능해집니다.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if body.get("message"):
            return str(body["message"])
    return str(getattr(exc, "message", None) or exc)


def describe_failure(exc: Exception) -> str:
    """SDK 예외를 원인이 드러나는 한국어 한 줄로 바꿉니다."""
    import anthropic

    model = settings.bedrock_model_id
    if isinstance(exc, anthropic.AuthenticationError):
        return "Bedrock 자격증명이 유효하지 않습니다."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return f"Bedrock 모델 접근이 거부되었습니다({model}): {upstream_message(exc)}"
    if isinstance(exc, anthropic.NotFoundError):
        return f"Bedrock 에서 모델을 쓸 수 없습니다({model}): {upstream_message(exc)}"
    if isinstance(exc, anthropic.RateLimitError):
        return "Bedrock 요청이 사용량 제한에 걸렸습니다."
    if isinstance(exc, anthropic.APIConnectionError):
        return f"Bedrock 에 연결하지 못했습니다: {exc}"
    if isinstance(exc, anthropic.APIStatusError):
        return f"Bedrock 호출이 실패했습니다(HTTP {exc.status_code}): {upstream_message(exc)}"
    return f"Bedrock 호출 중 오류가 발생했습니다: {exc}"


def extract_text(response: Any) -> str:
    """응답의 text 블록을 이어 붙입니다."""
    return "".join(
        block.text for block in response.content if block.type == "text"
    )


def schema_instruction(schema: dict[str, Any]) -> str:
    """스키마를 프롬프트로 강제하는 지시문을 만듭니다.

    Bedrock 은 구조화 출력(``response_format``)을 지원하지 않습니다. vLLM 경로가
    스키마로 키 이름을 못박는 것과 달리, 여기서는 프롬프트가 유일한 수단입니다.
    이 지시문이 없으면 모델이 키 이름을 스스로 지어내 ``suggested_message`` 같은
    필드가 통째로 비고, 카테고리도 다른 구조로 나옵니다(실측).

    "required 는 모두 채워라" 로 쓰면 이미지 분석 프롬프트의 "모르면 null" 과
    정면으로 부딪혀 모델이 금액·날짜를 지어냅니다. 그래서 요구하는 것은 키의
    **존재**이고, 값이 없을 때 넣을 것은 ``null`` 이라고 분리해 적습니다.
    """
    return (
        "설명이나 코드 블록 없이, 아래 JSON Schema 를 만족하는 JSON 객체 하나만 "
        "출력하세요. required 에 있는 키는 하나도 빠뜨리지 말고 모두 포함하되, "
        "null 이 허용된 키는 값을 확인할 수 없을 때 지어내지 말고 null 을 넣으세요. enum 이 있는 값은 "
        "목록의 문자열을 한 글자도 바꾸지 말고 그대로 복사하세요.\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )
