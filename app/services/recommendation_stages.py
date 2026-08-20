"""추천 생성을 세 단계로 나눠 Bedrock 에 비동기로 부릅니다.

단일 호출(``qwen_service.recommend_simple``)은 카테고리·이유·요약·감사 메시지를
한 번에 500자 안팎 씁니다. 실측에서 이 호출 하나가 11~12초였고, 그 뒤에 상품
검색 6~9초가 붙어 ``/recommend`` 가 최대 19.7초였습니다.

그런데 상품 검색이 기다리는 것은 카테고리 **이름**뿐입니다. 가격 범위는
``recommendation_policy.price_range`` 가 규칙으로 정하고 검색어 씨앗은
``SAFE_EXAMPLES`` 에서 나오므로, 모델에게 받을 것이 20자밖에 없습니다. 나머지
480자를 다 쓸 때까지 검색이 출발하지 못하는 것이 지연의 대부분이었습니다.

    t=0  ┌─ plan (카테고리만) ─┬─▶ 상품 검색 ────┐
         │                     └─▶ prose (이유·요약) ─┤
         └─ message (감사 메시지) ──────────────────┴─▶ 병합 → normalize

단계를 **줄줄이 세우는** 하니스는 왕복이 더해져 느려집니다. 여기서는 서로를
기다리지 않는 것끼리 옆으로 늘어놓으므로 생성 시간이 나뉩니다.

세 호출은 서로 독립적으로 실패합니다. 어느 하나가 죽어도 나머지는 그대로
나가고, 빈 자리는 ``normalize_recommendation`` 의 기존 폴백이 채웁니다. 단일
호출은 JSON 이 한 번 깨지면 네 필드가 **한꺼번에** 템플릿으로 떨어졌습니다.
"""

import logging
from typing import Any

from app.core.config import settings
from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services import bedrock_client, prompt
from app.services.model_response_parser import ModelResponseParseError, parse_json_object

logger = logging.getLogger(__name__)

# 단계별 출력 상한. 단일 호출의 2,048 은 네 필드를 한꺼번에 담으려는 값이라
# 단계마다 그대로 쓰면 잘림 감지가 무의미해집니다. plan 은 카테고리 3개면
# 120 토큰 안팎이고, 나머지는 여유를 둡니다.
_MAX_TOKENS = {"plan": 400, "prose": 1_200, "message": 800}

# qwen_service 와 같은 사정입니다. 최신 모델은 샘플링 파라미터를, 일부 모델·호출
# 경로는 구조화 출력을 400 으로 거부합니다. 거부는 모델을 바꿨을 때만 생기고
# 프로세스 내내 같으므로 모듈에 기억해 둡니다.
_accepts_sampling = True
_accepts_structured_output = True


def _drop_rejected(payload: dict[str, Any], exc: Exception) -> list[str]:
    """거부된 파라미터를 payload 에서 빼고 그 이름을 돌려줍니다. 뺄 것이 없으면 빈 목록."""
    global _accepts_sampling, _accepts_structured_output
    message = bedrock_client.upstream_message(exc)
    if "output_config" in message and "output_config" in payload:
        _accepts_structured_output = False
        payload.pop("output_config")
        return ["구조화 출력"]
    sampling = [k for k in ("temperature", "top_p", "top_k") if k in payload]
    if sampling:
        _accepts_sampling = False
        for key in sampling:
            payload.pop(key)
        return [f"샘플링 파라미터({', '.join(sampling)})"]
    return []


async def _call(stage: str, messages: list[dict[str, str]], schema: dict) -> dict[str, Any]:
    """한 단계를 부르고 파싱된 JSON 을 돌려줍니다. 실패하면 빈 dict.

    빈 dict 를 돌려주는 것이 이 함수의 계약입니다. 예외를 올리면 한 단계의 실패가
    추천 전체를 죽이는데, 그러면 단일 호출과 견고성이 같아져 나눈 뜻이 없습니다.
    호출 측은 빈 자리를 ``normalize_recommendation`` 의 폴백에 맡깁니다.
    """
    import anthropic

    system = next(m["content"] for m in messages if m["role"] == "system")
    # 구조화 출력에 실을 수 없는 maxItems·minLength 는 이 지시문만 요구할 수 있습니다.
    system += "\n\n" + bedrock_client.schema_instruction(schema)

    payload: dict[str, Any] = {
        "model": settings.bedrock_model_id,
        "max_tokens": _MAX_TOKENS[stage],
        "system": system,
        "messages": [m for m in messages if m["role"] != "system"],
    }
    if _accepts_sampling:
        payload["temperature"] = settings.bedrock_temperature
    if _accepts_structured_output:
        payload["output_config"] = bedrock_client.output_config(schema)

    client = bedrock_client.get_async_client()
    try:
        try:
            response = await client.messages.create(**payload)
        except anthropic.BadRequestError as exc:
            dropped = _drop_rejected(payload, exc)
            if not dropped:
                raise
            logger.warning(
                "%s 가 %s 를 거부해 빼고 재시도합니다: %s",
                settings.bedrock_model_id,
                ", ".join(dropped),
                bedrock_client.upstream_message(exc),
            )
            response = await client.messages.create(**payload)
    except Exception as exc:
        logger.warning("추천 %s 단계 호출 실패. 이 단계는 폴백으로 대체합니다: %s", stage, exc)
        return {}

    if response.stop_reason == "max_tokens":
        logger.warning(
            "추천 %s 단계가 max_tokens(%d)에서 잘렸습니다.", stage, _MAX_TOKENS[stage]
        )
    # 출력 토큰을 남기는 이유: 이 설계의 전제가 "지연은 출력 토큰에 비례한다" 입니다
    # (실측 고정비 약 1.2초 + 약 53 tok/s). 어느 단계가 예상보다 길게 쓰고 있는지는
    # 이 값으로만 보이고, 실제로 1차 측정에서 plan 이 지시를 무시하고 길게 써
    # 분할이 단일보다 느렸습니다.
    usage = getattr(response, "usage", None)
    logger.info(
        "추천 %s 단계 토큰=%s/%s",
        stage,
        getattr(usage, "input_tokens", "?"),
        getattr(usage, "output_tokens", "?"),
    )
    try:
        return parse_json_object(bedrock_client.extract_text(response))
    except ModelResponseParseError:
        logger.warning("추천 %s 단계 JSON 파싱 실패. 이 단계는 폴백으로 대체합니다.", stage)
        return {}


class RecommendationStages:
    """세 단계를 각각 부르는 얇은 껍데기.

    모듈 함수가 아니라 객체인 이유는 벤치마크가 단계별 시간을 재려고 메서드를
    감싸기 때문입니다(``scripts/benchmark_split.py``). 운영 코드에 타이머를 심으면
    지금 재고 싶은 경계와 나중에 재고 싶은 경계가 달라 그때마다 코드를 고칩니다.
    """

    async def plan(self, request: SimpleGiftRecommendationRequest) -> dict[str, Any]:
        """1단계: 카테고리와 점수. 상품 검색이 이 결과만으로 출발합니다."""
        return await _call(
            "plan", prompt.build_plan_messages(request), prompt.build_plan_schema()
        )

    async def prose(
        self, request: SimpleGiftRecommendationRequest, categories: list[dict]
    ) -> dict[str, Any]:
        """2단계: 카테고리별 이유와 요약. 1단계 결과에 의존합니다."""
        if not categories:
            return {}
        return await _call(
            "prose",
            prompt.build_prose_messages(request, categories),
            prompt.build_prose_schema(),
        )

    async def message(self, request: SimpleGiftRecommendationRequest) -> dict[str, Any]:
        """3단계: 감사 메시지. 카테고리에 의존하지 않아 1단계와 동시에 출발합니다."""
        return await _call(
            "message", prompt.build_message_messages(request), prompt.build_message_schema()
        )


recommendation_stages = RecommendationStages()
