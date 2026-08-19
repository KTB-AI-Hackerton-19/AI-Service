"""검색 결과가 답례 선물로 추천할 만한 상품인지 판정합니다.

키워드 사전만으로는 한계가 분명했습니다. 부분 문자열 매칭이라 "차"가 "차량"·"주차"에,
"떡"이 "떡메모지"에 걸려 무관한 상품이 통과하고, 반대로 사전에 없는 브랜드 표기
("스타벅스 다크 로스트 아메리카노")는 정상 상품인데도 탈락했습니다. 1만원대 답례
선물의 상당수가 기프티콘·브랜드 상품이라 이 누락이 특히 아팠습니다.

그래서 의미 판단은 모델에 맡깁니다. 후보를 한 번에 묶어 한 번만 부릅니다. 실측에서
24건 일괄 판정이 3회 모두 정확했고, 건별 호출(fan-out)과 정확도는 같은데 입력
토큰은 6.4배 적었습니다. 지연 차이는 0.6초로, 요청 수 24배를 감수할 이유가 없습니다.

모델을 못 쓰면 키워드 방식으로 돌아갑니다. 추천이 필터 하나 때문에 죽으면 안 됩니다.
"""

import json
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "쇼핑 검색 결과가 해당 카테고리의 답례 선물로 추천할 만한 상품인지 판정한다. "
    "포장재·용기 등 선물 자체가 아닌 물건은 제외한다."
)

# 통과한 번호만 받습니다. 항목마다 {n, keep} 객체를 받으면 출력이 12건 기준 141
# 토큰인데 번호 배열은 18 토큰이고, 그만큼 생성 시간도 짧습니다(2.2초 -> 1.7초, 실측).
# 정확도는 같았습니다. 구조화 출력을 빼면 1.5초까지 줄지만 정답 2건을 빠뜨렸습니다.
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {
            "type": "array",
            "items": {"type": "integer", "description": "추천 가능한 항목의 번호"},
        }
    },
    "required": ["keep"],
    "additionalProperties": False,
}


def is_available() -> bool:
    """모델 판정을 쓸 수 있는 상태인지.

    Bedrock 에서만 씁니다. vLLM(Gemma-12B)·MLX 는 구조화 출력과 판정 일관성을
    기대하기 어렵고, mock 은 네트워크로 나가면 안 됩니다.
    """
    return settings.product_llm_filter_enabled and settings.model_backend == "bedrock"


async def judge(items: list[tuple[str, str]]) -> dict[int, bool] | None:
    """(카테고리, 제목) 목록을 한 번에 판정합니다.

    Args:
        items: 판정할 (카테고리, 상품 제목) 목록.

    Returns:
        인덱스별 판정 결과. 호출이 실패하면 ``None`` 이며, 이때 호출 측은 전부
        키워드로 판정해야 합니다.

        통과 번호만 받으므로 "모델이 빠뜨린 항목"과 "모델이 뺀 항목"을 구분할 수
        없습니다. 실측에서 누락은 없었고, 빠뜨려도 상품 하나가 후보에서 빠질 뿐
        추천이 깨지지는 않습니다. 그 위험보다 매 요청 0.5초가 더 큽니다.
    """
    if not items:
        return {}

    from app.services import bedrock_client

    lines = "\n".join(
        f"{index + 1}. [{category}] {title}" for index, (category, title) in enumerate(items)
    )
    try:
        response = await bedrock_client.get_async_client().messages.create(
            model=settings.bedrock_model_id,
            max_tokens=settings.product_llm_filter_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"각 항목을 판정하세요. 번호를 빠짐없이 모두 포함하세요.\n{lines}",
                }
            ],
            output_config={"format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        numbers = json.loads(text)["keep"]
    except Exception as exc:
        logger.warning("상품 판정 호출 실패. 키워드 판정으로 대체합니다: %s", exc)
        return None

    keep = {int(number) - 1 for number in numbers}
    result = {index: index in keep for index in range(len(items))}
    logger.info("상품 판정 %d건 중 %d건 통과", len(items), sum(result.values()))
    return result
