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
    "쇼핑 검색 결과를 답례 선물 후보로 판정한다. "
    "각 줄은 '[카테고리] 상품명' 형식이며 두 조건을 모두 만족해야 통과다.\n"
    "1) 상품이 그 카테고리에 실제로 속할 것. "
    "[커피·차] 나주배 세트, [생활용품] 학습 교재처럼 라벨과 물건이 어긋나면 제외한다.\n"
    "2) 그 카테고리의 선물로 상대에게 그대로 줄 수 있을 것. "
    "중고, 대량 도매·업소용, 성인용품, 부품·리필·액세서리 단품, "
    "포장재·용기처럼 선물 자체가 아닌 물건은 제외한다.\n"
    "제외는 둘 중 하나가 분명할 때만 한다. 국내 쇼핑몰 제목에 흔한 "
    "대괄호 브랜드([센터커피])와 용량·규격 표기(170g, 10g X 15개, 30수)는 "
    "판매자가 붙인 검색어이지 제외 사유가 아니다. 애매하면 통과시킨다."
)
# 1) 을 따로 세운 이유: 예전 프롬프트는 부적합 **유형**만 나열하고 라벨 일치는
# "그 카테고리의 선물로" 한 마디에 기대고 있었습니다. 실측에서
# "[커피·차] [선물] 명품 나주배 세트 5kg" 이 그 문장을 통과해 1만원 예산의 유일한
# 추천으로 나갔습니다. 라벨 대조를 별도 조건으로 못박고 실측 사례를 예로 답니다.
# 입력 토큰은 요청당 60 토큰 안팎 늘어납니다(실측 입력 1,076~1,254 기준 +5%).
#
# 마지막 문단은 그 다음 라운드에 붙였습니다. 5차 실측에서 판정이 반대쪽으로
# 넘어가 **정상 상품 둘**을 떨어뜨렸습니다.
#     [커피·차] [센터커피] 디카페인 드립백 세트 (10g X 15개)
#     [생활용품] 송월타월 고급수건 답례품 프레디 170g 코마사 30수 두꺼운
# 둘 다 라벨과 물건이 정확히 맞고 그대로 선물할 수 있습니다. 공통점은 제목에
# 붙은 대괄호 브랜드와 용량·규격 나열뿐입니다.
#
# 무엇이 이렇게 만들었는지: 위 두 조건은 "통과하려면 무엇을 만족해야 하는가" 로
# 쓰여 있어 판정이 **적합성 증명**이 됩니다. 증명이 기본값이면 제목이 특이한
# 항목은 증명에 실패해 제외로 떨어집니다. 여기에 1) 이 "어긋난 것을 찾아라" 를
# 더했고, product_filter_temperature=0.0 이라 경계에 있는 항목이 실행마다
# 통과 쪽으로 굴러떨어질 여지도 없앴습니다. 세 가지가 같은 방향으로 겹쳤습니다.
#
# 그래서 규칙은 그대로 두고 **기본값만 뒤집습니다**. 제외는 근거가 분명할 때만
# 하고 나머지는 통과입니다. 후보 부족이 이 라운드의 P0 이라 오탈락 한 건이
# 오통과 한 건보다 비쌉니다(오통과는 상품 카드 3장 중 한 장이 살짝 어긋나는
# 것이고, 오탈락은 그 자리를 아예 비웁니다).
#
# 키워드로 되살리는 방법은 쓰지 않았습니다. 같은 실행에서 모델과 키워드가 갈린
# 다섯 건 중 모델이 맞은 것이 셋입니다 — 조의 문구가 붙은 "송월타월 조문 조의
# 답례품", 포장재인 "추석 명절 띠지세트B(케이스+띠지)", 부품인 "넝쿨식물지지대"
# 는 모두 카테고리 낱말("타월"·"수건"·"식물")을 갖고 있어 키워드로 되살리면
# 함께 돌아옵니다. 되살리기는 이 표본에서 이득보다 손해가 큽니다.

# Claude 는 temperature 와 top_p 를 동시에 받지 않으므로 temperature 만 보냅니다.
# 또 Opus 4.6+ / Sonnet 5 등은 샘플링 파라미터 자체를 400 으로 거부합니다
# (qwen_service._call_bedrock 과 같은 사정). BEDROCK_MODEL_ID 는 바꿔 가며 쓰는
# 값이므로 거부당하면 한 번만 감지해 내리고, 판정 자체는 계속 돌게 합니다.
_accepts_sampling = True

# 번호 배열만 받습니다. 항목마다 {n, keep} 객체를 받으면 출력이 12건 기준 141
# 토큰인데 번호 배열은 18 토큰이고, 그만큼 생성 시간도 짧습니다(2.2초 -> 1.7초, 실측).
# 정확도는 같았습니다. 구조화 출력을 빼면 1.5초까지 줄지만 정답 2건을 빠뜨렸습니다.
#
# 통과 번호만 받으면 "모델이 뺀 항목"과 "모델이 빠뜨린 항목"을 구분할 수 없습니다.
# 둘을 같게 다루면 필터가 한쪽으로 망가집니다. 빠진 것을 통과로 보면 부적합 상품이
# 그대로 나가고, 탈락으로 보면 멀쩡한 상품이 조용히 사라집니다. 그래서 통과와 제외를
# 따로 받아, 어느 쪽에도 없는 번호만 "모델이 판정하지 않은 것"으로 처리합니다.
# 배열이 하나 늘어도 출력은 12건 기준 36 토큰 안팎입니다.
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "답례 선물로 적합한 항목의 번호",
        },
        "drop": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "적합하지 않은 항목의 번호",
        },
    },
    "required": ["keep", "drop"],
    "additionalProperties": False,
}


def _indexes(raw: object, count: int) -> set[int]:
    """모델이 낸 1-based 번호를 인덱스로 바꿉니다. 범위 밖·형식 오류는 버립니다."""
    if not isinstance(raw, list):
        return set()
    indexes: set[int] = set()
    for number in raw:
        try:
            index = int(number) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < count:
            indexes.add(index)
    return indexes


def is_available() -> bool:
    """모델 판정을 쓸 수 있는 상태인지.

    Bedrock 에서만 씁니다. vLLM(Gemma-12B)·MLX 는 구조화 출력과 판정 일관성을
    기대하기 어렵고, mock 은 네트워크로 나가면 안 됩니다.
    """
    return settings.product_llm_filter_enabled and settings.model_backend == "bedrock"


async def _create(params: dict[str, object]):
    """판정을 호출하되 샘플링 파라미터가 거부되면 한 번만 빼고 재시도합니다.

    거부는 모델을 바꿨을 때만 생기고 프로세스 내내 같으므로 ``_accepts_sampling``
    에 기억해 둡니다. 재시도 없이 예외를 그대로 올리면 판정이 통째로 죽고 키워드
    폴백으로 떨어져, 모델을 바꾼 것만으로 필터 품질이 조용히 내려앉습니다.
    """
    global _accepts_sampling
    import anthropic

    from app.services import bedrock_client

    client = bedrock_client.get_async_client()
    try:
        return await client.messages.create(**params)
    except anthropic.BadRequestError:
        if "temperature" not in params:
            raise
        logger.warning(
            "%s 가 판정 temperature 를 거부해 빼고 재시도합니다.", settings.bedrock_model_id
        )
        _accepts_sampling = False
        params.pop("temperature")
        return await client.messages.create(**params)


async def judge(items: list[tuple[str, str]]) -> dict[int, bool] | None:
    """(카테고리, 제목) 목록을 한 번에 판정합니다.

    Args:
        items: 판정할 (카테고리, 상품 제목) 목록.

    Returns:
        모델이 판정한 인덱스만 담은 사전(통과 ``True`` / 제외 ``False``). 모델이
        번호를 빠뜨렸거나 통과·제외 양쪽에 넣어 자기모순이면 그 인덱스는 키가
        없으며, 호출 측이 키워드로 판정합니다. 호출 자체가 실패하면 ``None`` 이고
        이때는 전부 키워드로 판정해야 합니다.
    """
    if not items:
        return {}

    lines = "\n".join(
        f"{index + 1}. [{category}] {title}" for index, (category, title) in enumerate(items)
    )
    params: dict[str, object] = {
        "model": settings.bedrock_model_id,
        "max_tokens": settings.product_llm_filter_max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    "적합한 항목의 번호만 keep 에, 부적합한 항목의 번호만 drop 에 "
                    f"넣으세요. 두 배열을 합치면 모든 번호가 한 번씩 나와야 합니다.\n{lines}"
                ),
            }
        ],
        "output_config": {"format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
    }
    # 이 호출은 판정이지 창작이 아닙니다. 같은 제목이 실행마다 다른 판정을 받으면
    # 안 되므로 greedy 로 둡니다(근거는 config.product_filter_temperature 주석).
    if _accepts_sampling:
        params["temperature"] = settings.product_filter_temperature

    try:
        response = await _create(params)
        text = "".join(block.text for block in response.content if block.type == "text")
        payload = json.loads(text)
    except Exception as exc:
        logger.warning("상품 판정 호출 실패. 키워드 판정으로 대체합니다: %s", exc)
        return None

    keep = _indexes(payload.get("keep"), len(items))
    drop = _indexes(payload.get("drop"), len(items))
    result: dict[int, bool] = {index: True for index in keep - drop}
    result.update({index: False for index in drop - keep})
    usage = getattr(response, "usage", None)
    logger.info(
        "상품 판정 %d건 중 통과 %d건, 제외 %d건, 미판정 %d건(키워드로 판정) 토큰=%s/%s",
        len(items),
        sum(result.values()),
        len(result) - sum(result.values()),
        len(items) - len(result),
        getattr(usage, "input_tokens", "?"),
        getattr(usage, "output_tokens", "?"),
    )
    return result
