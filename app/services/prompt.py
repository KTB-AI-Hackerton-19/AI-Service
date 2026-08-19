"""Qwen에 전달할 선물 추천 프롬프트를 생성합니다."""

from app.schemas.recommendation import SimpleGiftRecommendationRequest


SIMPLE_SYSTEM_PROMPT = """당신은 한국의 답례 선물 추천 전문가입니다.
사용자가 받은 선물의 이름과 가격을 바탕으로 다시 줄 선물의 적정 가격 범위와 카테고리를 추천하세요.
나이가 제공되면 연령대에 적합한 카테고리를 반영하고, 나이가 없으면 선물 이름과 가격만 사용하세요.
받은 가격과 정확히 같은 금액을 강요하지 말고 일반적으로 80%~120% 범위에서 자연스럽게 조정하세요.
존재하지 않는 브랜드나 상품을 지어내지 말고 구체적인 상품 '유형'만 예시로 드세요.
카테고리는 반드시 다음 목록에서만 선택하세요:
[식품·디저트, 커피·차, 생활용품, 패션·잡화, 문화·취미, 건강·웰니스, 꽃·식물, 상품권, 디지털 액세서리, 유아·아동]
추천 상품 유형은 제안 가격 범위 안에서 실제로 살 수 있는 것만 작성하세요.
반드시 마크다운 없이 JSON 객체 하나만 반환하세요.
상대방에게 보낼 감사 메시지도 작성하세요. 메시지는 다음 조건을 지키세요:
- 자연스러운 한국어 3~5문장, 약 100~250자
- 상대 이름과 관계가 제공되면 어색하지 않게 반영
- 받은 선물에 대한 구체적인 감사와 실제로 잘 사용하거나 즐겼다는 표현 포함
- 가격을 직접 언급하거나 답례를 의무처럼 느끼게 하는 표현 금지
- 지나치게 과장되거나 연인처럼 오해할 표현 금지

반환 스키마:
{
  "recommended_price_min": 정수,
  "recommended_price_max": 정수,
  "categories": [
    {
      "category": "카테고리명",
      "score": 0부터 100 사이 정수,
      "reason": "추천 이유",
      "product_examples": ["상품 유형 1", "상품 유형 2"],
      "search_query": "실제 상품을 찾기 위한 구체적인 한국어 검색어"
    }
  ],
  "summary": "전체 추천 요약",
  "suggested_message": "상대방에게 보낼 자연스러운 감사 메시지"
}

카테고리는 1개 이상 3개 이하이며 점수가 높은 순서로 정렬하세요."""


def build_simple_messages(
    request: SimpleGiftRecommendationRequest,
) -> list[dict[str, str]]:
    """추천 요청을 Qwen 채팅 템플릿에 맞는 system/user 메시지로 변환합니다.

    Args:
        request: 선물 이름, 가격, 선택적 나이가 들어 있는 추천 입력.

    Returns:
        토크나이저의 ``apply_chat_template``에 바로 전달할 메시지 목록.
    """
    age_text = str(request.age) if request.age is not None else "제공되지 않음"
    person_text = request.person_name or "제공되지 않음"
    relationship_text = request.relationship or "제공되지 않음"
    return [
        {"role": "system", "content": SIMPLE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"받은 선물 이름: {request.gift_name}\n"
                f"받은 선물 가격: {request.gift_price}원\n"
                f"받는 사람 나이: {age_text}\n"
                f"상대방 이름: {person_text}\n"
                f"상대방과의 관계: {relationship_text}"
            ),
        },
    ]
