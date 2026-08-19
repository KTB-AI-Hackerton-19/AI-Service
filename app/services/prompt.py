"""선물 추천 프롬프트와 강제 출력 스키마를 생성합니다.

이미지 추출이 ``vision_prompt`` 를 쓰는 것과 대칭입니다. 두 프롬프트 모두 같은
vLLM 서버의 같은 모델(Gemma4-12B-QAT)로 갑니다.

카테고리 목록은 ``recommendation_policy.ALLOWED_CATEGORIES`` 하나에서 나옵니다.
프롬프트와 스키마가 각자 목록을 들고 있으면 반드시 어긋납니다.
"""

from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services.recommendation_policy import ALLOWED_CATEGORIES

_CATEGORY_LIST = ", ".join(ALLOWED_CATEGORIES)

SIMPLE_SYSTEM_PROMPT = f"""당신은 한국의 답례 선물 추천 전문가입니다.
사용자가 받은 것의 이름과 가격을 바탕으로 다시 줄 선물의 적정 가격 범위와 카테고리를 추천하세요.
나이와 성별이 제공되면 연령대와 성별에 적합한 카테고리를 반영하고, 없으면 나머지 정보만 사용하세요.
사용자가 예산이나 카테고리를 직접 지정했다면 반드시 그 안에서 추천하세요.
받은 가격과 정확히 같은 금액을 강요하지 말고 일반적으로 80%~120% 범위에서 자연스럽게 조정하세요.
존재하지 않는 브랜드나 상품을 지어내지 말고 구체적인 상품 '유형'만 예시로 드세요.
카테고리는 반드시 다음 목록에서만 선택하세요:
[{_CATEGORY_LIST}]
추천 상품 유형은 제안 가격 범위 안에서 실제로 살 수 있는 것만 작성하세요.
반드시 마크다운 없이 JSON 객체 하나만 반환하세요.

메시지는 다음 조건을 지키세요:
- 자연스러운 한국어 4~6문장, 150~250자 (130자 미만이면 폐기되고 기본 문구로 대체됩니다)
- 상대 이름과 관계가 제공되면 어색하지 않게 반영
- 받은 것에 대한 구체적인 감사와 실제로 잘 사용하거나 즐겼다는 표현 포함
- 가격을 직접 언급하거나 답례를 의무처럼 느끼게 하는 표현 금지
- 지나치게 과장되거나 연인처럼 오해할 표현 금지
- **당신은 사용자 본인의 입장에서 씁니다.** 상대방이 사용자에게 하는 말을 쓰면 안 됩니다.

카테고리는 1개 이상 3개 이하이며 점수가 높은 순서로 정렬하세요."""

# 청첩장·부고장은 "받은 선물" 이 아니라 "앞으로 참석하고 축의·조의할 일정" 입니다.
# 이 안내가 없으면 모델이 사용자를 신랑신부 쪽으로 착각해 하객에게 감사하는 문장을 씁니다.
_INVITATION_NOTE = """
[중요] 사용자는 이 행사에 **초대받은 하객**입니다. 사용자가 주인공이 아닙니다.
- 메시지는 사용자가 주최자에게 보내는 **축하 인사**로 작성하세요.
- "참석해 주셔서 감사합니다" 처럼 주최자가 하객에게 하는 말은 절대 쓰지 마세요.
- 가격 범위는 답례 선물이 아니라 축의금·조의금의 적정 수준으로 보세요."""

_MULTI_NOTE = """
[중요] 여러 사람에게 한 번에 받았습니다. 사람마다 금액이 다르므로
가격 범위는 가장 적게 준 사람과 가장 많이 준 사람을 모두 감당할 수 있게 넓게 잡으세요.
메시지는 특정 한 사람이 아니라 여러 사람에게 두루 쓸 수 있는 표현으로 작성하세요."""


def build_recommendation_schema() -> dict:
    """vLLM ``response_format={"type": "json_schema"}`` 에 그대로 넣는 스키마.

    카테고리를 enum 으로 못박아 두면 모델이 목록 밖의 값을 만들 수 없습니다.
    ``recommendation_policy`` 의 보정은 그래도 남겨 두지만, 이 스키마가 있으면
    보정이 주 경로가 아니라 안전망으로 물러납니다.
    """
    return {
        "type": "object",
        "properties": {
            "recommended_price_min": {"type": "integer", "minimum": 0},
            "recommended_price_max": {"type": "integer", "minimum": 0},
            "categories": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": list(ALLOWED_CATEGORIES)},
                        "score": {"type": "integer", "minimum": 0, "maximum": 100},
                        "reason": {"type": "string"},
                        "product_examples": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 3,
                        },
                    },
                    "required": ["category", "score", "reason"],
                    "additionalProperties": False,
                },
            },
            "summary": {"type": "string"},
            # 정책이 130자 미만을 폐기하므로 스키마에서도 미리 못박습니다.
            # 없으면 모델이 100자짜리를 만들어 매번 기본 문구로 대체됩니다(실측).
            "suggested_message": {"type": "string", "minLength": 130},
        },
        "required": [
            "recommended_price_min",
            "recommended_price_max",
            "categories",
            "summary",
            "suggested_message",
        ],
        "additionalProperties": False,
    }


def build_simple_messages(
    request: SimpleGiftRecommendationRequest,
) -> list[dict[str, str]]:
    """추천 요청을 채팅 템플릿에 맞는 system/user 메시지로 변환합니다.

    Args:
        request: 선물 이름, 가격, 선택적 나이가 들어 있는 추천 입력.

    Returns:
        토크나이저의 ``apply_chat_template`` 또는 OpenAI 호환 API 에 바로 넣을 메시지 목록.
    """
    system = SIMPLE_SYSTEM_PROMPT
    if request.record_type == "event_invitation":
        system += _INVITATION_NOTE
    elif len(request.received_amounts) > 1:
        system += _MULTI_NOTE

    age_text = str(request.age) if request.age is not None else "제공되지 않음"
    person_text = request.person_name or "제공되지 않음"
    relationship_text = request.relationship or "제공되지 않음"

    gender_text = {"male": "남성", "female": "여성"}.get(request.gender, "제공되지 않음")

    lines = [
        f"받은 것: {request.gift_name}",
        f"금액: {request.gift_price}원",
        f"받는 사람 나이: {age_text}",
        f"받는 사람 성별: {gender_text}",
        f"상대방 이름: {person_text}",
        f"상대방과의 관계: {relationship_text}",
    ]
    if request.event:
        lines.append(f"계기: {request.event}")
    if request.budget_min is not None or request.budget_max is not None:
        low = f"{request.budget_min:,}원" if request.budget_min is not None else "제한 없음"
        high = f"{request.budget_max:,}원" if request.budget_max is not None else "제한 없음"
        lines.append(f"사용자가 지정한 예산: {low} ~ {high}")
    if request.preferred_categories:
        lines.append(
            "사용자가 고른 카테고리(이 안에서만 고르세요): " + ", ".join(request.preferred_categories)
        )
    if request.interests:
        lines.append(f"상대방 관심사: {', '.join(request.interests)}")
    if request.dislikes:
        lines.append(f"상대방이 싫어하는 것(피하세요): {', '.join(request.dislikes)}")
    if len(request.received_amounts) > 1:
        detail = ", ".join(
            f"{name} {amount:,}원"
            for name, amount in zip(
                request.people or ["이름 미상"] * len(request.received_amounts),
                request.received_amounts,
                strict=False,
            )
        )
        lines.append(f"여러 사람에게 받음({len(request.received_amounts)}명): {detail}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]
