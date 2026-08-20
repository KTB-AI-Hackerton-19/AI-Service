"""선물 추천 프롬프트와 강제 출력 스키마를 생성합니다.

이미지 추출이 ``vision_prompt`` 를 쓰는 것과 대칭입니다. 두 프롬프트 모두 같은
vLLM 서버의 같은 모델(Gemma4-12B-QAT)로 갑니다.

카테고리 목록은 ``recommendation_policy.ALLOWED_CATEGORIES`` 하나에서 나옵니다.
프롬프트와 스키마가 각자 목록을 들고 있으면 반드시 어긋납니다.

가격 범위와 상품 유형은 모델에게 시키지 않습니다. 어차피 ``recommendation_policy``
가 규칙값과 ``SAFE_EXAMPLES`` 로 덮어쓰던 값이라, 생성해 봐야 출력 토큰만큼
지연만 늘었습니다.
"""

from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services.recommendation_policy import (
    ALLOWED_CATEGORIES,
    TARGET_MESSAGE_LENGTH,
    is_condolence,
)

_CATEGORY_LIST = ", ".join(ALLOWED_CATEGORIES)

SIMPLE_SYSTEM_PROMPT = f"""당신은 한국의 답례 선물 추천 전문가입니다.
사용자가 받은 것을 보고 답례로 줄 선물의 카테고리와 감사 메시지를 만드세요.
가격은 서비스가 규칙으로 계산하므로 당신은 금액을 정하거나 언급하지 않습니다.
나이·성별은 카테고리 선택에만 쓰고, 문장에 옮기거나 사람을 평가하듯 쓰지 마세요.
카테고리는 반드시 다음 목록에서만 점수가 높은 순서로 선택하세요:
[{_CATEGORY_LIST}]
summary 와 reason 에는 어떤 카테고리를 왜 권하는지만 쓰고, 특정 상품·브랜드·금액을 약속하지 마세요.
반드시 마크다운 없이 JSON 객체 하나만 반환하세요.

메시지는 다음 조건을 지키세요:
- 자연스러운 한국어로 답변하세요.
- **사용자 본인의 입장**에서 상대에게 직접 건네는 말입니다. 상대를 3인칭으로 서술하거나
  상대가 사용자에게 하는 말을 쓰면 안 됩니다
- 상대 이름과 관계는 받은 그대로 쓰고, 이름을 줄이거나 애칭으로 바꾸지 마세요. 이름은 상대방과의 관계를 고려해서 작성하고, 만약 관계 정보가 제공되지 않았다면 "이름님" 으로 씁니다
- 친구와 같은 가까운 관계는 친근한 반말, 관계 정보가 제공되지 않았다면 따뜻하고 부담 없는 존댓말을 사용하세요.
- 아직 써 보기 전일 수 있으므로 "잘 쓰겠다", "기대된다" 같은 미래형 표현도 괜찮습니다.
- 답례는 아직 고르는 중이니 선물을 준비한다거나 주겠다는 말은 사용하지 말고, 선물에 대한 고마움 정도만 자연스러운 한국어로 표현하세요."""

# 청첩장은 "받은 선물" 이 아니라 "앞으로 참석하고 축의할 일정" 입니다.
# 이 안내가 없으면 모델이 사용자를 신랑신부 쪽으로 착각해 하객에게 감사하는 문장을 씁니다.
_INVITATION_NOTE = """
[중요] 사용자는 이 행사에 **초대받은 하객**입니다. 사용자가 주인공이 아닙니다.
- 메시지는 사용자가 주최자에게 보내는 **축하 인사**로 작성하세요.
- "참석해 주셔서 감사합니다" 처럼 주최자가 하객에게 하는 말은 절대 쓰지 마세요.
- 카테고리는 축의금과 함께 전하기 좋은 것으로 고르세요."""

# 부고장도 이미지 추출에서는 청첩장과 같은 event_invitation 으로 옵니다.
# 갈라 주지 않으면 유족에게 "진심으로 축하드려요!" 가 나갑니다.
_CONDOLENCE_INVITATION_NOTE = """
[중요] 이것은 부고 소식입니다. 사용자는 조문하는 쪽이고, 상을 당한 유족이 아닙니다.
- 메시지는 사용자가 유족에게 보내는 **조의 인사**입니다.
- 축하·기쁨·설렘을 나타내는 말과 느낌표, 이모지를 절대 쓰지 마세요.
- 짧고 담백한 존댓말로 쓰고, 고인이나 사고 경위를 캐묻지 마세요.
- "감사합니다" 는 소식을 알려 주신 것에 대해서만 쓰세요.
- 카테고리는 조의금과 함께 전할 수 있는 담백한 것으로 고르세요."""

# 조의금·부의금을 받은 쪽입니다. 답례 인사도 축하 문구가 섞이면 안 됩니다.
_CONDOLENCE_NOTE = """
[중요] 이번 일은 상을 당한 일이고, 상을 치른 쪽이 사용자 본인입니다.
- 메시지는 조의를 표해 주신 분께 드리는 **감사 인사**입니다.
- 축하·기쁨·설렘을 나타내는 말과 느낌표, 이모지를 절대 쓰지 마세요.
- 정중하고 담백한 존댓말로 쓰고, 경황이 없어 직접 인사드리지 못한 점을 함께 전하세요."""

_MULTI_NOTE = """
[중요] 여러 사람에게 한 번에 받았습니다. 사람마다 금액이 다르므로
가격 범위는 가장 적게 준 사람과 가장 많이 준 사람을 모두 감당할 수 있게 넓게 잡으세요.
메시지는 특정 한 사람이 아니라 여러 사람에게 두루 쓸 수 있는 표현으로 작성하세요."""


def build_recommendation_schema() -> dict:
    """vLLM ``response_format={"type": "json_schema"}`` 에 그대로 넣는 스키마.

    카테고리를 enum 으로 못박아 두면 모델이 목록 밖의 값을 만들 수 없습니다.
    ``recommendation_policy`` 의 보정은 그래도 남겨 두지만, 이 스키마가 있으면
    보정이 주 경로가 아니라 안전망으로 물러납니다.

    가격 범위와 상품 유형은 정책이 무조건 덮어쓰므로 아예 요구하지 않습니다.
    """
    return {
        "type": "object",
        "properties": {
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
                    },
                    "required": ["category", "score", "reason"],
                    "additionalProperties": False,
                },
            },
            "summary": {"type": "string"},
            # 폐기 기준(MIN_MESSAGE_LENGTH)이 아니라 사람이 읽기 좋은 목표를 요구합니다.
            # Bedrock 은 이 스키마를 프롬프트 텍스트로도 넣고(minLength 는 구조화 출력에
            # 실을 수 없습니다) 모델이 이 숫자를 그대로 읽습니다. 위 산문의 요구와 **같은 값**이어야
            # 합니다. 3차에는 산문 150 / 스키마 130 으로 갈려 있었고, 4차에는 둘 다 160
            # 이었는데 실측 9건 중 아무도 160 에 닿지 못했습니다.
            #
            # 5차 실측 4건: 109 · 126 · 138 · 148자, 4 · 5 · 4 · 4문장. 문장 수 요구는
            # 그때까지 "모델이 지키는 조건" 으로 적혀 있었지만 5차 4건 모두 5문장에
            # 못 미쳤습니다. 산문의 요구를 실측대로 4~6문장으로 내렸습니다(토큰 변화
            # 없음). 길이 140 은 그대로 둡니다. 4건 중 1건만 넘겼지만, 못 지킨 지시를
            # 지우면 요구가 함께 내려가 문장이 더 짧아질 뿐이고 그 결과를 확인할 실측이
            # 없습니다. 넷 다 폐기선 90 을 넉넉히 넘겨 사용자에게 나가는 데는 지장이
            # 없으므로, 숫자를 바꾸는 대신 실측을 여기 적어 둡니다.
            "suggested_message": {"type": "string", "minLength": TARGET_MESSAGE_LENGTH},
        },
        "required": [
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
    invitation = request.record_type == "event_invitation"
    if is_condolence(request.event, request.gift_name):
        system += _CONDOLENCE_INVITATION_NOTE if invitation else _CONDOLENCE_NOTE
    elif invitation:
        system += _INVITATION_NOTE
    # 여러 명에게 청첩장을 받는 일도 있으므로 위 안내와 배타적이지 않습니다.
    if len(request.received_amounts) > 1:
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
        # 시스템 프롬프트에서 예산 지시를 뺀 대신, 예산이 있을 때만 여기에 붙입니다.
        lines.append(f"사용자가 지정한 예산: {low} ~ {high} (이 가격대에서 살 수 있는 카테고리로 고르세요)")
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
