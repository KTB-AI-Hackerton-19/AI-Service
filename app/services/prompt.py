"""선물 추천 프롬프트와 강제 출력 스키마를 생성합니다.

이미지 추출이 ``vision_prompt`` 를 쓰는 것과 대칭입니다. 두 프롬프트 모두 같은
모델로 갑니다 — 기본은 Bedrock 의 Claude Sonnet 4.6 이고, ``MODEL_BACKEND=vllm``
이면 같은 vLLM 서버의 Gemma4-12B-QAT 입니다.

여기서 만드는 프롬프트는 두 경로가 공유합니다.
- 단일 호출: ``build_simple_messages`` + ``build_recommendation_schema``
- 분할 호출: ``build_plan_*`` / ``build_prose_*`` / ``build_message_*``
  (``recommendation_stages``, ``RECOMMENDATION_SPLIT_CALLS=true`` 일 때)

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

# 프롬프트를 네 조각으로 나눠 둔 이유: 분할 호출(``build_plan_messages`` 등)이
# 자기 단계에 필요한 규칙만 골라 싣습니다. 조각을 다시 이어 붙인 것이
# ``SIMPLE_SYSTEM_PROMPT`` 이므로 단일 호출 경로의 프롬프트는 한 글자도 달라지지
# 않습니다(``tests/test_prompt_split.py`` 가 이 동일성을 지킵니다).
_ROLE = """당신은 한국의 답례 선물 추천 전문가입니다.
사용자가 받은 것을 보고 답례로 줄 선물의 카테고리와 감사 메시지를 만드세요.
가격은 서비스가 규칙으로 계산하므로 당신은 금액을 정하거나 언급하지 않습니다.
나이·성별은 카테고리 선택에만 쓰고, 문장에 옮기거나 사람을 평가하듯 쓰지 마세요."""

_CATEGORY_RULES = f"""카테고리는 반드시 다음 목록에서만 점수가 높은 순서로 선택하세요:
[{_CATEGORY_LIST}]
summary 와 reason 에는 어떤 카테고리를 왜 권하는지만 쓰고, 특정 상품·브랜드·금액을 약속하지 마세요."""

_JSON_RULE = "반드시 마크다운 없이 JSON 객체 하나만 반환하세요."

_MESSAGE_RULES = """메시지는 다음 조건을 지키세요:
- 자연스러운 한국어로 답변하세요.
- **사용자 본인의 입장**에서 상대에게 직접 건네는 말입니다. 상대를 3인칭으로 서술하거나
  상대가 사용자에게 하는 말을 쓰면 안 됩니다
- 상대 이름과 관계는 받은 그대로 쓰고, 이름을 줄이거나 애칭으로 바꾸지 마세요. 이름은 상대방과의 관계를 고려해서 작성하고, 만약 관계 정보가 제공되지 않았다면 "이름님" 으로 씁니다
- 친구와 같은 가까운 관계는 친근한 반말, 관계 정보가 제공되지 않았다면 따뜻하고 부담 없는 존댓말을 사용하세요.
- 아직 써 보기 전일 수 있으므로 "잘 쓰겠다", "기대된다" 같은 미래형 표현도 괜찮습니다.
- 답례는 아직 고르는 중이니 선물을 준비한다거나 주겠다는 말은 사용하지 말고, 선물에 대한 고마움 정도만 자연스러운 한국어로 표현하세요."""

SIMPLE_SYSTEM_PROMPT = f"{_ROLE}\n{_CATEGORY_RULES}\n{_JSON_RULE}\n\n{_MESSAGE_RULES}"

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
            # minLength 는 구조화 출력에 실을 수 없어(bedrock_client 의
            # UNSUPPORTED_SCHEMA_KEYWORDS) schema_instruction 이 프롬프트 텍스트로
            # 넣는 경로로만 모델에 닿습니다.
            #
            # 지금은 이 숫자가 길이를 요구하는 **유일한** 자리입니다. 위 _MESSAGE_RULES
            # 에서 길이·문장 수 요구가 빠졌기 때문입니다. 그래서 실측 메시지 길이는
            # 이 값이 아니라 모델이 자연스럽게 쓰는 길이(약 125~135자)로 수렴합니다.
            # 길이를 다시 강제하려면 산문에도 같은 값을 적어야 하고, 그때는 두 곳이
            # 어긋나지 않게 이 상수 하나만 참조하세요.
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
    return [
        {"role": "system", "content": SIMPLE_SYSTEM_PROMPT + _notes(request)},
        {"role": "user", "content": "\n".join(_user_lines(request))},
    ]


def _notes(request: SimpleGiftRecommendationRequest) -> str:
    """상황별 안내문. 분할 호출도 이 함수 하나를 공유합니다.

    조의·청첩장 안내는 카테고리 지시와 메시지 지시를 함께 담고 있어 어느 단계에도
    통째로 붙입니다. 갈라 놓으면 안내가 늘 때마다 두 곳을 고쳐야 하고, 한쪽을
    빠뜨리면 유족에게 축하 문구가 나가는 종류의 사고가 조용히 돌아옵니다.
    입력 토큰만 늘 뿐 생성 시간에는 영향이 없습니다.
    """
    notes = ""
    invitation = request.record_type == "event_invitation"
    if is_condolence(request.event, request.gift_name):
        notes += _CONDOLENCE_INVITATION_NOTE if invitation else _CONDOLENCE_NOTE
    elif invitation:
        notes += _INVITATION_NOTE
    # 여러 명에게 청첩장을 받는 일도 있으므로 위 안내와 배타적이지 않습니다.
    if len(request.received_amounts) > 1:
        notes += _MULTI_NOTE
    return notes


def _user_lines(request: SimpleGiftRecommendationRequest) -> list[str]:
    """모델에게 넘길 입력 블록. 단일 호출과 분할 호출이 같은 값을 씁니다."""
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

    return lines


# ─────────────────────────── 분할 호출용 프롬프트 ───────────────────────────
#
# 왜 나누는가: 단일 호출은 카테고리 이름(약 20자)부터 감사 메시지(약 130자)까지
# 500자 안팎을 한 번에 씁니다. 그런데 상품 검색이 기다리는 것은 맨 앞의 카테고리
# **이름**뿐입니다(가격 범위는 ``recommendation_policy.price_range`` 가 규칙으로
# 정하고, 검색어 씨앗은 ``SAFE_EXAMPLES`` 에서 꺼냅니다). 나머지 480자를 다 쓸
# 때까지 검색이 출발하지 못하는 것이 지연의 대부분입니다.
#
# 그래서 셋으로 나눕니다.
#   plan    카테고리와 점수만. 검색은 이것만 나오면 출발할 수 있습니다.
#   prose   카테고리별 이유와 summary. 카테고리에 의존하므로 plan 뒤에 옵니다.
#   message 감사 메시지. 카테고리에 의존하지 않으므로 plan 과 동시에 출발합니다.
#
# message 를 떼어낼 수 있는 근거는 ``_MESSAGE_RULES`` 자신입니다. 마지막 줄이
# "답례는 아직 고르는 중이니 선물을 준비한다거나 주겠다는 말은 사용하지 말고" 라고
# 못박고 있어, 이 문장은 애초에 카테고리를 쓰면 안 되는 출력입니다. 즉 컨텍스트에서
# 카테고리를 빼는 것은 모델이 **써서는 안 되는 입력**을 빼는 것입니다.

# plan 에서 reason 을 받지 않는 이유(실측으로 뒤집은 판단):
#
# 처음에는 한 구절(20자)을 받았습니다. 단일 호출에서는 2·3순위 카테고리가 1순위의
# reason 을 보고 결정되므로, 점수만 받으면 그 자기회귀 조건이 사라진다고 봤습니다.
# "한 구절이면 시간에 거의 영향이 없다" 는 것이 그 판단의 전제였습니다.
#
# 실측은 달랐습니다. reason 이 있는 plan 이 134 토큰 / 3.92초였고, 그중 60 토큰
# 안팎이 reason 입니다. plan 은 **검색이 출발하기 전까지의 순수 대기**라 여기서
# 늘어난 1.6초가 그대로 응답 시간이 됩니다. 전체가 10초 안팎인 경로에서 1.6초는
# "거의 영향 없음" 이 아닙니다.
#
# 대신 조건화 손실은 벤치마크의 '카테고리 일치' 지표로 감시합니다. 단일 호출과
# 같은 입력에서 고른 카테고리가 갈리기 시작하면 이 결정을 되돌릴 자리입니다.
_PLAN_RULES = """카테고리를 최대 3개까지 점수가 높은 순서로 고르세요.
설명은 쓰지 마세요. 카테고리와 점수만 필요합니다."""

# 길이를 못박는 이유는 실측입니다. 처음에는 "2~3문장" 이라고만 썼는데 prose 가
# 444 토큰(8.2초)을 써서 단일 호출(512 토큰, 8.8초)과 거의 같아졌습니다. 나누는
# 의미가 사라진 자리입니다. 같은 입력에서 단일 호출이 자연스럽게 쓰던 이유는
# 평균 55자였으므로, 그 길이를 그대로 요구합니다.
_PROSE_RULES = """이미 선정된 카테고리에 대한 설명만 쓰세요. 카테고리를 새로 고르거나 바꾸지 마세요.
reason 은 그 카테고리를 왜 권하는지 50자 안팎 한 문장으로 적으세요.
summary 는 추천 전체를 두 문장 안으로, 100자 안팎으로 요약하세요."""

_MESSAGE_ROLE = """당신은 한국의 답례 선물 추천 전문가입니다.
사용자가 받은 것을 보고 상대에게 보낼 감사 메시지를 만드세요.
가격은 서비스가 규칙으로 계산하므로 당신은 금액을 정하거나 언급하지 않습니다.
나이·성별은 문장에 옮기거나 사람을 평가하듯 쓰지 마세요."""


def build_plan_schema() -> dict:
    """1단계: 카테고리와 점수만 받는 스키마. 상품 검색이 이 결과만으로 출발합니다."""
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
                    },
                    "required": ["category", "score"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["categories"],
        "additionalProperties": False,
    }


def build_prose_schema() -> dict:
    """2단계: 카테고리별 이유와 요약을 받는 스키마.

    카테고리를 다시 고르게 하지 않으려고 ``reasons`` 안의 ``category`` 도 enum 으로
    둡니다. 호출 측은 이 값을 **이름으로 대조해** 1단계 결과에 붙이므로, 모델이
    순서를 바꿔 내도 이유가 엉뚱한 카테고리에 붙지 않습니다.
    """
    return {
        "type": "object",
        "properties": {
            "reasons": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": list(ALLOWED_CATEGORIES)},
                        "reason": {"type": "string"},
                    },
                    "required": ["category", "reason"],
                    "additionalProperties": False,
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["reasons", "summary"],
        "additionalProperties": False,
    }


def build_message_schema() -> dict:
    """3단계: 감사 메시지만 받는 스키마.

    ``minLength`` 는 구조화 출력에 실을 수 없어 ``bedrock_client._drop_unsupported``
    가 걷어내지만, 같은 스키마가 ``schema_instruction`` 으로 프롬프트에도 들어가
    거기서는 살아 있습니다. 단일 호출과 같은 사정입니다.
    """
    return {
        "type": "object",
        "properties": {
            "suggested_message": {"type": "string", "minLength": TARGET_MESSAGE_LENGTH},
        },
        "required": ["suggested_message"],
        "additionalProperties": False,
    }


def build_plan_messages(
    request: SimpleGiftRecommendationRequest,
) -> list[dict[str, str]]:
    """1단계 메시지. 카테고리 규칙만 싣고 메시지 규칙은 뺍니다."""
    system = f"{_ROLE}\n{_CATEGORY_RULES}\n{_PLAN_RULES}\n{_JSON_RULE}" + _notes(request)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(_user_lines(request))},
    ]


def build_prose_messages(
    request: SimpleGiftRecommendationRequest,
    categories: list[dict],
) -> list[dict[str, str]]:
    """2단계 메시지. 1단계가 고른 카테고리를 입력으로 받습니다.

    Args:
        request: 1단계와 같은 추천 입력.
        categories: 1단계 결과. ``category`` 와 ``reason`` 을 씁니다.
    """
    system = f"{_ROLE}\n{_CATEGORY_RULES}\n{_PROSE_RULES}\n{_JSON_RULE}" + _notes(request)
    chosen = "\n".join(
        f"- {item.get('category')} (점수 {item.get('score')}): {item.get('reason', '')}".rstrip(": ")
        for item in categories
    )
    lines = _user_lines(request) + ["", "선정된 카테고리:", chosen]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def build_message_messages(
    request: SimpleGiftRecommendationRequest,
) -> list[dict[str, str]]:
    """3단계 메시지. 카테고리 규칙과 목록을 싣지 않습니다.

    ``_user_lines`` 는 그대로 씁니다. 거기 들어가는 "사용자가 고른 카테고리" 는
    모델이 만든 추천이 아니라 **사용자가 입력한 값**이라 감사 메시지 작성자가
    알아도 되는 맥락이고, 빼면 단일 호출과 입력이 달라집니다.
    """
    system = f"{_MESSAGE_ROLE}\n{_JSON_RULE}\n\n{_MESSAGE_RULES}" + _notes(request)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(_user_lines(request))},
    ]
