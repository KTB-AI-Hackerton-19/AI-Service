"""모델 출력을 Giftie의 가격·카테고리 안전 정책에 맞게 보정합니다."""

import hashlib
import logging
import re
from collections.abc import Sequence
from typing import Any

from app.schemas.recommendation import (
    MessageSource,
    ProductSuggestion,
    SimpleGiftRecommendationRequest,
)
from app.services.price_policy import ceil_price, floor_price

logger = logging.getLogger(__name__)

CATEGORY_ALIASES = {
    "식품/음료": "식품·디저트",
    "음식": "식품·디저트",
    "식품": "식품·디저트",
    "디저트": "식품·디저트",
    "커피": "커피·차",
    "디지털 기기": "디지털 액세서리",
    "전자기기": "디지털 액세서리",
    "패션": "패션·잡화",
    "화장품": "뷰티·화장품",
    "화장품·스킨케어": "뷰티·화장품",
    "스킨케어": "뷰티·화장품",
    "뷰티": "뷰티·화장품",
    "향수": "뷰티·화장품",
    "문화": "문화·취미",
    "취미": "문화·취미",
}
SAFE_EXAMPLES = {
    "식품·디저트": ["프리미엄 디저트 세트", "제철 과일 세트"],
    "커피·차": ["스페셜티 드립백 세트", "프리미엄 티 세트"],
    "생활용품": ["고급 타월 세트", "보온·보냉 텀블러"],
    "뷰티·화장품": ["핸드크림·립밤 세트", "향수 미니어처 세트"],
    "패션·잡화": ["카드지갑", "파우치·에코백"],
    "문화·취미": ["도서·문구 세트", "전시·공연 관람권"],
    "건강·웰니스": ["건강 간식 세트", "마사지·스트레칭 소품"],
    "꽃·식물": ["미니 꽃다발", "관리하기 쉬운 화분"],
    "상품권": ["외식 상품권", "문화생활 상품권"],
    "디지털 액세서리": ["휴대폰 거치대", "충전 케이블 세트"],
    "유아·아동": ["연령별 그림책", "창의 놀이 세트"],
}


ALLOWED_CATEGORIES = tuple(SAFE_EXAMPLES)
"""추천에 허용된 카테고리. 프롬프트와 구조화 출력 스키마가 이 목록 하나를 공유합니다."""

MIN_MESSAGE_LENGTH = 90
"""이보다 짧은 모델 메시지만 폐기합니다. **degenerate 방어선**이지 품질 기준이 아닙니다.

130 → 90. 130 은 목적을 넘어 정상 출력을 버리고 있었습니다. 4차 실측에서 모델이 쓴
109·113·118·121자 네 건이 전부 여기서 폐기돼 템플릿 비율이 4/4 가 됐습니다.
109자짜리 한국어 감사 메시지는 그 자체로 부족하지 않습니다.

이 값이 막아야 하는 것은 "감사합니다" 한 마디, 빈 문자열, 한 줄짜리입니다.
실측 9건에서 모델이 쓴 **문장 하나**의 최대 길이가 53자였으므로(v3 giftdata),
한 줄짜리는 아무리 길어도 이 선에 닿지 못합니다. 반대로 실측에서 가장 짧았던
정상 출력은 109자라 17% 여유가 남습니다. 빈 문자열은 위 분기가 따로 잡습니다.

측정 표본(모델이 쓴 것만, 템플릿 제외):
    109 · 113 · 118 · 121 (4차) · 130 · 133 · 146 (2차) · 138 · 143 (3차)
"""

TARGET_MESSAGE_LENGTH = 140
"""프롬프트가 모델에게 요구하는 길이. 폐기선과 **독립된** 값입니다.

``MIN + 30`` 뺄셈 결속을 걷어냅니다. 두 값은 서로 다른 질문에 답합니다.
MIN 은 "무엇이 degenerate 인가", TARGET 은 "무엇이 잘 읽히는가" 입니다.
결속을 두면 폐기선을 90 으로 내리는 순간 요구가 120 으로 함께 **내려갑니다**.
방향이 정반대라 뺄셈으로 묶을 수 있는 관계가 아닙니다.
(TARGET > MIN 이라는 구조적 요구는 남습니다. 테스트가 지킵니다.)

160 → 140. 160 은 실측 9건 중 **단 한 건도 도달하지 못한** 값이라 지시가 아니라
잡음이었습니다. 요구를 150(3차)에서 160(4차)으로 올렸더니 출력이 138·143 에서
109·121 로 오히려 짧아졌습니다. 글자 수는 모델이 셀 수 없는 단위라, 도달 불가능한
숫자를 주면 그 조건을 통째로 버리고 남은 **셀 수 있는** 조건("4~6문장")만 지킵니다.
140 은 실측이 두 번(143·146) 넘어선 값이라 지시로 기능할 수 있습니다.

실제 지렛대는 문장 수입니다. 같은 표본에서 4문장은 109~143자, 5문장은 138·146자로
갈렸습니다. 그래서 prompt.py 의 요구를 "5~6문장"으로 올렸습니다.
"""

_MIN_PRICE = 1_000

# ── summary 의 금액 언급 ───────────────────────────────────────────────────
# summary 는 상품 검색보다 **먼저** 만들어집니다. 그래서 모델이 summary 에 금액을
# 적으면 그건 실제 결과가 아니라 추측입니다. 실측에서 summary 는 "8,000~12,000원
# 범위의 상품권..." 이었는데 실제로 나간 상품은 35,000원 한 건이었고, 같은 응답의
# rationale.product_basis 는 "0개가 8,000원 ~ 12,000원 안에 듭니다",
# warnings 는 "1개는 제안 가격대를 벗어납니다" 였습니다. 한 응답 안에서 summary 만
# 다른 말을 한 것입니다.
#
# 프롬프트에도 금지 지시를 넣었지만(prompt.py), 구조화 출력은 키와 타입만 보장할 뿐
# 문장 안에 무엇을 쓰는지는 못 막습니다. 여기서 결정론적으로 한 번 더 막습니다.
# 금액은 recommended_price_min/max 와 rationale 이 사실대로 말하므로,
# summary 에서 지워도 사용자가 잃는 정보가 없습니다.
_MONEY_PATTERN = re.compile(r"\d[\d,]*\s*[만천억]?\s*원")
DEFAULT_SUMMARY = "받은 선물과 가격대를 고려한 답례 추천입니다."

# ── summary 의 카테고리 언급 ───────────────────────────────────────────────
# 금액과 같은 이유로 카테고리도 검색 전에는 알 수 없습니다. 실측에서 모델은
# 커피·차(85), 식품·디저트(70), 생활용품(60) 을 고르고 summary 에 "커피나 차
# 관련 제품으로 답례하는 것을 추천합니다" 라고 썼는데, 예산 안에 든 후보가 최저
# 점수 카테고리에만 남아 화면에는 생활용품 볼펜 한 개가 나갔습니다.
#
# 카테고리까지 금지하면 summary 에 남는 내용이 없으므로 금지하지 않습니다.
# 대신 검색이 끝난 **뒤에** 실제로 상품이 나온 카테고리를 한 문장 덧붙여,
# 헤드라인과 화면이 어긋난 채로 나가지 않게 합니다.
_SUMMARY_MAX_LENGTH = 500


def shipped_categories(products: Sequence[ProductSuggestion]) -> list[str]:
    """상품이 실제로 하나라도 나온 카테고리를 처음 나온 순서로 돌려줍니다.

    카테고리를 모르는 상품(``category is None``)은 셀 수 없으므로 뺍니다.
    """
    return list(dict.fromkeys(p.category for p in products if p.category))


def reconcile_summary(
    summary: str,
    categories: Sequence[str],
    products: Sequence[ProductSuggestion],
) -> str:
    """모델 summary 를 화면에 실제로 나가는 상품과 맞춥니다.

    상품이 추천 카테고리를 모두 덮으면 어긋날 것이 없으므로 그대로 둡니다.
    한 카테고리에서만 나왔거나 추천 밖 카테고리가 섞였을 때만 사실을 덧붙입니다.
    상품이 하나도 없으면 summary 가 반박당할 대상이 없고,
    ``rationale.product_basis`` 가 "검색 결과가 없어" 라고 이미 말합니다.
    """
    shipped = shipped_categories(products)
    if not shipped or set(shipped) == set(categories):
        return summary
    tail = f"이번에 찾은 상품은 {', '.join(shipped)}입니다."
    room = _SUMMARY_MAX_LENGTH - len(tail) - 1
    return f"{summary[:room].rstrip()} {tail}"


# ── "{이름}님께 …해 주셔서" ───────────────────────────────────────────────
# "주시-" 는 주는 사람을 높이는 형태라 주는 사람이 주어입니다. 그 자리에 여격
# 조사 "님께" 가 오면 "니니즈에게 선물해 주셔서" 로 읽혀 주체와 대상이 뒤집힙니다.
# 실측에서 같은 이미지·같은 요청인데 한 번은 "님께서", 한 번은 "님께" 였습니다.
# 모델 출력에 달린 확률 문제라 프롬프트로는 빈도만 낮출 뿐 없앨 수 없습니다.
#
# 교정 대상을 아는 이름으로 못박습니다. 아무 명사에나 걸면 같은 응답 안의 정상
# 문장("니니즈님께 감사의 마음을 담아 …")까지 바꿔 버립니다. "주셔/주셨/주신" 이
# 문장 부호를 넘지 않고 가까이 뒤따를 때만 고칩니다.
_GIVER_VERB_WINDOW = 30


def fix_giver_particle(text: str, person_name: str | None) -> str:
    """주는 사람을 가리키는 "{이름}님께" 를 "{이름}님께서" 로 바로잡습니다."""
    name = (person_name or "").strip()
    if not text or not name:
        return text
    pattern = re.compile(
        re.escape(name) + rf"님께(?!서)(?=[^.!?]{{0,{_GIVER_VERB_WINDOW}}}?주[셔셨신])"
    )
    return pattern.sub(f"{name}님께서", text)


# ── "{성을 뗀 이름}님" ────────────────────────────────────────────────────
# 3차 실측: person_name 이 "김민수" 인데 모델이 "민수님, ..." 으로 시작했습니다.
# 1차에도 같은 계열("김영삼" → "영삼이")이 있어 프롬프트에 이름을 줄이지 말라는
# 지시를 넣었지만, 3차에서도 4회 중 1회 어겼습니다. 확률 문제라 지시로는 못 없앱니다.
#
# 좁게 잡습니다. **아는 이름의 뒷부분이 "~님" 꼴로 나온 경우**만 되돌립니다.
#  - 이름이 실제로 "민수" 면 뗄 성이 없어 후보가 안 생기므로 손대지 않습니다.
#  - "춤추는 니니즈" 처럼 온전히 쓴 이름은 아래 정규식이 **긴 후보부터** 시도하므로
#    통째로 먹혀 그 자리에서 끝납니다. "춤추는 춤추는 니니즈님" 이 될 수 없습니다.
#  - 앞 글자가 한글·영문·숫자면 건너뜁니다. 이 한 줄이 "교수님"(→"교김민수님"),
#    "사장님", 그리고 동명이인 "박민수님" 을 전부 막습니다.
_NAME_BOUNDARY = r"(?<![0-9A-Za-z가-힣])"

# 한 글자 후보("수님")는 "교수님"·"부처님"류를 때리므로 두 글자부터 봅니다.
_MIN_NAME_FRAGMENT = 2


def _name_forms(name: str) -> list[str]:
    """온전한 이름을 맨 앞에 두고, 그 뒤로 성을 뗀 형태를 긴 것부터 늘어놓습니다."""
    tails = [name[i:] for i in range(1, len(name) - 1)]
    return [name] + [
        tail for tail in tails if len(tail) >= _MIN_NAME_FRAGMENT and not tail[0].isspace()
    ]


def fix_shortened_name(text: str, person_name: str | None) -> str:
    """"민수님" 처럼 줄여 부른 이름을 받은 그대로("김민수님")로 되돌립니다."""
    name = (person_name or "").strip()
    # 두 글자 이름은 뗄 성이 없습니다. 여기서 걸러야 실제 이름이 "민수" 인 사람을
    # 건드릴 여지가 아예 사라집니다.
    if not text or len(name) <= _MIN_NAME_FRAGMENT:
        return text
    forms = _name_forms(name)
    if len(forms) == 1:
        return text
    alternatives = "|".join(re.escape(form) for form in forms)
    return re.sub(rf"{_NAME_BOUNDARY}(?:{alternatives})님", f"{name}님", text)


def fix_person_name(text: str, person_name: str | None) -> str:
    """이름 관련 교정을 정해진 순서로 겁니다.

    줄여 부른 이름을 먼저 펴야 합니다. "민수님께 ... 주셔서" 를 그대로 두면
    ``fix_giver_particle`` 이 온전한 이름("김민수님께")을 찾다가 놓칩니다.
    """
    return fix_giver_particle(fix_shortened_name(text, person_name), person_name)


# ── "고마우시-" ────────────────────────────────────────────────────────────
# 5차 실측 gift 콜드: "따뜻한 마음 전해주셔서 고마우신데, 저도 …"
#
# "-시-" 는 주체를 높이는 어미라 "고마우시-" 는 상대를 고마움을 느끼는 쪽으로
# 만듭니다. 이 서비스가 만드는 문장은 언제나 **사용자가 상대에게** 고마움을
# 전하는 말이라(프롬프트: "사용자 본인의 입장에서 상대에게 직접 건네는 말"),
# 고마움의 주체는 항상 화자입니다. 그래서 이 자리의 "-시-" 는 늘 틀립니다.
#
# 프롬프트로 막지 않는 이유는 fix_giver_particle 과 같습니다. 어미 하나에 걸린
# 확률 문제라 지시는 빈도만 낮추고, 외부 호출 없이 확인할 수도 없습니다.
# 결정론적 치환은 여기서 바로 검증됩니다.
#
# 좁게 잡습니다. **"고맙다" 의 ㅂ불규칙 존칭 어간("고마우시")으로 시작하는
# 어형만** 바꿉니다. 이 어간은 "고맙다 + -시-" 로만 만들어지므로 다른 낱말을
# 때릴 수 없습니다. 표에 없는 어형은 건드리지 않고 그대로 둡니다 — 지어낸
# 교정으로 정상 문장을 망가뜨리느니 하나를 놓치는 편이 낫습니다.
#
# "감사하시-" 는 넣지 않았습니다. "감사하다" 는 "고맙다" 와 달리 상대를 주체로
# 세우는 쓰임("감사하신 분")이 굳어 있어 같은 규칙으로 다룰 수 없고, 실측에도
# 없습니다.
_WRONG_HONORIFIC = {
    "고마우신데": "고마운데",  # 실측된 어형
    "고마우셔서": "고마워서",
    "고마우시고": "고맙고",
    "고마우십니다": "고맙습니다",
    "고마우시지만": "고맙지만",
}
_WRONG_HONORIFIC_PATTERN = re.compile(
    "|".join(sorted(map(re.escape, _WRONG_HONORIFIC), key=len, reverse=True))
)


def fix_wrong_honorific(text: str) -> str:
    """고마움의 주체를 상대로 만드는 "고마우시-" 를 화자 기준으로 되돌립니다."""
    if not text:
        return text
    return _WRONG_HONORIFIC_PATTERN.sub(lambda m: _WRONG_HONORIFIC[m.group(0)], text)


def polish_message(text: str, person_name: str | None) -> str:
    """사용자에게 나갈 한국어 문장에 거는 교정을 한 자리에 모읍니다.

    모든 백엔드가 ``normalize_recommendation`` 을 거치므로, 여기 한 곳만 보면
    무엇이 문장에 손을 대는지 전부 알 수 있습니다.
    """
    return fix_wrong_honorific(fix_person_name(text, person_name))


# ── 조의 판정 ──────────────────────────────────────────────────────────────
# vision 추출은 청첩장과 부고장을 똑같은 event_invitation 으로 분류합니다.
# 계기 텍스트로 갈라내지 않으면 유족에게 "진심으로 축하드려요!" 가 그대로 나갑니다.
CONDOLENCE_KEYWORDS = (
    "조의",
    "부의",
    "부고",
    "근조",
    "조문",
    "문상",
    "빈소",
    "발인",
    "장례",
    "상례",
    "별세",
    "타계",
    "영면",
    "추모",
    "명복",
    "상갓집",
)
# "부친상", "조모상" 처럼 위 낱말이 하나도 없는 표기를 잡습니다.
_BEREAVEMENT_PATTERN = re.compile(
    # 띄어쓰기를 허용하면 "장인 상품권" 같은 이름까지 조의로 잡힙니다. 늘 붙여 씁니다.
    r"(부친|모친|선친|조부|조모|외조부|외조모|장인|장모|시부|시모|빙부|빙모|백부|숙부|고모|이모)상"
)


def is_condolence(*texts: str | None) -> bool:
    """계기·분류·이름 중 하나라도 조의 계열이면 참을 돌려줍니다."""
    joined = " ".join(text for text in texts if text)
    if not joined:
        return False
    if any(keyword in joined for keyword in CONDOLENCE_KEYWORDS):
        return True
    return bool(_BEREAVEMENT_PATTERN.search(joined))


# ── 한국어 조사 ────────────────────────────────────────────────────────────
# "사촌 형로서", "카테고리을(를)" 같은 문장이 그대로 사용자 화면에 나갑니다.


def has_final_consonant(word: str) -> bool:
    """마지막 글자에 받침이 있으면 참. 한글이 아니면 없는 것으로 봅니다."""
    text = (word or "").strip()
    if not text:
        return False
    last = text[-1]
    if "가" <= last <= "힣":
        return (ord(last) - 0xAC00) % 28 != 0
    return False


def object_particle(word: str) -> str:
    """을/를 을 받침으로 결정합니다."""
    return "을" if has_final_consonant(word) else "를"


def role_particle(word: str) -> str:
    """(으)로서 를 받침으로 결정합니다. ㄹ 받침은 "로서" 입니다("딸로서")."""
    text = (word or "").strip()
    last = text[-1] if text else ""
    if "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28 not in (0, 8):
        return "으로서"
    return "로서"


def price_range(request: SimpleGiftRecommendationRequest) -> tuple[int, int]:
    """답례 가격 범위를 정합니다.

    한 건이면 받은 금액의 80~120% 입니다. 여러 사람에게 받았다면 각 금액의
    최저 80% 부터 최고 120% 까지로 넓힙니다. 축의금을 5만원 준 사람과 20만원 준 사람에게
    같은 가격대를 권하면 한쪽에는 과하고 다른 쪽에는 모자라기 때문입니다.

    반올림은 ``price_policy`` 에 맡깁니다. 여기서 상한까지 내림하면 12,300원을 받았을 때
    상한이 14,000원(113.8%)이 되어 "80~120%" 라는 근거 문장이 거짓말이 됩니다.
    """
    # 사용자가 예산을 직접 지정했으면 그대로 씁니다. 받은 금액에서 유추할 이유가 없습니다.
    if request.budget_min is not None or request.budget_max is not None:
        minimum = max(request.budget_min or _MIN_PRICE, _MIN_PRICE)
        maximum = max(request.budget_max or minimum, minimum)
        return minimum, maximum

    amounts = [a for a in request.received_amounts if a > 0] or [request.gift_price]
    minimum = floor_price(min(amounts))
    return minimum, max(minimum, ceil_price(max(amounts)))


def normalize_recommendation(
    request: SimpleGiftRecommendationRequest,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    """가격을 안전 범위로 고정하고 허용된 카테고리와 예시만 반환합니다."""
    minimum, maximum = price_range(request)
    categories: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_categories = parsed.get("categories", [])
    if not isinstance(raw_categories, list):
        raw_categories = []

    for item in raw_categories:
        if not isinstance(item, dict):
            continue
        raw_category = str(item.get("category", "")).strip()
        category = CATEGORY_ALIASES.get(raw_category, raw_category)
        if category not in SAFE_EXAMPLES or category in seen:
            continue
        seen.add(category)
        try:
            score = int(item.get("score", 50))
        except (TypeError, ValueError):
            score = 50
        categories.append(
            {
                "category": category,
                "score": min(max(score, 0), 100),
                "reason": str(
                    item.get("reason", "관계와 가격대를 고려한 추천입니다.")
                )[:300],
                # 상품 유형은 모델에게 시키지 않고 여기서 채웁니다. 모델이 만들어도
                # 어차피 이 값으로 덮어썼고, 그 출력 토큰이 그대로 지연이었습니다.
                # tasks/recommendation.py 가 이 값을 상품 검색 씨앗으로 씁니다.
                "product_examples": SAFE_EXAMPLES[category],
                "search_query": str(
                    item.get(
                        "search_query",
                        f"{category} 답례 선물 {minimum}원 {maximum}원",
                    )
                )[:200],
            }
        )

    allowed = {CATEGORY_ALIASES.get(c, c) for c in request.preferred_categories}
    if allowed:
        narrowed = [c for c in categories if c["category"] in allowed]
        if narrowed:
            categories = narrowed

    # 점수가 높은 순으로 세웁니다. 이 순서는 화면 순서로 끝나지 않습니다.
    # tasks/recommendation.py 가 이 목록 그대로 상품 검색을 부르고, product_search 의
    # _interleave 가 카테고리별 결과를 이 순서로 번갈아 뽑으며, _select_by_price 의
    # 정렬이 안정 정렬이라 **조건이 같은 상품 사이에서는 이 순서가 그대로 남습니다**.
    # 즉 여기서 정렬하지 않으면 점수 60짜리 카테고리의 상품이 85짜리보다 앞에 설 수
    # 있습니다(실측: summary 는 "커피·차를 최우선", 첫 상품은 생활용품 볼펜).
    #
    # 자르기 전에 정렬해야 상위 3개가 "점수 상위 3개" 가 됩니다. 모델이 낸 순서대로
    # 자르면 4번째의 90점이 1번째의 50점에 밀려 사라집니다.
    # sorted 는 안정 정렬이라 동점이면 모델이 낸 순서를 그대로 둡니다.
    categories.sort(key=lambda c: -c["score"])

    if not categories:
        categories.append(
            {
                "category": "상품권",
                "score": 70,
                "reason": "취향 정보가 부족할 때 선택 실패 가능성이 낮습니다.",
                "product_examples": SAFE_EXAMPLES["상품권"],
                "search_query": f"답례 상품권 {minimum}원 {maximum}원",
            }
        )
    suggested_message = str(parsed.get("suggested_message", "")).strip()
    # 소형 모델이 지나치게 짧거나 문맥이 빈약한 문장을 만들면 안정적인
    # 장문 템플릿으로 교체해 사용자에게 항상 충분한 메시지를 제공합니다.
    message_source = MessageSource.MODEL
    if len(suggested_message) < MIN_MESSAGE_LENGTH:
        # 교체 사실을 응답까지 들고 나갑니다. 응답의 generated_by 는 추천
        # 백엔드(BEDROCK_CLAUDE)를 말할 뿐 이 교체를 드러내지 않습니다. 3차 실측에서
        # 4건 중 2건이 여기로 떨어졌는데 로그에도 응답에도 아무 흔적이 없어,
        # 폴백 문구가 모델 출력으로 오독됐습니다.
        #
        # 로그는 서버에만 남고 백엔드는 못 봅니다. 그래서 로그와 별개로
        # message_source 를 함께 냅니다.
        if suggested_message:
            message_source = MessageSource.TEMPLATE_TOO_SHORT
            logger.warning(
                "모델 메시지가 %d자로 degenerate 방어선(%d자)에 못 미쳐 기본 문구로 대체합니다. "
                "이 선은 한 줄짜리·빈 문장만 막으라고 있는 값이므로, 잦다면 "
                "TARGET_MESSAGE_LENGTH 를 올리지 말고 프롬프트의 문장 수 요구를 보세요.",
                len(suggested_message),
                MIN_MESSAGE_LENGTH,
            )
        else:
            # 모델을 아예 부르지 않는 mock 경로와 JSON 파싱 실패가 여기입니다.
            # 위와 원인이 다르므로 값도 나눠 둡니다. 둘을 한 값으로 묶으면
            # "프롬프트 길이를 올릴 일" 과 "형식이 깨진 일" 을 구분할 수 없습니다.
            message_source = MessageSource.TEMPLATE_NO_OUTPUT
        suggested_message = _default_message(request)

    return {
        "recommended_price_min": minimum,
        "recommended_price_max": maximum,
        "categories": categories[:3],
        "summary": _summary(request, parsed),
        # 모든 백엔드가 이 함수를 거치므로 문장 교정도 여기 한 곳에만 둡니다.
        "suggested_message": polish_message(
            suggested_message[:500], request.person_name
        ),
        "message_source": message_source,
    }


def _summary(request: SimpleGiftRecommendationRequest, parsed: dict[str, Any]) -> str:
    """모델 요약을 그대로 쓰되, 금액을 말하면 기본 문구로 바꿉니다.

    summary 는 검색 전에 생성되므로 금액을 알 수 없습니다. 적힌 금액은 추측이고,
    같은 응답의 ``rationale`` 이 실제 값을 말하므로 둘이 어긋납니다(_MONEY_PATTERN 주석).
    """
    summary = str(parsed.get("summary", "")).strip()
    if not summary or _MONEY_PATTERN.search(summary):
        return DEFAULT_SUMMARY
    return polish_message(summary[:500], request.person_name)


def _default_message(request: SimpleGiftRecommendationRequest) -> str:
    """모델 메시지가 없거나 너무 짧을 때 사용할 기본 문구.

    받은 것의 종류에 따라 문장이 달라야 합니다. 청첩장에 "선물해 주신 청첩장 고마웠어요"
    라고 쓰면 어색하고, 여러 사람에게 받았는데 한 사람 이름을 넣으면 나머지에게는 못 씁니다.
    조의는 유족에게 나가는 글이라 축하 문구가 섞이면 사고입니다.

    여섯 종 모두 한 문단 안에서 종결어미를 섞지 않습니다. 3차 실측 문장이
    "고마웠어요 → 좋았습니다 → 들어요 → 인사드릴게요" 로 해요체와 합쇼체를 오갔는데,
    그 문장은 모델이 아니라 이 파일이 만든 것이었습니다. 프롬프트를 늘릴 일이 아닙니다.
    """
    if is_condolence(request.event, request.gift_name):
        if request.record_type == "event_invitation":
            return _condolence_visit_message(request)
        return _condolence_thanks_message(request)
    if request.record_type == "event_invitation":
        return _invitation_message(request)
    if len(request.received_amounts) > 1:
        return _group_message(request)
    return _single_gift_message(request)


# ── 폴백 문장의 다양성 ────────────────────────────────────────────────────
# 3차 실측에서 서로 다른 두 요청(선물·관계가 다름)이 이름과 품목만 바뀐 **같은
# 문장**으로 나갔습니다. 둘 다 이 함수의 출력이었습니다. 데모에서 두 응답을 나란히
# 놓으면 즉시 들통납니다.
#
# 폐기선을 90 으로 내려 폴백 빈도 자체는 크게 줄었지만, 파싱 실패나 모델 장애로
# 폴백만 나가는 순간은 남습니다. 그때 전부 같은 문장이면 안 되므로 틀 자체를 늘립니다.
#
# 고르는 기준은 입력의 해시입니다. 난수로 고르면 같은 요청을 두 번 보냈을 때 답이
# 달라져 재현이 안 되고 테스트도 못 씁니다. 파이썬 내장 ``hash`` 는 프로세스마다
# 시드가 달라(PYTHONHASHSEED) 서버를 재시작하면 문장이 바뀌므로 쓰지 않습니다.
#
# 한 변형 안에서는 종결어미를 섞지 않습니다. 실측 문장이
# "고마웠어요 → 좋았습니다 → 들어요" 로 해요체와 합쇼체를 오갔습니다.
def _variant(request: SimpleGiftRecommendationRequest, count: int) -> int:
    """같은 입력이면 항상 같은 번호를 주는 안정적인 선택."""
    key = "\x1f".join(
        (request.person_name or "", request.gift_name or "", request.relationship or "")
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % count


def _single_gift_message(request: SimpleGiftRecommendationRequest) -> str:
    """한 사람에게 선물이나 현금을 받은 기본 경우.

    품목을 가리지 않아야 합니다. "덕분에 잘 사용하고 있고, 볼 때마다" 는 케이크와
    커피 기프티콘에는 맞지 않고, 답례도 아직 준비하지 않았으므로 완료형으로 쓰지 않습니다.
    """
    greeting = f"{request.person_name}님, " if request.person_name else ""
    received = (request.gift_name or "").strip()
    if not received or received == "받은 선물":
        received = "선물"
    # 관계가 있으면 "친구로서" 처럼 조사를 받침에 맞춰 붙입니다. 모든 변형이 같은
    # 자리를 쓰므로, 변형을 더 늘려도 조사 처리가 한 곳에 남습니다.
    rel = request.relationship
    variants = SINGLE_GIFT_VARIANTS
    return variants[_variant(request, len(variants))](greeting, received, rel)


def _thanks_warm(greeting: str, received: str, rel: str | None) -> str:
    """해요체. 3차 실측에 나갔던 틀에서 종결어미만 해요체로 통일했습니다."""
    context = (
        f"늘 {rel}{role_particle(rel)} 살뜰히 챙겨 주시는 마음이 느껴져서"
        if rel
        else "세심하게 신경 써 주신 마음이 느껴져서"
    )
    return (
        f"{greeting}지난번에 {received} 챙겨 주셔서 정말 고마웠어요. "
        f"{context} 받고 나서도 한참 기분이 좋았어요. "
        "덕분에 요즘 하루하루 더 힘이 나고, 문득 생각날 때마다 고마운 마음이 들어요. "
        "저도 그 마음에 보답하고 싶어 답례를 고르고 있으니, 조만간 얼굴 보고 인사드릴게요."
    )


def _thanks_formal(greeting: str, received: str, rel: str | None) -> str:
    """합쇼체."""
    context = (
        f"{rel}{role_particle(rel)} 늘 마음을 써 주시는 분이라"
        if rel
        else "무엇 하나 허투루 넘기지 않고 챙겨 주셔서"
    )
    return (
        f"{greeting}{received} 정말 잘 받았습니다. "
        f"{context} 마음까지 함께 받은 것 같아 하루 종일 든든했습니다. "
        "바쁘신 중에도 저를 먼저 떠올려 주셨다는 것이 오래 기억에 남습니다. "
        "저도 그 마음에 답하고 싶어 어떤 것이 좋을지 천천히 살펴보고 있습니다. "
        "정리되는 대로 다시 연락드리겠습니다."
    )


def _thanks_delighted(greeting: str, received: str, rel: str | None) -> str:
    """해요체."""
    context = (
        f"{rel}{role_particle(rel)} 늘 살뜰히 신경 써 주시는 게 느껴져서"
        if rel
        else "바쁜 와중에 따로 시간을 내 주신 것이 느껴져서"
    )
    return (
        f"{greeting}얼마 전에 {received} 받고 정말 반가웠어요. "
        f"{context} 그날 하루가 내내 따뜻했어요. "
        "이렇게 잊지 않고 마음 써 주시는 분이 곁에 있다는 게 새삼 고맙게 느껴져요. "
        "저도 그 고마움을 어떻게 전할지 즐겁게 고민하고 있어요. "
        "정리되면 얼굴 뵙고 직접 인사드릴게요."
    )


def _thanks_steady(greeting: str, received: str, rel: str | None) -> str:
    """합쇼체."""
    context = (
        f"{rel}{role_particle(rel)} 챙겨 주시는 마음이 한결같아서"
        if rel
        else "먼저 마음 써 주신 것이 그대로 느껴져서"
    )
    return (
        f"{greeting}{received} 보내 주셔서 진심으로 감사합니다. "
        f"{context} 받아 든 순간 마음이 참 든든했습니다. "
        "이렇게 마음을 전해 주시니 요즘 부쩍 기운이 납니다. "
        "저도 같은 마음을 돌려 드리고 싶어 무엇이 좋을지 찾아보는 중입니다. "
        "정해지면 자리 만들어 인사드리겠습니다."
    )


SINGLE_GIFT_VARIANTS = (_thanks_warm, _thanks_formal, _thanks_delighted, _thanks_steady)
"""폴백 문장 틀. 변형을 늘리면 테스트가 자동으로 새 변형까지 검사합니다."""


def _invitation_message(request: SimpleGiftRecommendationRequest) -> str:
    """청첩장·초대장을 받은 경우. 사용자는 주인공이 아니라 하객입니다."""
    greeting = f"{request.person_name}님, " if request.person_name else ""
    occasion = request.event or "좋은 소식"
    return (
        f"{greeting}{occasion} 소식 전해 주셔서 정말 기뻤어요. "
        "정성스럽게 준비하신 초대장도 잘 받았어요. "
        "잊지 않고 마음 써서 알려 주신 덕분에 저까지 하루 종일 설렜어요. "
        "소중한 자리에 함께할 수 있어 영광이에요. "
        "그날까지 준비하시느라 바쁘시겠지만 건강 꼭 챙기시고, "
        "좋은 모습으로 뵐게요. 진심으로 축하드려요!"
    )


def _condolence_visit_message(request: SimpleGiftRecommendationRequest) -> str:
    """부고를 받은 경우. 유족에게 보내는 글이라 과장·이모지·느낌표를 쓰지 않습니다."""
    greeting = f"{request.person_name}님, " if request.person_name else ""
    return (
        f"{greeting}삼가 조의를 표합니다. 소식을 듣고 마음이 많이 무거웠습니다. "
        "경황 없으신 중에도 알려 주셔서 감사합니다. "
        "상심이 크시겠지만 끼니 거르지 마시고 건강 잘 챙기시길 바랍니다. "
        "빈소에 들러 조금이나마 곁을 지키겠습니다. 곧 찾아뵙겠습니다."
    )


def _condolence_thanks_message(request: SimpleGiftRecommendationRequest) -> str:
    """조의를 받은 경우. 사용자가 상을 치른 쪽이고, 받은 마음에 답하는 글입니다."""
    single = len(request.received_amounts) <= 1
    greeting = f"{request.person_name}님, " if request.person_name and single else ""
    return (
        f"{greeting}지난 장례 기간 동안 보내 주신 위로와 마음, 깊이 감사드립니다. "
        "경황이 없어 직접 찾아뵙고 인사드리지 못한 점 너그러이 헤아려 주시기 바랍니다. "
        "보내 주신 마음 덕분에 큰 힘을 얻어 무사히 장례를 마쳤습니다. "
        "감사한 마음 오래 기억하겠습니다. 늘 건강하시길 바랍니다."
    )


def _group_message(request: SimpleGiftRecommendationRequest) -> str:
    """여러 사람에게 받은 경우. 특정 이름 없이 두루 쓸 수 있어야 합니다."""
    occasion_context = f"{request.event} 때 " if request.event else ""
    return (
        f"{occasion_context}보내 주신 따뜻한 마음 덕분에 정말 큰 힘을 얻었습니다. "
        "바쁘신 중에도 이렇게 챙겨 주셔서 진심으로 감사드립니다. "
        "덕분에 잘 지내고 있고, 그 마음 오래 기억하겠습니다. "
        "조만간 감사한 마음을 담아 작은 인사라도 전할 수 있으면 좋겠습니다."
    )
