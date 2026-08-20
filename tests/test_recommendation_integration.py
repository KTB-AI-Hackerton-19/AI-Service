"""추천이 이미지 분석과 같은 엔진·같은 규칙으로 통합됐는지 확인합니다."""

import logging
import os
import re
import subprocess
import sys

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")

from datetime import date

import pytest

from app.schemas.agent import GiftData, GiftRecordItem
from app.schemas.recommendation import (
    CategoryRecommendation,
    Gender,
    MessageSource,
    ProductSuggestion,
    SimpleGiftRecommendationRequest,
    SimpleGiftRecommendationResponse,
)
from app.services.price_policy import calculate_recommended_price_range
from app.services.prompt import build_recommendation_schema, build_simple_messages
from app.services.recommendation_policy import (
    ALLOWED_CATEGORIES,
    CATEGORY_ALIASES,
    DEFAULT_SUMMARY,
    MIN_MESSAGE_LENGTH,
    SAFE_EXAMPLES,
    TARGET_MESSAGE_LENGTH,
    SINGLE_GIFT_VARIANTS,
    fix_giver_particle,
    fix_person_name,
    fix_shortened_name,
    fix_wrong_honorific,
    normalize_recommendation,
    price_range,
    reconcile_summary,
)
from app.services.product_search import SearchStats
from app.services.tasks.recommendation import (
    RecommendationPreparationService,
    build_request,
    recommendation_preparation_service,
)


def gift(**kwargs) -> GiftData:
    base = {
        "gift_name": "스타벅스 아이스 아메리카노",
        "gift_price": 12300,
        "person_name": "김수현",
        "relationship": "대학 동기",
        "received_at": date(2026, 5, 9),
    }
    base.update(kwargs)
    return GiftData(**base)


def record(name: str, price: int | None, direction: str = "received", **kwargs) -> GiftRecordItem:
    return GiftRecordItem(
        record_id=f"r-{name}",
        record_type="money",
        direction=direction,
        person_name=name,
        gift_name="축의금",
        price=price,
        confidence=1.0,
        **kwargs,
    )


class TestSchemaAndPromptShareOneCategoryList:
    # 백엔드가 저장하는 카테고리 목록입니다. 여기 없는 값은 백엔드에서 "기타" 로
    # 떨어지므로, 이 목록에서 벗어나는 순간 분류가 조용히 사라집니다.
    BACKEND_CATEGORIES = {"디저트", "꽃·식물", "패션·잡화", "상품권", "생활용품"}

    def test_the_allowed_list_is_what_the_backend_stores(self):
        """추천도 기록도 이 목록 하나에 맞춥니다(``gift_data_policy`` 도 같은 값을 씁니다).

        "기타" 는 여기 없습니다. 그 이름으로는 상품을 검색할 수 없고, 기록 쪽도
        매칭에 실패하면 모델 원문을 그대로 넘겨 백엔드가 스스로 기타로 분류합니다.
        """
        assert set(ALLOWED_CATEGORIES) == self.BACKEND_CATEGORIES

    def test_every_allowed_category_has_search_seeds(self):
        """씨앗이 없으면 검색이 카테고리명만으로 돌아갑니다."""
        for category in ALLOWED_CATEGORIES:
            assert SAFE_EXAMPLES[category]

    def test_every_alias_resolves_into_the_allowed_list(self):
        """목록 밖을 가리키는 별칭은 조용히 버려집니다."""
        for source, target in CATEGORY_ALIASES.items():
            assert target in ALLOWED_CATEGORIES, source

    def test_schema_enum_matches_policy(self):
        """프롬프트와 스키마가 각자 목록을 들고 있으면 반드시 어긋납니다."""
        schema = build_recommendation_schema()
        enum = schema["properties"]["categories"]["items"]["properties"]["category"]["enum"]
        assert enum == list(ALLOWED_CATEGORIES)

    def test_prompt_lists_the_same_categories(self):
        messages = build_simple_messages(SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000))
        system = messages[0]["content"]
        for category in ALLOWED_CATEGORIES:
            assert category in system


class TestOptionalGender:
    def test_gift_gender_is_forwarded_to_recommendation_request(self):
        request = build_request(gift(gender="female"))
        assert request.gender is Gender.FEMALE

    def test_prompt_uses_gender_when_provided(self):
        request = build_request(gift(gender="male"))
        user_prompt = build_simple_messages(request)[1]["content"]
        assert "받는 사람 성별: 남성" in user_prompt

    def test_prompt_omits_gender_when_missing(self):
        request = build_request(gift(gender=None))
        user_prompt = build_simple_messages(request)[1]["content"]
        assert "받는 사람 성별: 제공되지 않음" in user_prompt

    def test_schema_requires_message_and_summary(self):
        required = build_recommendation_schema()["required"]
        assert "suggested_message" in required
        assert "summary" in required


class TestPriceRange:
    def test_single_record_keeps_80_120(self):
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=50000)
        assert price_range(req) == (40000, 60000)

    def test_multiple_records_span_min_to_max(self):
        """5만원 준 사람과 20만원 준 사람에게 같은 가격대를 권하면 한쪽은 과합니다."""
        req = SimpleGiftRecommendationRequest(
            gift_name="축의금", gift_price=200000, received_amounts=[50000, 100000, 200000]
        )
        assert price_range(req) == (40000, 240000)

    def test_zero_amounts_are_ignored(self):
        req = SimpleGiftRecommendationRequest(
            gift_name="축의금", gift_price=50000, received_amounts=[0, 0]
        )
        assert price_range(req) == (40000, 60000)

    def test_low_price_keeps_a_usable_span(self):
        """1,000원 하한을 걸면 저가 선물의 범위가 한 점으로 붙습니다.

        받은 금액 경로에는 하한을 걸지 않습니다. 대신 100원 단위로 계산해
        400원 ~ 600원처럼 80~120%가 그대로 살아 있는 범위를 냅니다.
        (사용자가 예산을 직접 지정한 경로에는 1,000원 하한이 그대로 있습니다.)
        """
        req = SimpleGiftRecommendationRequest(gift_name="사탕", gift_price=500)
        assert price_range(req) == (400, 600)

    def test_user_budget_still_has_a_floor(self):
        req = SimpleGiftRecommendationRequest(gift_name="사탕", budget_min=500, budget_max=3000)
        assert price_range(req) == (1000, 3000)


class TestPriceRangeReallyIs80To120:
    """근거 문장이 "80% ~ 120%" 라고 단언하므로 계산도 그래야 합니다.

    상한까지 내림하던 시절 3,000원은 2,000~3,000원(66.7%~100%),
    12,300원은 9,000~14,000원(73.2%~113.8%)이 나왔습니다. 딱 떨어지는
    50,000원만 검증해서 이 결함이 오래 남아 있었습니다.
    """

    def test_low_price_3000(self):
        req = SimpleGiftRecommendationRequest(gift_name="사탕 한 봉지", gift_price=3000)
        assert price_range(req) == (2400, 3600)

    def test_non_round_price_12300(self):
        req = SimpleGiftRecommendationRequest(gift_name="아이스 아메리카노", gift_price=12300)
        assert price_range(req) == (9000, 15000)

    def test_round_price_50000(self):
        req = SimpleGiftRecommendationRequest(gift_name="상품권", gift_price=50000)
        assert price_range(req) == (40000, 60000)

    @pytest.mark.parametrize("price", [3000, 12300, 50000, 23333, 1101, 200000])
    def test_ceiling_never_falls_below_120_percent(self, price):
        """상한을 내림하면 "120%까지"라고 쓴 근거가 거짓이 됩니다."""
        req = SimpleGiftRecommendationRequest(gift_name="선물", gift_price=price)
        low, high = price_range(req)
        assert high >= price * 1.2
        assert low <= price * 0.8

    def test_policy_and_mock_agree(self):
        """mock 추천과 실사용 경로가 같은 계산을 써야 데모에서 값이 갈리지 않습니다."""
        for price in (3000, 12300, 50000):
            req = SimpleGiftRecommendationRequest(gift_name="선물", gift_price=price)
            assert price_range(req) == calculate_recommended_price_range(price)


class TestBuildRequestFromGiftData:
    def test_single_record_sends_no_multi_context(self):
        req = build_request(gift())
        assert req.received_amounts == []
        assert req.people == []
        assert req.gift_price == 12300

    def test_multi_record_sends_amounts_and_people(self):
        data = gift(
            gift_name="축의금",
            gift_price=200000,
            person_name="최은비",
            records=[
                record("김도윤", 100000),
                record("박서준", 50000),
                record("최은비", 200000),
            ],
        )
        req = build_request(data)

        assert req.received_amounts == [100000, 50000, 200000]
        assert req.people == ["김도윤", "박서준", "최은비"]
        assert price_range(req) == (40000, 240000)

    def test_sent_records_are_excluded(self):
        """거래내역의 출금 건은 답례 대상이 아닙니다."""
        data = gift(
            gift_name="축의금",
            gift_price=200000,
            records=[
                record("김도윤", 100000),
                record("최은비", 200000),
                record("카카오페이", 38900, direction="sent"),
            ],
        )
        req = build_request(data)
        assert req.received_amounts == [100000, 200000]
        assert "카카오페이" not in req.people

    def test_deselected_records_are_excluded(self):
        data = gift(
            gift_name="축의금",
            gift_price=200000,
            records=[
                record("김도윤", 100000),
                record("박서준", 50000, selected=False),
                record("최은비", 200000),
            ],
        )
        req = build_request(data)
        assert req.received_amounts == [100000, 200000]

    def test_record_type_and_event_are_passed(self):
        data = gift(record_type="event_invitation", event="결혼")
        req = build_request(data)
        assert req.record_type == "event_invitation"
        assert req.event == "결혼"


class TestPromptContext:
    def test_invitation_gets_guest_side_instruction(self):
        """청첩장을 받은 사용자는 하객입니다. 신랑신부가 아닙니다."""
        req = SimpleGiftRecommendationRequest(
            gift_name="결혼 청첩장", gift_price=50000, record_type="event_invitation"
        )
        system = build_simple_messages(req)[0]["content"]
        assert "초대받은 하객" in system
        assert "참석해 주셔서 감사합니다" in system  # 금지 예시로 명시돼 있어야 함

    def test_multi_record_gets_multi_instruction(self):
        req = SimpleGiftRecommendationRequest(
            gift_name="축의금", gift_price=200000, received_amounts=[50000, 200000]
        )
        system = build_simple_messages(req)[0]["content"]
        assert "여러 사람에게 한 번에 받았습니다" in system

    def test_single_gift_gets_neither(self):
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        system = build_simple_messages(req)[0]["content"]
        assert "초대받은 하객" not in system
        assert "여러 사람에게 한 번에 받았습니다" not in system

    def test_user_writes_the_message_not_the_counterpart(self):
        """실측에서 모델이 상대방 입장으로 메시지를 쓰는 혼동이 있었습니다."""
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        system = build_simple_messages(req)[0]["content"]
        assert "사용자 본인의 입장" in system

    def test_multi_record_user_message_lists_people(self):
        req = SimpleGiftRecommendationRequest(
            gift_name="축의금",
            gift_price=200000,
            received_amounts=[100000, 200000],
            people=["김도윤", "최은비"],
        )
        user = build_simple_messages(req)[1]["content"]
        assert "김도윤 100,000원" in user
        assert "최은비 200,000원" in user


@pytest.mark.parametrize("record_type", ["gift", "money", "receipt", "unknown"])
def test_non_invitation_types_do_not_get_guest_note(record_type):
    req = SimpleGiftRecommendationRequest(
        gift_name="선물", gift_price=30000, record_type=record_type
    )
    assert "초대받은 하객" not in build_simple_messages(req)[0]["content"]


class TestCategoriesAreOrderedByScore:
    """이 순서는 화면 순서로 끝나지 않고 상품 순서까지 정합니다.

    tasks/recommendation.py 가 이 목록 그대로 검색을 부르고, product_search 의
    _interleave 가 이 순서로 결과를 번갈아 뽑으며, _select_by_price 의 정렬이 안정
    정렬이라 가격 조건이 같은 상품 사이에서는 이 순서가 그대로 남습니다.
    """

    @staticmethod
    def req() -> SimpleGiftRecommendationRequest:
        return SimpleGiftRecommendationRequest(gift_name="커피 금액권", gift_price=10000)

    def test_a_lower_scoring_category_does_not_lead(self):
        """실측 카테고리 점수(커피·차 85 / 식품·디저트 70 / 생활용품 60)입니다.

        커피·차와 식품·디저트는 백엔드 목록에 맞추면서 둘 다 디저트로 접혔습니다.
        정렬을 보는 테스트라 세 카테고리가 서로 달라야 해서, 실측 점수는 그대로 두고
        이름만 지금 목록의 셋으로 옮깁니다.
        """
        parsed = {
            "categories": [
                {"category": "생활용품", "score": 60, "reason": "ㄱ"},
                {"category": "디저트", "score": 85, "reason": "ㄴ"},
                {"category": "꽃·식물", "score": 70, "reason": "ㄷ"},
            ]
        }
        result = normalize_recommendation(self.req(), parsed)

        assert [c["category"] for c in result["categories"]] == [
            "디저트",
            "꽃·식물",
            "생활용품",
        ]

    def test_the_top_three_are_the_top_three_by_score(self):
        """자르기 전에 정렬해야 4번째의 90점이 1번째의 50점에 밀려 사라지지 않습니다."""
        parsed = {
            "categories": [
                {"category": "생활용품", "score": 50, "reason": "ㄱ"},
                {"category": "패션·잡화", "score": 55, "reason": "ㄴ"},
                {"category": "꽃", "score": 60, "reason": "ㄷ"},
                {"category": "디저트", "score": 90, "reason": "ㄹ"},
            ]
        }
        result = normalize_recommendation(self.req(), parsed)

        # "꽃" 은 CATEGORY_ALIASES 가 "꽃·식물" 로 정규화합니다.
        assert [c["category"] for c in result["categories"]] == [
            "디저트",
            "꽃·식물",
            "패션·잡화",
        ]

    def test_a_tie_keeps_the_order_the_model_gave(self):
        parsed = {
            "categories": [
                {"category": "디저트", "score": 80, "reason": "ㄱ"},
                {"category": "생활용품", "score": 80, "reason": "ㄴ"},
            ]
        }
        result = normalize_recommendation(self.req(), parsed)

        assert [c["category"] for c in result["categories"]] == ["디저트", "생활용품"]


class TestModelIsNotAskedForDiscardedOutput:
    """모델이 만들어도 정책이 무조건 덮어쓰는 필드는 요구하지 않습니다.

    Bedrock 경로는 출력 토큰이 그대로 생성 시간이라 순수한 지연 낭비입니다.
    """

    def test_schema_does_not_ask_for_the_price_range(self):
        schema = build_recommendation_schema()
        assert "recommended_price_min" not in schema["properties"]
        assert "recommended_price_max" not in schema["properties"]
        assert "recommended_price_min" not in schema["required"]

    def test_schema_does_not_ask_for_product_examples(self):
        item = build_recommendation_schema()["properties"]["categories"]["items"]
        assert "product_examples" not in item["properties"]

    def test_prompt_does_not_ask_the_model_to_set_prices(self):
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        system = build_simple_messages(req)[0]["content"]
        assert "80%~120%" not in system
        assert "금액을 정하거나 언급하지 않습니다" in system

    def test_policy_still_fills_product_examples(self):
        """상품 검색(tasks/recommendation.py)이 이 값을 검색어 씨앗으로 읽습니다.

        모델에게 시키지 않을 뿐, 키가 사라지면 검색이 카테고리명만으로 돌아갑니다.
        """
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        parsed = {"categories": [{"category": "디저트", "score": 90, "reason": "이유"}]}
        result = normalize_recommendation(req, parsed)

        assert result["categories"][0]["product_examples"] == SAFE_EXAMPLES["디저트"]

    def test_fallback_category_also_has_examples(self):
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        result = normalize_recommendation(req, {})
        assert result["categories"][0]["product_examples"] == SAFE_EXAMPLES["상품권"]


class TestMessageLengthThresholdIsOneNumber:
    """폐기선과 모델에게 요구하는 길이는 **다른 숫자**이고, 서로 독립입니다.

    산문과 스키마가 각자 숫자를 들면(3차: 산문 150 / 스키마 130) 모델은 둘 중
    아무거나 따릅니다. 그래서 모델에게 나가는 두 자리는 TARGET 하나에서 나옵니다.

    반대로 폐기선(MIN)까지 같은 식에 묶으면 안 됩니다. MIN 은 "무엇이 degenerate
    인가", TARGET 은 "무엇이 잘 읽히는가" 라 방향이 반대입니다. 4차에는 둘을
    ``MIN + 30`` 으로 묶어 요구를 160 으로 올렸는데, 실측 출력이 138·143자에서
    109~121자로 **짧아지고** 템플릿 비율이 2/4 에서 4/4 가 됐습니다.
    """

    def test_schema_asks_for_the_target_not_the_floor(self):
        schema = build_recommendation_schema()
        assert schema["properties"]["suggested_message"]["minLength"] == TARGET_MESSAGE_LENGTH

    def test_the_target_stays_above_the_discard_floor(self):
        """스키마가 폐기선 아래를 요구하면 규격을 지킨 출력이 버려집니다."""
        assert TARGET_MESSAGE_LENGTH > MIN_MESSAGE_LENGTH

    def test_the_two_numbers_are_not_bound_by_arithmetic(self):
        """뺄셈으로 묶으면 폐기선을 내리는 순간 요구가 함께 **내려갑니다**.

        90 + 30 = 120 이 아니어야 합니다. 4차의 ``TARGET = MIN + 30`` 이 그
        결속이었고, 두 값이 답하는 질문이 달라 같이 움직여서는 안 됩니다.
        """
        assert TARGET_MESSAGE_LENGTH != MIN_MESSAGE_LENGTH + 30

    def test_the_prompt_asks_for_a_sentence_count_the_model_can_count(self):
        """실측에서 길이를 가른 것은 글자 수가 아니라 문장 수였습니다.

        같은 표본에서 4문장은 109~143자, 5문장은 138·146자였습니다. 글자는 모델이
        셀 수 없고 문장은 셀 수 있으므로, 분량을 늘리는 지렛대는 이쪽입니다.

        요구는 5차 실측(4 · 5 · 4 · 4문장)에 맞춰 4~6문장입니다. 예전 "5~6문장" 은
        4건 모두가 어긴 숫자였고, 지시와 실제가 어긋난 채 남으면 다음 사람이 이
        값을 "지켜지는 조건" 으로 읽습니다. 넷 다 폐기선 90 을 넘겨 사용자에게는
        영향이 없었습니다.
        """
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        system = build_simple_messages(req)[0]["content"]
        assert "4~6문장" in system

    def test_the_sentence_floor_is_not_above_what_the_model_produces(self):
        """5차 실측 최솟값이 4문장입니다. 요구가 그보다 높으면 아무도 못 지킵니다."""
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        system = build_simple_messages(req)[0]["content"]
        floor = int(re.search(r"한국어 (\d+)~\d+문장", system).group(1))
        assert floor <= 4

    def test_the_prose_and_the_schema_quote_the_same_number(self):
        """산문이 "150자", 스키마가 "160자" 면 모델은 둘 중 아무거나 따릅니다."""
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        system = build_simple_messages(req)[0]["content"]
        assert f"{TARGET_MESSAGE_LENGTH}~250자" in system

    def test_message_one_char_short_is_discarded(self):
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        short = "가" * (MIN_MESSAGE_LENGTH - 1)
        assert normalize_recommendation(req, {"suggested_message": short})["suggested_message"] != short

    def test_message_at_the_threshold_is_kept(self):
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        exact = "가" * MIN_MESSAGE_LENGTH
        assert normalize_recommendation(req, {"suggested_message": exact})["suggested_message"] == exact

    def test_prompt_does_not_leak_the_discard_rule(self):
        """"폐기된다"고 알려 주면 모델이 "정말 정말" 같은 패딩으로 분량을 채웁니다."""
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        system = build_simple_messages(req)[0]["content"]
        assert "폐기" not in system
        assert str(MIN_MESSAGE_LENGTH) not in system

    def test_default_messages_clear_the_threshold(self):
        """기본 문구가 임계값보다 짧으면 대체해 놓고도 분량 요구를 스스로 어깁니다."""
        cases = [
            SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000),
            SimpleGiftRecommendationRequest(
                gift_name="청첩장", gift_price=50000, record_type="event_invitation", event="결혼"
            ),
            SimpleGiftRecommendationRequest(
                gift_name="부고장", gift_price=50000, record_type="event_invitation", event="조의"
            ),
            SimpleGiftRecommendationRequest(
                gift_name="조의금", gift_price=50000, record_type="money", event="부친상"
            ),
            SimpleGiftRecommendationRequest(
                gift_name="축의금", gift_price=200000, received_amounts=[50000, 200000], event="결혼"
            ),
        ]
        for req in cases:
            assert len(normalize_recommendation(req, {})["suggested_message"]) >= MIN_MESSAGE_LENGTH


# ── 실측 표본 ────────────────────────────────────────────────────────────
# 2~4차 E2E 에서 **모델이 직접 쓴** 메시지입니다(템플릿으로 대체된 건 제외).
# 폐기선을 정하는 근거라 길이와 원문을 함께 남깁니다.
_MEASURED_MODEL_MESSAGES = (
    "민수님이 맛있는 스타벅스 케이크를 선물해 주셔서 정말 감사합니다. "
    "정성스러운 선물 덕분에 즐거운 시간을 보낼 수 있었어요. "
    "저도 마음을 담아 선물을 준비하고 있으니 받아 주세요. "
    "앞으로도 좋은 대학 동기로 지낼 수 있으면 좋겠습니다.",  # 2차 130자
    "춤추는 니니즈님, 소중한 선물 정말 감사합니다. "
    "맛있는 커피를 즐길 수 있게 챙겨주셔서 기분이 좋네요. "
    "마음이 담긴 선물을 받으니 더욱 의미 있습니다. "
    "저도 감사의 마음을 담아 좋은 선물로 답례하고 싶습니다. "
    "앞으로도 좋은 인연 이어가길 바랍니다.",  # 3차 138자
    "민수님, 맛있는 스타벅스 케이크를 보내주셔서 정말 감사합니다. "
    "정성스러운 선물이라 더 의미 있게 느껴졌어요. "
    "앞으로도 좋은 대학 동기로 지낼 수 있어서 고맙고, 작은 선물로 감사의 마음을 전하고 싶습니다. "
    "앞으로도 자주 연락하면서 지낼 수 있으면 좋겠어요.",  # 3차 143자
    "김영삼님, 예쁜 꽃을 보내주셔서 정말 감사합니다. "
    "꽃을 받으니 기분이 한결 밝아졌어요. "
    "님의 따뜻한 마음이 담긴 선물이라 더욱 소중하게 느껴집니다. "
    "저도 감사의 마음을 담아 선물을 준비하고 있으니 기대해 주세요. "
    "앞으로도 좋은 친구로 지낼 수 있으면 좋겠습니다.",  # 2차 146자
)

# 4차에서 폐기선 130 에 걸려 **전부 버려진** 출력의 길이입니다. 원문은 응답에
# 남지 않았지만(템플릿으로 교체됨) 로그가 길이를 남겼습니다.
_DISCARDED_IN_ROUND_FOUR = (109, 113, 118, 121)

# 실측에서 모델이 쓴 문장 **하나**의 최대 길이입니다(3차 giftdata 세 번째 문장).
# 폐기선은 이 위에 있어야 "한 줄짜리"를 잡습니다.
_LONGEST_MEASURED_SENTENCE = 53

# 폐기 장치가 원래 막으려던 것들입니다. 폐기선을 내려도 이건 통과하면 안 됩니다.
_DEGENERATE_MESSAGES = (
    "",
    "   ",
    "감사합니다",
    "감사합니다!",
    "선물 감사합니다. 잘 쓸게요.",
    "감사합니다. 감사합니다. 감사합니다. 감사합니다.",
    # 한 줄짜리. 실측 최장 문장(53자)보다 길게 잡아도 폐기선에 닿지 않습니다.
    "춤추는 니니즈님, 지난번에 빽다방 모바일 금액권 1만원권 보내 주셔서 정말 감사합니다.",
)


class TestTheFloorCatchesDegenerateOutputOnly:
    """폐기선은 **degenerate 방어선**이지 품질 기준이 아닙니다.

    4차 실측에서 130 이 그 선을 넘었습니다. 모델이 쓴 109·113·118·121자 네 건을
    모두 버려 템플릿 비율이 4/4 가 됐고, 사용자에게는 모델 문장이 한 번도 나가지
    않았습니다. 109자짜리 한국어 감사 메시지는 그 자체로 부족하지 않습니다.

    그래서 폐기선을 90 으로 내렸습니다. 이 클래스는 내린 뒤에도 원래 목적
    (빈 문자열·"감사합니다" 한 마디·한 줄짜리)이 그대로 지켜지는지 고정합니다.
    """

    req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)

    @pytest.mark.parametrize("degenerate", _DEGENERATE_MESSAGES)
    def test_degenerate_output_never_reaches_the_user(self, degenerate):
        normalized = normalize_recommendation(self.req, {"suggested_message": degenerate})
        assert normalized["message_source"] is not MessageSource.MODEL
        assert normalized["suggested_message"] != degenerate

    @pytest.mark.parametrize("degenerate", _DEGENERATE_MESSAGES)
    def test_degenerate_output_is_below_the_floor_by_construction(self, degenerate):
        """표본이 우연히 폐기선을 넘어 테스트가 무의미해지는 것을 막습니다."""
        assert len(degenerate.strip()) < MIN_MESSAGE_LENGTH

    @pytest.mark.parametrize("measured", _MEASURED_MODEL_MESSAGES)
    def test_measured_model_messages_reach_the_user(self, measured):
        normalized = normalize_recommendation(self.req, {"suggested_message": measured})
        assert normalized["message_source"] is MessageSource.MODEL
        assert normalized["suggested_message"] == measured

    @pytest.mark.parametrize("measured", _MEASURED_MODEL_MESSAGES)
    def test_the_same_message_one_sentence_shorter_still_reaches_the_user(self, measured):
        """4차 출력(109~121자)은 실측 문장에서 한 문장이 덜한 정도였습니다.

        그 길이의 메시지가 버려진 것이 이번 결함입니다. 마지막 한 문장을 덜어낸
        실제 문장으로 고정해, 폐기선이 다시 올라가면 여기서 걸리게 합니다.
        """
        shorter = measured.rstrip(".").rsplit(". ", 1)[0] + "."
        normalized = normalize_recommendation(self.req, {"suggested_message": shorter})
        assert normalized["message_source"] is MessageSource.MODEL

    def test_the_floor_sits_below_every_measured_model_output(self):
        """실측 정상 출력을 하나라도 버리면 폐기선이 목적을 넘은 것입니다."""
        shortest = min(_DISCARDED_IN_ROUND_FOUR + tuple(len(m) for m in _MEASURED_MODEL_MESSAGES))
        assert MIN_MESSAGE_LENGTH < shortest

    def test_the_floor_sits_above_the_longest_measured_single_sentence(self):
        """이 위에 있어야 "한 줄짜리"가 폐기선을 넘지 못합니다."""
        assert MIN_MESSAGE_LENGTH > _LONGEST_MEASURED_SENTENCE

    def test_the_target_is_a_length_the_model_has_actually_reached(self):
        """도달 불가능한 요구는 지시가 아니라 잡음입니다.

        4차의 160 은 실측 9건 중 아무도 넘지 못한 값이었고, 그 라운드의 출력은
        오히려 3차보다 짧아졌습니다. 요구는 실측이 넘어선 범위 안에 있어야 합니다.
        """
        assert TARGET_MESSAGE_LENGTH <= max(len(m) for m in _MEASURED_MODEL_MESSAGES)


class TestCondolenceNeverGetsCongratulated:
    """청첩장과 부고장이 같은 event_invitation 으로 옵니다. 갈라내지 못하면 사고입니다."""

    def condolence(self, **kwargs) -> SimpleGiftRecommendationRequest:
        base = {
            "gift_name": "부고장",
            "gift_price": 50000,
            "record_type": "event_invitation",
            "event": "조의",
        }
        base.update(kwargs)
        return SimpleGiftRecommendationRequest(**base)

    @pytest.mark.parametrize("event", ["조의", "부고", "부친상", "장례", "근조"])
    def test_prompt_takes_the_condolence_branch(self, event):
        system = build_simple_messages(self.condolence(event=event))[0]["content"]
        assert "조의 인사" in system
        assert "축하 인사" not in system
        assert "초대받은 하객" not in system

    def test_wedding_invitation_still_takes_the_celebration_branch(self):
        req = self.condolence(gift_name="청첩장", event="결혼")
        system = build_simple_messages(req)[0]["content"]
        assert "초대받은 하객" in system
        assert "부고 소식" not in system

    def test_received_condolence_money_gets_its_own_note(self):
        req = self.condolence(gift_name="조의금", record_type="money", event="모친상")
        system = build_simple_messages(req)[0]["content"]
        assert "감사 인사" in system
        assert "축하·기쁨·설렘을 나타내는 말과 느낌표, 이모지를 절대 쓰지 마세요" in system
        assert "축하 인사" not in system

    @pytest.mark.parametrize("event", ["조의", "부고", "부친상"])
    def test_fallback_message_has_no_celebration(self, event):
        message = normalize_recommendation(self.condolence(event=event), {})["suggested_message"]
        assert "축하" not in message
        assert "기뻤" not in message and "기쁘게" not in message
        assert "!" not in message
        assert "조의" in message or "위로" in message

    def test_condolence_money_fallback_is_a_thank_you_not_a_visit_note(self):
        req = self.condolence(gift_name="조의금", record_type="money", event="조의")
        message = normalize_recommendation(req, {})["suggested_message"]
        assert "감사" in message
        assert "축하" not in message and "!" not in message

    def test_wedding_fallback_still_congratulates(self):
        req = self.condolence(gift_name="청첩장", event="결혼")
        message = normalize_recommendation(req, {})["suggested_message"]
        assert "축하" in message


class TestFallbackMessageReadsLikeKorean:
    def test_relationship_particle_follows_the_final_consonant(self):
        with_batchim = SimpleGiftRecommendationRequest(
            gift_name="케이크", gift_price=30000, relationship="사촌 형"
        )
        without = SimpleGiftRecommendationRequest(
            gift_name="케이크", gift_price=30000, relationship="친구"
        )
        assert "사촌 형으로서" in normalize_recommendation(with_batchim, {})["suggested_message"]
        assert "친구로서" in normalize_recommendation(without, {})["suggested_message"]

    def test_gift_name_is_not_forced_into_a_present_shaped_sentence(self):
        """gift_name 에는 "현금", "생일 축하금" 같은 값도 들어옵니다."""
        req = SimpleGiftRecommendationRequest(
            gift_name="생일 축하금", gift_price=50000, record_type="money"
        )
        message = normalize_recommendation(req, {})["suggested_message"]
        assert "선물해 주신 생일 축하금" not in message
        assert "생일 축하금" in message

    def test_message_does_not_claim_the_gift_was_already_used(self):
        """받은 당일 캡처가 주 시나리오라 "잘 사용하고 있고" 는 단정입니다."""
        req = SimpleGiftRecommendationRequest(gift_name="케이크 기프티콘", gift_price=30000)
        message = normalize_recommendation(req, {})["suggested_message"]
        assert "잘 사용하고 있" not in message
        assert "볼 때마다" not in message

    def test_message_does_not_claim_a_return_gift_is_ready(self):
        """아직 고르는 중입니다. 완료형으로 쓰면 답례를 의무처럼 만듭니다."""
        req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
        message = normalize_recommendation(req, {})["suggested_message"]
        assert "준비했으니" not in message

    def test_group_message_does_not_say_wedding_e(self):
        """"결혼에 보내주신" 은 어색합니다."""
        req = SimpleGiftRecommendationRequest(
            gift_name="축의금", gift_price=200000, event="결혼", received_amounts=[50000, 200000]
        )
        message = normalize_recommendation(req, {})["suggested_message"]
        assert "결혼에 보내" not in message
        assert "결혼 때 보내" in message


def test_prompt_allows_future_tense_thanks():
    """받은 당일 캡처가 주 시나리오라 "이미 잘 썼다"고 단정하게 하면 안 됩니다."""
    req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
    system = build_simple_messages(req)[0]["content"]
    assert "미래형" in system


def test_invitation_and_multi_notes_are_not_mutually_exclusive():
    """여러 명에게 청첩장을 받는 일도 있습니다."""
    req = SimpleGiftRecommendationRequest(
        gift_name="결혼 청첩장",
        gift_price=200000,
        record_type="event_invitation",
        event="결혼",
        received_amounts=[50000, 200000],
    )
    system = build_simple_messages(req)[0]["content"]
    assert "초대받은 하객" in system
    assert "여러 사람에게 한 번에 받았습니다" in system


# ---------------------------------------------------------------- 실측 대응 지시
# 아래 문구들은 개선 전 HEAD 로 돌린 E2E 에서 Bedrock Haiku 가 실제로 낸 문장의
# 결함에 하나씩 대응합니다. 지시를 지우면 그 결함이 그대로 돌아옵니다.
MEASURED_DEFECTS = [
    # (실측 문장, 대응 지시 문구)
    ("춤추는 니니즈에게 빽다방 금액권을 받아서 — 수신자를 3인칭으로 서술", "3인칭으로 서술"),
    ("영삼이가 — 입력된 이름 김영삼을 애칭으로 변형", "애칭으로 바꾸지 마세요"),
    ("영삼이가 — 이름을 부르는 형태가 정해져 있지 않음", '"이름님" 으로 씁니다'),
    ("①② 존댓말 / ③ 반말 — 같은 서비스에서 말투가 흔들림", "존댓말로 통일"),
    ("미안한 마음에 이것을 준비했으니 — 감사가 아니라 사과", "사과하는 말"),
    ("선물을 준비했는데 / 받아줘서 고마워 — 아직 고르는 중인 답례를 완료형으로", "이미 준비했거나 건넸다고 쓰지 말고"),
    ("뭔가 특별한 것을 골라봤어 — 정해지지 않은 답례를 지어냄", "무엇을 줄지 지어내지 마세요"),
    ("잘 써먹고 있어요 — 감사 편지에 속된 표현", "속되거나 낮잡는 말투"),
    ("summary 가 실제 상품·금액과 어긋남(검색 전에 생성되므로 알 수 없음)", "특정 상품·브랜드·금액을 약속하지 마세요"),
    ("32세 남성도 충분히 감상하고 관리할 수 있는 — 사용자를 낮춰보는 사유", "사람을 평가하듯 쓰지 마세요"),
]


@pytest.mark.parametrize(("defect", "directive"), MEASURED_DEFECTS)
def test_prompt_addresses_each_measured_defect(defect, directive):
    req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)
    system = build_simple_messages(req)[0]["content"]
    assert directive in system, f"실측 결함에 대응하는 지시가 사라졌습니다: {defect}"


def test_budget_line_carries_the_category_constraint():
    """시스템 프롬프트에서 예산 지시를 뺀 자리를, 예산이 있을 때만 사용자 메시지가 메웁니다."""
    with_budget = SimpleGiftRecommendationRequest(
        gift_name="케이크", gift_price=30000, budget_min=10000, budget_max=20000
    )
    without = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)

    assert "이 가격대에서 살 수 있는 카테고리" in build_simple_messages(with_budget)[1]["content"]
    assert "이 가격대" not in build_simple_messages(without)[1]["content"]


class TestFallbackObeysTheSameRulesAsTheModel:
    """모델에게 시킨 규칙을 폴백이 어기면 어느 쪽이 나갔는지에 따라 품질이 갈립니다."""

    cases = [
        SimpleGiftRecommendationRequest(
            gift_name="케이크", gift_price=30000, person_name="김민수", relationship="사촌 형"
        ),
        SimpleGiftRecommendationRequest(gift_name="선물", gift_price=30000),
        SimpleGiftRecommendationRequest(
            gift_name="청첩장",
            gift_price=50000,
            record_type="event_invitation",
            event="결혼",
            person_name="박지훈",
        ),
        SimpleGiftRecommendationRequest(
            gift_name="부고장",
            gift_price=50000,
            record_type="event_invitation",
            event="조의",
            person_name="박지훈",
        ),
        SimpleGiftRecommendationRequest(
            gift_name="조의금",
            gift_price=50000,
            record_type="money",
            event="부친상",
            person_name="이서준",
        ),
        SimpleGiftRecommendationRequest(
            gift_name="축의금", gift_price=200000, event="결혼", received_amounts=[50000, 200000]
        ),
    ]

    @pytest.mark.parametrize("req", cases)
    def test_every_sentence_is_polite(self, req):
        """실측 ③ 은 반말이었습니다. 폴백은 전부 존댓말로 통일돼 있어야 합니다."""
        message = normalize_recommendation(req, {})["suggested_message"]
        endings = [s.strip()[-1] for s in re.split(r"[.!?]", message) if s.strip()]
        assert all(e in "요다오" for e in endings), message

    @pytest.mark.parametrize("req", cases)
    def test_no_apology_and_no_completed_return_gift(self, req):
        message = normalize_recommendation(req, {})["suggested_message"]
        for banned in ("미안", "죄송", "준비했으니", "준비했는데", "받아주세요", "받아줘서"):
            assert banned not in message, f"{banned} in {message}"

    def test_name_is_called_with_the_honorific(self):
        req = SimpleGiftRecommendationRequest(
            gift_name="케이크", gift_price=30000, person_name="춤추는 니니즈"
        )
        message = normalize_recommendation(req, {})["suggested_message"]
        assert message.startswith("춤추는 니니즈님,")

    def test_recipient_is_not_described_in_third_person(self):
        """실측 ①② 는 "{이름}에게 …" 로 시작해 제3자에게 설명하는 글이 됐습니다."""
        req = SimpleGiftRecommendationRequest(
            gift_name="케이크", gift_price=30000, person_name="김민수"
        )
        message = normalize_recommendation(req, {})["suggested_message"]
        assert "김민수에게" not in message


class TestSummaryNeverPromisesAnAmount:
    """summary 는 상품 검색보다 먼저 만들어집니다. 거기 적힌 금액은 추측입니다.

    실측에서 summary 는 "8,000~12,000원 범위의 상품권..." 이었는데 실제로 나간 상품은
    35,000원 한 건이었고, 같은 응답의 rationale 은
    product_basis "0개가 8,000원 ~ 12,000원 안에 듭니다",
    warnings "1개는 제안 가격대를 벗어납니다" 로 사실을 말하고 있었습니다.
    한 응답 안에서 summary 만 다른 말을 했습니다.
    """

    def req(self) -> SimpleGiftRecommendationRequest:
        return SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)

    @pytest.mark.parametrize(
        "summary",
        [
            "8,000~12,000원 범위의 상품권을 추천합니다.",
            "3만원대 디저트 세트가 무난합니다.",
            "1만 5천원 정도의 커피를 권합니다.",
        ],
    )
    def test_summary_with_an_amount_is_replaced(self, summary):
        result = normalize_recommendation(self.req(), {"summary": summary})
        assert result["summary"] == DEFAULT_SUMMARY

    def test_summary_without_an_amount_is_kept(self):
        """금액만 막습니다. 카테고리 설명은 모델이 쓴 그대로 나가야 합니다."""
        summary = "관계와 취향을 고려해 커피·차와 식품·디저트를 권합니다."
        assert normalize_recommendation(self.req(), {"summary": summary})["summary"] == summary

    def test_missing_summary_falls_back(self):
        assert normalize_recommendation(self.req(), {})["summary"] == DEFAULT_SUMMARY

    def test_blank_summary_falls_back(self):
        """빈 문자열은 스키마(min_length=1)에서 터집니다. 여기서 먼저 채웁니다."""
        assert normalize_recommendation(self.req(), {"summary": "   "})["summary"] == DEFAULT_SUMMARY

    def test_fallback_summary_promises_nothing_either(self):
        """폴백이 금액을 약속하면 막아 놓고 스스로 어기는 셈입니다."""
        assert not re.search(r"\d", DEFAULT_SUMMARY)

    def test_summary_cannot_contradict_the_rationale_price_facts(self):
        """rationale 은 실제 값을 말합니다. summary 가 금액을 못 쓰면 어긋날 수 없습니다."""
        result = normalize_recommendation(
            self.req(), {"summary": "8,000~12,000원 범위의 상품권을 추천합니다."}
        )
        assert not re.search(r"\d[\d,]*\s*[만천억]?\s*원", result["summary"])


# ---------------------------------------------------------------- summary vs 실제 상품
# summary 는 상품 검색보다 **먼저** 만들어집니다. 금액과 같은 이유로 카테고리도
# 그 시점에는 알 수 없습니다. 실측 /from-image 는 커피·차(85)·식품·디저트(70)·
# 생활용품(60) 을 고르고 summary 에 "커피나 차 관련 제품으로 답례하는 것을
# 추천합니다" 라고 썼는데, 예산 안에 든 후보가 최저 점수 카테고리에만 남아 화면에는
# 생활용품 볼펜 한 개가 나갔습니다.

def suggestion(category: str | None, price: int = 9800) -> ProductSuggestion:
    return ProductSuggestion(
        title="상품",
        url="https://gift.kakao.com/product/1",
        source="카카오 선물하기",
        category=category,
        price=price,
        price_verified=False,
    )


class TestSummaryMatchesTheProductsThatShip:
    MEASURED_SUMMARY = (
        "커피 음료권을 선물해주신 춤추는 니니즈님께 감사하며, "
        "커피나 차 관련 제품으로 답례하는 것을 추천합니다."
    )
    MEASURED_CATEGORIES = ["커피·차", "식품·디저트", "생활용품"]

    def test_measured_contradiction_is_reconciled(self):
        result = reconcile_summary(
            self.MEASURED_SUMMARY, self.MEASURED_CATEGORIES, [suggestion("생활용품")]
        )

        assert result.startswith(self.MEASURED_SUMMARY)
        assert result.endswith("이번에 찾은 상품은 생활용품입니다.")

    def test_full_coverage_is_left_alone(self):
        """어긋나지 않았으면 모델 문장을 그대로 내보냅니다."""
        summary = "같은 카테고리의 꽃 관련 상품으로 답례하는 것을 추천합니다."
        assert reconcile_summary(summary, ["꽃·식물"], [suggestion("꽃·식물")]) == summary

    def test_no_products_is_left_alone(self):
        """반박할 상품이 없습니다. rationale.product_basis 가 검색 실패를 말합니다."""
        assert reconcile_summary(self.MEASURED_SUMMARY, self.MEASURED_CATEGORIES, []) == (
            self.MEASURED_SUMMARY
        )

    def test_products_without_a_category_are_left_alone(self):
        assert reconcile_summary(
            self.MEASURED_SUMMARY, self.MEASURED_CATEGORIES, [suggestion(None)]
        ) == self.MEASURED_SUMMARY

    def test_every_shipped_category_is_named_once(self):
        result = reconcile_summary(
            self.MEASURED_SUMMARY,
            self.MEASURED_CATEGORIES,
            [suggestion("생활용품"), suggestion("커피·차"), suggestion("생활용품")],
        )
        assert result.endswith("이번에 찾은 상품은 생활용품, 커피·차입니다.")

    def test_reconciled_summary_still_fits_the_schema(self):
        """summary 는 max_length=500 입니다. 덧붙인 사실이 잘려 나가면 안 됩니다."""
        result = reconcile_summary("가" * 500, self.MEASURED_CATEGORIES, [suggestion("생활용품")])

        assert len(result) <= 500
        assert result.endswith("이번에 찾은 상품은 생활용품입니다.")

    def test_reconciled_summary_still_promises_no_amount(self):
        """지난 라운드의 금액 금지를 되돌리면 안 됩니다."""
        result = reconcile_summary(
            self.MEASURED_SUMMARY, self.MEASURED_CATEGORIES, [suggestion("생활용품", 9800)]
        )
        assert not re.search(r"\d[\d,]*\s*[만천억]?\s*원", result)


async def test_response_summary_and_rationale_are_reconciled_after_the_search(monkeypatch):
    """함수만 고치고 응답 경로에 붙이지 않으면 사용자에게는 아무것도 바뀌지 않습니다."""

    class StubSearch:
        is_available = True

        async def search(self, targets, low, high, *, stats=None):
            return [suggestion("생활용품")]

    monkeypatch.setattr("app.services.tasks.recommendation.product_search", StubSearch())
    info = await recommendation_preparation_service.prepare(
        GiftData(gift_name="빽다방 금액권", gift_price=10000, person_name="춤추는 니니즈")
    )
    recommendation = info.recommend_gift

    # mock 백엔드는 디저트(90)·생활용품(82) 을 고릅니다. 상품은 생활용품에서만 나왔습니다.
    assert [c.category for c in recommendation.categories] == ["디저트", "생활용품"]
    assert recommendation.summary.endswith("이번에 찾은 상품은 생활용품입니다.")
    assert "이 가격대에서 상품이 나온 것은 생활용품입니다" in recommendation.rationale.category_basis


# ---------------------------------------------------------------- 주는 사람의 조사
# "주시-" 는 주는 사람을 높이는 형태라 주는 사람이 주어입니다. 그 자리에 여격 조사
# "님께" 가 오면 "니니즈에게 선물해 주셔서" 로 읽혀 주체와 대상이 뒤집힙니다.
# 실측 4회 중 1회에서 나왔습니다. 같은 이미지·같은 요청인데 실행마다 갈렸습니다.

class TestGiverIsMarkedAsTheSubject:
    MEASURED = (
        "춤추는 니니즈님께 빽다방 금액권을 선물해주셔서 정말 감사합니다. "
        "덕분에 앞으로 맛있는 커피를 즐길 수 있을 것 같아 기대됩니다. "
        "저도 춤추는 니니즈님께 감사의 마음을 담아 선물을 준비하고 있으니 기꺼이 받아주세요. "
        "앞으로도 좋은 인연 이어가길 바랍니다."
    )

    def test_measured_defect_is_corrected(self):
        fixed = fix_giver_particle(self.MEASURED, "춤추는 니니즈")
        assert fixed.startswith("춤추는 니니즈님께서 빽다방 금액권을 선물해주셔서")

    def test_the_real_dative_in_the_same_message_is_untouched(self):
        """과잉 교정 금지. 같은 문장 안의 "님께 감사의 마음을 담아" 는 정상입니다."""
        fixed = fix_giver_particle(self.MEASURED, "춤추는 니니즈")
        assert "저도 춤추는 니니즈님께 감사의 마음을 담아" in fixed

    def test_already_correct_output_is_not_doubled(self):
        """실측 웜/giftdata 실행은 처음부터 옳았습니다. 건드리면 "님께서서" 가 됩니다."""
        good = "김민수님께서 맛있는 스타벅스 케이크를 선물해주셔서 정말 감사합니다."
        assert fix_giver_particle(good, "김민수") == good

    def test_a_name_shaped_greeting_is_not_touched(self):
        warm = "춤추는 니니즈님, 소중한 빽다방 금액권을 주셔서 정말 감사합니다."
        assert fix_giver_particle(warm, "춤추는 니니즈") == warm

    def test_only_the_known_name_is_corrected(self):
        """아무 명사에나 걸면 사용자가 쓰지 않은 문장까지 바꿉니다."""
        text = "사장님께 선물해주셔서 감사합니다."
        assert fix_giver_particle(text, "김민수") == text

    def test_correction_does_not_cross_a_sentence_boundary(self):
        text = "김민수님께 인사를 전합니다. 도와주셔서 고맙습니다."
        assert fix_giver_particle(text, "김민수") == text

    def test_missing_person_name_is_a_no_op(self):
        assert fix_giver_particle(self.MEASURED, None) == self.MEASURED

    def test_normalize_applies_it_to_the_model_message(self):
        """모든 백엔드가 normalize_recommendation 을 거칩니다. 교정도 여기 한 곳입니다."""
        req = SimpleGiftRecommendationRequest(
            gift_name="빽다방 금액권", gift_price=10000, person_name="춤추는 니니즈"
        )
        result = normalize_recommendation(req, {"suggested_message": self.MEASURED})
        assert result["suggested_message"].startswith("춤추는 니니즈님께서")

    def test_normalize_applies_it_to_the_summary(self):
        req = SimpleGiftRecommendationRequest(
            gift_name="빽다방 금액권", gift_price=10000, person_name="춤추는 니니즈"
        )
        summary = "춤추는 니니즈님께 금액권을 선물해주셔서 감사한 마음을 담아 답례를 권합니다."
        result = normalize_recommendation(req, {"summary": summary})
        assert result["summary"].startswith("춤추는 니니즈님께서")

    def test_fallback_message_never_needs_the_correction(self):
        """폴백은 "{이름}님," 으로 시작합니다. 교정 대상이 아예 없어야 합니다."""
        req = SimpleGiftRecommendationRequest(
            gift_name="케이크", gift_price=30000, person_name="김민수", relationship="친구"
        )
        message = normalize_recommendation(req, {})["suggested_message"]
        assert "김민수님께 " not in message


class TestNameIsNotShortened:
    """3차 실측: person_name 이 "김민수" 인데 모델이 "민수님, ..." 으로 불렀습니다.

    1차의 "김영삼" → "영삼이" 와 같은 계열입니다. 그때 프롬프트에 이름을 줄이지 말라는
    지시를 넣었는데도 3차에서 4회 중 1회 어겼습니다. 확률 문제라 지시로는 못 없앱니다.

    아래 negative 케이스가 이 교정의 전부입니다. 과잉 교정이 원래 결함보다 나쁩니다.
    """

    MEASURED = "민수님, 맛있는 스타벅스 케이크를 보내주셔서 정말 감사합니다."

    def test_measured_defect_is_corrected(self):
        assert fix_shortened_name(self.MEASURED, "김민수").startswith("김민수님,")

    def test_already_correct_output_is_not_doubled(self):
        """긴 후보부터 시도하지 않으면 "김김민수님" 이 됩니다."""
        good = "김민수님, 맛있는 케이크 고맙습니다."
        assert fix_shortened_name(good, "김민수") == good

    def test_a_person_actually_named_minsu_is_untouched(self):
        """입력이 "민수" 면 뗄 성이 없습니다. 여기서 손대면 순수한 손해입니다."""
        text = "민수님, 맛있는 케이크 고맙습니다."
        assert fix_shortened_name(text, "민수") == text

    def test_a_whole_nickname_is_not_duplicated(self):
        """"춤추는 니니즈" 는 통째로 먹혀야 합니다. "춤추는 춤추는 니니즈님" 이 되면 사고입니다."""
        text = "춤추는 니니즈님, 소중한 금액권 고맙습니다."
        assert fix_shortened_name(text, "춤추는 니니즈") == text

    def test_a_clipped_nickname_is_restored(self):
        text = "니니즈님, 소중한 금액권 고맙습니다."
        assert fix_shortened_name(text, "춤추는 니니즈").startswith("춤추는 니니즈님,")

    @pytest.mark.parametrize("text", ["교수님께 인사드렸어요.", "사장님, 고맙습니다.", "선생님께 배웠어요."])
    def test_common_honorifics_are_untouched(self, text):
        """"수님" 을 잡으면 "교수님" 이 "교김민수님" 이 됩니다."""
        assert fix_shortened_name(text, "김민수") == text

    def test_a_different_person_with_the_same_given_name_is_untouched(self):
        """앞 글자가 한글이면 건너뛰므로 동명이인이 안전합니다."""
        text = "박민수님도 자리에 함께 계셨어요."
        assert fix_shortened_name(text, "김민수") == text

    def test_a_latin_name_is_not_cut_mid_word(self):
        text = "Giftie님, 고맙습니다."
        assert fix_shortened_name(text, "Giftie") == text

    def test_only_the_honorific_form_is_touched(self):
        """"민수 씨" 까지 건드리면 교정 범위가 통제 불능이 됩니다."""
        text = "민수 씨가 챙겨 주셨어요."
        assert fix_shortened_name(text, "김민수") == text

    def test_missing_person_name_is_a_no_op(self):
        assert fix_shortened_name(self.MEASURED, None) == self.MEASURED

    def test_normalize_applies_it_to_the_model_message(self):
        req = SimpleGiftRecommendationRequest(
            gift_name="스타벅스 케이크", gift_price=30000, person_name="김민수"
        )
        message = normalize_recommendation(req, {"suggested_message": self.MEASURED * 3})[
            "suggested_message"
        ]
        assert message.startswith("김민수님,")
        # 한 번만 고치고 마는 것이 아니라 문단 전체에서 같은 형태로 부릅니다.
        assert message.count("김민수님") == message.count("민수님")

    def test_normalize_applies_it_to_the_summary(self):
        req = SimpleGiftRecommendationRequest(
            gift_name="케이크", gift_price=30000, person_name="김민수"
        )
        summary = "민수님이 주신 케이크에 대한 답례를 권합니다."
        assert normalize_recommendation(req, {"summary": summary})["summary"].startswith("김민수님")

    def test_the_shortened_name_is_expanded_before_the_particle_is_fixed(self):
        """순서가 뒤집히면 조사 교정이 온전한 이름을 찾다가 놓칩니다."""
        text = "민수님께 맛있는 케이크를 선물해 주셔서 정말 감사합니다."
        assert fix_person_name(text, "김민수").startswith("김민수님께서")

    def test_fallback_message_never_needs_the_correction(self):
        """폴백은 이름을 온전히 씁니다. 교정 대상이 아예 없어야 합니다."""
        req = SimpleGiftRecommendationRequest(
            gift_name="케이크", gift_price=30000, person_name="김민수", relationship="친구"
        )
        message = normalize_recommendation(req, {})["suggested_message"]
        assert message.startswith("김민수님")


class TestFallbackDoesNotCloneItself:
    """3차 실측에서 서로 다른 두 요청이 이름과 품목만 바뀐 같은 문장으로 나갔습니다.

    둘 다 모델 출력이 아니라 _single_gift_message 의 출력이었습니다.
    데모에서 두 응답을 나란히 놓으면 즉시 들통납니다.
    """

    WARM = SimpleGiftRecommendationRequest(
        gift_name="빽다방 모바일 금액권 1만원권", gift_price=10000, person_name="춤추는 니니즈"
    )
    RECOMMEND = SimpleGiftRecommendationRequest(
        gift_name="꽃", gift_price=23333, person_name="김영삼", relationship="친구"
    )

    def message(self, req) -> str:
        return normalize_recommendation(req, {})["suggested_message"]

    def test_the_two_measured_requests_no_longer_share_a_sentence_frame(self):
        warm, recommend = self.message(self.WARM), self.message(self.RECOMMEND)
        shared = "덕분에 요즘 하루하루 더 힘이 나고, 문득 생각날 때마다 고마운 마음이 들어요."
        assert not (shared in warm and shared in recommend), "두 응답이 같은 틀입니다"

    def test_the_same_request_always_gets_the_same_message(self):
        """난수로 고르면 재현이 안 됩니다. 같은 입력이면 언제나 같은 답이어야 합니다."""
        assert self.message(self.WARM) == self.message(self.WARM)

    def test_the_choice_does_not_depend_on_the_process_hash_seed(self):
        """내장 hash 를 쓰면 서버를 재시작할 때마다 문장이 바뀝니다."""
        script = (
            "from app.schemas.recommendation import SimpleGiftRecommendationRequest as R;"
            "from app.services.recommendation_policy import normalize_recommendation as n;"
            "print(n(R(gift_name='꽃', gift_price=23333, person_name='김영삼',"
            " relationship='친구'), {})['suggested_message'])"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            ).stdout
            for seed in ("0", "1", "12345")
        }
        assert len(runs) == 1, "PYTHONHASHSEED 에 따라 문장이 달라집니다"

    def test_varied_inputs_reach_several_different_frames(self):
        requests = [
            SimpleGiftRecommendationRequest(
                gift_name=gift, gift_price=30000, person_name=name, relationship=rel
            )
            for name, gift, rel in (
                ("춤추는 니니즈", "빽다방 모바일 금액권 1만원권", None),
                ("김영삼", "꽃", "친구"),
                ("김민수", "스타벅스 케이크", "대학 동기"),
                ("박지훈", "핸드크림", "직장 동료"),
                ("이서준", "커피 기프티콘", "사촌 형"),
                ("최유진", "디저트 세트", "선배"),
            )
        ]
        openings = {self.message(req).split(".")[0].split("님, ")[-1] for req in requests}
        assert len(openings) >= 3, f"6개 입력이 {len(openings)}종의 틀로만 갈립니다"

    @pytest.mark.parametrize("build", SINGLE_GIFT_VARIANTS)
    def test_every_variant_obeys_the_shared_rules(self, build):
        """변형을 늘릴 때 한 개만 규칙을 어겨도 그 요청만 품질이 떨어집니다."""
        message = build("김민수님, ", "스타벅스 케이크", "사촌 형")
        assert message.startswith("김민수님, ")
        assert "스타벅스 케이크" in message
        assert "사촌 형으로서" in message
        assert "선물해 주신 스타벅스 케이크" not in message
        for banned in ("미안", "죄송", "준비했으니", "준비했는데", "받아주세요", "받아줘서", "볼 때마다"):
            assert banned not in message, f"{banned} in {message}"

    @pytest.mark.parametrize("build", SINGLE_GIFT_VARIANTS)
    def test_every_variant_clears_the_threshold_without_a_name(self, build):
        """이름도 관계도 없는 최단 입력이 임계값을 못 넘으면 폴백이 폴백을 부릅니다."""
        assert len(build("", "선물", None)) >= MIN_MESSAGE_LENGTH


class TestFallbackKeepsOneSpeechLevel:
    """3차 실측 "고마웠어요 → 좋았습니다 → 들어요" 는 모델이 아니라 이 파일이 썼습니다.

    한 문단 안에서 해요체와 합쇼체가 번갈아 나오면 사람이 쓴 글로 읽히지 않습니다.
    프롬프트를 늘릴 일이 아니라 템플릿을 고칠 일이었습니다.
    """

    cases = [
        SimpleGiftRecommendationRequest(
            gift_name="케이크", gift_price=30000, person_name="김민수", relationship="사촌 형"
        ),
        SimpleGiftRecommendationRequest(gift_name="선물", gift_price=30000),
        SimpleGiftRecommendationRequest(
            gift_name="꽃", gift_price=23333, person_name="김영삼", relationship="친구"
        ),
        SimpleGiftRecommendationRequest(
            gift_name="빽다방 금액권", gift_price=10000, person_name="춤추는 니니즈"
        ),
        SimpleGiftRecommendationRequest(
            gift_name="청첩장", gift_price=50000, record_type="event_invitation", event="결혼"
        ),
        SimpleGiftRecommendationRequest(
            gift_name="부고장", gift_price=50000, record_type="event_invitation", event="조의"
        ),
        SimpleGiftRecommendationRequest(
            gift_name="조의금", gift_price=50000, record_type="money", event="부친상"
        ),
        SimpleGiftRecommendationRequest(
            gift_name="축의금", gift_price=200000, received_amounts=[50000, 200000], event="결혼"
        ),
    ]

    @pytest.mark.parametrize("req", cases)
    def test_one_message_does_not_mix_haeyo_and_hapsyo(self, req):
        message = normalize_recommendation(req, {})["suggested_message"]
        endings = {s.strip()[-1] for s in re.split(r"[.!?]", message) if s.strip()}
        assert len(endings) == 1, f"종결어미가 {endings} 로 섞였습니다: {message}"

    @pytest.mark.parametrize("build", SINGLE_GIFT_VARIANTS)
    def test_every_variant_holds_one_level(self, build):
        message = build("김민수님, ", "케이크", "친구")
        endings = {s.strip()[-1] for s in re.split(r"[.!?]", message) if s.strip()}
        assert len(endings) == 1, f"종결어미가 {endings} 로 섞였습니다: {message}"


class TestDiscardedModelMessageIsVisible:
    """3차 실측 4건 중 2건이 폴백이었는데 로그에도 응답에도 흔적이 없었습니다.

    응답의 generated_by 는 추천 백엔드(BEDROCK_CLAUDE)를 말할 뿐입니다. 그 값은
    JSON 파싱 성공 여부만 반영하므로, 길이 미달로 메시지가 통째로 교체돼도 그대로
    BEDROCK_CLAUDE 입니다. 그래서 폴백 문구가 모델 출력으로 오독됐습니다.
    """

    req = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)

    def test_a_discarded_model_message_is_logged_with_its_length(self, caplog):
        short = "가" * (MIN_MESSAGE_LENGTH - 1)
        with caplog.at_level(logging.WARNING, logger="app.services.recommendation_policy"):
            normalize_recommendation(self.req, {"suggested_message": short})
        assert str(MIN_MESSAGE_LENGTH - 1) in caplog.text

    def test_a_kept_message_logs_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger="app.services.recommendation_policy"):
            normalize_recommendation(self.req, {"suggested_message": "가" * MIN_MESSAGE_LENGTH})
        assert caplog.text == ""

    def test_a_missing_message_is_not_reported_as_a_discard(self, caplog):
        """모델을 아예 부르지 않는 MOCK 경로가 매 요청마다 경고를 찍으면 안 됩니다."""
        with caplog.at_level(logging.WARNING, logger="app.services.recommendation_policy"):
            normalize_recommendation(self.req, {})
        assert caplog.text == ""


    def test_a_kept_message_is_marked_as_model_written(self):
        normalized = normalize_recommendation(
            self.req, {"suggested_message": "가" * MIN_MESSAGE_LENGTH}
        )
        assert normalized["message_source"] is MessageSource.MODEL

    def test_a_discarded_message_is_marked_as_a_length_discard(self):
        """모델이 쓰긴 썼는데 짧아서 버린 경우. 프롬프트 길이를 올릴 자리입니다."""
        normalized = normalize_recommendation(
            self.req, {"suggested_message": "가" * (MIN_MESSAGE_LENGTH - 1)}
        )
        assert normalized["message_source"] is MessageSource.TEMPLATE_TOO_SHORT

    def test_no_model_message_is_marked_apart_from_a_length_discard(self):
        """파싱 실패·mock 은 원인이 달라 같은 값으로 묶으면 안 됩니다."""
        normalized = normalize_recommendation(self.req, {})
        assert normalized["message_source"] is MessageSource.TEMPLATE_NO_OUTPUT

    def test_every_template_value_is_distinguishable_from_the_model_one(self):
        """백엔드 판정 규칙: MODEL 이 아니면 전부 템플릿."""
        assert MessageSource.TEMPLATE_TOO_SHORT is not MessageSource.MODEL
        assert MessageSource.TEMPLATE_NO_OUTPUT is not MessageSource.MODEL


class TestResponseSeparatesBackendFromMessageAuthor:
    """generated_by 와 message_source 는 다른 것을 말합니다.

    실측에서 4건 중 2건이 폴백 템플릿이었는데 전부 generated_by=BEDROCK_CLAUDE 로
    표시됐습니다. 그 값은 JSON 파싱 성공 여부만 반영하므로, 파싱에 성공한 뒤 길이
    미달로 메시지만 교체되면 응답에 아무 흔적이 남지 않았습니다.
    """

    request = SimpleGiftRecommendationRequest(gift_name="케이크", gift_price=30000)

    @staticmethod
    def _response(message_source: MessageSource) -> SimpleGiftRecommendationResponse:
        return SimpleGiftRecommendationResponse(
            input_gift_name="케이크",
            input_gift_price=30000,
            input_age=None,
            recommended_price_min=24000,
            recommended_price_max=36000,
            categories=[
                CategoryRecommendation(
                    category="식품·디저트",
                    score=90,
                    reason="무난합니다",
                    product_examples=["프리미엄 디저트 세트"],
                )
            ],
            summary="요약",
            suggested_message="메시지" * 40,
            message_source=message_source,
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            # 파싱은 성공했으므로 폴백이 아닙니다. 실측에서 나온 조합 그대로입니다.
            source="BEDROCK_CLAUDE",
        )

    def _message(self, message_source: MessageSource) -> dict:
        info = RecommendationPreparationService._finalize(
            self.request, self._response(message_source), SearchStats()
        )
        return info.model_dump()["message"]

    def test_a_template_message_is_not_reported_as_model_written(self):
        message = self._message(MessageSource.TEMPLATE_TOO_SHORT)
        # 기존 필드는 그대로입니다. 백엔드가 이미 읽고 있습니다.
        assert message["generated_by"] == "BEDROCK_CLAUDE"
        # 새 필드가 교체 사실을 드러냅니다.
        assert message["message_source"] == MessageSource.TEMPLATE_TOO_SHORT

    def test_a_model_message_is_reported_as_model_written(self):
        message = self._message(MessageSource.MODEL)
        assert message["generated_by"] == "BEDROCK_CLAUDE"
        assert message["message_source"] == MessageSource.MODEL

    def test_the_two_fields_do_not_move_together(self):
        """같은 백엔드·같은 파싱 결과인데 메시지 출처만 갈리는 것이 요점입니다."""
        kept = self._message(MessageSource.MODEL)
        replaced = self._message(MessageSource.TEMPLATE_TOO_SHORT)
        assert kept["generated_by"] == replaced["generated_by"]
        assert kept["message_source"] != replaced["message_source"]

    def test_message_source_is_not_duplicated_inside_recommend_gift(self):
        """내부 전달값이라 recommend_gift 에는 싣지 않습니다(suggested_message 와 같음)."""
        info = RecommendationPreparationService._finalize(
            self.request, self._response(MessageSource.MODEL), SearchStats()
        )
        dumped = info.model_dump()["recommend_gift"]
        assert "message_source" not in dumped
        assert "suggested_message" not in dumped


# ------------------------------------------------------- 고마움의 주체는 화자입니다
# 5차 실측 gift 콜드: "따뜻한 마음 전해주셔서 고마우신데, 저도 …"
# "-시-" 는 주체를 높이는 어미라 "고마우시-" 는 상대를 고마움을 느끼는 쪽으로
# 만듭니다. 이 서비스의 문장은 언제나 사용자가 상대에게 건네는 말이므로 틀립니다.

class TestGratitudeBelongsToTheSpeaker:
    MEASURED = (
        "춤추는 니니즈님, 빽다방 금액권을 선물해주셔서 정말 감사합니다. "
        "덕분에 앞으로 커피 한 잔의 여유를 더 자주 누릴 수 있을 것 같아요. "
        "따뜻한 마음 전해주셔서 고마우신데, 저도 감사의 마음을 담아 선물을 준비해드리고 싶습니다. "
        "앞으로도 좋은 인연 이어가길 바랍니다."
    )

    def test_the_measured_defect_is_corrected(self):
        assert "따뜻한 마음 전해주셔서 고마운데, 저도" in fix_wrong_honorific(self.MEASURED)

    def test_nothing_else_in_the_measured_message_moves(self):
        fixed = fix_wrong_honorific(self.MEASURED)
        assert fixed.startswith("춤추는 니니즈님, 빽다방 금액권을 선물해주셔서 정말 감사합니다.")
        assert fixed.endswith("앞으로도 좋은 인연 이어가길 바랍니다.")

    @pytest.mark.parametrize(
        "text",
        [
            # 이미 옳은 형태. 두 번 걸어도 같아야 합니다.
            "따뜻한 마음 전해주셔서 고마운데, 저도 준비하고 있어요.",
            # "고마우신" 은 상대를 고마운 사람으로 가리키는 정상 쓰임입니다.
            "늘 고마우신 분이라 잊지 않고 있어요.",
            # 손댈 이유가 없는 다른 어형들.
            "정말 감사합니다.",
            "고마워서 어쩔 줄 모르겠어요.",
            "고맙습니다. 잘 쓰겠습니다.",
            "감사한데 어떻게 갚아야 할지 모르겠어요.",
            # 다른 낱말의 존칭. "고마우시" 로 시작하지 않습니다.
            "바쁘신데 챙겨 주셔서 감사합니다.",
            "건강하시고 늘 좋은 일만 있으시길 바랍니다.",
            "",
        ],
    )
    def test_correct_sentences_are_left_alone(self, text):
        assert fix_wrong_honorific(text) == text

    def test_the_correction_is_idempotent(self):
        once = fix_wrong_honorific(self.MEASURED)
        assert fix_wrong_honorific(once) == once

    def test_normalize_applies_it_to_the_model_message(self):
        """모든 백엔드가 normalize_recommendation 을 거칩니다."""
        req = SimpleGiftRecommendationRequest(
            gift_name="빽다방 금액권", gift_price=10000, person_name="춤추는 니니즈"
        )
        result = normalize_recommendation(req, {"suggested_message": self.MEASURED})
        assert "고마우신데" not in result["suggested_message"]
        assert "고마운데" in result["suggested_message"]

    def test_the_name_fix_and_the_honorific_fix_both_run(self):
        """한쪽만 걸리면 다른 결함이 조용히 남습니다."""
        req = SimpleGiftRecommendationRequest(
            gift_name="케이크", gift_price=30000, person_name="김민수"
        )
        # 폐기선(90자)을 넘겨야 모델 문장 그대로 교정 경로를 탑니다.
        broken = (
            "민수님께 맛있는 스타벅스 케이크를 선물해주셔서 정말 감사합니다. "
            "따뜻한 마음 전해주셔서 고마우신데, 저도 감사의 마음을 담아 답례를 고르는 중입니다. "
            "앞으로도 좋은 인연 이어가길 바랍니다."
        )
        message = normalize_recommendation(req, {"suggested_message": broken})["suggested_message"]
        assert message.startswith("김민수님께서 맛있는 스타벅스 케이크를 선물해주셔서")
        assert "따뜻한 마음 전해주셔서 고마운데, 저도" in message

    @pytest.mark.parametrize("req", TestFallbackObeysTheSameRulesAsTheModel.cases)
    def test_no_template_message_needs_the_correction(self, req):
        """폴백 문구에 교정 대상이 있으면 교정이 아니라 그 문구를 고쳐야 합니다."""
        message = normalize_recommendation(req, {})["suggested_message"]
        assert fix_wrong_honorific(message) == message
