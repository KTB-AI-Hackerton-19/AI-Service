"""추천 프롬프트를 분할 호출용으로 나눈 것이 단일 호출을 건드리지 않는지 확인합니다.

분할(``recommendation_stages``)은 프롬프트를 조각으로 나눠 단계마다 필요한 것만
싣습니다. 그 과정에서 두 가지가 조용히 깨질 수 있고, 둘 다 화면까지 나갑니다.

1. 단일 호출 프롬프트가 달라짐. 조각을 이어 붙인 것이 원본과 한 글자라도 다르면
   ``MODEL_BACKEND`` 를 되돌렸을 때 예전과 다른 추천이 나옵니다. 되돌리기가
   되돌리기가 아니게 되는 것이 이 회귀의 성질입니다.
2. 상황별 안내문 누락. ``_CONDOLENCE_NOTE`` 가 감사 메시지 단계에 안 실리면
   유족에게 축하 문구가 나갑니다. 단계가 넷이라 한 곳만 빠뜨리기 쉽습니다.
"""

import pytest

from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services import prompt
from app.services.recommendation_policy import ALLOWED_CATEGORIES
from app.services.tasks.recommendation import _merge_reasons


def request(**extra) -> SimpleGiftRecommendationRequest:
    return SimpleGiftRecommendationRequest(
        gift_name="베스킨라빈스 쿠폰", gift_price=20_000, person_name="이지수", **extra
    )


def system_of(messages: list[dict[str, str]]) -> str:
    return next(m["content"] for m in messages if m["role"] == "system")


class TestTheSingleCallPromptIsUnchanged:
    """조각내기가 단일 호출 경로에 영향을 주지 않아야 합니다."""

    def test_the_pieces_reassemble_into_the_original_prompt(self):
        assert prompt.SIMPLE_SYSTEM_PROMPT == (
            f"{prompt._ROLE}\n{prompt._CATEGORY_RULES}\n{prompt._JSON_RULE}"
            f"\n\n{prompt._MESSAGE_RULES}"
        )

    def test_the_single_call_carries_both_rule_sets(self):
        """단일 호출은 카테고리와 메시지를 한 번에 만들므로 규칙이 둘 다 있어야 합니다."""
        system = system_of(prompt.build_simple_messages(request()))
        assert prompt._CATEGORY_RULES in system
        assert prompt._MESSAGE_RULES in system


class TestEveryStageCarriesTheSituationNotes:
    """조의·청첩장 안내가 한 단계라도 빠지면 그 단계의 출력만 조용히 어긋납니다."""

    BUILDERS = [
        ("single", lambda r: prompt.build_simple_messages(r)),
        ("plan", lambda r: prompt.build_plan_messages(r)),
        ("prose", lambda r: prompt.build_prose_messages(r, [{"category": "상품권", "score": 70}])),
        ("message", lambda r: prompt.build_message_messages(r)),
    ]

    @pytest.mark.parametrize("name,build", BUILDERS, ids=[b[0] for b in BUILDERS])
    def test_condolence_note_reaches_every_stage(self, name, build):
        system = system_of(build(request(event="조의")))
        assert prompt._CONDOLENCE_NOTE in system

    @pytest.mark.parametrize("name,build", BUILDERS, ids=[b[0] for b in BUILDERS])
    def test_invitation_note_reaches_every_stage(self, name, build):
        system = system_of(build(request(record_type="event_invitation")))
        assert prompt._INVITATION_NOTE in system

    @pytest.mark.parametrize("name,build", BUILDERS, ids=[b[0] for b in BUILDERS])
    def test_multi_note_reaches_every_stage(self, name, build):
        system = system_of(build(request(received_amounts=[30_000, 50_000])))
        assert prompt._MULTI_NOTE in system


class TestTheMessageStageIsIndependentOfCategories:
    """분할이 성립하는 전제입니다. 메시지 단계는 카테고리를 몰라야 합니다.

    프롬프트가 이미 "답례는 아직 고르는 중이니 선물을 준비한다거나 주겠다는 말은
    사용하지 말고" 라고 금지하고 있으므로, 카테고리는 써서는 안 되는 입력입니다.
    """

    def test_the_message_stage_does_not_receive_the_category_list(self):
        system = system_of(prompt.build_message_messages(request()))
        assert prompt._CATEGORY_LIST not in system

    def test_the_message_stage_still_receives_the_message_rules(self):
        system = system_of(prompt.build_message_messages(request()))
        assert prompt._MESSAGE_RULES in system

    def test_the_user_block_is_identical_across_stages(self):
        """입력 블록까지 갈라지면 단계마다 다른 사실을 보고 씁니다."""
        req = request(relationship="친구", event="생일")
        blocks = {
            next(m["content"] for m in build(req) if m["role"] == "user")
            for build in (prompt.build_simple_messages, prompt.build_plan_messages,
                          prompt.build_message_messages)
        }
        assert len(blocks) == 1


class TestThePlanStageStaysSmall:
    """1단계는 검색이 출발하기 전까지의 순수 대기입니다. 여기서 늘어난 출력은
    그대로 응답 시간이 됩니다(실측: reason 을 받던 판은 134토큰 3.9초,
    빼고 나서 45토큰 2.0초).
    """

    def test_the_plan_schema_asks_only_for_a_category_and_a_score(self):
        properties = prompt.build_plan_schema()["properties"]["categories"]["items"]
        assert set(properties["properties"]) == {"category", "score"}

    def test_the_plan_schema_pins_the_category_enum(self):
        item = prompt.build_plan_schema()["properties"]["categories"]["items"]
        assert item["properties"]["category"]["enum"] == list(ALLOWED_CATEGORIES)

    def test_the_message_schema_asks_only_for_the_message(self):
        assert set(prompt.build_message_schema()["properties"]) == {"suggested_message"}


class TestReasonsAreMatchedByName:
    """이유는 화면에 그대로 나갑니다. 엉뚱한 카테고리에 붙으면 사용자가 봅니다."""

    def test_a_reordered_prose_response_still_lands_on_the_right_category(self):
        categories = [{"category": "상품권", "score": 88}, {"category": "커피·차", "score": 70}]
        prose = {"reasons": [
            {"category": "커피·차", "reason": "커피 이유"},
            {"category": "상품권", "reason": "상품권 이유"},
        ]}
        merged = _merge_reasons(categories, prose)
        assert [(c["category"], c["reason"]) for c in merged] == [
            ("상품권", "상품권 이유"), ("커피·차", "커피 이유")
        ]

    def test_a_category_the_prose_stage_skipped_is_left_alone(self):
        """빠진 자리는 normalize_recommendation 의 기본 문구가 채웁니다."""
        merged = _merge_reasons([{"category": "상품권", "score": 88}], {"reasons": []})
        assert "reason" not in merged[0]

    def test_an_empty_prose_response_does_not_lose_the_categories(self):
        categories = [{"category": "상품권", "score": 88}]
        assert _merge_reasons(categories, {}) == categories
