"""추천이 이미지 분석과 같은 엔진·같은 규칙으로 통합됐는지 확인합니다."""

import os

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")

from datetime import date

import pytest

from app.schemas.agent import GiftData, GiftRecordItem
from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services.prompt import build_recommendation_schema, build_simple_messages
from app.services.recommendation_policy import ALLOWED_CATEGORIES, price_range
from app.services.tasks.recommendation import build_request


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

    def test_never_below_floor(self):
        req = SimpleGiftRecommendationRequest(gift_name="사탕", gift_price=500)
        low, high = price_range(req)
        assert low >= 1000 and high >= low


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
