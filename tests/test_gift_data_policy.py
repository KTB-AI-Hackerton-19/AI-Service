"""추출 결과 -> GiftData 변환 정책 테스트."""

import os

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")

from datetime import date

import pytest

from app.core.config import settings
from app.schemas.agent import PriceBasis
from app.schemas.vision import Direction, ExtractedRecord, ExtractionResult, RecordType
from app.services.gift_data_policy import (
    build_gift_name,
    GiftDataPolicyError,
    build_gift_data,
    select_primary,
)


def record(**kwargs) -> ExtractedRecord:
    base = {
        "record_type": RecordType.GIFT,
        "direction": Direction.RECEIVED,
        "counterpart_name": "김수현",
        "occurred_date": date(2026, 3, 14),
        "item_name": "아이스 아메리카노",
        "amount": 12300,
        "confidence": 0.9,
    }
    base.update(kwargs)
    return ExtractedRecord(**base)


class TestSelectPrimary:
    def test_prefers_largest_received_amount(self):
        records = [record(amount=10000), record(amount=200000), record(amount=50000)]
        assert select_primary(records).amount == 200000

    def test_ignores_sent_records(self):
        records = [
            record(direction=Direction.SENT, amount=999999),
            record(direction=Direction.RECEIVED, amount=1000),
        ]
        assert select_primary(records).amount == 1000

    def test_falls_back_to_invitation(self):
        records = [
            record(direction=Direction.SENT, record_type=RecordType.RECEIPT, amount=98000),
            record(
                direction=Direction.UNKNOWN,
                record_type=RecordType.EVENT_INVITATION,
                amount=None,
                item_name=None,
                event="결혼식",
            ),
        ]
        assert select_primary(records).record_type is RecordType.EVENT_INVITATION

    def test_empty(self):
        assert select_primary([]) is None


class TestBuildGiftData:
    def test_single_record(self):
        result = ExtractionResult(image_kind="kakao_gift", records=[record(brand="스타벅스")])
        build = build_gift_data(result)

        assert build.gift_data.gift_name == "스타벅스 아이스 아메리카노"
        assert build.gift_data.gift_price == 12300
        assert build.gift_data.person_name == "김수현"
        assert build.gift_data.received_at == date(2026, 3, 14)
        assert build.price_basis is PriceBasis.STATED
        assert build.dropped_records == []

    def test_multiple_records_are_all_kept(self):
        """다건 이미지의 모든 건이 records 에 담기고, 평면 필드는 대표 1건입니다."""
        result = ExtractionResult(
            image_kind="bank_statement",
            records=[
                record(counterpart_name="김도윤", amount=100000),
                record(counterpart_name="박서준", amount=50000),
                record(counterpart_name="최은비", amount=200000),
            ],
        )
        build = build_gift_data(result)

        assert build.gift_data.gift_price == 200000  # 대표 = 금액 최대
        assert build.gift_data.person_name == "최은비"
        assert len(build.gift_data.records) == 3
        assert [r.person_name for r in build.gift_data.records] == ["김도윤", "박서준", "최은비"]
        assert [r.price for r in build.gift_data.records] == [100000, 50000, 200000]
        assert all(r.selected for r in build.gift_data.records)

    def test_missing_price_stays_missing(self):
        """카테고리로 추정하지 않습니다. 모르는 금액은 비운 채로 내보냅니다."""
        result = ExtractionResult(
            image_kind="invitation",
            records=[
                record(
                    record_type=RecordType.EVENT_INVITATION,
                    direction=Direction.UNKNOWN,
                    amount=None,
                    item_name=None,
                    event="결혼식",
                )
            ],
        )
        build = build_gift_data(result)

        assert build.gift_data.gift_price is None
        assert build.price_basis is PriceBasis.UNKNOWN
        assert build.gift_data.records[0].price is None

    def test_missing_price_is_reported_without_guessing(self, monkeypatch):
        """금액이 없는 청첩장도 502 가 아니라 정상 응답이 되어야 합니다.

        다만 값을 지어내지는 않습니다. 브랜드를 모르는 카테고리 추정가는 실제와
        몇 배씩 어긋나는데(TWG Tea 를 10,000원으로 추정, 실제 3~7만원) 사용자는
        그 값을 사실로 받아들입니다.
        """
        monkeypatch.setattr(settings, "strict_price", False)
        result = ExtractionResult(
            image_kind="invitation",
            records=[
                record(
                    record_type=RecordType.EVENT_INVITATION,
                    direction=Direction.UNKNOWN,
                    amount=None,
                    item_name=None,
                    occurred_date=None,
                    event_date=date(2026, 6, 20),
                    event="결혼식",
                    counterpart_name="박지훈",
                )
            ],
        )
        build = build_gift_data(result)

        assert build.gift_data.gift_price is None
        assert build.price_basis is PriceBasis.UNKNOWN
        # 이름에 "(금액 미상)" 같은 표시를 덧붙이지 않습니다. 이름은 이름이어야 합니다.
        assert "금액" not in build.gift_data.gift_name
        assert build.gift_data.target_date == date(2026, 6, 20)
        assert any("확인하지 못했습니다" in w for w in build.warnings)

    def test_strict_price_rejects_missing_amount(self, monkeypatch):
        monkeypatch.setattr(settings, "strict_price", True)
        result = ExtractionResult(records=[record(amount=None)])

        with pytest.raises(GiftDataPolicyError, match="금액"):
            build_gift_data(result)

    def test_no_records_raises(self):
        with pytest.raises(GiftDataPolicyError, match="찾지 못했습니다"):
            build_gift_data(ExtractionResult(records=[]))

    def test_long_name_is_truncated(self):
        result = ExtractionResult(records=[record(item_name="가" * 500, brand=None)])
        assert len(build_gift_data(result).gift_data.gift_name) <= 200

    def test_long_person_name_is_truncated(self):
        result = ExtractionResult(records=[record(counterpart_name="나" * 100)])
        assert len(build_gift_data(result).gift_data.person_name) <= 50


class TestGiftNameStaysAName:
    """이름에는 이름만 넣습니다. 상태 표시나 중복 브랜드가 섞이면 안 됩니다."""

    def test_brand_is_not_repeated_when_already_in_the_item_name(self):
        built = build_gift_name(
            record(brand="TWG Tea", item_name="TWG Tea Teabags Collection", amount=None)
        )
        assert built == "TWG Tea Teabags Collection"

    def test_brand_is_prefixed_when_missing(self):
        built = build_gift_name(record(brand="스타벅스", item_name="아메리카노 T", amount=None))
        assert built == "스타벅스 아메리카노 T"


class TestInvitationNameNamesTheRightDocument:
    """청첩장은 결혼에만 쓰는 말입니다. 유족 화면에 "조의 청첩장" 이 나가면 사고입니다."""

    def invitation(self, **kwargs) -> ExtractedRecord:
        return record(
            record_type=RecordType.EVENT_INVITATION,
            direction=Direction.UNKNOWN,
            item_name=None,
            amount=None,
            **kwargs,
        )

    @pytest.mark.parametrize("event", ["조의", "부고", "부친상", "장례", "근조"])
    def test_condolence_never_becomes_a_wedding_invitation(self, event):
        assert build_gift_name(self.invitation(event=event)) == "부고장"

    def test_condolence_is_detected_from_the_memo(self):
        built = build_gift_name(self.invitation(event=None, memo="삼가 고인의 명복을 빕니다"))
        assert built == "부고장"

    def test_wedding_keeps_the_word_invitation(self):
        assert build_gift_name(self.invitation(event="결혼")) == "결혼 청첩장"

    def test_other_events_are_not_called_wedding_invitations(self):
        assert build_gift_name(self.invitation(event="돌잔치")) == "돌잔치 초대장"

    def test_missing_event_does_not_invent_one(self):
        assert build_gift_name(self.invitation(event=None)) == "초대장"


class TestMoneyNameIsAKindNotAnOccasion:
    """선물 이름 자리에 "생일" 이 들어가면 "선물해 주신 생일" 이 됩니다."""

    def money(self, **kwargs) -> ExtractedRecord:
        return record(record_type=RecordType.MONEY, item_name=None, amount=50000, **kwargs)

    def test_occasion_gets_a_kind_appended(self):
        assert build_gift_name(self.money(event="생일")) == "생일 축하금"

    def test_wedding_money_is_congratulatory_money(self):
        assert build_gift_name(self.money(event="결혼")) == "결혼 축의금"

    def test_condolence_money_is_condolence_money(self):
        assert build_gift_name(self.money(event="조의")) == "조의금"
        assert build_gift_name(self.money(event="부친상")) == "부친상 조의금"

    def test_category_from_the_image_is_used_as_the_kind(self):
        assert build_gift_name(self.money(event="결혼", category="축의금")) == "결혼 축의금"

    def test_nothing_known_stays_cash(self):
        assert build_gift_name(self.money()) == "현금"


class TestGiftWithoutAnItemName:
    def test_occasion_alone_is_not_a_gift_name(self):
        built = build_gift_name(record(item_name=None, brand=None, event="생일", category=None))
        assert built == "생일 선물"
