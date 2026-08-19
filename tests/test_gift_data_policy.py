"""추출 결과 -> GiftData 변환 정책 테스트."""

import os

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")

from datetime import date

import pytest

from app.core.config import settings
from app.schemas.vision import Direction, ExtractedRecord, ExtractionResult, PriceBasis, RecordType
from app.services.gift_data_policy import (
    GiftDataPolicyError,
    build_gift_data,
    estimate_price,
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


class TestEstimatePrice:
    @pytest.mark.parametrize(
        "category,expected",
        [("조의금", 50_000), ("축의금", 50_000), ("기프티콘/음료", 10_000), ("상품권", 50_000)],
    )
    def test_by_category(self, category, expected):
        assert estimate_price(record(category=category, amount=None)) == expected

    def test_default(self):
        assert estimate_price(record(category="알수없음", item_name=None, amount=None, event=None)) == 30_000


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

    def test_multiple_records_report_dropped(self):
        result = ExtractionResult(
            image_kind="bank_statement",
            records=[record(amount=100000), record(amount=50000), record(amount=200000)],
        )
        build = build_gift_data(result)

        assert build.gift_data.gift_price == 200000
        assert len(build.dropped_records) == 2
        assert any("대표 1건만 전달" in w for w in build.warnings)

    def test_missing_price_is_estimated_and_marked(self, monkeypatch):
        """금액이 없는 청첩장도 502 가 아니라 정상 응답이 되어야 합니다."""
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

        assert build.gift_data.gift_price == 50_000
        assert build.price_basis is PriceBasis.ESTIMATED
        assert "(금액 미상)" in build.gift_data.gift_name
        assert build.gift_data.target_date == date(2026, 6, 20)
        assert any("추정했습니다" in w for w in build.warnings)

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
