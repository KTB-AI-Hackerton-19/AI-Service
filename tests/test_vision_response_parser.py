"""VLM 출력 정규화 테스트. vLLM 없이 돕니다."""

import os

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")

from datetime import date

import pytest

from app.schemas.vision import Direction, RecordType
from app.services.vision_response_parser import (
    deduplicate,
    parse_amount_value,
    parse_date_value,
    parse_extraction,
)

TODAY = date(2026, 5, 10)


class TestParseDateValue:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2026-03-14", date(2026, 3, 14)),
            ("2026.03.14", date(2026, 3, 14)),
            ("2026년 3월 14일", date(2026, 3, 14)),
            ("26.03.14", date(2026, 3, 14)),
            ("3/14", date(2026, 3, 14)),
            ("3월 14일", date(2026, 3, 14)),
        ],
    )
    def test_various_formats(self, raw, expected):
        assert parse_date_value(raw, 2026) == expected

    @pytest.mark.parametrize("raw", [None, "", "null", "미상", "2026-13-45", 12345])
    def test_unreadable_becomes_none(self, raw):
        assert parse_date_value(raw, 2026) is None


class TestParseAmountValue:
    @pytest.mark.parametrize(
        "raw,expected",
        [(12300, 12300), ("12,300원", 12300), ("₩ 100,000", 100000), (12300.0, 12300)],
    )
    def test_valid(self, raw, expected):
        assert parse_amount_value(raw) == expected

    @pytest.mark.parametrize("raw", [-6000, "-6,000", 0, "0", None, "금액미상", True])
    def test_zero_or_negative_is_none(self, raw):
        """영수증 할인 줄의 음수 금액이 선물 금액으로 올라오면 안 됩니다."""
        assert parse_amount_value(raw) is None


class TestParseExtraction:
    def test_single_gift(self):
        payload = {
            "image_kind": "kakao_gift",
            "records": [
                {
                    "record_type": "gift",
                    "direction": "received",
                    "counterpart_name": "김수현",
                    "occurred_date": "2026-03-14",
                    "item_name": "아이스 아메리카노",
                    "brand": "스타벅스",
                    "amount": 12300,
                    "confidence": 0.9,
                }
            ],
        }
        result = parse_extraction(payload, TODAY)

        assert len(result.records) == 1
        record = result.records[0]
        assert record.counterpart_name == "김수현"
        assert record.occurred_date == date(2026, 3, 14)
        assert record.amount == 12300
        assert record.needs_review is False
        assert result.warnings == []

    def test_bank_statement_keeps_every_row(self):
        """계좌 거래내역은 여러 건이 나옵니다. 하나도 잃지 않아야 합니다."""
        payload = {
            "image_kind": "bank_statement",
            "records": [
                {
                    "record_type": "money",
                    "direction": "received",
                    "counterpart_name": name,
                    "occurred_date": "5/9",
                    "amount": amount,
                    "confidence": 0.95,
                }
                for name, amount in [("김도윤", 100000), ("박서준", 50000), ("최은비", 200000)]
            ]
            + [
                {
                    "record_type": "money",
                    "direction": "sent",
                    "counterpart_name": "카카오페이",
                    "occurred_date": "5/8",
                    "amount": 38900,
                    "confidence": 0.95,
                }
            ],
        }
        result = parse_extraction(payload, TODAY)

        assert len(result.records) == 4
        assert result.records[0].occurred_date == date(2026, 5, 9)
        assert result.records[-1].direction is Direction.SENT

    def test_receipt_discount_and_total_rows_are_dropped(self):
        payload = {
            "image_kind": "receipt",
            "records": [
                {"record_type": "receipt", "item_name": "조 말론 라임바질 30ml", "amount": 98000, "confidence": 0.9},
                {"record_type": "receipt", "item_name": "할인", "amount": -6000, "confidence": 0.9},
                {"record_type": "receipt", "item_name": "합계", "amount": 114000, "confidence": 0.9},
                {"record_type": "receipt", "item_name": "부가세", "amount": 11400, "confidence": 0.9},
            ],
        }
        result = parse_extraction(payload, TODAY)

        assert [r.item_name for r in result.records] == ["조 말론 라임바질 30ml"]
        assert result.records[0].direction is Direction.SENT

    def test_invitation_future_date_moves_to_event_date(self):
        payload = {
            "image_kind": "invitation",
            "records": [
                {
                    "record_type": "event_invitation",
                    "counterpart_name": "박지훈",
                    "occurred_date": "2026-06-20",
                    "event": "결혼식",
                    "confidence": 0.9,
                }
            ],
        }
        result = parse_extraction(payload, TODAY)

        assert result.records[0].event_date == date(2026, 6, 20)
        assert result.records[0].occurred_date is None

    def test_low_confidence_and_missing_fields_flag_review(self):
        payload = {
            "image_kind": "other",
            "records": [
                {
                    "record_type": "money",
                    "direction": "received",
                    "counterpart_name": None,
                    "occurred_date": None,
                    "amount": None,
                    "confidence": 0.2,
                }
            ],
        }
        result = parse_extraction(payload, TODAY)

        record = result.records[0]
        assert record.needs_review is True
        assert len(record.review_reasons) >= 3

    def test_unknown_enum_falls_back(self):
        payload = {
            "image_kind": "other",
            "records": [{"record_type": "선물", "direction": "받음", "amount": 1000, "confidence": 0.8}],
        }
        result = parse_extraction(payload, TODAY)

        assert result.records[0].record_type is RecordType.UNKNOWN
        assert result.records[0].direction is Direction.UNKNOWN

    def test_missing_records_key(self):
        result = parse_extraction({}, TODAY)
        assert result.records == []
        assert "records 배열이 없습니다" in result.warnings[0]

    def test_empty_records_warns(self):
        result = parse_extraction({"image_kind": "other", "records": []}, TODAY)
        assert result.records == []
        assert result.warnings


class TestDeduplicate:
    def test_keeps_higher_confidence(self):
        payload = {
            "image_kind": "gift_list",
            "records": [
                {
                    "record_type": "gift",
                    "direction": "received",
                    "counterpart_name": "김수현",
                    "occurred_date": "2026-03-14",
                    "item_name": "아메리카노",
                    "amount": 12300,
                    "confidence": confidence,
                }
                for confidence in (0.6, 0.95)
            ],
        }
        result = parse_extraction(payload, TODAY)

        assert len(result.records) == 1
        assert result.records[0].confidence == 0.95

    def test_distinct_records_are_kept(self):
        payload = {
            "image_kind": "gift_list",
            "records": [
                {
                    "record_type": "gift",
                    "direction": "received",
                    "counterpart_name": name,
                    "occurred_date": "2026-03-14",
                    "amount": 10000,
                    "confidence": 0.9,
                }
                for name in ("김수현", "이준호")
            ],
        }
        assert len(parse_extraction(payload, TODAY).records) == 2

    def test_empty_list(self):
        assert deduplicate([]) == []
