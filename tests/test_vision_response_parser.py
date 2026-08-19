"""VLM 출력 정규화 테스트. vLLM 없이 돕니다."""

import os

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")

from datetime import date

import pytest

from app.schemas.vision import Direction, ExtractedRecord, ExtractionResult, RecordType
from app.services.vision_response_parser import (
    clean_person_name,
    deduplicate,
    flag_review,
    parse_amount_value,
    parse_date_value,
    parse_extraction,
    refresh_review_flags,
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
            ("3.15", date(2026, 3, 15)),
            ("3-15", date(2026, 3, 15)),
            ("2026년 3월 14일 오전 10:30", date(2026, 3, 14)),
        ],
    )
    def test_various_formats(self, raw, expected):
        assert parse_date_value(raw, 2026) == expected

    @pytest.mark.parametrize("raw", [None, "", "null", "미상", "2026-13-45", 12345])
    def test_unreadable_becomes_none(self, raw):
        assert parse_date_value(raw, 2026) is None

    @pytest.mark.parametrize(
        "raw",
        [
            "오전 10:30",   # 실측: 2026-10-30 이 됐습니다
            "오후 12:05",   # 실측: 2026-12-05 가 됐습니다
            "12,300원",     # 실측: 2026-12-30 이 됐습니다
            "오전 7:53",
            "오후 3:30 결제",
            "₩12,300",
            "12,300",
        ],
    )
    def test_clock_and_amount_are_never_dates(self, raw):
        """시각·금액이 날짜로 둔갑하면 캘린더 등록과 알림 예약까지 그대로 흘러갑니다."""
        assert parse_date_value(raw, 2026) is None


class TestCleanPersonName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("홍길동 님", "홍길동"),
            ("김수현 님", "김수현"),
            ("박서준씨", "박서준"),
            ("박서준 씨", "박서준"),
            ("홍길동선생님", "홍길동"),
            ("님", None),
        ],
    )
    def test_honorific_is_removed(self, raw, expected):
        assert clean_person_name(raw) == expected

    @pytest.mark.parametrize("raw", ["김선배", "박군", "이양", "최후배", "김선생님", "정고객님"])
    def test_short_name_is_not_eaten(self, raw):
        """붙여 쓴 호칭을 떼면 한 글자만 남는 이름은 호칭이 아니라 이름입니다."""
        assert clean_person_name(raw) == raw


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

    def test_clock_string_does_not_become_a_date(self):
        """모델이 시각을 occurred_date 에 넣어도 날짜로 승격시키지 않습니다."""
        payload = {
            "image_kind": "kakao_gift",
            "records": [
                {
                    "record_type": "gift",
                    "direction": "received",
                    "counterpart_name": "김수현",
                    "occurred_date": "오전 10:30",
                    "event_date": None,
                    "item_name": "아메리카노",
                    "amount": 12300,
                    "confidence": 0.95,
                }
            ],
        }
        result = parse_extraction(payload, TODAY)

        record = result.records[0]
        assert record.occurred_date is None
        assert record.event_date is None
        assert record.needs_review is True

    def test_memo_wording_does_not_drop_a_real_gift(self):
        """영수증 잡음 필터는 상품명만 봅니다. 메모의 "할인"은 상품 줄이 아닙니다."""
        payload = {
            "image_kind": "kakao_gift",
            "records": [
                {
                    "record_type": "gift",
                    "direction": "received",
                    "counterpart_name": "김수현",
                    "occurred_date": "2026-03-14",
                    "item_name": "조 말론 라임바질 30ml",
                    "category": "향수",
                    "memo": "할인받아서 샀어",
                    "amount": 98000,
                    "confidence": 0.95,
                }
            ],
        }
        result = parse_extraction(payload, TODAY)
        assert [r.item_name for r in result.records] == ["조 말론 라임바질 30ml"]

    def test_unknown_enum_falls_back(self):
        payload = {
            "image_kind": "other",
            "records": [{"record_type": "선물", "direction": "받음", "amount": 1000, "confidence": 0.8}],
        }
        result = parse_extraction(payload, TODAY)

        assert result.records[0].record_type is RecordType.UNKNOWN
        assert result.records[0].direction is Direction.UNKNOWN

    def test_honorific_suffix_is_stripped(self):
        """실측에서 선물함 목록이 "김수현 님" 으로 나왔습니다. 인물 매칭이 깨집니다."""
        payload = {
            "image_kind": "gift_list",
            "records": [
                {
                    "record_type": "gift",
                    "direction": "received",
                    "counterpart_name": name,
                    "occurred_date": "2026-03-14",
                    "amount": amount,
                    "confidence": 0.95,
                }
                for name, amount in [("김수현 님", 12300), ("박서준씨", 132000)]
            ],
        }
        result = parse_extraction(payload, TODAY)
        assert [r.counterpart_name for r in result.records] == ["김수현", "박서준"]

    def test_invitation_account_rows_are_dropped(self):
        """청첩장의 계좌 안내는 받은 돈이 아니라 보낼 곳입니다."""
        payload = {
            "image_kind": "invitation",
            "records": [
                {
                    "record_type": "event_invitation",
                    "counterpart_name": "박지훈, 이서연",
                    "event_date": "2026-06-20",
                    "event": "결혼",
                    "confidence": 0.95,
                },
                {
                    "record_type": "money",
                    "counterpart_name": "박지훈",
                    "category": "축의금",
                    "amount": None,
                    "memo": "신랑측 국민 123-45-678901",
                    "confidence": 0.95,
                },
                {
                    "record_type": "money",
                    "counterpart_name": "이서연",
                    "category": "축의금",
                    "amount": None,
                    "memo": "신부측 신한 110-234-567890",
                    "confidence": 0.95,
                },
            ],
        }
        result = parse_extraction(payload, TODAY)
        assert len(result.records) == 1
        assert result.records[0].record_type is RecordType.EVENT_INVITATION

    def test_real_condolence_transfer_is_kept(self):
        """금액이 있는 송금은 청첩장 이미지에서도 실제 기록입니다."""
        payload = {
            "image_kind": "invitation",
            "records": [
                {
                    "record_type": "money",
                    "direction": "received",
                    "counterpart_name": "정예린",
                    "occurred_date": "2026-04-27",
                    "amount": 100000,
                    "confidence": 0.95,
                }
            ],
        }
        result = parse_extraction(payload, TODAY)
        assert len(result.records) == 1
        assert result.records[0].amount == 100000

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


class TestReviewReasons:
    def test_filled_price_replaces_missing_amount_reason(self):
        """검색으로 금액을 채운 뒤에는 "금액을 확인하지 못했습니다"가 남으면 안 됩니다."""
        record = ExtractedRecord(
            record_type=RecordType.MONEY,
            direction=Direction.RECEIVED,
            counterpart_name="김도윤",
            occurred_date=date(2026, 5, 9),
            confidence=0.95,
        )
        flag_review(record, TODAY)
        assert any("금액을 확인하지 못했습니다" in reason for reason in record.review_reasons)

        record.amount = 36000
        record.price_searched = True
        refresh_review_flags(ExtractionResult(records=[record]), TODAY)

        assert not any("금액을 확인하지 못했습니다" in reason for reason in record.review_reasons)
        # 검색으로 채운 값은 다른 용량·구성일 수 있고, 이 값이 답례 가격대의 기준이 됩니다.
        assert any("검색으로 채운" in reason for reason in record.review_reasons)
        assert record.needs_review is True

    def test_clean_record_has_no_reasons(self):
        record = ExtractedRecord(
            record_type=RecordType.GIFT,
            direction=Direction.RECEIVED,
            counterpart_name="김수현",
            occurred_date=date(2026, 3, 14),
            amount=12300,
            confidence=0.95,
        )
        flag_review(record, TODAY)
        assert record.needs_review is False
        assert record.review_reasons == []
