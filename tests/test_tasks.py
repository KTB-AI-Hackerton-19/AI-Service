"""선물 기록 · 캘린더 · 알림 작업 테스트."""

import os

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")

from datetime import date, datetime, timedelta

import pytest

from app.core.config import settings
from app.schemas.agent import GiftData, GiftRecordItem, TaskStatus
from app.services.calendar_mcp_client import CalendarMcpError
from app.services.reciprocity_schedule import resolve_schedule
from app.services.tasks.calendar import calendar_preparation_service
from app.services.tasks.gift_record import gift_record_preparation_service
from app.services.tasks.notification import notification_preparation_service

WORKFLOW_ID = "wf-test"
TODAY = date(2026, 5, 10)


def gift(**kwargs) -> GiftData:
    base = {
        "gift_name": "스타벅스 아이스 아메리카노",
        "gift_price": 12300,
        "person_name": "김수현",
        "relationship": "대학 동기",
        "received_at": date(2026, 5, 9),
        "target_date": date(2026, 9, 12),
    }
    base.update(kwargs)
    return GiftData(**base)


def multi_gift() -> GiftData:
    """계좌 거래내역처럼 여러 건이 들어 있는 입력."""
    records = [
        GiftRecordItem(
            record_id=f"r{i}",
            record_type="money",
            direction="received",
            person_name=name,
            gift_name="축의금",
            price=price,
            category="축의금",
            event="결혼",
            received_at=date(2026, 5, 9),
            confidence=1.0,
        )
        for i, (name, price) in enumerate([("김도윤", 100000), ("박서준", 50000), ("최은비", 200000)])
    ]
    return gift(gift_name="축의금", gift_price=200000, person_name="최은비", records=records)


class TestReciprocitySchedule:
    def test_uses_given_target_date(self):
        schedule = resolve_schedule(gift(), today=TODAY)
        assert schedule.target_date == date(2026, 9, 12)
        assert schedule.prepare_date == date(2026, 9, 5)  # 답례일 - 7일
        assert schedule.notify_at.hour == 10
        assert schedule.is_target_estimated is False

    def test_derives_target_from_received_at(self):
        schedule = resolve_schedule(gift(target_date=None), today=TODAY)
        assert schedule.target_date == date(2026, 6, 8)  # 5/9 + 30일
        assert schedule.is_target_estimated is True

    def test_past_target_is_pushed_to_tomorrow(self):
        """지난 날짜로 일정을 잡으면 알림이 울리지 않습니다."""
        schedule = resolve_schedule(gift(target_date=date(2026, 1, 1)), today=TODAY)
        assert schedule.target_date == TODAY + timedelta(days=1)
        assert schedule.is_target_estimated is True

    def test_prepare_date_never_in_the_past(self):
        schedule = resolve_schedule(gift(target_date=TODAY + timedelta(days=2)), today=TODAY)
        assert schedule.prepare_date == TODAY

    def test_notify_time_is_pushed_when_10am_already_passed(self):
        """준비일이 오늘인데 오전 10시가 지났으면 알림이 영영 울리지 않습니다."""
        now = datetime(2026, 5, 10, 14, 1)
        schedule = resolve_schedule(gift(target_date=TODAY + timedelta(days=2)), now=now)

        assert schedule.notify_at == datetime(2026, 5, 10, 15, 0)
        assert schedule.calendar_start_time == "15:00"
        assert schedule.prepare_date == TODAY

    def test_notify_time_stays_10am_when_future(self):
        now = datetime(2026, 5, 10, 14, 1)
        schedule = resolve_schedule(gift(), now=now)  # 답례일 9/12

        assert schedule.notify_at == datetime(2026, 9, 5, 10, 0)
        assert schedule.calendar_start_time == "10:00"

    def test_late_night_push_rolls_into_next_day(self):
        now = datetime(2026, 5, 10, 23, 30)
        schedule = resolve_schedule(gift(target_date=TODAY + timedelta(days=2)), now=now)

        assert schedule.notify_at == datetime(2026, 5, 11, 0, 0)
        assert schedule.prepare_date == date(2026, 5, 11)


class TestGiftRecord:
    async def test_keeps_original_gift_data_fields(self):
        """기존 계약을 읽는 쪽이 깨지지 않아야 합니다."""
        prepared = await gift_record_preparation_service.prepare(gift(), WORKFLOW_ID)

        assert prepared.status is TaskStatus.SUCCESS
        payload = prepared.payload
        assert payload["gift_name"] == "스타벅스 아이스 아메리카노"
        assert payload["gift_price"] == 12300
        assert payload["received_at"] == "2026-05-09"
        assert payload["target_date"] == "2026-09-12"
        assert payload["workflowId"] == WORKFLOW_ID

    async def test_adds_derived_fields(self):
        prepared = await gift_record_preparation_service.prepare(gift(), WORKFLOW_ID)
        payload = prepared.payload

        assert payload["direction"] == "received"  # GiftData 원본 필드
        assert payload["currency"] == "KRW"
        assert payload["recordCount"] == 1
        assert payload["receivedCount"] == 1
        assert payload["totalAmount"] == 12300
        assert "김수현" in payload["summary"]
        assert payload["resolvedTargetDate"] == "2026-09-12"
        assert payload["targetDateEstimated"] is False

    async def test_missing_target_date_stays_none_in_original_field(self):
        prepared = await gift_record_preparation_service.prepare(gift(target_date=None), WORKFLOW_ID)
        payload = prepared.payload

        assert payload["target_date"] is None  # 원본은 그대로
        assert payload["resolvedTargetDate"] is not None  # 계산값은 별도 키
        assert payload["targetDateEstimated"] is True


class TestCalendar:
    async def test_draft_only_by_default(self, monkeypatch):
        """준비 단계는 초안까지입니다. 등록은 사용자 승인 후 /confirm 에서 합니다."""
        monkeypatch.setattr(settings, "google_access_token", "ya29.fake-token")
        monkeypatch.setattr(settings, "calendar_auto_register", False)
        called = False

        async def should_not_be_called(**_kwargs):
            nonlocal called
            called = True
            return {}

        monkeypatch.setattr(
            "app.services.tasks.calendar.calendar_mcp_client.create_event", should_not_be_called
        )
        prepared = await calendar_preparation_service.prepare(gift(), WORKFLOW_ID)

        payload = prepared.payload
        assert called is False, "토큰이 있어도 승인 전에는 등록하지 않아야 합니다"
        assert prepared.status is TaskStatus.SUCCESS
        assert payload["registered"] is False
        assert payload["provider"] == "GOOGLE_MCP_DRAFT"
        assert payload["title"] == "김수현님 답례 준비"
        assert payload["date"] == "2026-09-05"
        assert payload["targetDate"] == "2026-09-12"
        assert "eventId" not in payload

    async def test_auto_register_when_enabled(self, monkeypatch):
        """승인 UI 가 없는 개발 단계용 스위치."""
        monkeypatch.setattr(settings, "google_access_token", "ya29.fake-token")
        monkeypatch.setattr(settings, "calendar_auto_register", True)
        captured = {}

        async def fake_create_event(**kwargs):
            captured.update(kwargs)
            return {"event_id": "evt-1", "html_link": "https://calendar.google.com/e/1"}

        monkeypatch.setattr(
            "app.services.tasks.calendar.calendar_mcp_client.create_event", fake_create_event
        )
        prepared = await calendar_preparation_service.prepare(gift(), WORKFLOW_ID)
        payload = prepared.payload

        assert payload["registered"] is True
        assert payload["provider"] == "GOOGLE_MCP"
        assert payload["eventId"] == "evt-1"
        assert captured["access_token"] == "ya29.fake-token"
        assert captured["start_date"] == "2026-09-05"

    async def test_auto_register_failure_still_returns_draft(self, monkeypatch):
        """캘린더가 막혀도 나머지 세 작업 결과는 살아야 합니다."""
        monkeypatch.setattr(settings, "google_access_token", "ya29.fake-token")
        monkeypatch.setattr(settings, "calendar_auto_register", True)

        async def failing_create_event(**_kwargs):
            raise CalendarMcpError("MCP 서버에 연결할 수 없습니다")

        monkeypatch.setattr(
            "app.services.tasks.calendar.calendar_mcp_client.create_event", failing_create_event
        )
        prepared = await calendar_preparation_service.prepare(gift(), WORKFLOW_ID)

        assert prepared.status is TaskStatus.SUCCESS  # 작업 자체는 실패가 아님
        assert prepared.payload["registered"] is False
        assert "연결할 수 없습니다" in prepared.payload["registerError"]
        assert prepared.payload["date"] == "2026-09-05"

    async def test_anonymous_person_uses_placeholder(self):
        prepared = await calendar_preparation_service.prepare(gift(person_name=None), WORKFLOW_ID)
        assert prepared.payload["title"] == "상대방 답례 준비"

    async def test_multi_record_lists_everyone_in_description(self):
        """축의금 4건을 받았다고 캘린더에 일정 4개가 뜨면 방해가 됩니다. 하나로 묶습니다."""
        prepared = await calendar_preparation_service.prepare(multi_gift(), WORKFLOW_ID)
        payload = prepared.payload

        assert payload["title"] == "김도윤님 외 2명 답례 준비"
        assert "총 350,000원" in payload["description"]
        for name in ("김도윤", "박서준", "최은비"):
            assert name in payload["description"]


class TestNotification:
    async def test_schedules_at_prepare_date_10am(self):
        prepared = await notification_preparation_service.prepare(gift(), WORKFLOW_ID)
        payload = prepared.payload

        assert payload["scheduledAt"] == "2026-09-05T10:00:00"
        assert payload["notifications"][0]["type"] == "RECIPROCITY_PREPARE"
        assert "김수현" in payload["notifications"][0]["body"]
        assert payload["workflowId"] == WORKFLOW_ID

    async def test_keeps_legacy_top_level_keys(self):
        prepared = await notification_preparation_service.prepare(gift(), WORKFLOW_ID)
        assert "title" in prepared.payload
        assert "scheduledAt" in prepared.payload

    async def test_multi_record_body_mentions_everyone(self):
        prepared = await notification_preparation_service.prepare(multi_gift(), WORKFLOW_ID)
        notification = prepared.payload["notifications"][0]
        assert "김도윤님 외 2명" in notification["body"]
        assert notification["recipientCount"] == 3

    async def test_matches_calendar_date(self):
        """알림과 캘린더가 서로 다른 날짜를 쓰면 사용자에게는 버그로 보입니다."""
        data = gift()
        calendar_payload = (await calendar_preparation_service.prepare(data, WORKFLOW_ID)).payload
        noti_payload = (await notification_preparation_service.prepare(data, WORKFLOW_ID)).payload

        assert noti_payload["scheduledAt"].startswith(calendar_payload["date"])


@pytest.mark.parametrize("price", [1, 100_000_000])
async def test_extreme_prices_are_accepted(price):
    prepared = await gift_record_preparation_service.prepare(gift(gift_price=price), WORKFLOW_ID)
    assert prepared.payload["gift_price"] == price
