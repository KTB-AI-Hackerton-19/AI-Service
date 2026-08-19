"""Google Calendar MCP 서버와 클라이언트 테스트.

MCP SDK 2.0 의 인메모리 전송을 써서 네트워크 없이 실제 MCP 왕복을 검증합니다.
Google API 호출만 가짜로 바꿉니다.
"""

import os

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")

import pytest

from app.services.calendar_mcp_client import CalendarMcpClient, CalendarMcpError
from mcp_servers import google_calendar
from mcp_servers.google_calendar import _build_event_body, mcp


class TestBuildEventBody:
    def test_timed_event(self):
        body = _build_event_body(
            summary="김수현님 답례 준비",
            description="설명",
            all_day=False,
            start_date="2026-09-05",
            end_date=None,
            start_time="10:00",
            duration_minutes=30,
            timezone="Asia/Seoul",
            reminders_minutes=[0, 1440],
        )

        assert body["start"] == {"dateTime": "2026-09-05T10:00:00", "timeZone": "Asia/Seoul"}
        assert body["end"] == {"dateTime": "2026-09-05T10:30:00", "timeZone": "Asia/Seoul"}
        assert body["reminders"]["useDefault"] is False
        assert [o["minutes"] for o in body["reminders"]["overrides"]] == [0, 1440]

    def test_all_day_end_date_is_exclusive(self):
        """Google 종일 일정의 end.date 는 배타적이라 하루짜리도 다음 날을 넣어야 합니다."""
        body = _build_event_body(
            summary="s",
            description="",
            all_day=True,
            start_date="2026-09-05",
            end_date=None,
            start_time=None,
            duration_minutes=30,
            timezone="Asia/Seoul",
            reminders_minutes=None,
        )

        assert body["start"] == {"date": "2026-09-05"}
        assert body["end"] == {"date": "2026-09-06"}
        assert "reminders" not in body

    def test_out_of_range_reminders_are_dropped(self):
        """Google 은 0~40320 분만 허용합니다."""
        body = _build_event_body(
            summary="s",
            description="",
            all_day=False,
            start_date="2026-09-05",
            end_date=None,
            start_time="10:00",
            duration_minutes=30,
            timezone="Asia/Seoul",
            reminders_minutes=[-540, 0, 40_320, 50_000],
        )

        assert [o["minutes"] for o in body["reminders"]["overrides"]] == [0, 40_320]

    def test_zero_duration_is_clamped(self):
        body = _build_event_body(
            summary="s",
            description="",
            all_day=False,
            start_date="2026-09-05",
            end_date=None,
            start_time="10:00",
            duration_minutes=0,
            timezone="Asia/Seoul",
            reminders_minutes=None,
        )
        assert body["end"]["dateTime"] == "2026-09-05T10:01:00"


class TestMcpRoundTrip:
    """인메모리 MCP 전송으로 클라이언트-서버 왕복을 실제로 태웁니다."""

    async def test_create_event(self, monkeypatch):
        captured = {}

        async def fake_call_google(method, path, access_token, *, json_body=None, params=None):
            captured.update(
                {"method": method, "path": path, "token": access_token, "body": json_body}
            )
            return {
                "id": "evt-123",
                "htmlLink": "https://calendar.google.com/event?eid=abc",
                "summary": json_body["summary"],
                "start": json_body["start"],
                "status": "confirmed",
            }

        monkeypatch.setattr(google_calendar, "_call_google", fake_call_google)

        client = CalendarMcpClient(server=mcp)
        result = await client.create_event(
            access_token="ya29.fake-token",
            summary="김수현님 답례 준비",
            start_date="2026-09-05",
            description="답례를 준비할 시간입니다",
        )

        assert result["event_id"] == "evt-123"
        assert result["html_link"] == "https://calendar.google.com/event?eid=abc"
        assert captured["method"] == "POST"
        assert captured["path"] == "/calendars/primary/events"
        assert captured["token"] == "ya29.fake-token"
        assert captured["body"]["summary"] == "김수현님 답례 준비"

    async def test_google_error_surfaces_as_mcp_error(self, monkeypatch):
        async def failing_call_google(*_args, **_kwargs):
            raise google_calendar.CalendarApiError("Google Calendar API 401: invalid token")

        monkeypatch.setattr(google_calendar, "_call_google", failing_call_google)

        client = CalendarMcpClient(server=mcp)
        with pytest.raises(CalendarMcpError):
            await client.create_event(
                access_token="expired", summary="s", start_date="2026-09-05"
            )

    async def test_list_tool_names(self):
        client = CalendarMcpClient(server=mcp)
        names = await client.list_tool_names()
        assert set(names) == {
            "create_event",
            "update_event",
            "get_event",
            "delete_event",
            "list_events",
        }

    async def test_get_event_returns_reminders(self, monkeypatch):
        """알림이 실제로 걸렸는지는 등록 응답만으로는 알 수 없어 따로 조회합니다."""

        async def fake_call_google(method, path, access_token, **_kwargs):
            assert method == "GET"
            return {
                "id": "evt-1",
                "summary": "김수현님 답례 준비",
                "status": "confirmed",
                "description": "답례를 준비할 시간입니다.",
                "reminders": {
                    "useDefault": False,
                    "overrides": [{"method": "popup", "minutes": 0}, {"method": "popup", "minutes": 1440}],
                },
            }

        monkeypatch.setattr(google_calendar, "_call_google", fake_call_google)

        client = CalendarMcpClient(server=mcp)
        result = await client.call_tool("get_event", {"access_token": "t", "event_id": "evt-1"})

        minutes = sorted(o["minutes"] for o in result["reminders"]["overrides"])
        assert minutes == [0, 1440]
        assert result["status"] == "confirmed"

    async def test_delete_event(self, monkeypatch):
        async def fake_call_google(*_args, **_kwargs):
            return {}

        monkeypatch.setattr(google_calendar, "_call_google", fake_call_google)

        client = CalendarMcpClient(server=mcp)
        result = await client.call_tool(
            "delete_event", {"access_token": "t", "event_id": "evt-1"}
        )
        assert result == {"deleted": True, "event_id": "evt-1"}
