"""사용자 승인 이후의 확정 흐름 테스트.

핵심 검증: 준비 단계에서는 캘린더에 등록되지 않고, /confirm 에서만 등록된다는 것.
그리고 사용자가 고친 값이 확정 결과에 그대로 반영된다는 것.
"""

import os

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.calendar_mcp_client import CalendarMcpError

client = TestClient(app)
HEADERS = {"X-API-KEY": "test-key"}


@pytest.fixture
def created_events(monkeypatch):
    """MCP 로 나가는 등록 호출을 가로챕니다."""
    calls: list[dict] = []

    async def fake_create_event(**kwargs):
        calls.append(kwargs)
        return {
            "event_id": f"evt-{len(calls)}",
            "html_link": f"https://calendar.google.com/e/{len(calls)}",
        }

    monkeypatch.setattr(
        "app.services.tasks.calendar.calendar_mcp_client.create_event", fake_create_event
    )
    monkeypatch.setattr(settings, "calendar_auto_register", False)
    return calls


def prepare(**overrides) -> dict:
    """준비 단계를 돌려 확정에 넣을 응답을 얻습니다."""
    gift_data = {
        "gift_name": "스타벅스 케이크",
        "gift_price": 35000,
        "person_name": "김민수",
        "relationship": "대학 동기",
        "received_at": "2026-08-19",
        "target_date": "2026-09-10",
    }
    gift_data.update(overrides)
    response = client.post(
        "/api/v1/agent/from-gift-data", headers=HEADERS, json={"gift_data": gift_data}
    )
    assert response.status_code == 200
    return response.json()


def confirm_body(prepared: dict, **overrides) -> dict:
    body = {
        "workflow_id": prepared["workflow_id"],
        "gift_data": prepared["gift_data"]["payload"],
        "calendar": prepared["calendar_info"]["payload"],
        "approved": True,
        "google_access_token": "ya29.fake-token",
    }
    body.update(overrides)
    return body


class TestConfirmFlow:
    def test_preparation_does_not_register(self, created_events):
        prepared = prepare()
        assert prepared["requires_confirmation"] is True
        assert prepared["calendar_info"]["payload"]["registered"] is False
        assert created_events == []

    def test_confirm_registers_calendar(self, created_events):
        prepared = prepare()
        response = client.post("/api/v1/agent/confirm", headers=HEADERS, json=confirm_body(prepared))

        assert response.status_code == 200
        body = response.json()
        assert body["approved"] is True
        assert body["workflow_id"] == prepared["workflow_id"]

        calendar = body["calendar_info"]["payload"]
        assert calendar["registered"] is True
        assert calendar["provider"] == "GOOGLE_MCP"
        assert calendar["eventId"] == "evt-1"
        assert len(created_events) == 1
        assert created_events[0]["access_token"] == "ya29.fake-token"

    def test_rejection_registers_nothing(self, created_events):
        prepared = prepare()
        response = client.post(
            "/api/v1/agent/confirm", headers=HEADERS, json=confirm_body(prepared, approved=False)
        )

        assert response.status_code == 200
        body = response.json()
        assert body["approved"] is False
        assert body["calendar_info"]["payload"]["registered"] is False
        assert created_events == []

    def test_register_calendar_false_keeps_draft(self, created_events):
        prepared = prepare()
        response = client.post(
            "/api/v1/agent/confirm",
            headers=HEADERS,
            json=confirm_body(prepared, register_calendar=False),
        )

        body = response.json()
        assert body["approved"] is True
        assert body["calendar_info"]["payload"]["registered"] is False
        assert created_events == []

    def test_user_edits_are_reflected(self, created_events):
        prepared = prepare()
        edited = confirm_body(prepared)
        edited["gift_data"]["gift_price"] = 40000
        edited["gift_data"]["person_name"] = "김민수(대학)"
        edited["calendar"]["title"] = "민수한테 답례하기"
        edited["calendar"]["date"] = "2026-09-01"

        response = client.post("/api/v1/agent/confirm", headers=HEADERS, json=edited)
        body = response.json()

        assert body["gift_data"]["payload"]["gift_price"] == 40000
        assert body["gift_data"]["payload"]["person_name"] == "김민수(대학)"
        assert created_events[0]["summary"] == "민수한테 답례하기"
        assert created_events[0]["start_date"] == "2026-09-01"

    def test_calendar_omitted_is_recomputed_from_edits(self, created_events):
        """일정을 안 보내면 수정된 기록으로 다시 계산합니다."""
        prepared = prepare()
        body = confirm_body(prepared)
        body["calendar"] = None
        body["gift_data"]["person_name"] = "박서준"
        body["gift_data"]["target_date"] = "2026-12-25"

        response = client.post("/api/v1/agent/confirm", headers=HEADERS, json=body)
        calendar = response.json()["calendar_info"]["payload"]

        assert calendar["title"] == "박서준님 답례 준비"
        assert calendar["date"] == "2026-12-18"  # 12/25 - 7일

    def test_missing_token_reports_error_without_crashing(self, created_events, monkeypatch):
        monkeypatch.setattr(settings, "google_access_token", "")
        prepared = prepare()
        body = confirm_body(prepared)
        body.pop("google_access_token")

        response = client.post("/api/v1/agent/confirm", headers=HEADERS, json=body)

        assert response.status_code == 200
        calendar = response.json()["calendar_info"]["payload"]
        assert calendar["registered"] is False
        assert "token" in calendar["registerError"]
        assert created_events == []

    def test_mcp_failure_is_reported_not_raised(self, monkeypatch):
        async def failing(**_kwargs):
            raise CalendarMcpError("MCP 서버에 연결할 수 없습니다")

        monkeypatch.setattr(settings, "calendar_auto_register", False)
        monkeypatch.setattr("app.services.tasks.calendar.calendar_mcp_client.create_event", failing)

        prepared = prepare()
        response = client.post("/api/v1/agent/confirm", headers=HEADERS, json=confirm_body(prepared))

        assert response.status_code == 200
        calendar = response.json()["calendar_info"]["payload"]
        assert calendar["registered"] is False
        assert "연결할 수 없습니다" in calendar["registerError"]

    def test_api_key_is_required(self):
        response = client.post("/api/v1/agent/confirm", json={"workflow_id": "x", "gift_data": {}})
        assert response.status_code == 401


class TestMultiRecordConfirm:
    """다건 이미지에서 사용자가 일부만 남기는 경우."""

    def _multi_records(self) -> list[dict]:
        return [
            {
                "record_id": f"r{i}",
                "record_type": "money",
                "direction": direction,
                "person_name": name,
                "gift_name": "축의금",
                "price": price,
                "received_at": "2026-08-19",
                "category": "축의금",
                "confidence": 1.0,
                "selected": True,
            }
            for i, (name, price, direction) in enumerate(
                [
                    ("김도윤", 100000, "received"),
                    ("박서준", 50000, "received"),
                    ("최은비", 200000, "received"),
                    ("카카오페이", 38900, "sent"),
                ]
            )
        ]

    def test_all_records_flow_through(self, created_events):
        prepared = prepare(
            gift_name="축의금", gift_price=200000, person_name="최은비", records=self._multi_records()
        )
        body = confirm_body(prepared)
        response = client.post("/api/v1/agent/confirm", headers=HEADERS, json=body)
        payload = response.json()["gift_data"]["payload"]

        assert len(payload["records"]) == 4
        assert payload["recordCount"] == 4  # 저장은 4건 전부
        # 출금(sent)은 답례 대상이 아니므로 답례 대상 수·합계·명단에서 빠집니다.
        assert payload["receivedCount"] == 3
        assert payload["totalAmount"] == 350000
        assert "김도윤님 외 2명" in payload["summary"]

    def test_deselected_records_are_excluded(self, created_events):
        records = self._multi_records()
        records[1]["selected"] = False  # 박서준 건을 사용자가 뺌
        prepared = prepare(
            gift_name="축의금", gift_price=200000, person_name="최은비", records=records
        )
        response = client.post(
            "/api/v1/agent/confirm", headers=HEADERS, json=confirm_body(prepared)
        )
        payload = response.json()["gift_data"]["payload"]

        assert payload["receivedCount"] == 2
        assert payload["totalAmount"] == 300000
        assert "김도윤님 외 1명" in payload["summary"]

    def test_calendar_lists_selected_people_only(self, created_events):
        records = self._multi_records()
        records[2]["selected"] = False  # 최은비 제외
        prepared = prepare(
            gift_name="축의금", gift_price=200000, person_name="최은비", records=records
        )
        body = confirm_body(prepared)
        body["calendar"] = None  # 수정된 기록으로 다시 계산하게 함

        response = client.post("/api/v1/agent/confirm", headers=HEADERS, json=body)
        calendar = response.json()["calendar_info"]["payload"]

        assert "김도윤" in calendar["description"]
        assert "박서준" in calendar["description"]
        assert "최은비" not in calendar["description"]
        assert calendar["title"] == "김도윤님 외 1명 답례 준비"


def test_confirm_response_shape():
    """백엔드가 기대할 최상위 키."""
    prepared = prepare()
    body = confirm_body(prepared, register_calendar=False)
    response = client.post("/api/v1/agent/confirm", headers=HEADERS, json=body)

    assert set(response.json()) == {
        "workflow_id",
        "approved",
        "gift_data",
        "calendar_info",
        "noti_info",
    }


def test_received_date_normalization_survives_round_trip():
    """준비 응답을 그대로 되돌려줘도 날짜가 깨지지 않아야 합니다."""
    prepared = prepare(received_at="2026-08-19", target_date="2026-09-10")
    body = confirm_body(prepared, register_calendar=False)
    response = client.post("/api/v1/agent/confirm", headers=HEADERS, json=body)
    payload = response.json()["gift_data"]["payload"]

    assert payload["received_at"] == "2026-08-19"
    assert payload["target_date"] == "2026-09-10"
    assert date.fromisoformat(payload["resolvedTargetDate"]) == date(2026, 9, 10)
