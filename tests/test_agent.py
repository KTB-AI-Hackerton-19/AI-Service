import os

os.environ["MODEL_BACKEND"] = "mock"
os.environ["API_KEY"] = "test-key"

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.services.gift_agent_service import gift_agent_service

client = TestClient(app)
headers = {"X-API-KEY": "test-key"}


def test_swagger_has_exactly_two_business_endpoints():
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/api/v1/agent/from-heart-data",
        "/api/v1/agent/from-image",
    }


def test_prepare_from_heart_data():
    response = client.post(
        "/api/v1/agent/from-heart-data",
        headers=headers,
        json={
            "heart_data": {
                "gift_name": "스타벅스 케이크",
                "gift_price": 35000,
                "age": 29,
                "person_name": "김민수",
                "target_date": "2026-09-10",
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"마음데이터", "캘린더정보", "알림정보", "추천선물정보"}
    assert body["마음데이터"]["payload"]["gift_name"] == "스타벅스 케이크"
    assert body["추천선물정보"]["추천선물"]["input_age"] == 29


def test_prepare_from_image():
    response = client.post(
        "/api/v1/agent/from-image",
        headers=headers,
        json={"image_url": "https://example-bucket.s3.amazonaws.com/gift.png"},
    )
    assert response.status_code == 200
    assert response.json()["마음데이터"]["payload"]["gift_name"] == "이미지에서 추출된 선물"


def test_api_key_is_required():
    response = client.post(
        "/api/v1/agent/from-heart-data",
        json={"heart_data": {"gift_name": "케이크", "gift_price": 35000}},
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "invalid_date",
    ["", None, "날짜 아님", "2026-99-99", "19-08-2026"],
)
def test_invalid_or_empty_dates_are_treated_as_missing(invalid_date):
    response = client.post(
        "/api/v1/agent/from-heart-data",
        headers=headers,
        json={
            "heart_data": {
                "gift_name": "케이크",
                "gift_price": 30000,
                "received_at": invalid_date,
                "target_date": invalid_date,
            }
        },
    )
    assert response.status_code == 200
    payload = response.json()["마음데이터"]["payload"]
    assert payload["received_at"] is None
    assert payload["target_date"] is None


def test_one_task_failure_does_not_remove_other_results(monkeypatch):
    async def failed_calendar(*_args, **_kwargs):
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr(
        gift_agent_service.calendar_preparer,
        "prepare",
        failed_calendar,
    )
    response = client.post(
        "/api/v1/agent/from-heart-data",
        headers=headers,
        json={"heart_data": {"gift_name": "케이크", "gift_price": 30000}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["캘린더정보"]["status"] == "ERROR"
    assert body["마음데이터"]["status"] == "READY"
    assert body["알림정보"]["status"] == "READY"
    assert body["추천선물정보"]["status"] == "READY"
