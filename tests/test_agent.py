import os

os.environ["MODEL_BACKEND"] = "mock"
os.environ["API_KEY"] = "test-key"
# 자동 테스트가 외부 credits를 소비하거나 네트워크 상태에 의존하지 않게 합니다.
os.environ["PRODUCT_SEARCH_PROVIDER"] = "disabled"

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.services.gift_agent_service import gift_agent_service

client = TestClient(app)
headers = {"X-API-KEY": "test-key"}


def test_swagger_has_exactly_two_business_endpoints():
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/api/v1/agent/from-gift-data",
        "/api/v1/agent/from-image",
    }


def test_prepare_from_gift_data():
    response = client.post(
        "/api/v1/agent/from-gift-data",
        headers=headers,
        json={
            "gift_data": {
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
    assert set(body) == {
        "gift_data",
        "calendar_info",
        "noti_info",
        "recommend_gift_info",
    }
    assert body["gift_data"]["payload"]["gift_name"] == "스타벅스 케이크"
    assert body["recommend_gift_info"]["recommend_gift"]["input_age"] == 29
    assert "suggested_message" not in body["recommend_gift_info"]["recommend_gift"]
    message = body["recommend_gift_info"]["message"]
    assert len(message["content"]) >= 120
    assert "김민수" in message["content"]
    assert message["generated_by"] == "MOCK"


def test_low_price_keeps_meaningful_80_to_120_percent_range():
    response = client.post(
        "/api/v1/agent/from-gift-data",
        headers=headers,
        json={"gift_data": {"gift_name": "헤이", "gift_price": 1101}},
    )
    assert response.status_code == 200
    recommendation = response.json()["recommend_gift_info"]["recommend_gift"]
    assert recommendation["recommended_price_min"] == 800
    assert recommendation["recommended_price_max"] == 1400


@pytest.mark.parametrize("missing_age", [0, "0", "", "   ", None])
def test_zero_or_empty_age_is_treated_as_missing(missing_age):
    response = client.post(
        "/api/v1/agent/from-gift-data",
        headers=headers,
        json={
            "gift_data": {
                "gift_name": "선물",
                "gift_price": 1101,
                "age": missing_age,
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["gift_data"]["payload"]["age"] is None
    # 공개 응답은 response_model_exclude_none=True이므로 null 필드는 생략됩니다.
    assert body["recommend_gift_info"]["recommend_gift"].get("input_age") is None


def test_empty_optional_text_is_treated_as_missing():
    response = client.post(
        "/api/v1/agent/from-gift-data",
        headers=headers,
        json={
            "gift_data": {
                "gift_name": "  선물  ",
                "gift_price": 10000,
                "person_name": "   ",
                "relationship": "",
            }
        },
    )
    assert response.status_code == 200
    payload = response.json()["gift_data"]["payload"]
    assert payload["gift_name"] == "선물"
    assert payload["person_name"] is None
    assert payload["relationship"] is None


def test_prepare_from_image():
    response = client.post(
        "/api/v1/agent/from-image",
        headers=headers,
        json={"image_url": "https://example-bucket.s3.amazonaws.com/gift.png"},
    )
    assert response.status_code == 200
    assert response.json()["gift_data"]["payload"]["gift_name"] == "이미지에서 추출된 선물"


def test_api_key_is_required():
    response = client.post(
        "/api/v1/agent/from-gift-data",
        json={"gift_data": {"gift_name": "케이크", "gift_price": 35000}},
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "invalid_date",
    ["", None, "날짜 아님", "2026-99-99", "19-08-2026"],
)
def test_invalid_or_empty_dates_are_treated_as_missing(invalid_date):
    response = client.post(
        "/api/v1/agent/from-gift-data",
        headers=headers,
        json={
            "gift_data": {
                "gift_name": "케이크",
                "gift_price": 30000,
                "received_at": invalid_date,
                "target_date": invalid_date,
            }
        },
    )
    assert response.status_code == 200
    payload = response.json()["gift_data"]["payload"]
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
        "/api/v1/agent/from-gift-data",
        headers=headers,
        json={"gift_data": {"gift_name": "케이크", "gift_price": 30000}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["calendar_info"]["status"] == "ERROR"
    assert body["gift_data"]["status"] == "READY"
    assert body["noti_info"]["status"] == "READY"
    assert body["recommend_gift_info"]["status"] == "READY"
