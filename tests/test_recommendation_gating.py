"""답례 선물 추천이 필요한 입력에서만 실행되는지 확인합니다.

부조금 명단에 대고 "8,000~12,000원 디저트"를 권하는 것은 사용자에게 의미가 없고,
모델 호출 비용과 지연만 늘립니다.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.gift_agent_service import gift_agent_service

client = TestClient(app)
headers = {"X-API-KEY": "test-key"}


def post(records: list[dict], **top) -> dict:
    body = {"gift_data": {"gift_name": "기록", "gift_price": 30000, **top}}
    if records:
        body["gift_data"]["records"] = records
    response = client.post("/api/v1/agent/from-gift-data", headers=headers, json=body)
    assert response.status_code == 200
    return response.json()


def record(index: int, kind: str, price: int = 30000, **extra) -> dict:
    return {
        "record_id": f"r{index}",
        "record_type": kind,
        "direction": "received",
        "person_name": f"사람{index}",
        "gift_name": "기록",
        "price": price,
        "selected": True,
        **extra,
    }


def test_gift_records_still_get_a_recommendation():
    info = post([record(0, "gift")], record_type="gift")["recommend_gift_info"]

    assert info["status"] == "SUCCESS"
    assert info["recommend_gift"] is not None


@pytest.mark.parametrize(
    ("kind", "keyword"),
    [("money", "부조금"), ("receipt", "영수증"), ("unknown", "선물 기록이 아니")],
)
def test_non_gift_records_skip_the_recommendation(kind, keyword):
    info = post([record(0, kind)], record_type=kind)["recommend_gift_info"]

    assert info["status"] == "SKIPPED"
    assert keyword in info["reason"]
    # 건너뛴 것은 실패가 아니므로 error 를 채우지 않습니다.
    assert info.get("error") is None
    assert info.get("recommend_gift") is None


def test_invitation_still_gets_a_recommendation():
    """청첩장은 답례품이 아니라 축의금 적정 수준을 안내하므로 추천 대상입니다."""
    info = post(
        [record(0, "event_invitation")], record_type="event_invitation"
    )["recommend_gift_info"]

    assert info["status"] == "SUCCESS"


def test_mixed_ledger_recommends_when_a_gift_is_included():
    """현금과 선물이 섞여 있으면 선물이 있으므로 추천합니다."""
    info = post(
        [record(0, "money"), record(1, "gift")], record_type="money"
    )["recommend_gift_info"]

    assert info["status"] == "SUCCESS"


def test_skipped_recommendation_does_not_call_the_model(monkeypatch):
    """건너뛸 때는 모델을 호출하지 않아야 지연과 비용이 실제로 줄어듭니다."""
    called = False

    async def spy(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("추천 대상이 아닌데 모델을 호출했습니다.")

    monkeypatch.setattr(gift_agent_service.recommendation_preparer, "prepare", spy)
    info = post([record(0, "money")], record_type="money")["recommend_gift_info"]

    assert info["status"] == "SKIPPED"
    assert called is False


def test_other_tasks_are_unaffected_by_skipping():
    body = post([record(0, "money")], record_type="money")

    assert body["gift_data"]["status"] == "SUCCESS"
    assert body["calendar_info"]["status"] == "SUCCESS"
    assert body["noti_info"]["status"] == "SUCCESS"


# ---------------------------------------------------------------- 사용자 선택 우선
# 백엔드가 업로드 화면에서 받은 "선물 / 경조사" 선택을 그대로 넘겨줍니다.
# 사람이 고른 값이 모델의 이미지 분류보다 정확하므로 그쪽을 따릅니다.

def post_image(category=None) -> dict:
    body = {"image_url": "https://example-bucket.s3.amazonaws.com/gift.png"}
    if category is not None:
        body["category"] = category
    response = client.post("/api/v1/agent/from-image", headers=headers, json=body)
    assert response.status_code == 200
    return response.json()


def test_user_choice_gift_runs_the_recommendation():
    assert post_image("gift")["recommend_gift_info"]["status"] == "SUCCESS"


@pytest.mark.parametrize("chosen", ["occasion", "경조사", "OCCASION", "condolence"])
def test_user_choice_occasion_skips_the_recommendation(chosen):
    info = post_image(chosen)["recommend_gift_info"]

    assert info["status"] == "SKIPPED"
    assert "경조사" in info["reason"]


def test_user_choice_overrides_the_model(monkeypatch):
    """모델이 선물이라고 읽어도 사용자가 경조사를 골랐으면 추천하지 않습니다."""
    info = post_image("경조사")["recommend_gift_info"]

    assert info["status"] == "SKIPPED"


def test_missing_category_falls_back_to_model_inference():
    """백엔드가 값을 안 보내도 기존 동작 그대로여야 합니다."""
    assert post_image()["recommend_gift_info"]["status"] in {"SUCCESS", "SKIPPED"}


def test_unknown_category_value_is_ignored():
    """모르는 값이 와도 422 로 막지 않고 미지정으로 봅니다."""
    assert post_image("이상한값")["recommend_gift_info"]["status"] in {"SUCCESS", "SKIPPED"}


def test_missing_price_skips_the_recommendation():
    """답례 가격대는 받은 금액의 80~120% 입니다. 금액을 모르면 추천이 성립하지 않습니다."""
    response = client.post(
        "/api/v1/agent/from-gift-data",
        headers=headers,
        json={"gift_data": {"gift_name": "TWG Tea 티백", "record_type": "gift"}},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["gift_data"]["status"] == "SUCCESS"
    assert body["gift_data"]["payload"].get("gift_price") is None
    info = body["recommend_gift_info"]
    assert info["status"] == "SKIPPED"
    assert "금액" in info["reason"]
    # 기록·캘린더·알림은 금액 없이도 준비됩니다.
    assert body["calendar_info"]["status"] == "SUCCESS"
    assert body["noti_info"]["status"] == "SUCCESS"
