import os

os.environ["MODEL_BACKEND"] = "mock"
os.environ["API_KEY"] = "test-key"
# 자동 테스트가 외부 credits를 소비하거나 네트워크 상태에 의존하지 않게 합니다.
os.environ["PRODUCT_SEARCH_PROVIDER"] = "disabled"

import asyncio

from fastapi.testclient import TestClient
import pytest

from app.core.config import settings
from app.main import app
from app.routers.agent import recommend
from app.schemas.agent import GiftData, InputCategory, RecommendRequest
from app.services.gift_agent_service import gift_agent_service
from app.services.gift_data_policy import GiftDataPolicyError
from app.services.tasks.recommendation import recommendation_preparation_service

# 백엔드에 권장한 HTTP 타임아웃(README). 우리 최악 지연이 이보다 낮아야
# 백엔드가 먼저 끊는 일이 없습니다.
BACKEND_TIMEOUT_SECONDS = 90

client = TestClient(app)
headers = {"X-API-KEY": "test-key"}


def test_swagger_exposes_only_business_endpoints():
    """준비용 두 개와 확정용 한 개. 그 밖의 API 는 외부에 노출하지 않습니다."""
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/api/v1/agent/from-gift-data",
        "/api/v1/agent/from-image",
        "/api/v1/agent/confirm",
        "/api/v1/agent/recommend",
    }
    error_schema = paths["/api/v1/agent/from-image"]["post"]["responses"]["401"]
    assert "ApiErrorResponse" in str(error_schema)


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
    assert {
        "gift_data",
        "calendar_info",
        "noti_info",
        "recommend_gift_info",
        "workflow_id",
        "requires_confirmation",
    } == set(body)
    # 준비 단계에서는 캘린더에 등록하지 않습니다. 사용자 확인 뒤 /confirm 에서 등록합니다.
    assert body["requires_confirmation"] is True
    assert body["calendar_info"]["payload"]["registered"] is False
    assert body["gift_data"]["payload"]["gift_name"] == "스타벅스 케이크"
    assert body["recommend_gift_info"]["recommend_gift"]["input_age"] == 29
    assert "suggested_message" not in body["recommend_gift_info"]["recommend_gift"]
    message = body["recommend_gift_info"]["message"]
    assert len(message["content"]) >= 120
    assert "김민수" in message["content"]
    assert message["generated_by"] == "MOCK"
    # generated_by 는 추천 백엔드일 뿐이라 메시지를 누가 썼는지 말하지 않습니다.
    # mock 은 모델을 부르지 않으므로 문장은 정책 템플릿입니다.
    assert message["message_source"] == "TEMPLATE_NO_OUTPUT"


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


@pytest.mark.parametrize("missing_gender", ["", "   ", "unknown", "UNKNOWN", None])
def test_empty_or_unknown_gender_is_treated_as_missing(missing_gender):
    response = client.post(
        "/api/v1/agent/from-gift-data",
        headers=headers,
        json={
            "gift_data": {
                "gift_name": "선물",
                "gift_price": 30000,
                "gender": missing_gender,
            }
        },
    )
    assert response.status_code == 200
    assert response.json()["gift_data"]["payload"]["gender"] is None


@pytest.mark.parametrize(
    ("input_gender", "expected"),
    [("male", "male"), ("MALE", "male"), ("남성", "male"), ("female", "female"), ("여성", "female")],
)
def test_gender_is_normalized_and_preserved(input_gender, expected):
    response = client.post(
        "/api/v1/agent/from-gift-data",
        headers=headers,
        json={
            "gift_data": {
                "gift_name": "선물",
                "gift_price": 30000,
                "gender": input_gender,
            }
        },
    )
    assert response.status_code == 200
    assert response.json()["gift_data"]["payload"]["gender"] == expected


def test_prepare_from_image():
    # MODEL_BACKEND=mock 이면 이미지를 실제로 내려받지 않고 고정된 추출 결과를 씁니다.
    # 특정 mock 문자열이 아니라 네 작업이 모두 준비됐는지를 확인합니다.
    response = client.post(
        "/api/v1/agent/from-image",
        headers=headers,
        json={"image_url": "https://example-bucket.s3.amazonaws.com/gift.png"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["workflow_id"]
    assert body["gift_data"]["status"] == "SUCCESS"
    payload = body["gift_data"]["payload"]
    assert payload["gift_name"]
    assert payload["gift_price"] > 0
    assert body["calendar_info"]["status"] == "SUCCESS"
    assert body["noti_info"]["status"] == "SUCCESS"
    assert body["recommend_gift_info"]["status"] == "SUCCESS"


def test_api_key_is_required():
    response = client.post(
        "/api/v1/agent/from-gift-data",
        json={"gift_data": {"gift_name": "케이크", "gift_price": 35000}},
    )
    assert response.status_code == 401
    assert response.json() == {
        "status": "ERROR",
        "error_code": "INVALID_API_KEY",
        "detail": "유효하지 않은 AI 서비스 API 키입니다.",
    }


def test_request_validation_error_uses_common_error_format():
    response = client.post(
        "/api/v1/agent/from-gift-data",
        headers=headers,
        json={"gift_data": {"gift_name": "", "gift_price": -1}},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "ERROR"
    assert body["error_code"] == "VALIDATION_ERROR"
    assert body["detail"] == "요청 데이터 형식이 올바르지 않습니다. 입력값을 확인해 주세요."
    assert body["errors"]


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
    assert body["gift_data"]["status"] == "SUCCESS"
    assert body["noti_info"]["status"] == "SUCCESS"
    assert body["recommend_gift_info"]["status"] == "SUCCESS"


def test_image_without_records_keeps_the_real_reason(monkeypatch):
    """왜 실패했는지가 사용자 화면에서 사라지면 다시 찍어야 할지 알 수 없습니다."""

    async def no_records(*_args, **_kwargs):
        raise GiftDataPolicyError("이미지에서 선물·부조금 기록을 찾지 못했습니다.")

    monkeypatch.setattr(gift_agent_service.image_analyzer, "analyze", no_records)
    response = client.post(
        "/api/v1/agent/from-image",
        headers=headers,
        json={"image_url": "https://example-bucket.s3.amazonaws.com/blank.png"},
    )

    assert response.status_code == 502
    body = response.json()
    assert body["error_code"] == "IMAGE_ANALYSIS_FAILED"
    assert body["detail"] == "이미지에서 선물·부조금 기록을 찾지 못했습니다."


def test_user_category_reaches_the_image_analyzer(monkeypatch):
    """사용자가 고른 종류는 추천 skip 판정뿐 아니라 추출 프롬프트에도 전달돼야 합니다."""
    seen: list = []

    original = gift_agent_service.image_analyzer.analyze

    async def spy(image_url, category=None):
        seen.append(category)
        return await original(image_url, category)

    monkeypatch.setattr(gift_agent_service.image_analyzer, "analyze", spy)
    response = client.post(
        "/api/v1/agent/from-image",
        headers=headers,
        json={
            "image_url": "https://example-bucket.s3.amazonaws.com/gift.png",
            "category": "경조사",
        },
    )

    assert response.status_code == 200
    assert seen == [InputCategory.OCCASION]


def test_worst_case_request_budget_stays_under_the_backend_timeout():
    """이미지 분석과 후속 작업은 직렬입니다. 두 예산의 합이 우리 최악 지연입니다.

    예전에는 한 값(REQUEST_TIMEOUT_SECONDS)을 두 단계에 각각 걸어 최악이 그 2배였고,
    로컬 .env(60초) 기준 120초로 권장값 90초를 넘었습니다.
    """
    worst_case = settings.image_analysis_timeout_seconds + settings.task_timeout_seconds
    assert worst_case < BACKEND_TIMEOUT_SECONDS
    # 짧게 잡아 정상 요청을 죽이지 않도록, 각 단계에 최소한의 여유는 남깁니다.
    assert settings.image_analysis_timeout_seconds >= 30
    assert settings.task_timeout_seconds >= 20


async def test_each_stage_gets_its_own_budget(monkeypatch):
    """단계마다 다른 예산이 실제로 전달되는지 확인합니다(한 값을 두 번 걸지 않습니다)."""
    applied: list[float] = []
    real_wait_for = asyncio.wait_for

    async def spy(coroutine, timeout):
        applied.append(timeout)
        return await real_wait_for(coroutine, timeout)

    monkeypatch.setattr(asyncio, "wait_for", spy)
    monkeypatch.setattr(settings, "image_analysis_timeout_seconds", 45.0)
    monkeypatch.setattr(settings, "task_timeout_seconds", 30.0)

    await gift_agent_service.run_from_image("https://example-bucket.s3.amazonaws.com/gift.png")

    # 이미지 분석 1회 + 후속 작업 4개.
    assert applied[0] == 45.0
    assert applied[1:] == [30.0] * 4


async def test_gift_data_path_uses_only_the_task_budget(monkeypatch):
    """/from-gift-data 에는 이미지 분석 단계가 없으므로 후속 작업 예산만 씁니다."""
    applied: list[float] = []
    real_wait_for = asyncio.wait_for

    async def spy(coroutine, timeout):
        applied.append(timeout)
        return await real_wait_for(coroutine, timeout)

    monkeypatch.setattr(asyncio, "wait_for", spy)

    await gift_agent_service.run_from_gift_data(
        GiftData(gift_name="케이크", gift_price=30000)
    )

    assert applied == [settings.task_timeout_seconds] * 4


async def test_recommend_endpoint_uses_the_task_budget(monkeypatch):
    """/recommend 는 오케스트레이터를 거치지 않으므로 라우터가 직접 예산을 겁니다.

    이 경로에 예산이 없으면 상한이 없습니다. bedrock_timeout_seconds(90) ×
    bedrock_max_retries(2) 에 Tavily 까지 겹치면 백엔드 권장 타임아웃을 넘고,
    그때는 백엔드가 먼저 끊어 우리 오류 코드가 사용자에게 닿지 않습니다.
    """
    applied: list[float] = []
    real_wait_for = asyncio.wait_for

    async def spy(coroutine, timeout):
        applied.append(timeout)
        return await real_wait_for(coroutine, timeout)

    monkeypatch.setattr(asyncio, "wait_for", spy)

    await recommend(RecommendRequest(gift_price=30000))

    # 같은 일(모델 1회 + 상품 검색)을 하는 /from-image 의 추천 작업과 같은 값이어야
    # 합니다. 경로마다 다르면 "한 번의 추천에 허용된 시간"이 두 개가 됩니다.
    assert applied == [settings.task_timeout_seconds]


def test_recommend_over_budget_returns_504(monkeypatch):
    """예산을 넘기면 500 이 아니라 504 + UPSTREAM_TIMEOUT 입니다.

    프론트가 "다시 시도" 와 "오류" 를 구분할 수 있어야 합니다.
    """

    async def never_finishes(_request):
        await asyncio.sleep(30)

    monkeypatch.setattr(settings, "task_timeout_seconds", 0.05)
    monkeypatch.setattr(
        recommendation_preparation_service, "recommend_only", never_finishes
    )

    response = client.post(
        "/api/v1/agent/recommend",
        headers=headers,
        json={"budget_min": 18000, "budget_max": 27000},
    )

    assert response.status_code == 504
    assert response.json()["error_code"] == "UPSTREAM_TIMEOUT"


def test_recommend_budget_leaves_room_for_the_extract_wait():
    """Extract 대기는 /recommend 의 임계 경로입니다(gather 의 늦은 쪽이 응답 시간).

    실측 20.17초 중 약 8초가 이 대기였습니다. 이 값이 예산에 육박하면 가격 확인
    하나 때문에 추천 전체가 504 가 됩니다.
    """
    assert settings.tavily_extract_timeout_seconds < settings.task_timeout_seconds / 2
