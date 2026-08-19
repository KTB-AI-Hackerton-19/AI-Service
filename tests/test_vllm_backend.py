"""추천이 이미지 분석과 같은 vLLM 엔진을 쓰는지 확인합니다."""

import os

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")

import json

import httpx
import pytest
import respx

from app.core.config import settings
from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services.qwen_service import RecommendationGenerationError, qwen_service

VLLM_URL = "http://localhost:8001/v1/chat/completions"

RECOMMENDATION = {
    "recommended_price_min": 28000,
    "recommended_price_max": 42000,
    "categories": [
        {
            "category": "디저트/베이커리",
            "score": 88,
            "reason": "받은 선물과 결이 비슷하면서 부담이 없습니다.",
            "product_examples": ["케이크 교환권"],
        }
    ],
    "summary": "비슷한 가격대의 디저트가 무난합니다.",
    "suggested_message": "지난번 케이크 정말 고마웠어요. 마음이 오래 남아서 작은 답례를 준비했습니다.",
}


def vllm_response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 200},
        },
    )


@pytest.fixture
def vllm_backend(monkeypatch):
    monkeypatch.setattr(settings, "model_backend", "vllm")


def request_fixture() -> SimpleGiftRecommendationRequest:
    return SimpleGiftRecommendationRequest(
        gift_name="스타벅스 케이크",
        gift_price=35000,
        age=29,
        person_name="김민수",
        relationship="대학 동기",
    )


@respx.mock
def test_recommendation_uses_same_vllm_endpoint(vllm_backend):
    route = respx.post(VLLM_URL).mock(return_value=vllm_response(RECOMMENDATION))

    result = qwen_service.recommend_simple(request_fixture())

    assert result.source == "GEMMA_VLLM"
    assert result.model == "gemma4-12b-qat"
    assert result.recommended_price_min == 28000
    # 이미지 분석과 같은 모델 이름으로 같은 서버에 요청했는지
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == settings.vllm_model


@respx.mock
def test_vllm_failure_raises_recommendation_error(vllm_backend):
    respx.post(VLLM_URL).mock(return_value=httpx.Response(503, text="no capacity"))

    with pytest.raises(RecommendationGenerationError, match="503"):
        qwen_service.recommend_simple(request_fixture())


@respx.mock
def test_malformed_json_raises_recommendation_error(vllm_backend):
    respx.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "설명만 있고 JSON 없음"}}]})
    )

    with pytest.raises(RecommendationGenerationError):
        qwen_service.recommend_simple(request_fixture())


def test_mock_backend_still_works():
    """기존 mock 경로는 그대로 동작해야 합니다."""
    result = qwen_service.recommend_simple(request_fixture())
    assert result.source == "MOCK"
