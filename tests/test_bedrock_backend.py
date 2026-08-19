"""추천과 이미지 분석이 Amazon Bedrock 의 Claude 를 쓰는지 확인합니다.

실제 AWS 로는 나가지 않습니다. anthropic SDK 가 httpx 를 쓰므로 respx 로 가로채
요청 형태와 응답 매핑, 오류 변환을 검증합니다.
"""

import json

import anthropic
import httpx
import pytest
import respx

from app.core.config import settings
from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services import bedrock_client
from app.services.image_loader import LoadedImage
from app.services.qwen_service import RecommendationGenerationError, qwen_service
from app.services.vlm_service import VisionAnalysisError, vlm_extraction_service

# 레거시 InvokeModel 경로의 URL. 모델 ID 가 경로에 들어갑니다.
BEDROCK_URL_PATTERN = r"https://bedrock-runtime\..*\.amazonaws\.com/model/.*/invoke"

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
    "suggested_message": (
        "지난번 케이크 정말 고마웠어요. 덕분에 하루가 즐거웠습니다. "
        "마음이 오래 남아서 작은 답례를 준비했어요. 편하게 받아주시면 좋겠습니다."
    ),
}

REQUEST = SimpleGiftRecommendationRequest(
    gift_name="스타벅스 케이크", gift_price=35_000, age=29
)


def make_image() -> LoadedImage:
    """image_loader 가 정규화해 넘겨주는 형태의 최소 이미지."""
    data = b"\x89PNG\r\n\x1a\n"
    return LoadedImage(
        data=data, mime="image/png", width=8, height=8, downloaded_bytes=len(data)
    )


def bedrock_response(text: str, stop_reason: str = "end_turn") -> httpx.Response:
    """Messages API 성공 응답을 흉내 냅니다."""
    return httpx.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": settings.bedrock_model_id,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 420, "output_tokens": 180},
        },
    )


@pytest.fixture(autouse=True)
def bedrock_backend(monkeypatch):
    """이 파일의 모든 테스트를 bedrock 백엔드로 고정하고 클라이언트를 새로 만듭니다."""
    monkeypatch.setattr(settings, "model_backend", "bedrock")
    bedrock_client.reset_clients()
    yield
    bedrock_client.reset_clients()


@respx.mock
def test_recommendation_uses_bedrock_and_maps_response():
    route = respx.post(url__regex=BEDROCK_URL_PATTERN).mock(
        return_value=bedrock_response(json.dumps(RECOMMENDATION, ensure_ascii=False))
    )

    result = qwen_service.recommend_simple(REQUEST)

    body = json.loads(route.calls[0].request.content)
    # system 은 messages 가 아니라 별도 인자로 가야 합니다.
    assert "당신은 한국의 답례 선물 추천 전문가입니다" in body["system"]
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert "스타벅스 케이크" in body["messages"][0]["content"]
    # Bedrock 은 구조화 출력을 지원하지 않으므로 보내면 안 됩니다.
    assert "response_format" not in body
    assert "output_config" not in body
    assert settings.bedrock_model_id in str(route.calls[0].request.url)

    assert result.source == "BEDROCK_CLAUDE"
    assert result.model == settings.bedrock_model_id
    assert result.input_gift_name == "스타벅스 케이크"
    assert result.input_age == 29
    assert result.categories


@respx.mock
def test_broken_json_falls_back_instead_of_breaking_the_workflow():
    """추천 하나가 실패해도 기록·캘린더·알림까지 깨뜨리지 않습니다."""
    respx.post(url__regex=BEDROCK_URL_PATTERN).mock(
        return_value=bedrock_response("추천을 도와드릴게요! 먼저 예산을 알려주세요.")
    )

    result = qwen_service.recommend_simple(REQUEST)

    assert result.source == "BEDROCK_CLAUDE_FALLBACK"
    assert result.categories
    assert result.suggested_message


@respx.mock
def test_upstream_message_is_surfaced_in_the_error():
    """403/404 의 진짜 원인은 Bedrock 이 준 메시지에만 담깁니다."""
    respx.post(url__regex=BEDROCK_URL_PATTERN).mock(
        return_value=httpx.Response(
            404,
            json={"message": "Model use case details have not been submitted."},
        )
    )

    with pytest.raises(RecommendationGenerationError) as caught:
        qwen_service.recommend_simple(REQUEST)

    assert "use case details" in str(caught.value)


@respx.mock
def test_connection_failure_is_converted():
    respx.post(url__regex=BEDROCK_URL_PATTERN).mock(
        side_effect=httpx.ConnectError("boom")
    )

    with pytest.raises(RecommendationGenerationError):
        qwen_service.recommend_simple(REQUEST)


def test_api_style_selects_the_client_class(monkeypatch):
    """계정마다 열린 경로가 달라 이 값이 틀리면 모든 모델이 403 이 됩니다."""
    monkeypatch.setattr(settings, "bedrock_api_style", "invoke")
    bedrock_client.reset_clients()
    assert isinstance(bedrock_client.get_client(), anthropic.AnthropicBedrock)

    monkeypatch.setattr(settings, "bedrock_api_style", "mantle")
    bedrock_client.reset_clients()
    assert isinstance(bedrock_client.get_client(), anthropic.AnthropicBedrockMantle)

    monkeypatch.setattr(settings, "bedrock_api_style", "converse")
    bedrock_client.reset_clients()
    with pytest.raises(bedrock_client.BedrockClientError):
        bedrock_client.get_client()


def test_api_key_and_profile_cannot_be_used_together(monkeypatch):
    monkeypatch.setattr(settings, "bedrock_api_key", "token")
    monkeypatch.setattr(settings, "bedrock_aws_profile", "hackathon")
    bedrock_client.reset_clients()

    with pytest.raises(bedrock_client.BedrockClientError):
        bedrock_client.get_client()


EXTRACTION = {
    "image_kind": "gift_message",
    "records": [
        {
            "person_name": "김민수",
            "gift_name": "스타벅스 케이크",
            "price": 35000,
            "received_at": "2026-08-01",
        }
    ],
}


@respx.mock
async def test_image_analysis_uses_bedrock():
    route = respx.post(url__regex=BEDROCK_URL_PATTERN).mock(
        return_value=bedrock_response(json.dumps(EXTRACTION, ensure_ascii=False))
    )
    image = make_image()

    result = await vlm_extraction_service.extract(image)

    body = json.loads(route.calls[0].request.content)
    blocks = body["messages"][0]["content"]
    # Bedrock 은 URL 이미지 소스를 지원하지 않으므로 base64 로 실어야 합니다.
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["type"] == "base64"
    assert blocks[0]["source"]["media_type"] == "image/png"
    assert result.payload["image_kind"] == "gift_message"
    assert result.prompt_tokens == 420


@respx.mock
async def test_image_analysis_error_is_converted():
    respx.post(url__regex=BEDROCK_URL_PATTERN).mock(
        return_value=httpx.Response(403, json={"message": "no access"})
    )
    image = make_image()

    with pytest.raises(VisionAnalysisError) as caught:
        await vlm_extraction_service.extract(image)

    assert "no access" in str(caught.value)


@respx.mock
async def test_image_sampling_param_is_dropped_when_model_rejects_it():
    """이미지 분석도 추천과 동일하게 temperature 거부 시 한 번 재시도합니다."""
    vlm_extraction_service._bedrock_accepts_sampling = True
    route = respx.post(url__regex=BEDROCK_URL_PATTERN).mock(
        side_effect=[
            httpx.Response(400, json={"message": "temperature is not supported"}),
            bedrock_response(json.dumps(EXTRACTION, ensure_ascii=False)),
        ]
    )

    result = await vlm_extraction_service.extract(make_image())

    first = json.loads(route.calls[0].request.content)
    second = json.loads(route.calls[1].request.content)
    assert "temperature" in first
    assert "temperature" not in second
    assert result.payload["image_kind"] == "gift_message"
    assert not vlm_extraction_service._bedrock_accepts_sampling


@respx.mock
def test_sampling_params_are_dropped_when_the_model_rejects_them():
    """BEDROCK_MODEL_ID 를 최신 Claude 로 바꿔도 400 으로 죽지 않아야 합니다."""
    qwen_service._bedrock_accepts_sampling = True
    route = respx.post(url__regex=BEDROCK_URL_PATTERN).mock(
        side_effect=[
            httpx.Response(400, json={"message": "temperature is not supported"}),
            bedrock_response(json.dumps(RECOMMENDATION, ensure_ascii=False)),
        ]
    )

    result = qwen_service.recommend_simple(REQUEST)

    first = json.loads(route.calls[0].request.content)
    second = json.loads(route.calls[1].request.content)
    assert "temperature" in first
    # Claude 는 temperature 와 top_p 를 동시에 받지 않습니다.
    assert "top_p" not in first
    assert "temperature" not in second
    assert result.source == "BEDROCK_CLAUDE"

    # 한 번 내려간 뒤에는 다시 보내지 않습니다.
    respx.post(url__regex=BEDROCK_URL_PATTERN).mock(
        return_value=bedrock_response(json.dumps(RECOMMENDATION, ensure_ascii=False))
    )
    qwen_service.recommend_simple(REQUEST)
    assert not qwen_service._bedrock_accepts_sampling
