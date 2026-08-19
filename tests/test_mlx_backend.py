"""MLX 로컬 추론의 JSON 실패 복구 동작을 검증합니다."""

import json
import sys
from types import ModuleType

from app.core.config import settings
from app.schemas.recommendation import SimpleGiftRecommendationRequest
from app.services.qwen_service import QwenRecommendationService


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "prompt"


def install_fake_mlx(monkeypatch, outputs: list[str]) -> None:
    mlx_lm = ModuleType("mlx_lm")
    sample_utils = ModuleType("mlx_lm.sample_utils")

    def generate(*_args, **_kwargs):
        return outputs.pop(0)

    mlx_lm.generate = generate
    sample_utils.make_sampler = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)


def service() -> QwenRecommendationService:
    instance = QwenRecommendationService()
    instance._model = object()
    instance._tokenizer = FakeTokenizer()
    return instance


def request() -> SimpleGiftRecommendationRequest:
    return SimpleGiftRecommendationRequest(
        gift_name="꽃",
        gift_price=23333,
        age=32,
        gender="male",
        person_name="김영삼",
    )


def test_two_invalid_json_responses_use_safe_fallback(monkeypatch):
    monkeypatch.setattr(settings, "model_backend", "mlx")
    install_fake_mlx(monkeypatch, ["JSON이 아닌 답변", "다시 실패한 답변"])

    result = service().recommend_simple(request())

    assert result.source == "QWEN_MLX_FALLBACK"
    assert result.recommended_price_min == 18000
    assert result.recommended_price_max == 28000
    assert result.categories
    assert len(result.suggested_message) >= 120


def test_valid_json_keeps_mlx_source(monkeypatch):
    monkeypatch.setattr(settings, "model_backend", "mlx")
    payload = {
        "categories": [
            {
                "category": "꽃·식물",
                "score": 95,
                "reason": "받은 꽃과 자연스럽게 어울립니다.",
            }
        ],
        "summary": "꽃을 받은 맥락에 맞춘 추천입니다.",
        "suggested_message": "김영삼님, 따뜻한 마음으로 꽃을 챙겨 주셔서 정말 감사했어요. " * 3,
    }
    install_fake_mlx(monkeypatch, [json.dumps(payload, ensure_ascii=False)])

    result = service().recommend_simple(request())

    assert result.source == "QWEN_MLX"
    assert result.categories[0].category == "꽃·식물"
