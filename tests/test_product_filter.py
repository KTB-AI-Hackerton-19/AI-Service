"""검색 결과 판정이 모델 우선, 키워드 폴백으로 동작하는지 확인합니다.

키워드 사전은 부분 문자열 매칭이라 "차"가 "차량"에 걸리고 브랜드 표기는 놓칩니다.
모델 판정으로 바꾸되, 모델을 못 쓰는 상황에서도 추천이 죽지 않아야 합니다.
"""

import json
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.schemas.recommendation import ProductSuggestion
from app.services import product_filter
from app.services.product_search import filter_relevant


def item(title: str, category: str = "커피·차") -> ProductSuggestion:
    return ProductSuggestion(
        title=title, url=f"https://www.kurly.com/goods/{abs(hash(title)) % 9999}",
        source="컬리", category=category, kind="product", reason="",
    )


class FakeMessages:
    def __init__(self, verdicts=None, error=None):
        self.verdicts = verdicts
        self.error = error
        self.calls = []

    async def create(self, **payload):
        self.calls.append(payload)
        if self.error:
            raise self.error
        text = json.dumps({"keep": self.verdicts})
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(settings, "model_backend", "bedrock")
    monkeypatch.setattr(settings, "product_llm_filter_enabled", True)

    def install(verdicts=None, error=None):
        fake = FakeMessages(verdicts, error)
        from app.services import bedrock_client

        monkeypatch.setattr(
            bedrock_client, "get_async_client", lambda: SimpleNamespace(messages=fake)
        )
        return fake

    return install


@pytest.mark.asyncio
async def test_model_verdict_decides_what_survives(model):
    """키워드로는 '차량'이 '차'에 걸려 통과했습니다. 모델은 이를 걸러야 합니다."""
    fake = model([1])
    batches = [[item("모모스커피 드립백 선물세트"), item("차량용 방향제 선물세트")]]

    kept = await filter_relevant(batches, [None])

    assert [p.title for p in kept[0]] == ["모모스커피 드립백 선물세트"]
    # 검색 횟수와 무관하게 한 번만 부릅니다.
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_all_candidates_go_in_one_call(model):
    """카테고리가 여러 개여도 호출은 한 번이어야 합니다."""
    fake = model([1, 2, 3])
    batches = [[item("드립백")], [item("마카롱", "식품·디저트"), item("쿠키", "식품·디저트")]]

    kept = await filter_relevant(batches, [None, None])

    assert len(fake.calls) == 1
    assert [len(b) for b in kept] == [1, 2]


@pytest.mark.asyncio
async def test_numbers_outside_the_range_are_ignored(model):
    """모델이 없는 번호를 내도 다른 항목의 판정을 망치지 않아야 합니다."""
    model([1, 99])
    batches = [[item("모모스커피 드립백"), item("삼성 무선충전기")]]

    kept = await filter_relevant(batches, [None])

    assert [p.title for p in kept[0]] == ["모모스커피 드립백"]


@pytest.mark.asyncio
async def test_model_failure_falls_back_to_keywords(model):
    """모델 호출이 죽어도 추천은 계속돼야 합니다."""
    model(error=RuntimeError("bedrock down"))
    batches = [[item("모모스커피 드립백"), item("삼성 무선충전기")]]

    kept = await filter_relevant(batches, [None])

    assert [p.title for p in kept[0]] == ["모모스커피 드립백"]


@pytest.mark.asyncio
async def test_packaging_is_dropped_by_keyword_fallback(model):
    """포장재 제외도 판정에 포함됩니다. 폴백 경로에서는 금지어가 맡습니다."""
    model(error=RuntimeError("down"))
    batches = [[item("커피 선물용 쇼핑백 대형 10매")]]

    kept = await filter_relevant(batches, [None])

    assert kept[0] == []


@pytest.mark.asyncio
async def test_other_backends_never_call_the_model(monkeypatch):
    monkeypatch.setattr(settings, "model_backend", "vllm")

    def explode():
        raise AssertionError("bedrock 이 아닌데 모델을 불렀습니다.")

    from app.services import bedrock_client

    monkeypatch.setattr(bedrock_client, "get_async_client", explode)

    kept = await filter_relevant([[item("모모스커피 드립백")]], [None])

    assert len(kept[0]) == 1


@pytest.mark.asyncio
async def test_empty_input_makes_no_call(model):
    fake = model([])
    assert await filter_relevant([], []) == []
    assert fake.calls == []
