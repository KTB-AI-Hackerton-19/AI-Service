"""검색 결과 판정이 모델 우선, 키워드 폴백으로 동작하는지 확인합니다.

키워드 사전은 부분 문자열 매칭이라 "차"가 "차량"에 걸리고 브랜드 표기는 놓칩니다.
모델 판정으로 바꾸되, 모델을 못 쓰는 상황에서도 추천이 죽지 않아야 합니다.
"""

import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import settings
from app.schemas.recommendation import ProductSuggestion
from app.services import bedrock_client
from app.services import product_filter
from app.services import product_search
from app.services.product_search import filter_relevant


def item(title: str, category: str = "디저트") -> ProductSuggestion:
    return ProductSuggestion(
        title=title, url=f"https://www.kurly.com/goods/{abs(hash(title)) % 9999}",
        source="컬리", category=category, kind="product", reason="",
    )


class FakeMessages:
    def __init__(self, keep=None, drop=None, error=None):
        self.keep = keep
        self.drop = drop
        self.error = error
        self.calls = []

    async def create(self, **payload):
        self.calls.append(payload)
        if self.error:
            raise self.error
        text = json.dumps({"keep": self.keep or [], "drop": self.drop or []})
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=412, output_tokens=18),
        )


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(settings, "model_backend", "bedrock")
    monkeypatch.setattr(settings, "product_llm_filter_enabled", True)

    def install(keep=None, drop=None, error=None):
        fake = FakeMessages(keep, drop, error)
        from app.services import bedrock_client

        monkeypatch.setattr(
            bedrock_client, "get_async_client", lambda: SimpleNamespace(messages=fake)
        )
        return fake

    return install


@pytest.mark.asyncio
async def test_model_verdict_decides_what_survives(model):
    """키워드로는 '차량'이 '차'에 걸려 통과했습니다. 모델이 제외하면 그것으로 끝입니다."""
    fake = model(keep=[1], drop=[2])
    batches = [[item("모모스커피 드립백 선물세트"), item("차량용 방향제 선물세트")]]

    kept = await filter_relevant(batches, [None])

    assert [p.title for p in kept[0]] == ["모모스커피 드립백 선물세트"]
    # 검색 횟수와 무관하게 한 번만 부릅니다.
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_all_candidates_go_in_one_call(model):
    """카테고리가 여러 개여도 호출은 한 번이어야 합니다."""
    fake = model(keep=[1, 2, 3])
    batches = [[item("드립백")], [item("마카롱", "식품·디저트"), item("쿠키", "식품·디저트")]]

    kept = await filter_relevant(batches, [None, None])

    assert len(fake.calls) == 1
    assert [len(b) for b in kept] == [1, 2]


@pytest.mark.asyncio
async def test_numbers_outside_the_range_are_ignored(model):
    """모델이 없는 번호를 내도 다른 항목의 판정을 망치지 않아야 합니다."""
    model(keep=[1, 99], drop=[2, 0])
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
    fake = model()
    assert await filter_relevant([], []) == []
    assert fake.calls == []


@pytest.mark.asyncio
async def test_number_missing_from_both_lists_falls_back_to_keywords(model):
    """모델이 번호를 빠뜨리면 조용히 탈락시키지 않고 키워드로 다시 봅니다.

    통과 번호만 받던 시절에는 빠진 번호가 곧 탈락이라, 모델이 하나를 잊으면
    멀쩡한 상품이 이유 없이 사라졌습니다.
    """
    model(keep=[1], drop=[])  # 2·3번을 통째로 빠뜨렸습니다.
    batches = [
        [item("모모스커피 드립백"), item("에티오피아 원두 200g"), item("삼성 무선충전기")]
    ]

    kept = await filter_relevant(batches, [None])

    # 빠진 2번은 키워드('원두')가 살리고, 3번은 키워드도 걸러 냅니다.
    assert [p.title for p in kept[0]] == ["모모스커피 드립백", "에티오피아 원두 200g"]


@pytest.mark.asyncio
async def test_contradictory_number_falls_back_to_keywords(model):
    """통과와 제외 양쪽에 든 번호는 판정하지 않은 것으로 봅니다."""
    model(keep=[1, 2], drop=[2])
    batches = [[item("모모스커피 드립백"), item("삼성 무선충전기")]]

    kept = await filter_relevant(batches, [None])

    assert [p.title for p in kept[0]] == ["모모스커피 드립백"]


@pytest.mark.asyncio
async def test_prompt_separates_pass_and_reject(model):
    """'번호를 빠짐없이 모두 포함하세요' 는 전부 통과시키라는 말이라 필터가 무력화됐습니다."""
    fake = model(keep=[1])
    await filter_relevant([[item("모모스커피 드립백")]], [None])

    prompt = fake.calls[0]["messages"][0]["content"]
    assert "keep" in prompt and "drop" in prompt
    assert "빠짐없이 모두 포함" not in prompt


@pytest.mark.asyncio
async def test_system_prompt_states_what_is_unsuitable(model):
    """제목만 보고 판단하므로 무엇이 부적합한지가 프롬프트에 있어야 합니다."""
    fake = model(keep=[1])
    await filter_relevant([[item("모모스커피 드립백")]], [None])

    system = fake.calls[0]["system"]
    for unsuitable in ("중고", "도매", "성인", "포장재"):
        assert unsuitable in system


@pytest.mark.asyncio
async def test_broken_payload_falls_back_to_keywords(model):
    """번호 자리에 엉뚱한 값이 와도 판정이 죽지 않아야 합니다."""
    model(keep=["둘", None], drop=[{}])
    batches = [[item("모모스커피 드립백"), item("삼성 무선충전기")]]

    kept = await filter_relevant(batches, [None])

    assert [p.title for p in kept[0]] == ["모모스커피 드립백"]


@pytest.mark.asyncio
async def test_usage_is_logged_for_cost_tracking(model, caplog):
    """비전 호출만 토큰이 남고 판정 호출은 비용 추적이 안 됐습니다."""
    model(keep=[1])

    with caplog.at_level(logging.INFO, logger="app.services.product_filter"):
        await filter_relevant([[item("모모스커피 드립백")]], [None])

    assert any("토큰=412/18" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_usage_missing_does_not_break_the_verdict(model, monkeypatch, caplog):
    """usage 가 없는 응답에도 판정은 그대로 나와야 합니다."""
    fake = model(keep=[1])

    async def create_without_usage(**payload):
        fake.calls.append(payload)
        text = json.dumps({"keep": [1], "drop": []})
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

    monkeypatch.setattr(fake, "create", create_without_usage)

    with caplog.at_level(logging.INFO, logger="app.services.product_filter"):
        kept = await filter_relevant([[item("모모스커피 드립백")]], [None])

    assert [p.title for p in kept[0]] == ["모모스커피 드립백"]
    assert any("토큰=?/?" in record.message for record in caplog.records)


class TestSeasonVeto:
    """모델은 "선물로 줄 수 있는 물건인가"만 봅니다. 달력은 코드가 봅니다.

    실측(8월): 꽃 답례 판정에서 모델이 "크리스마스 트리 미니트리 풀세트"를 통과시켰고
    (판정 5건 중 통과 4건에 포함) 그것이 유일한 추천으로 나갔습니다.
    """

    TREE = "크리스마스 트리 미니트리 풀세트 눈꽃 / 홀리데이 선물 겨울 집들이 졸업 졸업식 꽃 선물 벽트리 장식 이사"

    @pytest.mark.asyncio
    async def test_model_approval_does_not_override_the_calendar(self, model, monkeypatch):
        monkeypatch.setattr(product_search, "_current_month", lambda: 8)
        model(keep=[1], drop=[])  # 모델은 통과시켰습니다.

        kept = await filter_relevant([[item(self.TREE, "꽃·식물")]], [None])

        assert kept[0] == []

    @pytest.mark.asyncio
    async def test_the_same_product_survives_in_december(self, model, monkeypatch):
        monkeypatch.setattr(product_search, "_current_month", lambda: 12)
        model(keep=[1], drop=[])

        kept = await filter_relevant([[item(self.TREE, "꽃·식물")]], [None])

        assert [p.title for p in kept[0]] == [self.TREE]

    @pytest.mark.asyncio
    async def test_out_of_season_items_are_never_sent_to_the_model(self, model, monkeypatch):
        """판정 프롬프트의 입력 토큰을 그만큼 아낍니다."""
        monkeypatch.setattr(product_search, "_current_month", lambda: 8)
        fake = model(keep=[1], drop=[])

        await filter_relevant([[item(self.TREE, "꽃·식물"), item("미니 꽃다발", "꽃·식물")]], [None])

        sent = fake.calls[0]["messages"][0]["content"]
        assert "미니 꽃다발" in sent
        assert "크리스마스" not in sent

    @pytest.mark.asyncio
    async def test_the_keyword_fallback_is_vetoed_too(self, monkeypatch):
        """모델을 못 쓰는 상황에서도 계절 판단은 그대로 적용됩니다."""
        monkeypatch.setattr(settings, "model_backend", "vllm")
        monkeypatch.setattr(product_search, "_current_month", lambda: 8)

        kept = await filter_relevant([[item(self.TREE, "꽃·식물")]], [None])

        assert kept[0] == []

    @pytest.mark.asyncio
    async def test_the_exclusion_is_logged_with_its_reason(self, model, monkeypatch, caplog):
        monkeypatch.setattr(product_search, "_current_month", lambda: 8)
        model(keep=[1], drop=[])

        with caplog.at_level(logging.INFO, logger="app.services.product_search"):
            await filter_relevant([[item(self.TREE, "꽃·식물")]], [None])

        assert any("철 지난 행사 상품 제외 8월 기준 '크리스마스'" in r.message for r in caplog.records)


class TestSamplingIsPinnedForJudgement:
    """실측 3차: 같은 제목의 판정이 실행마다 뒤집혔습니다.

    "[선물] 명품 나주배 세트 5kg(8-10과) 부모님 명절 선물" 이
    28,000~42,000원 요청에서는 제외됐는데(판정 16건 중 3건 제외에 포함)
    23초 뒤 8,000~12,000원 요청에서는 통과해 유일한 추천으로 나갔습니다.
    판정 프롬프트는 가격을 보지 않으므로 입력 차이로 설명되지 않습니다.

    지정하지 않으면 API 기본값 1.0 이 걸립니다. 이 호출은 판정이지 창작이 아닙니다.
    """

    @pytest.fixture(autouse=True)
    def remember_nothing(self, monkeypatch):
        """샘플링 거부 기억은 프로세스 단위라 테스트끼리 새지 않게 되돌립니다."""
        bedrock_client.reset_rejections()

    @pytest.mark.asyncio
    async def test_judgement_is_greedy(self, model):
        fake = model(keep=[1])

        await product_filter.judge([("디저트", "드립백 선물세트")])

        assert fake.calls[0]["temperature"] == settings.product_filter_temperature
        assert settings.product_filter_temperature == 0.0

    @pytest.mark.asyncio
    async def test_top_p_is_never_sent_alongside_temperature(self, model):
        """Claude 는 둘을 동시에 받으면 요청을 거부합니다."""
        fake = model(keep=[1])

        await product_filter.judge([("디저트", "드립백 선물세트")])

        assert "top_p" not in fake.calls[0]
        assert "top_k" not in fake.calls[0]

    @pytest.mark.asyncio
    async def test_a_model_that_rejects_sampling_still_gets_judged(self, model, monkeypatch):
        """Opus 4.6+ 등은 샘플링 파라미터 자체를 400 으로 거부합니다.

        거기서 예외를 그대로 올리면 판정이 통째로 죽고 키워드 폴백으로 떨어집니다.
        모델을 바꾼 것만으로 필터 품질이 조용히 내려앉으면 안 됩니다.
        """
        import anthropic

        fake = model(keep=[1])
        rejection = anthropic.BadRequestError(
            "temperature not supported",
            response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
            body=None,
        )
        calls: list[dict] = []
        original = fake.create

        async def create(**payload):
            calls.append(payload)
            if "temperature" in payload:
                raise rejection
            return await original(**payload)

        fake.create = create

        assert await product_filter.judge([("디저트", "드립백 선물세트")]) == {0: True}
        assert len(calls) == 2
        assert "temperature" in calls[0] and "temperature" not in calls[1]
        # 두 번째 요청부터는 처음부터 빼고 보냅니다.
        assert not bedrock_client.accepts(
            settings.bedrock_model_id, bedrock_client.SAMPLING
        )


class TestPromptChecksTheCategoryLabel:
    """실측: "[커피·차] [선물] 명품 나주배 세트 5kg" 이 판정을 통과했습니다.

    예전 프롬프트는 부적합 **유형**(중고·도매·성인용품·부품·포장재)만 나열하고
    라벨 일치는 "그 카테고리의 선물로" 한 마디에 기대고 있었습니다.

    실측 당시의 라벨은 커피·차였지만, 카테고리를 백엔드 목록에 맞추면서 커피·차가
    디저트에 접혔습니다. 같은 나주배 세트가 디저트에서는 **정상 통과**라 예시로 쓸 수
    없으므로, 프롬프트도 이 테스트도 여전히 어긋나는 라벨(꽃·식물)로 옮겼습니다.
    """

    def test_the_label_match_is_its_own_rule(self):
        assert "그 카테고리에 실제로 속할 것" in product_filter.SYSTEM_PROMPT

    def test_the_measured_mismatch_is_the_example(self):
        assert "[꽃·식물] 나주배 세트" in product_filter.SYSTEM_PROMPT

    def test_the_old_exclusion_types_are_still_there(self):
        """라벨 규칙을 더하면서 기존 기준을 잃으면 다른 오탐이 돌아옵니다."""
        for term in ("중고", "도매", "성인용품", "포장재"):
            assert term in product_filter.SYSTEM_PROMPT


class TestTheJudgementDidNotGetStricterThanIntended:
    """5차 실측: 라벨 대조 규칙을 넣은 뒤 정상 상품 둘이 함께 떨어졌습니다.

        추천 부적합 제외 category=커피·차 title=[센터커피] 디카페인 드립백 세트 (10g X 15개)
        추천 부적합 제외 category=생활용품 title=송월타월 고급수건 답례품 프레디 170g 코마사 30수 두꺼운

    커피 카테고리에서 커피 드립백을, 생활용품에서 수건 답례품을 버린 것입니다.
    두 제목의 공통점은 대괄호 브랜드와 용량·규격 나열뿐입니다.

    라벨 대조 자체는 남겨야 합니다. 나주배는 계속 걸려야 하기 때문입니다.
    아래는 그 둘을 한 클래스에 함께 못박습니다. 모델 판정은 외부 호출이라 여기서
    확인할 수 없으므로, 프롬프트가 무엇을 말하는지와 코드가 스스로 판정하는
    폴백 경로 양쪽을 고정합니다.
    """

    # 5차 로그가 잘못 버린 제목.
    WRONGLY_DROPPED = (
        ("디저트", "[센터커피] 디카페인 드립백 세트 (10g X 15개)"),
        ("생활용품", "송월타월 고급수건 답례품 프레디 170g 코마사 30수 두꺼운"),
    )
    # 같은 로그가 **정당하게** 버린 제목. 라벨과 물건이 어긋납니다.
    # 라벨은 실측 당시의 커피·차가 아니라 지금 목록의 이름입니다. 나주배는 디저트에서
    # 정상 통과라 어긋나는 라벨(꽃·식물)로 옮겼고, 나머지 둘은 디저트에서도 그대로 어긋납니다.
    RIGHTLY_DROPPED = (
        ("꽃·식물", "[선물] 명품 나주배 세트 5kg(8-10과) 부모님 명절 선물"),
        ("디저트", "[지갑벨트세트][선물추천] 닥스 지갑 벨트 선물 세트 4종"),
        ("디저트", "[4] 울트라 훼이셜 크림 125ml 더블 선물 세트"),
        ("꽃·식물", "당일배송 정품 미니 키티 산리오 쿠로미 마멜 폼폼푸린 인형"),
    )

    def test_the_label_rule_is_still_there(self):
        """나주배를 걸러 낸 규칙입니다. 완화하면 그 사례가 돌아옵니다."""
        assert "그 카테고리에 실제로 속할 것" in product_filter.SYSTEM_PROMPT
        assert "[꽃·식물] 나주배 세트" in product_filter.SYSTEM_PROMPT

    def test_the_default_is_to_pass(self):
        """제외를 증명 실패의 기본값이 아니라 분명한 근거가 있을 때로 못박습니다."""
        assert "분명할 때만" in product_filter.SYSTEM_PROMPT
        assert "애매하면 통과" in product_filter.SYSTEM_PROMPT

    def test_the_two_measured_title_shapes_are_named_as_not_a_reason(self):
        """오탈락 둘의 공통점이었던 표기입니다."""
        assert "대괄호 브랜드" in product_filter.SYSTEM_PROMPT
        assert "용량·규격" in product_filter.SYSTEM_PROMPT

    @pytest.mark.parametrize(("category", "title"), WRONGLY_DROPPED)
    @pytest.mark.asyncio
    async def test_a_wrongly_dropped_title_survives_the_code_path(
        self, monkeypatch, category, title
    ):
        monkeypatch.setattr(settings, "model_backend", "vllm")  # 모델을 못 쓰는 경로

        kept = await filter_relevant([[item(title, category)]], [None])

        assert [p.title for p in kept[0]] == [title]

    @pytest.mark.parametrize(("category", "title"), RIGHTLY_DROPPED)
    @pytest.mark.asyncio
    async def test_a_rightly_dropped_title_stays_out(self, monkeypatch, category, title):
        monkeypatch.setattr(settings, "model_backend", "vllm")

        kept = await filter_relevant([[item(title, category)]], [None])

        assert kept[0] == []

    @pytest.mark.asyncio
    async def test_the_model_still_has_the_last_word_on_a_real_mismatch(self, model):
        """폴백을 고정한다고 모델 판정을 무르는 것은 아닙니다.

        모델과 키워드가 갈린 5차 다섯 건 중 셋은 모델이 맞았습니다. 되살리기
        규칙을 넣지 않은 이유이므로, 모델의 제외가 그대로 유지되는지 봅니다.
        """
        model(keep=[], drop=[1])
        title = "송월타월 조문 조의 답례품 핸드타올 2매 크라프트"

        kept = await filter_relevant([[item(title, "생활용품")]], [None])

        assert kept[0] == []
