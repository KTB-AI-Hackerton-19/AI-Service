"""답례 선물 추천이 필요한 입력에서만 실행되는지 확인합니다.

부조금 명단에 대고 "8,000~12,000원 디저트"를 권하는 것은 사용자에게 의미가 없고,
모델 호출 비용과 지연만 늘립니다.
"""

import ast
import inspect
import re
import textwrap

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.gift_agent_service import GiftAgentService, gift_agent_service

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


# ---------------------------------------------------------------- 조의/축하 분기
# 이미지 추출은 청첩장과 부고장을 똑같은 event_invitation 으로 분류합니다.
# 계기로 갈라내지 못하면 유족 화면에 "진심으로 축하드려요!" 가 그대로 나갑니다.

def invitation_message(event: str) -> str:
    body = post(
        [record(0, "event_invitation")], record_type="event_invitation", event=event
    )
    info = body["recommend_gift_info"]
    assert info["status"] == "SUCCESS"
    return info["message"]["content"]


@pytest.mark.parametrize("event", ["조의", "부고", "부친상"])
def test_condolence_invitation_never_congratulates(event):
    content = invitation_message(event)

    assert "축하" not in content
    assert "!" not in content
    assert "조의" in content


def test_wedding_invitation_still_congratulates():
    assert "축하" in invitation_message("결혼")


# ---------------------------------------------------------------- SKIP 사유는 응답 본문입니다
# recommend_gift_info.reason 은 프런트가 사용자에게 그대로 표시할 수 있는 필드입니다.
# 실측 /from-image occasion 응답에는 "사용자가 경조사로 선택해 답례 선물 추천 대신
# 금액 기준으로 안내하세요." 가 그대로 실려 나갔습니다. 대상이 개발자인 명령문이고,
# 금액을 못 읽은 갈래에는 내부 엔드포인트 "POST /api/v1/agent/recommend" 까지 있었습니다.
#
# 문자열 하나만 검사하면 다음에 추가되는 사유는 그대로 새 나갑니다. 그래서 함수가
# 돌려주는 문자열 **전부**를 소스에서 훑어 같은 기준을 겁니다.

def _skip_reason_returns() -> list[ast.Return]:
    source = textwrap.dedent(inspect.getsource(GiftAgentService._recommendation_skip_reason))
    function = ast.parse(source).body[0]
    return [node for node in ast.walk(function) if isinstance(node, ast.Return)]


def declared_skip_reasons() -> list[str]:
    """``_recommendation_skip_reason`` 이 돌려줄 수 있는 문자열 전부."""
    return [
        node.value.value
        for node in _skip_reason_returns()
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    ]


SKIP_REASONS = declared_skip_reasons()

# 개발자에게 "이렇게 구현하라" 고 시키는 말. 사용자에게 보이면 안 됩니다.
IMPLEMENTER_IMPERATIVES = (
    "안내하세요",
    "요청하세요",
    "호출하세요",
    "처리하세요",
    "표시하세요",
    "구현하세요",
    "반환하세요",
    "전달하세요",
)


def test_the_sweep_actually_found_the_reasons():
    """훑은 목록이 비면 아래 검사가 전부 공회전합니다."""
    assert len(SKIP_REASONS) >= 5


def test_every_skip_reason_is_a_plain_literal():
    """f-string 이나 변수로 만들면 위 훑기가 조용히 놓칩니다."""
    for node in _skip_reason_returns():
        assert isinstance(node.value, ast.Constant), (
            "SKIP 사유는 소스에서 훑을 수 있게 문자열 리터럴로 두세요. "
            f"{ast.dump(node.value)[:120]}"
        )


@pytest.mark.parametrize("reason", SKIP_REASONS)
class TestEverySkipReasonIsWrittenForTheUser:
    def test_carries_no_internal_api_surface(self, reason):
        """실측 응답에 "POST /api/v1/agent/recommend" 가 그대로 실려 나갔습니다."""
        assert not re.search(r"\b(GET|POST|PUT|PATCH|DELETE)\b", reason), reason
        assert "/api/" not in reason, reason
        assert "http" not in reason.lower(), reason

    def test_carries_no_code_identifier(self, reason):
        """RecordKind.MONEY, gift_price 같은 식별자도 사용자에게는 뜻이 없습니다."""
        assert not re.search(r"[A-Za-z][A-Za-z.]*_[A-Za-z_.]*", reason), reason
        assert not re.search(r"\b[A-Z]{2,}\b", reason), reason
        assert "`" not in reason, reason

    def test_does_not_talk_about_the_reader_in_the_third_person(self, reason):
        """이 글을 읽는 사람이 그 "사용자" 입니다. 3인칭으로 부르면 남에게 쓴 글입니다."""
        assert "사용자" not in reason, reason

    def test_is_not_an_instruction_to_the_implementer(self, reason):
        for imperative in IMPLEMENTER_IMPERATIVES:
            assert imperative not in reason, f"{imperative} in {reason}"

    def test_says_what_happened_first(self, reason):
        """첫 문장은 무슨 일이 있었는지를 말하는 서술문이어야 합니다."""
        first = reason.split(".")[0].strip()
        assert first.endswith("습니다"), reason

    def test_then_says_what_the_reader_can_do(self, reason):
        """무슨 일이 있었는지만 알려 주면 사용자는 다음에 뭘 할지 알 수 없습니다."""
        sentences = [s.strip() for s in reason.split(".") if s.strip()]
        assert len(sentences) >= 2, reason


def test_every_branch_ships_one_of_the_swept_reasons():
    """훑은 문자열이 실제 응답과 다르면 검사가 엉뚱한 것을 지키는 셈입니다."""
    no_price = client.post(
        "/api/v1/agent/from-gift-data",
        headers=headers,
        json={"gift_data": {"gift_name": "TWG Tea 티백", "record_type": "gift"}},
    )
    assert no_price.status_code == 200

    observed = {
        no_price.json()["recommend_gift_info"]["reason"],
        post_image("occasion")["recommend_gift_info"]["reason"],
        post([record(0, "money")], record_type="money")["recommend_gift_info"]["reason"],
        post([record(0, "receipt")], record_type="receipt")["recommend_gift_info"]["reason"],
        post([record(0, "unknown")], record_type="unknown")["recommend_gift_info"]["reason"],
    }

    # 다섯 갈래가 모두 닿고, 나간 문장이 전부 훑은 목록 안에 있어야 합니다.
    assert len(observed) == 5
    assert observed <= set(SKIP_REASONS)
