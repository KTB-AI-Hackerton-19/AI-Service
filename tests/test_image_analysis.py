"""이미지 분석 종단 테스트.

S3 다운로드와 vLLM 호출을 respx 로 가로채, presigned URL 부터 GiftData 까지
실제 코드 경로를 그대로 태웁니다.
"""

import os

os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("API_KEY", "test-key")

import io
import ipaddress
import json
import socket
import threading

import httpx
import pytest
import respx
from PIL import Image

from app.core.config import settings
from app.schemas.agent import InputCategory
from app.services import image_loader as image_loader_module
from app.services import product_search
from app.services.image_loader import ImageLoadError, image_loader
from app.services.tasks.image_analysis import MAX_PRICE_LOOKUPS, image_analysis_service
from app.services.vlm_service import VisionAnalysisError

IMAGE_URL = "https://example-bucket.s3.ap-northeast-2.amazonaws.com/u1/gift.png?X-Amz-Signature=abc"
# 개발자의 .env 값에 의존하지 않도록 설정에서 파생시킵니다.
VLLM_URL = f"{settings.vllm_base_url.rstrip('/')}/v1/chat/completions"

# SSRF 검사가 호스트명을 실제로 해석하므로, 테스트에서는 해석 함수를 스텁으로 바꿉니다.
# 실제 DNS 조회를 하면 테스트가 네트워크 상태에 의존하고 오프라인 CI 에서 깨집니다.
PUBLIC_IP = ipaddress.ip_address("52.219.60.1")  # S3 서울 리전 대역의 공인 IP


def resolver(*addresses: str):
    """지정한 주소들만 돌려주는 이름 해석 스텁을 만듭니다."""

    async def _resolve(host: str, port: int):
        return [ipaddress.ip_address(a) for a in addresses]

    return _resolve


@pytest.fixture(autouse=True)
def stub_dns(monkeypatch):
    """기본은 공인 IP 로 해석됨. 개별 테스트가 필요하면 다시 덮어씁니다."""
    monkeypatch.setattr(image_loader_module, "_resolve_addresses", resolver(str(PUBLIC_IP)))


def png_bytes(width: int = 720, height: int = 1280) -> bytes:
    """벤치 이미지와 같은 크기의 PNG 를 만듭니다."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (240, 240, 240)).save(buffer, format="PNG")
    return buffer.getvalue()


def vllm_response(payload: dict, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": json.dumps(payload, ensure_ascii=False)},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 520, "completion_tokens": 130},
        },
    )


KAKAO_GIFT = {
    "image_kind": "kakao_gift",
    "records": [
        {
            "record_type": "gift",
            "direction": "received",
            "counterpart_name": "김수현",
            "occurred_date": "2026-03-14",
            "event_date": None,
            "item_name": "아이스 카페 아메리카노 T",
            "brand": "스타벅스",
            "category": "기프티콘/음료",
            "event": "생일",
            "amount": 12300,
            "memo": "생일 축하해!",
            "confidence": 0.95,
        }
    ],
}

BANK_STATEMENT = {
    "image_kind": "bank_statement",
    "records": [
        {
            "record_type": "money",
            "direction": "received",
            "counterpart_name": name,
            "occurred_date": "2026-05-09",
            "event_date": None,
            "item_name": None,
            "brand": None,
            "category": "축의금",
            "event": "결혼",
            "amount": amount,
            "memo": memo,
            "confidence": 0.93,
        }
        for name, amount, memo in [
            ("김도윤", 100000, "결혼 축하합니다"),
            ("박서준", 50000, None),
            ("최은비", 200000, None),
        ]
    ],
}


GIFT_LIST_NO_PRICE = {
    "image_kind": "gift_list",
    "records": [
        {
            "record_type": "gift",
            "direction": "received",
            "counterpart_name": name,
            "occurred_date": "2026-03-14",
            "event_date": None,
            "item_name": item,
            "brand": "TWG Tea",
            "category": "기프티콘/음료",
            "event": "생일",
            "amount": None,
            "memo": None,
            "confidence": 0.9,
        }
        for name, item in [
            ("김수현", "TWG Tea 티백 컬렉션"),
            ("박서준", "TWG Tea 실버문"),
            ("최은비", "TWG Tea 잉글리시 브렉퍼스트"),
            ("정예린", "TWG Tea 프렌치 얼그레이"),
            ("이준호", "TWG Tea 카모마일"),
        ]
    ],
}


@pytest.fixture
def vllm_backend(monkeypatch):
    monkeypatch.setattr(settings, "model_backend", "vllm")
    monkeypatch.setattr(settings, "google_access_token", "")
    return settings


@pytest.fixture
def price_lookup_spy(monkeypatch):
    """Tavily 를 호출하지 않고 검색 횟수만 기록합니다."""
    queries: list[str] = []

    async def fake_lookup_price(name: str, brand: str | None = None) -> int:
        queries.append(name)
        return 36000

    monkeypatch.setattr(settings, "product_price_lookup_enabled", True)
    monkeypatch.setattr(product_search, "lookup_price", fake_lookup_price)
    return queries


class TestAnalyze:
    @respx.mock
    async def test_single_gift_end_to_end(self, vllm_backend):
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )
        respx.post(VLLM_URL).mock(return_value=vllm_response(KAKAO_GIFT))

        gift_data = await image_analysis_service.analyze(IMAGE_URL)

        assert gift_data.gift_name == "스타벅스 아이스 카페 아메리카노 T"
        assert gift_data.gift_price == 12300
        assert gift_data.person_name == "김수현"
        assert gift_data.received_at.isoformat() == "2026-03-14"

    @respx.mock
    async def test_multi_record_picks_largest_received(self, vllm_backend):
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )
        respx.post(VLLM_URL).mock(return_value=vllm_response(BANK_STATEMENT))

        gift_data = await image_analysis_service.analyze(IMAGE_URL)

        # GiftData 가 1건만 표현하므로 금액이 가장 큰 건이 대표가 됩니다.
        assert gift_data.person_name == "최은비"
        assert gift_data.gift_price == 200000

    @respx.mock
    async def test_image_is_sent_as_data_url(self, vllm_backend):
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )
        route = respx.post(VLLM_URL).mock(return_value=vllm_response(KAKAO_GIFT))

        await image_analysis_service.analyze(IMAGE_URL)

        body = json.loads(route.calls.last.request.content)
        content = body["messages"][1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert body["response_format"]["type"] == "json_schema"
        assert body["temperature"] == 0.0

    @respx.mock
    async def test_vllm_error_raises(self, vllm_backend):
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )
        respx.post(VLLM_URL).mock(return_value=httpx.Response(500, text="engine dead"))

        with pytest.raises(VisionAnalysisError, match="500"):
            await image_analysis_service.analyze(IMAGE_URL)

    @respx.mock
    async def test_truncated_output_is_warned_not_fatal(self, vllm_backend, caplog):
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )
        respx.post(VLLM_URL).mock(return_value=vllm_response(KAKAO_GIFT, finish_reason="length"))

        gift_data = await image_analysis_service.analyze(IMAGE_URL)
        assert gift_data.gift_price == 12300
        assert any("max_tokens" in record.getMessage() for record in caplog.records)

    @respx.mock
    async def test_price_lookup_is_capped(self, vllm_backend, price_lookup_spy):
        """Tavily 는 검색 1회가 1크레딧입니다. 기록 수만큼 다 쏘면 안 됩니다."""
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )
        respx.post(VLLM_URL).mock(return_value=vllm_response(GIFT_LIST_NO_PRICE))

        await image_analysis_service.analyze(IMAGE_URL)

        assert len(GIFT_LIST_NO_PRICE["records"]) > MAX_PRICE_LOOKUPS
        assert len(price_lookup_spy) == MAX_PRICE_LOOKUPS

    @respx.mock
    async def test_price_lookup_is_skipped_for_occasion(self, vllm_backend, price_lookup_spy):
        """사용자가 경조사를 고르면 추천을 만들지 않으므로 검색 크레딧을 쓰지 않습니다."""
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )
        respx.post(VLLM_URL).mock(return_value=vllm_response(GIFT_LIST_NO_PRICE))

        await image_analysis_service.analyze(IMAGE_URL, InputCategory.OCCASION)

        assert price_lookup_spy == []

    @respx.mock
    async def test_searched_price_turns_on_review(self, vllm_backend, price_lookup_spy):
        """검색으로 채운 금액이 답례 가격대의 기준이 되므로 확인을 받아야 합니다."""
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )
        respx.post(VLLM_URL).mock(return_value=vllm_response(GIFT_LIST_NO_PRICE))

        gift_data = await image_analysis_service.analyze(IMAGE_URL)

        assert gift_data.gift_price == 36000
        assert gift_data.needs_review is True
        assert any("검색으로 채운" in reason for reason in gift_data.review_reasons)

    @respx.mock
    async def test_user_category_is_sent_as_a_hint(self, vllm_backend):
        """사용자가 이미 고른 종류를 모델에게 알려 줍니다. 비용도 지연도 늘지 않습니다."""
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )
        route = respx.post(VLLM_URL).mock(return_value=vllm_response(KAKAO_GIFT))

        await image_analysis_service.analyze(IMAGE_URL, InputCategory.OCCASION)

        prompt = json.loads(route.calls.last.request.content)["messages"][1]["content"][1]["text"]
        assert "경조사" in prompt
        # 사용자 선택이 이미지와 어긋나면 이미지를 따라야 합니다.
        assert "이미지가 분명히 다른 종류라면" in prompt

    @respx.mock
    async def test_no_category_sends_no_hint(self, vllm_backend):
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )
        route = respx.post(VLLM_URL).mock(return_value=vllm_response(KAKAO_GIFT))

        await image_analysis_service.analyze(IMAGE_URL)

        prompt = json.loads(route.calls.last.request.content)["messages"][1]["content"][1]["text"]
        assert "골라 올렸습니다" not in prompt

    async def test_mock_backend_does_not_touch_network(self):
        """MODEL_BACKEND=mock 이면 이미지를 내려받지 않습니다. respx 없이도 성공해야 합니다."""
        gift_data = await image_analysis_service.analyze(IMAGE_URL)
        assert gift_data.gift_price > 0


class TestImageLoader:
    @respx.mock
    async def test_oversized_image_is_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "image_max_bytes", 1024)
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )

        with pytest.raises(ImageLoadError, match="너무 큽니다"):
            await image_loader.load(IMAGE_URL)

    @respx.mock
    async def test_non_image_content_type_is_rejected(self):
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=b"<html/>", headers={"content-type": "text/html"})
        )

        with pytest.raises(ImageLoadError, match="이미지가 아닙니다"):
            await image_loader.load(IMAGE_URL)

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000/secret.png",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
        ],
    )
    async def test_internal_addresses_are_blocked(self, url):
        """presigned URL 을 가장한 내부망 요청(SSRF)을 막습니다."""
        with pytest.raises(ImageLoadError):
            await image_loader.load(url)

    @pytest.mark.parametrize(
        "resolved",
        [
            "169.254.169.254",  # EC2 IMDS. 배포 서버가 공인 EC2 라 그대로 자격증명이 샙니다.
            "10.0.0.5",  # VPC 사설 대역
            "127.0.0.1",  # 루프백
            "::ffff:169.254.169.254",  # IPv4-mapped IPv6 로 감싼 IMDS
        ],
    )
    async def test_domain_resolving_to_internal_address_is_blocked(
        self, monkeypatch, resolved
    ):
        """IP 리터럴만 막으면 도메인 하나로 우회됩니다. 해석 결과를 검사해야 합니다."""
        monkeypatch.setattr(image_loader_module, "_resolve_addresses", resolver(resolved))

        with pytest.raises(ImageLoadError, match="내부 주소"):
            await image_loader.load("https://metadata.attacker.example/gift.png")

    async def test_one_internal_address_among_many_blocks(self, monkeypatch):
        """공인 IP 를 섞어 돌려줘도 내부 주소가 하나라도 있으면 막아야 합니다."""
        monkeypatch.setattr(
            image_loader_module,
            "_resolve_addresses",
            resolver(str(PUBLIC_IP), "169.254.169.254"),
        )

        with pytest.raises(ImageLoadError, match="내부 주소"):
            await image_loader.load("https://mixed.attacker.example/gift.png")

    async def test_resolution_failure_is_blocked(self, monkeypatch):
        """해석 실패는 열어 두지 않습니다. 열면 재조회 때 사설 IP 를 주는 우회가 통합니다."""

        async def boom(host, port):
            raise socket.gaierror("nope")

        monkeypatch.setattr(image_loader_module, "_resolve_addresses", boom)

        with pytest.raises(ImageLoadError, match="해석하지 못했습니다"):
            await image_loader.load("https://nxdomain.attacker.example/gift.png")

    @respx.mock
    async def test_dns_is_not_resolved_when_switch_is_on(self, monkeypatch):
        """개발용 스위치를 켜면 지금처럼 검사를 통째로 건너뜁니다(조회도 하지 않습니다)."""
        monkeypatch.setattr(settings, "allow_private_image_hosts", True)

        async def never_called(host, port):
            raise AssertionError("스위치가 켜져 있으면 이름 해석을 하지 않아야 합니다.")

        monkeypatch.setattr(image_loader_module, "_resolve_addresses", never_called)
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )

        loaded = await image_loader.load(IMAGE_URL)
        assert loaded.width == 720

    async def test_ip_literal_needs_no_resolution(self, monkeypatch):
        """IP 리터럴은 해석할 것이 없으므로 DNS 를 부르지 않고 그대로 판정합니다."""

        async def never_called(host, port):
            raise AssertionError("IP 리터럴에는 이름 해석이 필요 없습니다.")

        monkeypatch.setattr(image_loader_module, "_resolve_addresses", never_called)

        with pytest.raises(ImageLoadError, match="내부 주소"):
            await image_loader.load("http://169.254.169.254/latest/meta-data/")

    @respx.mock
    async def test_normalize_runs_off_the_event_loop(self, monkeypatch):
        """PIL 재인코딩은 순수 CPU 작업입니다. 이벤트 루프 스레드에서 돌면 서버가 멈춥니다."""
        seen: dict[str, str] = {}
        original = image_loader_module._normalize

        def spy(blob):
            seen["thread"] = threading.current_thread().name
            return original(blob)

        monkeypatch.setattr(image_loader_module, "_normalize", spy)
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )

        await image_loader.load(IMAGE_URL)

        assert seen["thread"] != threading.current_thread().name

    async def test_file_scheme_blocked_even_when_private_hosts_allowed(self, monkeypatch):
        """개발용 스위치를 켜도 file:// 은 절대 허용하지 않습니다."""
        monkeypatch.setattr(settings, "allow_private_image_hosts", True)
        with pytest.raises(ImageLoadError, match="스킴"):
            await image_loader.load("file:///etc/passwd")

    @respx.mock
    async def test_localhost_allowed_when_switch_on(self, monkeypatch):
        """로컬 종단 테스트를 위해 개발용 스위치를 켜면 루프백을 허용합니다."""
        monkeypatch.setattr(settings, "allow_private_image_hosts", True)
        local_url = "http://127.0.0.1:9999/gift.png"
        respx.get(local_url).mock(
            return_value=httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        )

        loaded = await image_loader.load(local_url)
        assert loaded.width == 720

    @respx.mock
    async def test_large_image_is_resized_to_max_edge(self, monkeypatch):
        monkeypatch.setattr(settings, "image_max_edge", 640)
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(
                200, content=png_bytes(2000, 3000), headers={"content-type": "image/png"}
            )
        )

        loaded = await image_loader.load(IMAGE_URL)
        assert max(loaded.width, loaded.height) == 640

    @respx.mock
    async def test_bench_sized_image_is_not_resized(self):
        """벤치 이미지(720x1280)는 그대로여야 측정된 정확도가 유지됩니다."""
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(
                200, content=png_bytes(720, 1280), headers={"content-type": "image/png"}
            )
        )

        loaded = await image_loader.load(IMAGE_URL)
        assert (loaded.width, loaded.height) == (720, 1280)

    @respx.mock
    async def test_corrupt_image_is_rejected(self):
        respx.get(IMAGE_URL).mock(
            return_value=httpx.Response(200, content=b"not-a-png", headers={"content-type": "image/png"})
        )

        with pytest.raises(ImageLoadError, match="열 수 없습니다"):
            await image_loader.load(IMAGE_URL)


def test_exif_orientation_is_applied_before_resize():
    """폰 사진은 EXIF 로만 회전이 표시됩니다. 적용하지 않으면 모델이 옆으로 누운 이미지를 봅니다."""
    import io

    from PIL import Image

    from app.services.image_loader import _normalize

    # 가로로 저장된 이미지에 "시계 방향 90도 회전" EXIF 를 붙입니다.
    source = Image.new("RGB", (400, 200), "white")
    exif = source.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", exif=exif)

    loaded = _normalize(buffer.getvalue())

    # 회전이 적용되면 세로(200x400)가 되어야 합니다.
    assert (loaded.width, loaded.height) == (200, 400)
