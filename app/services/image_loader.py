"""S3 presigned URL 로 받은 이미지를 검증·정규화합니다.

presigned URL 을 vLLM 에 그대로 넘기지 않는 이유:
- vLLM 컨테이너가 S3 에 닿는다는 보장이 없습니다.
- 다운로드 크기와 제한 시간을 우리 쪽에서 통제할 수 없습니다.
- 리다이렉트를 타고 내부망으로 향하는 요청(SSRF)을 막을 수 없습니다.
"""

import asyncio
import ipaddress
import io
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps

from app.core.config import settings

logger = logging.getLogger(__name__)

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _is_internal(address: IPAddress) -> bool:
    """사설·루프백·링크로컬 등 외부로 나가면 안 되는 주소인지 판단합니다.

    ``::ffff:169.254.169.254`` 같은 IPv4-mapped IPv6 는 먼저 IPv4 로 되돌립니다.
    Python 버전에 따라 mapped 주소의 ``is_link_local`` 이 False 로 나와 EC2 IMDS 가
    그대로 통과할 수 있기 때문입니다.
    """
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


async def _resolve_addresses(host: str, port: int) -> list[IPAddress]:
    """호스트명을 실제 IP 로 해석합니다. 이벤트 루프를 막지 않습니다.

    ``socket.getaddrinfo`` 는 블로킹 호출이라 그대로 부르면 워커 1개짜리 서버의
    이벤트 루프가 DNS 응답을 기다리는 동안 통째로 멈춥니다.
    """
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses: list[IPAddress] = []
    for info in infos:
        # IPv6 sockaddr 의 주소에는 "fe80::1%en0" 처럼 스코프가 붙을 수 있습니다.
        raw = str(info[4][0]).split("%")[0]
        addresses.append(ipaddress.ip_address(raw))
    return addresses


class ImageLoadError(RuntimeError):
    """이미지를 내려받거나 열 수 없을 때 발생합니다."""


@dataclass(frozen=True)
class LoadedImage:
    """VLM 에 넣을 수 있게 정규화된 이미지."""

    data: bytes
    mime: str
    width: int
    height: int
    downloaded_bytes: int


async def _assert_public_url(url: str) -> None:
    """내부 주소로 향하는 요청을 차단합니다.

    호스트가 IP 리터럴이면 그 값을, 도메인이면 **해석된 모든 주소**를 같은 규칙으로
    검사합니다. 도메인을 무조건 통과시키면 ``169.254.169.254`` 로 해석되는 도메인
    하나로 EC2 IMDS 에 GET 이 나갑니다(배포 서버가 공인 EC2 입니다).

    해석 실패는 **차단**으로 처리합니다. 열어 두면 자기 네임서버를 가진 공격자가
    우리 조회에는 SERVFAIL 을, 이어지는 httpx 조회에는 사설 IP 를 돌려주는 것만으로
    검사를 통과시킬 수 있습니다. 반대로 진짜 일시적 DNS 장애라면 어차피 httpx 의
    조회도 실패하므로, 막아서 잃는 정상 요청은 없습니다.

    남는 한계: 검사와 실제 연결이 각각 조회하므로 그 사이에 응답이 바뀌는
    DNS rebinding 은 막지 못합니다. 완전히 막으려면 해석한 IP 로 직접 붙고 Host 를
    덮어써야 하는데, presigned URL 의 TLS/SNI 를 건드리게 되어 데모 범위를 넘습니다.
    리다이렉트를 통한 2차 우회는 ``follow_redirects=False`` 가 계속 막습니다.

    Args:
        url: 검사할 이미지 주소.

    Raises:
        ImageLoadError: 스킴이 http(s) 가 아니거나, 호스트가(또는 해석 결과 중 하나가)
            내부 주소이거나, 이름 해석에 실패한 경우.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        # 스킴 검사는 설정과 무관하게 항상 합니다. file:// 은 어떤 경우에도 허용하지 않습니다.
        raise ImageLoadError(f"허용되지 않는 스킴입니다: {parsed.scheme}")

    if settings.allow_private_image_hosts:
        logger.warning("allow_private_image_hosts=true 이므로 내부 주소 검사를 건너뜁니다: %s", parsed.hostname)
        return

    host = parsed.hostname or ""
    if not host:
        raise ImageLoadError("호스트가 없는 주소입니다.")

    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            # 포트가 범위를 벗어나면 urlparse 가 여기서 ValueError 를 던집니다.
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise ImageLoadError("포트 번호가 올바르지 않습니다.") from exc
        try:
            addresses = await _resolve_addresses(host, port)
        except (OSError, ValueError) as exc:
            raise ImageLoadError(f"이미지 주소의 이름을 해석하지 못했습니다: {host}") from exc
        if not addresses:
            raise ImageLoadError(f"이미지 주소의 이름을 해석하지 못했습니다: {host}")

    for address in addresses:
        if _is_internal(address):
            raise ImageLoadError(f"내부 주소로는 요청할 수 없습니다: {host} -> {address}")


def _normalize(blob: bytes) -> LoadedImage:
    """EXIF 회전을 적용하고, 장변을 설정값으로 줄여 PNG 로 다시 인코딩합니다.

    JPEG 가 아니라 PNG 로 재인코딩하는 이유는 스크린샷의 작은 글자가
    JPEG 압축에서 뭉개지면 추출 정확도가 그대로 떨어지기 때문입니다.

    Args:
        blob: 내려받은 원본 바이트.

    Returns:
        리사이즈된 PNG 이미지.

    Raises:
        ImageLoadError: 이미지로 열 수 없는 경우.
    """
    try:
        image = Image.open(io.BytesIO(blob))
        image.load()
    except Exception as exc:  # Pillow 는 형식마다 다른 예외를 던집니다.
        raise ImageLoadError(f"이미지를 열 수 없습니다: {exc}") from exc

    # 폰으로 찍은 사진은 센서 방향 그대로 저장되고 회전은 EXIF 로만 표시됩니다.
    # 이를 적용하지 않으면 세로로 찍은 장부·영수증이 옆으로 누운 채 모델에 전달되어
    # 손글씨와 금액을 잘못 읽습니다(실측). PNG 로 재인코딩하면 EXIF 가 사라지므로
    # 여기서 픽셀 자체를 돌려 둡니다.
    image = ImageOps.exif_transpose(image)

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    longest_edge = max(image.size)
    if longest_edge > settings.image_max_edge:
        ratio = settings.image_max_edge / longest_edge
        new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
        image = image.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return LoadedImage(
        data=buffer.getvalue(),
        mime="image/png",
        width=image.width,
        height=image.height,
        downloaded_bytes=len(blob),
    )


class ImageLoader:
    """presigned URL 을 받아 VLM 입력용 이미지로 바꿉니다."""

    async def load(self, image_url: str) -> LoadedImage:
        """이미지를 내려받아 검증하고 리사이즈합니다.

        Args:
            image_url: Spring Boot 가 전달한 S3 presigned URL.

        Returns:
            정규화된 PNG 이미지.

        Raises:
            ImageLoadError: 주소가 안전하지 않거나, 다운로드·디코딩에 실패한 경우.
        """
        await _assert_public_url(image_url)

        chunks: list[bytes] = []
        total = 0
        try:
            async with httpx.AsyncClient(
                timeout=settings.image_fetch_timeout_seconds,
                follow_redirects=False,
            ) as client:
                async with client.stream("GET", image_url) as response:
                    if response.status_code != 200:
                        raise ImageLoadError(f"이미지 다운로드 실패 HTTP {response.status_code}")

                    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
                    if content_type and not content_type.startswith("image/"):
                        raise ImageLoadError(f"이미지가 아닙니다: content-type={content_type}")

                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > settings.image_max_bytes:
                            raise ImageLoadError(
                                f"이미지가 너무 큽니다({settings.image_max_bytes} bytes 초과)"
                            )
                        chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise ImageLoadError(f"이미지 다운로드에 실패했습니다: {exc}") from exc

        # PIL 디코드 → EXIF 회전 → 리사이즈 → PNG 재인코딩은 순수 CPU 작업이라
        # 그대로 부르면 그동안 이벤트 루프 전체가 멈춥니다(워커 1개, Dockerfile 참고).
        # 한 요청 기준 0.1~0.5초지만 그 사이 다른 요청은 accept 조차 되지 않습니다.
        loaded = await asyncio.to_thread(_normalize, b"".join(chunks))
        logger.info(
            "이미지 로드 완료 %dx%d (%d bytes -> %d bytes)",
            loaded.width,
            loaded.height,
            loaded.downloaded_bytes,
            len(loaded.data),
        )
        return loaded


image_loader = ImageLoader()
