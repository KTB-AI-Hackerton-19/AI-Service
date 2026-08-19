"""S3 presigned URL 로 받은 이미지를 검증·정규화합니다.

presigned URL 을 vLLM 에 그대로 넘기지 않는 이유:
- vLLM 컨테이너가 S3 에 닿는다는 보장이 없습니다.
- 다운로드 크기와 제한 시간을 우리 쪽에서 통제할 수 없습니다.
- 리다이렉트를 타고 내부망으로 향하는 요청(SSRF)을 막을 수 없습니다.
"""

import ipaddress
import io
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


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


def _assert_public_url(url: str) -> None:
    """내부 주소로 향하는 요청을 차단합니다.

    Args:
        url: 검사할 이미지 주소.

    Raises:
        ImageLoadError: 스킴이 http(s) 가 아니거나 사설·루프백 주소인 경우.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        # 스킴 검사는 설정과 무관하게 항상 합니다. file:// 은 어떤 경우에도 허용하지 않습니다.
        raise ImageLoadError(f"허용되지 않는 스킴입니다: {parsed.scheme}")

    if settings.allow_private_image_hosts:
        logger.warning("allow_private_image_hosts=true 이므로 내부 주소 검사를 건너뜁니다: %s", parsed.hostname)
        return

    host = parsed.hostname or ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return  # 도메인 이름이면 통과합니다. S3 presigned URL 의 정상 경로입니다.

    if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
        raise ImageLoadError(f"내부 주소로는 요청할 수 없습니다: {host}")


def _normalize(blob: bytes) -> LoadedImage:
    """장변을 설정값으로 줄이고 PNG 로 다시 인코딩합니다.

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
        _assert_public_url(image_url)

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

        loaded = _normalize(b"".join(chunks))
        logger.info(
            "이미지 로드 완료 %dx%d (%d bytes -> %d bytes)",
            loaded.width,
            loaded.height,
            loaded.downloaded_bytes,
            len(loaded.data),
        )
        return loaded


image_loader = ImageLoader()
