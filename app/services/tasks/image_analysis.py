"""S3 선물 이미지를 공통 선물데이터로 변환하는 작업."""

import asyncio

from app.schemas.agent import GiftData


class ImageAnalysisService:
    """이미지 분석 담당자가 실제 구현으로 교체할 서비스."""

    async def analyze(self, image_url: str) -> GiftData:
        """S3 이미지 주소를 분석해 선물명·가격·나이를 반환합니다.

        Args:
            image_url: Spring Boot가 전달한 S3 HTTP(S) 주소 또는 presigned URL.

        Returns:
            후속 네 작업이 공통으로 사용할 ``GiftData``.

        Raises:
            Exception: S3 다운로드나 비전 모델 분석이 실패한 경우.
                오케스트레이터가 이를 안전한 502 응답으로 변환합니다.
        """
        # =====================================================================
        # IMPLEMENTATION POINT 1: 이미지 추출/분석 담당자가 수정할 곳
        # ---------------------------------------------------------------------
        # 1) image_url에서 이미지를 읽습니다.
        # 2) 비전 모델로 gift_name, gift_price, age를 추출합니다.
        # 3) 반드시 GiftData 타입으로 반환합니다.
        # 함수 이름, 입력 타입, 반환 타입은 다른 코드와의 계약이므로 유지하세요.
        # =====================================================================
        await asyncio.sleep(0)
        return GiftData(
            gift_name="이미지에서 추출된 선물",
            gift_price=30_000,
            age=None,
        )


image_analysis_service = ImageAnalysisService()
