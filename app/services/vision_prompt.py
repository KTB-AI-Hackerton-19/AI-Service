"""이미지 분석(VLM)이 사용하는 프롬프트와 강제 출력 스키마.

추천용 프롬프트가 ``app/services/prompt.py`` 에 있는 것과 같은 위치다.

설계 근거는 사전 실측(RTX 4090에서 L4 메모리 예산 시뮬레이션) 결과다.
- 필드 이름을 의미로 못박는다. ``date`` 하나로 두면 모델이 수령일과 행사일을 섞는다.
- 영수증의 할인·합계 줄은 프롬프트와 후처리 양쪽에서 걸러낸다.
- 이미지 한 장에 여러 건이 있을 수 있으므로 출력은 항상 배열이다.
  (계좌 거래내역 4~5건, 선물함 목록 4건, 영수증 3품목)

샘플링은 ``vision_temperature=0.0`` (greedy) 을 쓴다.
Gemma4-12B-QAT 로 실측했을 때 temperature 0.0 과 공식 권장값(1.0/0.95/64)의 추출 정확도는
6종 이미지 3회 반복에서 모두 1.000 으로 동일했고 MTP 채택 길이도 1.88 vs 1.87 로 차이가 없었다.
그럼에도 greedy 를 쓰는 이유는 재현성이다. 같은 이미지를 다시 올렸을 때 금액이 달라지면
사용자가 결과를 신뢰할 수 없고, 디버깅도 불가능해진다.
"""

IMAGE_KINDS = [
    "kakao_gift",
    "kakao_transfer",
    "bank_statement",
    "invitation",
    "receipt",
    "gift_list",
    "other",
]

RECORD_TYPES = ["gift", "money", "event_invitation", "receipt", "unknown"]
DIRECTIONS = ["received", "sent", "unknown"]

_RECORD_PROPERTIES = {
    "record_type": {"type": "string", "enum": RECORD_TYPES},
    "direction": {"type": "string", "enum": DIRECTIONS},
    "counterpart_name": {"type": ["string", "null"]},
    "occurred_date": {"type": ["string", "null"]},
    "event_date": {"type": ["string", "null"]},
    "item_name": {"type": ["string", "null"]},
    "brand": {"type": ["string", "null"]},
    "category": {"type": ["string", "null"]},
    "event": {"type": ["string", "null"]},
    "amount": {"type": ["integer", "null"]},
    "memo": {"type": ["string", "null"]},
    "confidence": {"type": "number"},
}

# 전체 필드를 required 로 두면 정확도가 가장 높지만 출력 토큰이 늘어난다.
# L4에서 Gemma4-12B 는 약 25 tok/s 라 출력 토큰이 그대로 지연이 된다.
_REQUIRED_FULL = list(_RECORD_PROPERTIES)
_REQUIRED_LEAN = [
    "record_type",
    "direction",
    "counterpart_name",
    "occurred_date",
    "amount",
    "confidence",
]


def build_extraction_schema(lean: bool = False) -> dict:
    """vLLM ``response_format={"type": "json_schema"}`` 에 그대로 넣는 스키마.

    Args:
        lean: True 면 required 필드를 6개로 줄여 출력 토큰을 깎는다.
            정확도가 떨어질 수 있으므로 기본값은 False 다.

    Returns:
        구조화 출력 강제에 사용할 JSON Schema.
    """
    return {
        "type": "object",
        "properties": {
            "image_kind": {"type": "string", "enum": IMAGE_KINDS},
            "records": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": _RECORD_PROPERTIES,
                    "required": _REQUIRED_LEAN if lean else _REQUIRED_FULL,
                    "additionalProperties": False,
                },
            },
        },
        "required": ["image_kind", "records"],
        "additionalProperties": False,
    }


SYSTEM_PROMPT = (
    "당신은 이미지에서 선물·부조금 기록을 추출하는 도구입니다. "
    "보이는 사실만 옮기고, 보이지 않는 값은 추측하지 않습니다. 요청한 JSON 만 출력합니다."
)

EXTRACTION_PROMPT = """이 이미지에서 선물·부조금과 관련된 기록을 빠짐없이 추출해 JSON으로 출력하세요.

이미지는 카카오톡 선물 메시지, 송금 메시지, 계좌 거래내역, 청첩장·부고장, 영수증, 선물함 목록 중 하나입니다.
image_kind 에 어떤 종류인지 넣으세요.

records 배열에 화면에 보이는 항목을 하나도 빠뜨리지 말고 담으세요.
거래내역이나 선물함처럼 여러 건이면 각 건을 별도 원소로 만들고, 선물 메시지 한 건이면 원소 1개짜리 배열로 만드세요.

각 원소의 필드:
- record_type: gift(물건·기프티콘) | money(현금·송금·축의금·조의금) | event_invitation(청첩장·부고장) | receipt(영수증) | unknown
- direction: received(내가 받음) | sent(내가 보냄) | unknown
- counterpart_name: 상대방 이름. 내 이름이 아니라 보낸 사람 또는 받는 사람
- occurred_date: 실제로 주고받은 날짜 (YYYY-MM-DD)
- event_date: 행사가 열리는 날짜 (YYYY-MM-DD). 청첩장의 예식일, 부고장의 발인일 등
- item_name: 선물·상품명
- brand: 브랜드·매장명
- category: 기프티콘/음료, 조의금, 축의금, 화장품 같은 분류
- event: 계기. 생일, 결혼, 조의, 출산, 집들이 등
- amount: 금액(원, 정수). 쉼표 없이 숫자만
- memo: 메시지에 적힌 문구나 거래 적요
- confidence: 이 원소를 얼마나 확신하는지 0~1 실수

규칙:
- 월·일은 보이는데 연도만 없을 때만 {year}년으로 간주하세요. 날짜 자체가 화면에
  없으면 occurred_date 와 event_date 를 null 로 두세요. 시각(오전 7:53)만 보이는
  것은 날짜가 아닙니다. 빠진 값은 나중에 사용자가 확인 화면에서 채웁니다.
- "내" 는 이 화면을 캡처한 사람, 즉 앱 사용자입니다. 입금·받은 선물은 received,
  출금·보낸 것은 sent 입니다.
- 메신저 대화 화면에서는 말풍선 위치로 방향을 판단하세요. 왼쪽에 붙어 있고 프로필
  사진이 함께 보이면 상대가 보낸 것이므로 received 이고, 오른쪽에 붙어 있고 프로필
  사진이 없으면 내가 보낸 것이므로 sent 입니다.
- 선물 카드의 "OO 선물을 보냈어요" 같은 문구는 보낸 사람 시점으로 인쇄된 고정 문구
  입니다. 이 문구만 보고 sent 로 판단하지 말고 반드시 말풍선 위치를 먼저 보세요.
- 메신저 화면 상단이나 프로필 옆에 보이는 대화 상대 이름을 counterpart_name 에
  넣으세요. 여러 명이 있는 단체방 이름으로 보이면 null 로 두세요.
- amount 에는 원화 금액만 넣으세요. "$25", "¥3,000" 처럼 외화로 적혀 있으면
  환산하지 말고 amount 를 null 로 둔 뒤 memo 에 보이는 그대로 적으세요.
- 정가와 할인가가 함께 보이면 실제로 결제한 금액(할인가, 최종가)을 amount 에 넣으세요.
- 수량이 여러 개면 화면에 보이는 결제 총액을 amount 에 넣으세요. 총액이 없고 단가만
  보이면 곱하지 말고 단가를 넣은 뒤 memo 에 수량을 적으세요.
- 영수증은 구매한 상품 각각을 원소로 만들되, 할인·적립·부가세·합계처럼 상품이 아닌 줄은 넣지 마세요.
- 청첩장·부고장에 적힌 계좌번호 안내는 앞으로 보낼 곳일 뿐 내가 받은 것이 아니므로 records 에 넣지 마세요.
- counterpart_name 에는 "님", "씨" 같은 호칭을 빼고 이름만 넣으세요.
- 확실하지 않은 필드는 null 로 두세요. 추측해서 채우지 마세요.
{hint}
JSON만 출력하세요."""

# 사용자가 업로드 화면에서 이미 고른 종류. 비용도 지연도 늘지 않는 힌트라
# 넣지 않을 이유가 없습니다. 다만 단정적으로 강제하면 사용자가 잘못 골랐을 때
# 모델이 이미지를 무시하게 되므로 "이미지가 우선" 을 함께 적습니다.
_CATEGORY_HINTS = {
    "gift": "참고: 사용자는 이 이미지를 '선물'로 골라 올렸습니다. 선물·기프티콘 기록일 가능성이 높습니다.",
    "occasion": (
        "참고: 사용자는 이 이미지를 '경조사'로 골라 올렸습니다. "
        "청첩장·부고장이거나 축의금·조의금 기록일 가능성이 높습니다."
    ),
}
_CATEGORY_HINT_TAIL = "이건 힌트일 뿐이니, 이미지가 분명히 다른 종류라면 보이는 대로 판단하세요."


def build_extraction_prompt(year: int, category: str | None = None) -> str:
    """연도와 사용자가 고른 종류를 채운 추출 프롬프트를 만듭니다.

    Args:
        year: 연도가 보이지 않을 때 사용할 기준 연도.
        category: 사용자가 업로드 화면에서 고른 종류(``gift`` / ``occasion``).
            모르면 ``None``.

    Returns:
        모델에 그대로 보낼 프롬프트 문자열.
    """
    hint = _CATEGORY_HINTS.get(str(category)) if category else None
    return EXTRACTION_PROMPT.format(
        year=year,
        hint=f"\n{hint} {_CATEGORY_HINT_TAIL}\n" if hint else "",
    )
