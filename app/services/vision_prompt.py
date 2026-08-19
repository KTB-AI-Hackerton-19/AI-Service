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
- 연도가 보이지 않으면 {year}년으로 간주하세요.
- 입금·받은 선물은 received, 출금·보낸 것은 sent 입니다.
- 영수증은 구매한 상품 각각을 원소로 만들되, 할인·적립·부가세·합계처럼 상품이 아닌 줄은 넣지 마세요.
- 청첩장·부고장에 적힌 계좌번호 안내는 앞으로 보낼 곳일 뿐 내가 받은 것이 아니므로 records 에 넣지 마세요.
- counterpart_name 에는 "님", "씨" 같은 호칭을 빼고 이름만 넣으세요.
- 확실하지 않은 필드는 null 로 두세요. 추측해서 채우지 마세요.

JSON만 출력하세요."""
