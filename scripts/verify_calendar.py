"""Google Calendar MCP 연동을 실제 계정으로 검증합니다.

생성 -> 조회 -> 삭제 순으로 돌리므로 캘린더에 흔적이 남지 않습니다.

사용법
    # 1) MCP 서버 기동
    python -m mcp_servers.google_calendar

    # 2) .env 에 GOOGLE_ACCESS_TOKEN 을 넣은 뒤
    python scripts/verify_calendar.py
"""

import asyncio
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.schemas.agent import GiftData  # noqa: E402
from app.services.calendar_mcp_client import CalendarMcpError, calendar_mcp_client  # noqa: E402
from app.services.tasks.calendar import calendar_preparation_service  # noqa: E402

MARKER = "[Giftie 연동 검증]"


def ok(message: str) -> None:
    print(f"  \033[32m✓\033[0m {message}")


def fail(message: str) -> None:
    print(f"  \033[31m✗\033[0m {message}")


async def main() -> int:
    token = settings.google_access_token
    if not token:
        fail("GOOGLE_ACCESS_TOKEN 이 비어 있습니다. .env 에 넣어 주세요.")
        return 1
    print(f"토큰 확인: {token[:12]}… (길이 {len(token)})")
    print(f"MCP 서버 : {settings.calendar_mcp_url}")
    print(f"캘린더   : {settings.google_calendar_id}\n")

    # 1) MCP 서버 연결
    print("1. MCP 서버 연결")
    try:
        tools = await calendar_mcp_client.list_tool_names()
    except CalendarMcpError as exc:
        fail(f"{exc}")
        print("\n   MCP 서버가 떠 있는지 확인하세요: python -m mcp_servers.google_calendar")
        return 1
    ok(f"툴 {len(tools)}개 노출: {', '.join(tools)}")

    # 2) 일정 생성
    print("\n2. 일정 생성 (create_event)")
    start = date.today() + timedelta(days=7)
    try:
        created = await calendar_mcp_client.create_event(
            access_token=token,
            summary=f"{MARKER} 답례 준비",
            description="Giftie AI 서비스의 MCP 연동 검증용 일정입니다. 검증이 끝나면 자동으로 삭제됩니다.",
            start_date=start.isoformat(),
            start_time="10:00",
            duration_minutes=30,
            reminders_minutes=[0, 24 * 60],
        )
    except CalendarMcpError as exc:
        fail(f"{exc}")
        return 1

    event_id = created.get("event_id")
    if not event_id:
        fail(f"event_id 가 없습니다: {created}")
        return 1
    ok(f"event_id={event_id}")
    ok(f"start={created.get('start')}")
    ok(f"link={created.get('html_link')}")

    # 3) 조회로 실제 등록 확인
    print("\n3. 조회로 실제 등록 확인 (list_events)")
    try:
        listed = await calendar_mcp_client.call_tool(
            "list_events",
            {
                "access_token": token,
                "time_min": f"{start.isoformat()}T00:00:00+09:00",
                "time_max": f"{(start + timedelta(days=1)).isoformat()}T00:00:00+09:00",
                "calendar_id": settings.google_calendar_id,
            },
        )
    except CalendarMcpError as exc:
        fail(f"{exc}")
        listed = {"events": []}

    found = [e for e in listed.get("events", []) if e.get("event_id") == event_id]
    if found:
        ok(f"캘린더에서 확인됨: {found[0].get('summary')}")
    else:
        fail("생성한 일정을 조회 결과에서 찾지 못했습니다")

    # 4) 서비스 경로 전체 (calendar.prepare)
    print("\n4. 서비스 경로 전체 (calendar_preparation_service.prepare)")
    gift = GiftData(
        gift_name="스타벅스 아이스 카페 아메리카노 T",
        gift_price=12300,
        person_name="김수현",
        relationship="대학 동기",
        received_at=date.today(),
        target_date=date.today() + timedelta(days=30),
    )
    prepared = await calendar_preparation_service.prepare(gift, "verify-workflow")
    payload = prepared.payload or {}
    if payload.get("registered"):
        ok(f"provider={payload.get('provider')} eventId={payload.get('eventId')}")
        ok(f"title={payload.get('title')} date={payload.get('date')} {payload.get('startTime')}")
        service_event_id = payload.get("eventId")
    else:
        fail(f"등록되지 않았습니다: {payload.get('registerError') or payload.get('provider')}")
        service_event_id = None

    # 5) 정리
    print("\n5. 검증용 일정 삭제 (delete_event)")
    for eid in filter(None, [event_id, service_event_id]):
        try:
            await calendar_mcp_client.call_tool(
                "delete_event",
                {"access_token": token, "event_id": eid, "calendar_id": settings.google_calendar_id},
            )
            ok(f"삭제됨 {eid}")
        except CalendarMcpError as exc:
            fail(f"삭제 실패 {eid}: {exc}")

    print("\n검증 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
