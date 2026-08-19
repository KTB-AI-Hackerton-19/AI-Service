"""OpenAPI 스펙을 파일로 내보냅니다.

백엔드에서 Java 클라이언트를 생성하거나 계약 변경을 리뷰할 때 씁니다.
서버를 띄우지 않아도 되므로 CI 에서도 돌릴 수 있습니다.

    python scripts/export_openapi.py            # docs/openapi.json 갱신
    python scripts/export_openapi.py --check    # 변경이 있으면 종료코드 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 스펙 생성에는 모델도 외부 서비스도 필요 없습니다.
os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("TAVILY_ENABLED", "false")

from app.main import app  # noqa: E402

OUTPUT = ROOT / "docs" / "openapi.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="파일이 최신인지만 확인합니다")
    args = parser.parse_args()

    spec = json.dumps(app.openapi(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not OUTPUT.exists():
            print(f"{OUTPUT} 가 없습니다. python scripts/export_openapi.py 를 실행하세요.")
            return 1
        if OUTPUT.read_text(encoding="utf-8") != spec:
            print(f"{OUTPUT} 가 코드와 다릅니다. python scripts/export_openapi.py 를 실행하세요.")
            return 1
        print(f"{OUTPUT} 최신 상태입니다.")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(spec, encoding="utf-8")
    paths = sorted(json.loads(spec)["paths"])
    print(f"{OUTPUT} 갱신 완료. 엔드포인트 {len(paths)}개")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
