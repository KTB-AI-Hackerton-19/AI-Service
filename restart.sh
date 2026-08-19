#!/usr/bin/env bash
# Giftie AI Service를 재시작합니다 (stop.sh 후 start.sh).
#
#   ./restart.sh
#   PORT=8000 ./restart.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$ROOT/stop.sh"
"$ROOT/start.sh"
