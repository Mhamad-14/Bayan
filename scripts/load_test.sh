#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v hey >/dev/null 2>&1; then
  echo "ERROR: hey is not installed."
  echo "On macOS install it with: brew install hey"
  exit 2
fi

mkdir -p artifacts/serving
OUT="artifacts/serving/hey_output.txt"

echo "Health before load:"
curl -fsS http://127.0.0.1:8000/health
echo
echo

hey -z 60s -c 16 -m POST -H 'Content-Type: application/json' \
  -d '{"text":"الخدمة ممتازة ولكن التأخير طويل"}' \
  http://127.0.0.1:8000/v1/classify | tee "$OUT"

P99_SECONDS="$(awk '$1 ~ /^99%+$/ && $2 == "in" {print $3}' "$OUT" | tail -1)"
if [[ -z "${P99_SECONDS}" ]]; then
  echo "Could not parse hey p99."
  exit 3
fi

P99_MS="$(python - <<PY
print(float("${P99_SECONDS}") * 1000.0)
PY
)"

if grep -q "Error distribution:" "$OUT"; then
  ERRORS="present — inspect artifacts/serving/hey_output.txt"
else
  ERRORS="0"
fi

python - <<PY
from pathlib import Path

p = Path("BENCHMARKS.md")
text = p.read_text(encoding="utf-8")
p99 = float("${P99_MS}")
errors = "${ERRORS}"

old = "- HTTP p99, 16 concurrent: pending load test"
new = f"- HTTP p99, 16 concurrent: {p99:.2f} ms; request errors: {errors}; target <= 40 ms: " + ("met" if p99 <= 40 and errors == "0" else "not met / inspect errors")
if old in text:
    text = text.replace(old, new)
else:
    text += "\n" + new + "\n"

p.write_text(text, encoding="utf-8")
print(new)
PY
