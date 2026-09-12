#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=4
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

echo
echo "===== 1. Baseline BEFORE ONNX/INT8 ====="
python scripts/benchmark_inference.py

echo
echo "===== 2. Export ONNX + INT8 + paired quality ====="
python scripts/export_onnx.py

echo
echo "===== 3. Benchmark optimisation ladder ====="
python scripts/benchmark_inference.py

echo
echo "===== 4. Serving contract ====="
pytest tests/test_serving_contract.py -q

echo
echo "===== 5. Start API and verify health ====="
uvicorn bayan.serving.api:app --host 127.0.0.1 --port 8000 \
  > artifacts/serving/uvicorn.log 2>&1 &
PID=$!

cleanup() {
  kill "$PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for i in {1..30}; do
  if curl -fsS http://127.0.0.1:8000/health >/tmp/bayan_health.json 2>/dev/null; then
    break
  fi
  sleep 1
done

cat /tmp/bayan_health.json
echo

if ! grep -q '"status":"ok"' /tmp/bayan_health.json; then
  echo "API did not become healthy. See artifacts/serving/uvicorn.log"
  exit 4
fi

echo
echo "===== 6. HTTP load test ====="
bash scripts/load_test.sh

echo
echo "===== 7. Final checks ====="
pytest tests/test_serving_contract.py -q
git diff --check
git status --short

echo
echo "Lab 7 run complete ✅"
