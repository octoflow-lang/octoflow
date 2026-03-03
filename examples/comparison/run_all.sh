#!/usr/bin/env bash
# OctoFlow vs PyTorch — Full Comparison Suite Runner
# Usage: bash run_all.sh [--flow-only | --py-only]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OCTOFLOW="${OCTOFLOW:-octoflow}"
PYTHON="${PYTHON:-python}"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FLOW_OUT="$RESULTS_DIR/octoflow_${TIMESTAMP}.txt"
PY_OUT="$RESULTS_DIR/pytorch_${TIMESTAMP}.txt"

SCENARIOS=(
  s1_data_pipeline
  s2_statistics
  s3_signal_processing
  s4_monte_carlo
  s5_ml_forward
  s6_data_transform
)

run_flow=true
run_py=true
if [[ "${1:-}" == "--flow-only" ]]; then run_py=false; fi
if [[ "${1:-}" == "--py-only" ]]; then run_flow=false; fi

echo "=== OctoFlow vs PyTorch Comparison Suite ==="
echo "Timestamp: $TIMESTAMP"
echo ""

# --- OctoFlow benchmarks ---
if $run_flow; then
  echo "--- Running OctoFlow benchmarks ---"
  : > "$FLOW_OUT"
  for s in "${SCENARIOS[@]}"; do
    echo -n "  $s ... "
    if "$OCTOFLOW" run "$SCRIPT_DIR/${s}.flow" 2>/dev/null | tee -a "$FLOW_OUT" | grep "BENCH|" > /dev/null; then
      echo "OK"
    else
      echo "FAIL"
    fi
  done
  echo "  Results: $FLOW_OUT"
  echo ""
fi

# --- PyTorch benchmarks ---
if $run_py; then
  echo "--- Running PyTorch benchmarks ---"
  : > "$PY_OUT"
  for s in "${SCENARIOS[@]}"; do
    echo -n "  $s ... "
    if "$PYTHON" "$SCRIPT_DIR/${s}.py" 2>/dev/null | tee -a "$PY_OUT" | grep "BENCH|" > /dev/null; then
      echo "OK"
    else
      echo "FAIL"
    fi
  done
  echo "  Results: $PY_OUT"
  echo ""
fi

# --- Generate comparison ---
if $run_flow && $run_py; then
  echo "--- Generating comparison table ---"
  "$PYTHON" "$SCRIPT_DIR/compare.py" "$FLOW_OUT" "$PY_OUT"
fi

echo ""
echo "Done."
