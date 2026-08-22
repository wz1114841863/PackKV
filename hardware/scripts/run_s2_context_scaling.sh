#!/usr/bin/env bash
set -euo pipefail

# S2 runs cycle and architectural-SRAM scaling only. It never writes frozen
# result directories and does not run DC or alter algorithm/RTL source.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARDWARE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHISEL_DIR="$HARDWARE_DIR/chisel"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SBT_BIN="${SBT_BIN:-sbt}"
RUN_TAG="${RUN_TAG:-s2_context_scaling_$(date +%Y%m%d_%H%M%S)}"
TOKEN_COUNTS="${TOKEN_COUNTS:-256,512,1024,2048,4096}"
FEATURE_DIM="${FEATURE_DIM:-128}"
BACKPRESSURE_MODES="${BACKPRESSURE_MODES:-none,periodic}"
RESULT_ROOT="${RESULT_ROOT:-$HARDWARE_DIR/evaluation/results/context_scaling/$RUN_TAG}"
RTL_ROOT="${RTL_ROOT:-$HARDWARE_DIR/rtl/generated/context_scaling/$RUN_TAG}"
RUN_CACTI="${RUN_CACTI:-false}"

case "$RUN_CACTI" in true|false) ;; *) echo "RUN_CACTI must be true or false" >&2; exit 2;; esac
IFS=',' read -r -a TOKENS <<< "$TOKEN_COUNTS"
for token in "${TOKENS[@]}"; do
  if (( token < 64 || token % 64 )); then
    echo "TOKEN_COUNTS entries must be multiples of 64 and at least 64: $token" >&2
    exit 2
  fi
done

mkdir -p "$RESULT_ROOT" "$RTL_ROOT"
OUTPUT_DIR="$RESULT_ROOT/cycles" FEATURE_DIM="$FEATURE_DIM" TOKEN_COUNTS="$TOKEN_COUNTS" \
ARCHITECTURES="full_v,jit_v_shared_writer_cg" BACKPRESSURE_MODES="$BACKPRESSURE_MODES" \
  bash "$SCRIPT_DIR/run_cycle_benchmark.sh"

for architecture in full_v jit_v_shared_writer_cg; do
  for token in "${TOKENS[@]}"; do
    target_dir="$RTL_ROOT/$architecture/t$token/dc_logic"
    if [[ -f "$target_dir/manifest.json" ]]; then
      echo "skip existing dc_logic export: $target_dir"
      continue
    fi
    mkdir -p "$target_dir"
    (
      cd "$CHISEL_DIR"
      "$SBT_BIN" "runMain briskkv.GenerateBriskKvSingleHeadTileTop \
        --target-dir $target_dir \
        --maximum-feature-dim $FEATURE_DIM \
        --maximum-tokens $token \
        --input-bits 24 \
        --scale-lanes 4 \
        --enable-stats false \
        --quant-architecture v1 \
        --attention-architecture $architecture \
        --vcs-compatibility false \
        --mode dc_logic"
    )
  done
done

"$PYTHON_BIN" "$HARDWARE_DIR/evaluation/context_scaling/summarize_context_scaling.py" \
  --cycle-summary "$RESULT_ROOT/cycles/cycle_benchmark_summary.csv" \
  --rtl-root "$RTL_ROOT" --output-dir "$RESULT_ROOT" --tokens "$TOKEN_COUNTS" \
  --feature-dim "$FEATURE_DIM" --backpressure periodic --clock-period-ns 2.0 \
  --frozen-full-cycle "$HARDWARE_DIR/evaluation/results/cycle_breakdown/2026081702_matched_replay_pipe_v1/full_v.csv" \
  --frozen-jit-cycle "$HARDWARE_DIR/evaluation/results/cycle_breakdown/2026081703_shared_writer_cg/jit_v_shared_writer_cg.csv"

if [[ "$RUN_CACTI" == true ]]; then
  for architecture in full_v jit_v_shared_writer_cg; do
    for token in "${TOKENS[@]}"; do
      "$PYTHON_BIN" "$HARDWARE_DIR/evaluation/mem/briskkv_memory_eval.py" \
        --memories-csv "$RTL_ROOT/$architecture/t$token/dc_logic/memories.csv" \
        --output-dir "$RESULT_ROOT/cacti/$architecture/t$token" \
        --technology-nm 22 --clock-period-ns 2.0 --maximum-bank-width 128
    done
  done
fi

printf 'S2 cycle/SRAM outputs: %s\n' "$RESULT_ROOT"
printf 'S2 generated dc_logic exports: %s\n' "$RTL_ROOT"
