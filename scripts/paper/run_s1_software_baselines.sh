#!/usr/bin/env bash
set -uo pipefail

# BRISK-KV paper S1 software-baseline runner.  This orchestrates existing
# evaluation entry points only; it does not modify PackKV algorithms or RTL.
#
# Examples:
#   MODE=pilot RUN_TAG=paper_s1_20260821_pilot bash scripts/paper/run_s1_software_baselines.sh
#   MODE=full LIMIT=200 RUN_TAG=paper_s1_20260821_full bash scripts/paper/run_s1_software_baselines.sh
#   STAGES=check RUN_TAG=paper_s1_20260821_pilot bash scripts/paper/run_s1_software_baselines.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-pilot}"
TASK="${TASK:-gsm8k}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CTX_LEN="${CTX_LEN:-4096}"
COLLECT_ROUND="${COLLECT_ROUND:-1}"
STAGES="${STAGES:-accuracy,cr,check}"
RUN_TAG="${RUN_TAG:-paper_s1_baseline_$(date +%Y%m%d_%H%M%S)_${MODE}}"
SUITE_ID="${SUITE_ID:-${RUN_TAG}-cr}"

BLOCK_SIZE=64
BUFFER_SIZE=192
PACK_SIZE=16
BUCKET_COUNT=4
BUCKET_SCORE_METHOD=k_sum
K_SCALE=0.03
V_SCALE=0.10

case "$MODE" in
  pilot)
    MODELS=("Qwen/Qwen3-4B")
    EVAL_LIMIT="${LIMIT:-20}"
    ;;
  full)
    MODELS=(
      "Qwen/Qwen3-4B"
      "Qwen/Qwen3-8B"
      "NousResearch/Meta-Llama-3.1-8B"
    )
    EVAL_LIMIT="${LIMIT:-}"
    ;;
  *)
    echo "MODE must be pilot or full, got: $MODE" >&2
    exit 2
    ;;
esac

RUN_ROOT="eval_logs/${RUN_TAG}"
LOG_ROOT="logs/${RUN_TAG}"
CSV_ROOT="csv_results/${RUN_TAG}"
ACCURACY_ROOT="${RUN_ROOT}/accuracy"
PAIR_ROOT="${ACCURACY_ROOT}/po2_repack_pair"
ACCURACY_SUMMARY="${CSV_ROOT}/accuracy_summary.csv"
PAIR_COMPARISON="${CSV_ROOT}/repack_accuracy_comparison.csv"
JOINT_SUMMARY="${CSV_ROOT}/joint_accuracy_cr_summary.csv"
CHECK_REPORT="${CSV_ROOT}/s1_check_report.json"
CR_SUMMARY="csv_results/Global_Macro_Summary_v11.csv"

contains_stage() {
  local target="$1"
  local stage
  local stage_list=()
  IFS=',' read -r -a stage_list <<< "$STAGES"
  for stage in "${stage_list[@]}"; do
    [[ "$stage" == "$target" ]] && return 0
  done
  return 1
}

validate_stages() {
  local stage
  local stage_list=()
  IFS=',' read -r -a stage_list <<< "$STAGES"
  for stage in "${stage_list[@]}"; do
    case "$stage" in
      accuracy|cr|check) ;;
      *)
        echo "STAGES accepts accuracy,cr,check; got: $stage" >&2
        exit 2
        ;;
    esac
  done
}

validate_stages

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$CSV_ROOT" "$ACCURACY_ROOT" "$PAIR_ROOT"

accuracy_output_dir() {
  local model_tag="$1"
  local variant="$2"
  if [[ "$variant" == po2_none || "$variant" == po2_bucket ]]; then
    printf '%s/%s/%s' "$PAIR_ROOT" "$model_tag" "$variant"
  else
    printf '%s/variants/%s/%s' "$ACCURACY_ROOT" "$model_tag" "$variant"
  fi
}

run_accuracy() {
  local model="$1"
  local variant="$2"
  local scale_method="$3"
  local repack_method="$4"
  local quant_enabled="$5"
  local model_tag="${model//\//-}"
  local output_dir
  output_dir="$(accuracy_output_dir "$model_tag" "$variant")"
  local log_file="${LOG_ROOT}/accuracy/${model_tag}_${variant}.log"
  local limit_args=()
  local quant_args=()

  if [[ -n "$(find "$output_dir" -type f -name results.json -print -quit 2>/dev/null)" && \
        -n "$(find "$output_dir" -type f -name packkv_config.json -print -quit 2>/dev/null)" ]]; then
    echo "skip accuracy (complete): model=$model variant=$variant"
    return 0
  fi
  if [[ -n "$EVAL_LIMIT" ]]; then
    limit_args+=(--limit "$EVAL_LIMIT")
  fi
  if [[ "$quant_enabled" == false ]]; then
    quant_args+=(--no_quant)
  fi
  mkdir -p "$output_dir" "$(dirname "$log_file")"

  echo "accuracy: model=$model variant=$variant task=$TASK limit=${EVAL_LIMIT:-full}"
  "$PYTHON_BIN" scripts/accuracy/accuracy_eval.py \
    -m "$model" -t "$TASK" -o "$output_dir" -b "$BATCH_SIZE" \
    --quant_method PackKV \
    --scale_method "$scale_method" --repack_method "$repack_method" \
    --k_scale "$K_SCALE" --v_scale "$V_SCALE" \
    --block_size "$BLOCK_SIZE" --buffer_size "$BUFFER_SIZE" --pack_size "$PACK_SIZE" \
    --bucket_count "$BUCKET_COUNT" --bucket_score_method "$BUCKET_SCORE_METHOD" \
    --no-high_precision_zero_point \
    "${quant_args[@]}" "${limit_args[@]}" \
    2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  if (( status != 0 )); then
    echo "accuracy failed: model=$model variant=$variant exit=$status" >&2
  fi
  return "$status"
}

run_cr() {
  local model="$1"
  local variant="$2"
  local scale_method="$3"
  local repack_method="$4"
  local model_tag="${model//\//-}"
  local run_id="${SUITE_ID}-${model_tag}-${variant}"
  local log_file="${LOG_ROOT}/cr/${model_tag}_${variant}.log"
  local done_file="${LOG_ROOT}/cr/${model_tag}_${variant}.done"

  if [[ -f "$done_file" ]]; then
    echo "skip CR (complete): model=$model variant=$variant"
    return 0
  fi
  mkdir -p "$(dirname "$log_file")"

  echo "CR: model=$model variant=$variant ctx=$CTX_LEN suite=$SUITE_ID"
  "$PYTHON_BIN" scripts/cr/cr_eval.py \
    -m "$model" --ctx_len "$CTX_LEN" --collect_round "$COLLECT_ROUND" \
    --suite_id "$SUITE_ID" --run_id "$run_id" \
    --quant_method PackKV \
    --scale_method "$scale_method" --repack_method "$repack_method" \
    --k_scale "$K_SCALE" --v_scale "$V_SCALE" \
    --block_size "$BLOCK_SIZE" --buffer_size "$BUFFER_SIZE" --pack_size "$PACK_SIZE" \
    --bucket_count "$BUCKET_COUNT" --bucket_score_method "$BUCKET_SCORE_METHOD" \
    --no-high_precision_zero_point \
    2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  if (( status == 0 )); then
    touch "$done_file"
  else
    echo "CR failed: model=$model variant=$variant exit=$status" >&2
  fi
  return "$status"
}

failed=0
if contains_stage accuracy; then
  for model in "${MODELS[@]}"; do
    run_accuracy "$model" fp continuous NONE false || failed=$((failed + 1))
    run_accuracy "$model" continuous continuous NONE true || failed=$((failed + 1))
    run_accuracy "$model" po2_none po2_nearest NONE true || failed=$((failed + 1))
    run_accuracy "$model" po2_bucket po2_nearest BUCKET true || failed=$((failed + 1))
  done

  "$PYTHON_BIN" scripts/accuracy/summarize_pa_accuracy.py \
    --root "$ACCURACY_ROOT" --output "$ACCURACY_SUMMARY" || failed=$((failed + 1))
  "$PYTHON_BIN" scripts/accuracy/compare_repack_accuracy.py \
    --root "$PAIR_ROOT" --output "$PAIR_COMPARISON" || failed=$((failed + 1))
fi

if contains_stage cr; then
  for model in "${MODELS[@]}"; do
    run_cr "$model" continuous continuous NONE || failed=$((failed + 1))
    run_cr "$model" po2_none po2_nearest NONE || failed=$((failed + 1))
    run_cr "$model" po2_bucket po2_nearest BUCKET || failed=$((failed + 1))
  done
fi

if contains_stage check; then
  "$PYTHON_BIN" scripts/paper/join_check_s1_software_baselines.py \
    --accuracy-summary "$ACCURACY_SUMMARY" \
    --cr-summary "$CR_SUMMARY" \
    --suite-id "$SUITE_ID" \
    --task "$TASK" --ctx-len "$CTX_LEN" \
    --output "$JOINT_SUMMARY" --report "$CHECK_REPORT" \
    "${MODELS[@]}" || failed=$((failed + 1))
fi

echo "S1 run tag: $RUN_TAG"
echo "S1 suite id: $SUITE_ID"
echo "Accuracy summary: $ACCURACY_SUMMARY"
echo "PO2 repack comparison: $PAIR_COMPARISON"
echo "Joint summary: $JOINT_SUMMARY"
echo "Check report: $CHECK_REPORT"
echo "Failure count: $failed"
exit "$failed"
