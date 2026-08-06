#!/usr/bin/env bash
set -uo pipefail

# 验证普通 po2_nearest 在 NONE 与 BUCKET 重排下的端到端 GSM8K 精度。
# LIMIT 默认 200；显式设置 LIMIT= 可运行完整 GSM8K。
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B}"
TASK="${TASK:-gsm8k}"
LIMIT="${LIMIT-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL_TAG="${MODEL//\//-}"

if [[ -n "$LIMIT" ]]; then
  SCOPE="limit-${LIMIT}"
else
  SCOPE="full"
fi
ROOT="${ROOT:-eval_logs/repack_accuracy_validation/${MODEL_TAG}/${SCOPE}}"
LOG_ROOT="${LOG_ROOT:-logs/repack_accuracy_validation/${MODEL_TAG}/${SCOPE}}"
SUMMARY="${SUMMARY:-${ROOT}/accuracy_summary.csv}"
COMPARISON="${COMPARISON:-${ROOT}/repack_accuracy_comparison.csv}"

# label:repack_method:k_scale:v_scale
CONFIGS=(
  "nearest_conservative_none:NONE:0.03:0.10"
  "nearest_conservative_bucket:BUCKET:0.03:0.10"
  "nearest_compression_none:NONE:0.04:0.12"
  "nearest_compression_bucket:BUCKET:0.04:0.12"
)

mkdir -p "$ROOT" "$LOG_ROOT"

run_one() {
  local label="$1"
  local repack_method="$2"
  local k_scale="$3"
  local v_scale="$4"
  local output_dir="${ROOT}/${label}"
  local log_file="${LOG_ROOT}/${label}.log"
  local limit_args=()

  if [[ -n "$(find "$output_dir" -name results.json -print -quit 2>/dev/null)" ]]; then
    echo "跳过已有结果: $label"
    return 0
  fi
  if [[ -n "$LIMIT" ]]; then
    limit_args+=(--limit "$LIMIT")
  fi

  echo "=================================================="
  echo "运行: $label"
  echo "模型: $MODEL | 任务: $TASK | 样本数: ${LIMIT:-full}"
  echo "po2_nearest K=$k_scale V=$v_scale | repack=$repack_method"
  echo "=================================================="

  "$PYTHON_BIN" scripts/accuracy/accuracy_eval.py \
    -m "$MODEL" \
    -t "$TASK" \
    -o "$output_dir" \
    -b "$BATCH_SIZE" \
    --quant_method PackKV \
    --scale_method po2_nearest \
    --repack_method "$repack_method" \
    --k_scale "$k_scale" \
    --v_scale "$v_scale" \
    --block_size 64 \
    --buffer_size 192 \
    --pack_size 16 \
    --bucket_count 4 \
    --bucket_score_method k_sum \
    --no-high_precision_zero_point \
    "${limit_args[@]}" \
    2>&1 | tee "$log_file"

  local status=${PIPESTATUS[0]}
  if (( status != 0 )); then
    echo "失败: $label (exit=$status)" >&2
  fi
  return "$status"
}

failed=0
for config in "${CONFIGS[@]}"; do
  IFS=: read -r label repack_method k_scale v_scale <<< "$config"
  run_one "$label" "$repack_method" "$k_scale" "$v_scale" || failed=$((failed + 1))
done

"$PYTHON_BIN" scripts/accuracy/summarize_pa_accuracy.py \
  --root "$ROOT" \
  --output "$SUMMARY" || failed=$((failed + 1))

"$PYTHON_BIN" scripts/accuracy/compare_repack_accuracy.py \
  --root "$ROOT" \
  --output "$COMPARISON" || failed=$((failed + 1))

echo "=================================================="
echo "重排精度验证完成，失败任务数: $failed"
echo "结果目录: $ROOT"
echo "日志目录: $LOG_ROOT"
echo "精度汇总: $SUMMARY"
echo "NONE/BUCKET 逐题配对: $COMPARISON"
echo "=================================================="

exit "$failed"
