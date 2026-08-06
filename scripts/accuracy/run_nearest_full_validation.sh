#!/usr/bin/env bash
set -uo pipefail

# po2_nearest 全量 GSM8K 候选验证。
# 默认不传 --limit；可用 LIMIT=20 做独立目录的冒烟测试。
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B}"
TASK="${TASK:-gsm8k}"
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL_TAG="${MODEL//\//-}"

if [[ -n "$LIMIT" ]]; then
  SCOPE="limit-${LIMIT}"
  ROOT="${ROOT:-eval_logs/nearest_full_validation/${MODEL_TAG}/${SCOPE}}"
else
  SCOPE="full"
  # 沿用论文复现目录，使新汇总同时包含已有 FP、proxy 和 continuous 全量结果。
  ROOT="${ROOT:-eval_logs/paper_reproduction_full}"
fi

LOG_ROOT="${LOG_ROOT:-logs/nearest_full_validation/${MODEL_TAG}/${SCOPE}}"
SUMMARY="${SUMMARY:-${ROOT}/accuracy_summary_nearest_validation.csv}"

# label:scale_method:k_scale:v_scale
CONFIGS=(
  "nearest_proxy_001:po2_nearest:0.01:0.01"
  "continuous_conservative_003_010:continuous:0.03:0.10"
  "nearest_conservative_003_010:po2_nearest:0.03:0.10"
  "nearest_compression_004_012:po2_nearest:0.04:0.12"
)

mkdir -p "$ROOT" "$LOG_ROOT"

run_one() {
  local label="$1"
  local scale_method="$2"
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
  echo "scale_method: $scale_method | K=$k_scale | V=$v_scale"
  echo "repack_method: NONE | high_precision_zero_point: False"
  echo "=================================================="

  "$PYTHON_BIN" scripts/accuracy/accuracy_eval.py \
    -m "$MODEL" \
    -t "$TASK" \
    -o "$output_dir" \
    -b "$BATCH_SIZE" \
    --quant_method PackKV \
    --scale_method "$scale_method" \
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
  IFS=: read -r label scale_method k_scale v_scale <<< "$config"
  run_one \
    "$label" \
    "$scale_method" \
    "$k_scale" \
    "$v_scale" || failed=$((failed + 1))
done

"$PYTHON_BIN" scripts/accuracy/summarize_pa_accuracy.py \
  --root "$ROOT" \
  --output "$SUMMARY"

summary_status=$?
if (( summary_status != 0 )); then
  echo "汇总失败 (exit=$summary_status)" >&2
  failed=$((failed + 1))
fi

echo "=================================================="
echo "验证完成，失败任务数: $failed"
echo "结果目录: $ROOT"
echo "日志目录: $LOG_ROOT"
echo "汇总文件: $SUMMARY"
echo "=================================================="

exit "$failed"
