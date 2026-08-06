#!/usr/bin/env bash
set -uo pipefail

# 量化策略精度筛选：默认在 GSM8K 前 200 个样本上比较 continuous 与 2^k。
# 可通过 LIMIT=20 做冒烟测试；不同 LIMIT 自动写入不同目录，避免相互跳过。
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B}"
TASK="${TASK:-gsm8k}"
LIMIT="${LIMIT:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"

MODEL_TAG="${MODEL//\//-}"
ROOT="${ROOT:-eval_logs/quant_strategy_screen/${MODEL_TAG}/limit-${LIMIT}}"
LOG_ROOT="${LOG_ROOT:-logs/quant_strategy_screen/${MODEL_TAG}/limit-${LIMIT}}"
SUMMARY="${SUMMARY:-${ROOT}/accuracy_summary.csv}"

# 默认运行全部方法；可用 METHODS="continuous po2_nearest" 分批执行。
read -r -a SCALE_METHODS <<< "${METHODS:-continuous po2_nearest po2_ceil po2_floor}"

# label:k_scale:v_scale
CONFIGS=(
  "proxy_001:0.01:0.01"
  "k_only_004:0.04:0.01"
  "k_only_006:0.06:0.01"
  "k_only_008:0.08:0.01"
  "v_only_012:0.01:0.12"
  "v_only_018:0.01:0.18"
  "v_only_024:0.01:0.24"
  "joint_003_010:0.03:0.10"
  "joint_004_010:0.04:0.10"
  "joint_004_012:0.04:0.12"
  "joint_006_012:0.06:0.12"
  "joint_006_018:0.06:0.18"
)

mkdir -p "$ROOT" "$LOG_ROOT"

run_one() {
  local scale_method="$1"
  local config_label="$2"
  local k_scale="$3"
  local v_scale="$4"
  local run_label="${scale_method}__${config_label}"
  local output_dir="${ROOT}/${run_label}"
  local log_file="${LOG_ROOT}/${run_label}.log"

  # 已有成功结果时跳过，支持中断后续跑。
  if [[ -n "$(find "$output_dir" -name results.json -print -quit 2>/dev/null)" ]]; then
    echo "跳过已有结果: $run_label"
    return 0
  fi

  echo "=================================================="
  echo "运行: $run_label"
  echo "模型: $MODEL | 任务: $TASK | 样本数: $LIMIT"
  echo "scale_method: $scale_method | K=$k_scale | V=$v_scale"
  echo "repack_method: NONE | high_precision_zero_point: False"
  echo "=================================================="

  "$PYTHON_BIN" scripts/accuracy/accuracy_eval.py \
    -m "$MODEL" \
    -t "$TASK" \
    -o "$output_dir" \
    -b "$BATCH_SIZE" \
    --limit "$LIMIT" \
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
    2>&1 | tee "$log_file"

  local status=${PIPESTATUS[0]}
  if (( status != 0 )); then
    echo "失败: $run_label (exit=$status)" >&2
  fi
  return "$status"
}

failed=0
for scale_method in "${SCALE_METHODS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    IFS=: read -r config_label k_scale v_scale <<< "$config"
    run_one \
      "$scale_method" \
      "$config_label" \
      "$k_scale" \
      "$v_scale" || failed=$((failed + 1))
  done
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
echo "筛选完成，失败任务数: $failed"
echo "结果目录: $ROOT"
echo "日志目录: $LOG_ROOT"
echo "汇总文件: $SUMMARY"
echo "=================================================="

exit "$failed"
