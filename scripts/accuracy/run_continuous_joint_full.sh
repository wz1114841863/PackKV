#!/usr/bin/env bash
set -uo pipefail

# Llama-3.1-8B / GSM8K continuous 联合 K/V 量化补充实验。
# 默认不传 --limit，因此执行完整任务；可用 LIMIT=20 做冒烟测试。
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B}"
TASK="${TASK:-gsm8k}"
LIMIT="${LIMIT:-}"
ROOT="${ROOT:-eval_logs/paper_reproduction_full}"
LOG_ROOT="${LOG_ROOT:-logs/paper_reproduction_full}"
SUMMARY="${SUMMARY:-${ROOT}/accuracy_summary_with_joint.csv}"

mkdir -p "$ROOT" "$LOG_ROOT"

run_one() {
  local k_scale="$1"
  local v_scale="$2"
  local label="joint_continuous_k${k_scale}_v${v_scale}"
  local output_dir="${ROOT}/${label}"
  local log_file="${LOG_ROOT}/${label}.log"
  local limit_args=()

  # 支持中断后续跑，避免覆盖已经完成的 lm-eval 结果。
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
  echo "量化: PackKV continuous | K=$k_scale | V=$v_scale"
  echo "重排: NONE"
  echo "=================================================="

  "$PYTHON_BIN" scripts/accuracy/accuracy_eval.py \
    -m "$MODEL" \
    -t "$TASK" \
    -o "$output_dir" \
    -b 1 \
    --quant_method PackKV \
    --scale_method continuous \
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

run_one 0.04 0.12 || failed=$((failed + 1))
run_one 0.06 0.12 || failed=$((failed + 1))
run_one 0.06 0.18 || failed=$((failed + 1))

"$PYTHON_BIN" scripts/accuracy/summarize_pa_accuracy.py \
  --root "$ROOT" \
  --output "$SUMMARY"

summary_status=$?
if (( summary_status != 0 )); then
  echo "汇总失败 (exit=$summary_status)" >&2
  failed=$((failed + 1))
fi

echo "=================================================="
echo "实验完成，失败任务数: $failed"
echo "结果目录: $ROOT"
echo "日志目录: $LOG_ROOT"
echo "汇总文件: $SUMMARY"
echo "=================================================="

exit "$failed"
