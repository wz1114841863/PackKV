#!/usr/bin/env bash
set -uo pipefail

# 使用当前 Conda/系统环境；不会自动 source .venv，也不会删除 lm-eval 缓存。
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-sanity}"
TASK="${TASK:-gsm8k}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_ROOT="${LOG_ROOT:-logs/accuracy_pa}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_logs/accuracy_pa}"

case "$MODE" in
  sanity)
    MODELS=("Qwen/Qwen3-4B")
    EVAL_LIMIT="${LIMIT:-20}"
    CONFIGS=("continuous:continuous:NA:NA" "nearest:po2_nearest:NA:NA" "pa_010_010:po2_pack_aware:0.10:0.10")
    ;;
  pilot)
    MODELS=("Qwen/Qwen3-4B")
    EVAL_LIMIT="${LIMIT:-200}"
    CONFIGS=(
      "fp:no_quant:NA:NA"
      "continuous:continuous:NA:NA"
      "nearest:po2_nearest:NA:NA"
      "pa_005_005:po2_pack_aware:0.05:0.05"
      "pa_010_010:po2_pack_aware:0.10:0.10"
      "pa_025_010:po2_pack_aware:0.25:0.10"
      "pa_025_025:po2_pack_aware:0.25:0.25"
    )
    ;;
  full)
    MODELS=(
      "Qwen/Qwen3-4B"
      "Qwen/Qwen3-8B"
      "NousResearch/Meta-Llama-3-8B"
      "mistralai/Ministral-8B-Instruct-2410"
    )
    # full 默认跑完整任务；可用 LIMIT=200 先缩短。
    EVAL_LIMIT="${LIMIT:-}"
    CONFIGS=(
      "fp:no_quant:NA:NA"
      "continuous:continuous:NA:NA"
      "nearest:po2_nearest:NA:NA"
      "pa_005_005:po2_pack_aware:0.05:0.05"
      "pa_010_010:po2_pack_aware:0.10:0.10"
      "pa_025_010:po2_pack_aware:0.25:0.10"
      "pa_025_025:po2_pack_aware:0.25:0.25"
    )
    ;;
  *)
    echo "用法: $0 [sanity|pilot|full]" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_ROOT/$MODE" "$OUTPUT_ROOT/$MODE"

run_one() {
  local model="$1" label="$2" method="$3" k_budget="$4" v_budget="$5"
  local model_tag="${model//\//-}"
  local run_tag="${model_tag}_${TASK}_${label}"
  local log_file="$LOG_ROOT/$MODE/${run_tag}.log"
  local output_dir="$OUTPUT_ROOT/$MODE/$run_tag"
  local method_args=(--scale_method "$method")
  local budget_args=()
  local quant_args=()
  local limit_args=()

  if [[ "$method" == "no_quant" ]]; then
    method_args=(--scale_method continuous)
    quant_args+=(--no_quant)
  elif [[ "$method" == "po2_pack_aware" ]]; then
    budget_args+=(--k_error_budget "$k_budget" --v_error_budget "$v_budget")
  fi
  if [[ -n "$EVAL_LIMIT" ]]; then
    limit_args+=(--limit "$EVAL_LIMIT")
  fi

  echo "运行: model=$model task=$TASK config=$label limit=${EVAL_LIMIT:-full}"
  "$PYTHON_BIN" scripts/accuracy/accuracy_eval.py \
    -m "$model" \
    -t "$TASK" \
    -o "$output_dir" \
    -b "$BATCH_SIZE" \
    --quant_method PackKV \
    --k_scale 0.1 \
    --v_scale 0.1 \
    --block_size 64 \
    --buffer_size 192 \
    --pack_size 16 \
    --bucket_count 4 \
    --bucket_score_method k_sum \
    --no-high_precision_zero_point \
    "${method_args[@]}" \
    "${budget_args[@]}" \
    "${quant_args[@]}" \
    "${limit_args[@]}" \
    2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  if (( status != 0 )); then
    echo "失败: model=$model config=$label (exit=$status)" >&2
  fi
  return "$status"
}

failed=0
for model in "${MODELS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    IFS=: read -r label method k_budget v_budget <<< "$config"
    run_one "$model" "$label" "$method" "$k_budget" "$v_budget" || failed=$((failed + 1))
  done
done

"$PYTHON_BIN" scripts/accuracy/summarize_pa_accuracy.py \
  --root "$OUTPUT_ROOT/$MODE" \
  --output "$OUTPUT_ROOT/$MODE/accuracy_summary.csv" || true

echo "完成: mode=$MODE, 失败任务数=$failed"
echo "结果目录: $OUTPUT_ROOT/$MODE"
echo "汇总文件: $OUTPUT_ROOT/$MODE/accuracy_summary.csv"
echo "日志目录: $LOG_ROOT/$MODE"
exit "$failed"
