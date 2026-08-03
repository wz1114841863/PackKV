#!/usr/bin/env bash
set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LIMIT="${LIMIT:-200}"
ROOT="eval_logs/scale_sensitivity"
LOG_ROOT="logs/scale_sensitivity"

MODELS=(
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
)

SCALES=(0.01 0.03 0.05 0.07 0.10)
METHODS=(continuous po2_nearest)

mkdir -p "$ROOT" "$LOG_ROOT"

run_eval() {
  local model="$1"
  local label="$2"
  local method="$3"
  local k_scale="$4"
  local v_scale="$5"
  local model_tag="${model//\//-}"
  local tag="${model_tag}_${label}"
  local output="${ROOT}/${tag}"
  local log="${LOG_ROOT}/${tag}.log"
  local extra_args=()

  if [[ "$label" == "fp" ]]; then
    extra_args+=(--no_quant)
  fi

  echo "运行: model=$model config=$label"

  "$PYTHON_BIN" scripts/accuracy/accuracy_eval.py \
    -m "$model" \
    -t gsm8k \
    -o "$output" \
    -b 1 \
    --limit "$LIMIT" \
    --quant_method PackKV \
    --k_scale "$k_scale" \
    --v_scale "$v_scale" \
    --scale_method "$method" \
    --block_size 64 \
    --buffer_size 192 \
    --pack_size 16 \
    --bucket_count 4 \
    --bucket_score_method k_sum \
    --no-high_precision_zero_point \
    "${extra_args[@]}" \
    2>&1 | tee "$log"

  return "${PIPESTATUS[0]}"
}

failed=0

for model in "${MODELS[@]}"; do
  # 每个模型只跑一次全精度基线
  run_eval "$model" fp continuous 0.01 0.01 || failed=$((failed + 1))

  for scale in "${SCALES[@]}"; do
    for method in "${METHODS[@]}"; do
      label="${method}_k${scale}_v${scale}"
      run_eval "$model" "$label" "$method" "$scale" "$scale" \
        || failed=$((failed + 1))
    done
  done
done

echo "完成,失败任务数: $failed"
echo "结果目录: $ROOT"
echo "日志目录: $LOG_ROOT"
exit "$failed"
