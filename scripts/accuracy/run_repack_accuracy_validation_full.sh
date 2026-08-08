MODEL="Qwen/Qwen3-4B"
MODEL_TAG="Qwen-Qwen3-4B"
ROOT="eval_logs/pa_decision_full/${MODEL_TAG}"
LOG_ROOT="logs/pa_decision_full/${MODEL_TAG}"

mkdir -p "$ROOT" "$LOG_ROOT"

run_one () {
  label="$1"
  method="$2"
  k_budget="$3"
  v_budget="$4"

  echo "开始运行: $label"

  python scripts/accuracy/accuracy_eval.py \
    -m "$MODEL" \
    -t gsm8k \
    -o "$ROOT/$label" \
    -b 1 \
    --quant_method PackKV \
    --scale_method "$method" \
    --repack_method BUCKET \
    --k_scale 0.03 \
    --v_scale 0.10 \
    --k_error_budget "$k_budget" \
    --v_error_budget "$v_budget" \
    --block_size 64 \
    --buffer_size 192 \
    --pack_size 16 \
    --bucket_count 4 \
    --bucket_score_method k_sum \
    --no-high_precision_zero_point \
    2>&1 | tee "$LOG_ROOT/${label}.log"

  status="${PIPESTATUS[0]}"
  if [ "$status" -ne 0 ]; then
    echo "运行失败: $label,exit=$status"
    return "$status"
  fi
}

run_one nearest       po2_nearest    0     0
run_one pa_k005_v000  po2_pack_aware 0.05  0
