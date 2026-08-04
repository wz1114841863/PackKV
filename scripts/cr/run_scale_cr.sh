#!/usr/bin/env bash
set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

MODELS=(
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
)

SCALES=(0.01 0.03 0.05 0.07 0.10)
METHODS=(continuous po2_nearest)

mkdir -p logs/cr_scale_sensitivity

failed=0

for model in "${MODELS[@]}"; do
  model_tag="${model//\//-}"

  for scale in "${SCALES[@]}"; do
    for method in "${METHODS[@]}"; do
      tag="${model_tag}_${method}_k${scale}_v${scale}"

      python scripts/cr/cr_eval.py \
        -m "$model" \
        --ctx_len 4096 \
        --collect_round 1 \
        --quant_method PackKV \
        --repack_method BUCKET \
        --scale_method "$method" \
        --k_scale "$scale" \
        --v_scale "$scale" \
        --block_size 64 \
        --buffer_size 192 \
        --pack_size 16 \
        --bucket_count 4 \
        --bucket_score_method k_sum \
        --no-high_precision_zero_point \
        --suite_id scale_sensitivity \
        2>&1 | tee "logs/cr_scale_sensitivity/${tag}.log"

      status=${PIPESTATUS[0]}
      if (( status != 0 )); then
        failed=$((failed + 1))
      fi
    done
  done
done

echo "完成,失败任务数: $failed"
exit "$failed"
