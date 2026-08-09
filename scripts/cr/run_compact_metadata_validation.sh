#!/usr/bin/env bash
set -uo pipefail

# 在三个模型、多个互不重叠 WikiText Cache 样本上比较 native 与真实窄字段
# quant metadata。第二种格式同时执行 metadata + bucket + bitstream round-trip。
PYTHON_BIN="${PYTHON_BIN:-python}"
CTX_LEN="${CTX_LEN:-8192}"
COLLECT_ROUND="${COLLECT_ROUND:-3}"
ROUNDTRIP_LAYERS="${ROUNDTRIP_LAYERS:-4}"
ROUNDTRIP_BLOCKS="${ROUNDTRIP_BLOCKS:-2}"
SUITE_ID="${SUITE_ID:-compact_metadata_ctx8192_v1}"
LOG_ROOT="${LOG_ROOT:-logs/compact_metadata_validation}"

DEFAULT_MODEL_LIST="Qwen/Qwen3-4B Qwen/Qwen3-8B NousResearch/Meta-Llama-3.1-8B"
read -r -a MODELS <<< "${MODEL_LIST:-$DEFAULT_MODEL_LIST}"

mkdir -p "$LOG_ROOT"
failed=0

for model in "${MODELS[@]}"; do
  model_tag="${model//\//-}"
  for metadata_format in native po2_compact; do
    run_id="${SUITE_ID}-${model_tag}-${metadata_format}"
    log_file="${LOG_ROOT}/${model_tag}_${metadata_format}_ctx-${CTX_LEN}.log"
    compact_args=()
    if [[ "$metadata_format" == "po2_compact" ]]; then
      compact_args+=(
        --k_zero_point_bits 7
        --v_zero_point_bits 5
        --exponent_bits 4
        --verify_roundtrip
        --roundtrip_layers "$ROUNDTRIP_LAYERS"
        --roundtrip_blocks "$ROUNDTRIP_BLOCKS"
      )
    fi

    echo "运行: model=$model ctx=$CTX_LEN samples=$COLLECT_ROUND metadata=$metadata_format"
    "$PYTHON_BIN" scripts/cr/cr_eval.py \
      -m "$model" \
      --ctx_len "$CTX_LEN" \
      --collect_round "$COLLECT_ROUND" \
      --suite_id "$SUITE_ID" \
      --run_id "$run_id" \
      --quant_method PackKV \
      --scale_method po2_nearest \
      --repack_method BUCKET \
      --k_scale 0.03 \
      --v_scale 0.10 \
      --block_size 64 \
      --buffer_size 192 \
      --pack_size 16 \
      --bucket_count 4 \
      --bucket_score_method k_sum \
      --no-high_precision_zero_point \
      --quant_metadata_format "$metadata_format" \
      "${compact_args[@]}" \
      2>&1 | tee "$log_file"

    status=${PIPESTATUS[0]}
    if (( status != 0 )); then
      echo "失败: model=$model metadata=$metadata_format (exit=$status)" >&2
      failed=$((failed + 1))
    fi
  done
done

echo "完成，失败任务数: $failed"
echo "宏观汇总: csv_results/Global_Macro_Summary_v11.csv"
echo "逐层汇总: csv_results/Layer_Detail_v6.csv"
echo "日志目录: $LOG_ROOT"
exit "$failed"
