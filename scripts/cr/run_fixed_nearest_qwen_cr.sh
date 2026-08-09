#!/usr/bin/env bash
set -uo pipefail

# 最终候选配置的 Qwen CR 补充实验：唯一变量为 NONE/BUCKET。
# 不激活或切换环境；使用调用者当前环境中的 Python。
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-Qwen/Qwen3-4B Qwen/Qwen3-8B}"
CTX_LEN="${CTX_LEN:-4096}"
COLLECT_ROUND="${COLLECT_ROUND:-1}"
BLOCK_SIZE=64
BUFFER_SIZE=192
PACK_SIZE=16
BUCKET_COUNT=4
BUCKET_SCORE_METHOD=k_sum
SCALE_METHOD=po2_nearest
K_SCALE=0.03
V_SCALE=0.10
SUITE_ID="${SUITE_ID:-fixed-nearest-003-010-qwen-cr}"
LOG_ROOT="${LOG_ROOT:-logs/cr_fixed_nearest_003_010}"

mkdir -p "$LOG_ROOT"

run_one() {
  local model="$1"
  local repack_method="$2"
  local model_tag="${model//\//-}"
  local run_id="${SUITE_ID}-${model_tag}-${repack_method,,}"
  local model_log_dir="${LOG_ROOT}/${model_tag}/ctx-${CTX_LEN}"
  local log_file="${model_log_dir}/${repack_method,,}.log"
  local done_file="${model_log_dir}/${repack_method,,}.done"
  mkdir -p "$model_log_dir"

  if [[ -f "$done_file" ]]; then
    echo "跳过已有成功结果: $model $repack_method"
    return 0
  fi

  echo "=================================================="
  echo "模型: $model | ctx_len: $CTX_LEN | repack: $repack_method"
  echo "po2_nearest | K=0.03 V=0.10 | block=64 buffer=192 pack=16"
  echo "BUCKET 固定参数: bucket_count=4 bucket_score_method=k_sum"
  echo "=================================================="

  "$PYTHON_BIN" scripts/cr/cr_eval.py \
    -m "$model" \
    --ctx_len "$CTX_LEN" \
    --collect_round "$COLLECT_ROUND" \
    --suite_id "$SUITE_ID" \
    --run_id "$run_id" \
    --quant_method PackKV \
    --scale_method "$SCALE_METHOD" \
    --k_scale "$K_SCALE" \
    --v_scale "$V_SCALE" \
    --repack_method "$repack_method" \
    --block_size "$BLOCK_SIZE" \
    --buffer_size "$BUFFER_SIZE" \
    --pack_size "$PACK_SIZE" \
    --bucket_count "$BUCKET_COUNT" \
    --bucket_score_method "$BUCKET_SCORE_METHOD" \
    --no-high_precision_zero_point \
    2>&1 | tee "$log_file"

  local status=${PIPESTATUS[0]}
  if (( status == 0 )); then
    touch "$done_file"
  else
    echo "失败: $model $repack_method (exit=$status)" >&2
  fi
  return "$status"
}

failed=0
for model in $MODELS; do
  for repack_method in NONE BUCKET; do
    run_one "$model" "$repack_method" || failed=$((failed + 1))
  done
done

echo "=================================================="
echo "固定候选 Qwen CR 完成，失败任务数: $failed"
echo "宏观汇总: csv_results/Global_Macro_Summary_v10.csv"
echo "逐层汇总: csv_results/Layer_Detail_v5.csv"
echo "筛选 Suite_ID=$SUITE_ID"
echo "=================================================="
exit "$failed"
