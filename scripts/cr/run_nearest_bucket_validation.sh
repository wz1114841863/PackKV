#!/usr/bin/env bash
set -uo pipefail

# 对 nearest 候选补跑硬件友好的 BUCKET 重排。
# 与 run_nearest_candidate_cr.sh 的四个 NONE 配置逐一配对；唯一算法差异应为
# RepackMethod.BUCKET（bucket_count=4, bucket_score_method=k_sum）。
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B}"
CTX_LEN="${CTX_LEN:-4096}"
COLLECT_ROUND="${COLLECT_ROUND:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
BUFFER_SIZE="${BUFFER_SIZE:-192}"
PACK_SIZE="${PACK_SIZE:-16}"
BUCKET_COUNT="${BUCKET_COUNT:-4}"
MODEL_TAG="${MODEL//\//-}"
SUITE_ID="${SUITE_ID:-nearest_bucket_cr}"
LOG_ROOT="${LOG_ROOT:-logs/cr_nearest_bucket/${MODEL_TAG}/ctx-${CTX_LEN}/buckets-${BUCKET_COUNT}}"

# label:scale_method:k_scale:v_scale
CONFIGS=(
  "continuous_conservative_003_010:continuous:0.03:0.10"
  "nearest_conservative_003_010:po2_nearest:0.03:0.10"
  "continuous_compression_004_012:continuous:0.04:0.12"
  "nearest_compression_004_012:po2_nearest:0.04:0.12"
)

if (( BUCKET_COUNT < 2 || BUCKET_COUNT > BLOCK_SIZE || (BUCKET_COUNT & (BUCKET_COUNT - 1)) != 0 )); then
  echo "BUCKET_COUNT 必须是 [2, BLOCK_SIZE] 范围内的 2 的幂" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"

run_one() {
  local label="$1"
  local scale_method="$2"
  local k_scale="$3"
  local v_scale="$4"
  local run_id="${SUITE_ID}-${label}-bucket${BUCKET_COUNT}"
  local log_file="${LOG_ROOT}/${label}.log"
  local done_file="${LOG_ROOT}/${label}.done"

  if [[ -f "$done_file" ]]; then
    echo "跳过已有成功结果: $label"
    return 0
  fi

  echo "=================================================="
  echo "运行 BUCKET CR: $label"
  echo "模型: $MODEL | ctx_len: $CTX_LEN"
  echo "scale_method: $scale_method | K=$k_scale | V=$v_scale"
  echo "repack_method: BUCKET | buckets: $BUCKET_COUNT | score: k_sum"
  echo "high_precision_zero_point: False"
  echo "=================================================="

  "$PYTHON_BIN" scripts/cr/cr_eval.py \
    -m "$MODEL" \
    --ctx_len "$CTX_LEN" \
    --collect_round "$COLLECT_ROUND" \
    --suite_id "$SUITE_ID" \
    --run_id "$run_id" \
    --quant_method PackKV \
    --repack_method BUCKET \
    --scale_method "$scale_method" \
    --k_scale "$k_scale" \
    --v_scale "$v_scale" \
    --block_size "$BLOCK_SIZE" \
    --buffer_size "$BUFFER_SIZE" \
    --pack_size "$PACK_SIZE" \
    --bucket_count "$BUCKET_COUNT" \
    --bucket_score_method k_sum \
    --no-high_precision_zero_point \
    2>&1 | tee "$log_file"

  local status=${PIPESTATUS[0]}
  if (( status == 0 )); then
    touch "$done_file"
  else
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

echo "=================================================="
echo "BUCKET CR 对照完成，失败任务数: $failed"
echo "日志目录: $LOG_ROOT"
echo "宏观汇总: csv_results/Global_Macro_Summary_v11.csv"
echo "逐层汇总: csv_results/Layer_Detail_v6.csv"
echo "请按 Suite_ID=$SUITE_ID 筛选本轮四条记录"
echo "=================================================="

exit "$failed"
