#!/usr/bin/env bash
set -uo pipefail

# 与 nearest 全量精度候选对应的离线压缩率对照；统一关闭重排以隔离量化。
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B}"
CTX_LEN="${CTX_LEN:-4096}"
COLLECT_ROUND="${COLLECT_ROUND:-1}"
LOG_ROOT="${LOG_ROOT:-logs/cr_nearest_candidate}"
SUITE_ID="${SUITE_ID:-nearest_candidate_cr}"

# label:scale_method:k_scale:v_scale
CONFIGS=(
  "continuous_conservative_003_010:continuous:0.03:0.10"
  "nearest_conservative_003_010:po2_nearest:0.03:0.10"
  "continuous_compression_004_012:continuous:0.04:0.12"
  "nearest_compression_004_012:po2_nearest:0.04:0.12"
)

mkdir -p "$LOG_ROOT"

run_one() {
  local label="$1"
  local scale_method="$2"
  local k_scale="$3"
  local v_scale="$4"
  local log_file="${LOG_ROOT}/${label}.log"
  local done_file="${LOG_ROOT}/${label}.done"

  if [[ -f "$done_file" ]]; then
    echo "跳过已有成功结果: $label"
    return 0
  fi

  echo "=================================================="
  echo "运行 CR: $label"
  echo "模型: $MODEL | ctx_len: $CTX_LEN"
  echo "scale_method: $scale_method | K=$k_scale | V=$v_scale"
  echo "repack_method: NONE | high_precision_zero_point: False"
  echo "=================================================="

  "$PYTHON_BIN" scripts/cr/cr_eval.py \
    -m "$MODEL" \
    --ctx_len "$CTX_LEN" \
    --collect_round "$COLLECT_ROUND" \
    --suite_id "$SUITE_ID" \
    --run_id "${SUITE_ID}-${label}" \
    --quant_method PackKV \
    --repack_method NONE \
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
echo "CR 对照完成，失败任务数: $failed"
echo "日志目录: $LOG_ROOT"
echo "宏观汇总: csv_results/Global_Macro_Summary_v11.csv"
echo "逐层汇总: csv_results/Layer_Detail_v6.csv"
echo "=================================================="

exit "$failed"
