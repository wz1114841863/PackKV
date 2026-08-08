#!/usr/bin/env bash
set -uo pipefail

# 固定最终 BUCKET 配置，筛选 packing-aware nearest/ceil 量化。
# sanity: Qwen3-4B, ctx=4096；full: 三个模型。
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-sanity}"
CTX_LEN="${CTX_LEN:-4096}"
LOG_ROOT="${LOG_ROOT:-logs/rq2_pack_aware_final}"
SUITE_ID="${SUITE_ID:-RQ2_pack_aware_final_${MODE}}"
mkdir -p "$LOG_ROOT/$MODE"

COMMON_ARGS=(
  --ctx_len "$CTX_LEN"
  --collect_round 1
  --quant_method PackKV
  --repack_method BUCKET
  --block_size 64
  --buffer_size 192
  --pack_size 16
  --bucket_count 4
  --bucket_score_method k_sum
  --k_scale 0.03
  --v_scale 0.10
  --no-high_precision_zero_point
)

if [[ "$MODE" == "sanity" ]]; then
  MODELS=("Qwen/Qwen3-4B")
elif [[ "$MODE" == "full" ]]; then
  MODELS=(
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "NousResearch/Meta-Llama-3.1-8B"
  )
else
  echo "用法: $0 [sanity|full]" >&2
  exit 2
fi

# 优先把预算给 K；V 对 BUCKET 压缩贡献很小，最后一个点才允许少量 V 预算。
BUDGET_PAIRS=(
  "0.000:0.000"
  "0.005:0.000"
  "0.010:0.000"
  "0.020:0.000"
  "0.050:0.000"
  "0.020:0.010"
)

run_one() {
  local model="$1"
  local method="$2"
  local k_budget="$3"
  local v_budget="$4"
  local model_tag="${model//\//-}"
  local tag="${model_tag}_ctx-${CTX_LEN}_${method}"
  local extra_args=()
  if [[ "$method" == "po2_pack_aware" ]]; then
    tag="${tag}_kerr-${k_budget}_verr-${v_budget}"
    extra_args+=(--k_error_budget "$k_budget" --v_error_budget "$v_budget")
  fi
  local log_file="$LOG_ROOT/$MODE/${tag}.log"
  local done_file="$LOG_ROOT/$MODE/${tag}.done"
  if [[ -f "$done_file" ]]; then
    echo "跳过已有成功结果: $tag"
    return 0
  fi

  echo "运行: model=$model ctx=$CTX_LEN method=$method K预算=$k_budget V预算=$v_budget"
  "$PYTHON_BIN" scripts/cr/cr_eval.py \
    -m "$model" \
    "${COMMON_ARGS[@]}" \
    --scale_method "$method" \
    --suite_id "$SUITE_ID" \
    --run_id "${SUITE_ID}-${tag}" \
    "${extra_args[@]}" \
    2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  if (( status == 0 )); then
    touch "$done_file"
  else
    echo "失败: $tag (exit=$status)" >&2
  fi
  return "$status"
}

failed=0
for model in "${MODELS[@]}"; do
  run_one "$model" po2_nearest "NA" "NA" || failed=$((failed + 1))
  run_one "$model" po2_ceil "NA" "NA" || failed=$((failed + 1))
  for pair in "${BUDGET_PAIRS[@]}"; do
    IFS=: read -r k_budget v_budget <<< "$pair"
    run_one "$model" po2_pack_aware "$k_budget" "$v_budget" || failed=$((failed + 1))
  done
done

echo "=================================================="
echo "packing-aware 筛选完成: mode=$MODE, 失败任务数=$failed"
echo "固定配置: po2 0.03/0.10 + BUCKET(k_sum,4), ctx=$CTX_LEN"
echo "宏观汇总: csv_results/Global_Macro_Summary_v9.csv"
echo "筛选 Suite_ID=$SUITE_ID"
echo "=================================================="
exit "$failed"
