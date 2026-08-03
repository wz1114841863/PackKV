#!/usr/bin/env bash
set -uo pipefail

# 当前 Conda/系统环境中的 python；如有需要可用 PYTHON_BIN 覆盖。
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-sanity}"
CTX_LEN="${CTX_LEN:-4096}"
LOG_ROOT="${LOG_ROOT:-logs/rq2_pack_aware}"
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
  --k_scale 0.1
  --v_scale 0.1
  --no-high_precision_zero_point
)

if [[ "$MODE" == "sanity" ]]; then
  MODELS=("Qwen/Qwen3-4B")
  BUDGET_PAIRS=("0.10:0.10")
elif [[ "$MODE" == "full" ]]; then
  MODELS=(
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "NousResearch/Meta-Llama-3-8B"
    "mistralai/Ministral-8B-Instruct-2410"
  )
  # 预算约束整层 selected SSE 相对 nearest SSE 的增幅。
  BUDGET_PAIRS=(
    "0.00:0.00"
    "0.05:0.05"
    "0.10:0.10"
    "0.25:0.25"
    "0.25:0.10"
    "0.50:0.25"
  )
else
  echo "用法: $0 [sanity|full]" >&2
  exit 2
fi

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
  echo "运行: model=$model ctx=$CTX_LEN scale=$method K预算=$k_budget V预算=$v_budget"
  "$PYTHON_BIN" scripts/cr/cr_eval.py \
    -m "$model" \
    "${COMMON_ARGS[@]}" \
    --scale_method "$method" \
    --suite_id "RQ2_layer_budget_${MODE}" \
    "${extra_args[@]}" \
    2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  if (( status != 0 )); then
    echo "失败: model=$model scale=$method K预算=$k_budget V预算=$v_budget (exit=$status)" >&2
  fi
  return "$status"
}

failed=0
for model in "${MODELS[@]}"; do
  # 两个固定端点用于解释自适应结果。
  run_one "$model" po2_nearest "NA" "NA" || failed=$((failed + 1))
  run_one "$model" po2_ceil "NA" "NA" || failed=$((failed + 1))
  for pair in "${BUDGET_PAIRS[@]}"; do
    IFS=: read -r k_budget v_budget <<< "$pair"
    run_one "$model" po2_pack_aware "$k_budget" "$v_budget" || failed=$((failed + 1))
  done
done

echo "完成: mode=$MODE, 失败任务数=$failed"
echo "宏观汇总: csv_results/Global_Macro_Summary_v9.csv"
echo "统一逐层明细: csv_results/Layer_Detail_v4.csv"
exit "$failed"
