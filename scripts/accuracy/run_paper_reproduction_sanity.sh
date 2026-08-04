#!/usr/bin/env bash
set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="NousResearch/Meta-Llama-3.1-8B"
LIMIT="${LIMIT:-20}"
ROOT="eval_logs/paper_reproduction_sanity"
LOG_ROOT="logs/paper_reproduction_sanity"

mkdir -p "$ROOT" "$LOG_ROOT"

run_one() {
  local label="$1"
  local quant_method="$2"
  local k_scale="$3"
  local v_scale="$4"
  local no_quant="$5"

  local output_dir="${ROOT}/${label}"
  local log_file="${LOG_ROOT}/${label}.log"
  local extra_args=()

  if [[ "$no_quant" == "yes" ]]; then
    extra_args+=(--no_quant)
  fi

  echo "运行: $label, quant=$quant_method, K=$k_scale, V=$v_scale"

  "$PYTHON_BIN" scripts/accuracy/accuracy_eval.py \
    -m "$MODEL" \
    -t gsm8k \
    -o "$output_dir" \
    -b 1 \
    --limit "$LIMIT" \
    --quant_method "$quant_method" \
    --scale_method continuous \
    --k_scale "$k_scale" \
    --v_scale "$v_scale" \
    --block_size 64 \
    --buffer_size 192 \
    --pack_size 16 \
    --no-high_precision_zero_point \
    "${extra_args[@]}" \
    2>&1 | tee "$log_file"

  return "${PIPESTATUS[0]}"
}

failed=0

# 真正的未量化基线
run_one fp PackKV 0.01 0.01 yes || failed=$((failed + 1))

# 官方代码用于近似未压缩精度的 .01/.01 代理基线
run_one proxy_001 PackKV 0.01 0.01 no || failed=$((failed + 1))

# 论文 Llama3.1-8B GSM8K K-token turning point 约 0.063
run_one k_token_006 PackKV 0.06 0.01 no || failed=$((failed + 1))

# 论文 V-token turning point 约 0.173
run_one v_token_018 PackKV 0.01 0.18 no || failed=$((failed + 1))

# 论文 KIVI K-channel turning point 约 0.117
run_one k_channel_012 KIVI 0.12 0.01 no || failed=$((failed + 1))

echo "完成,失败任务数: $failed"
echo "结果目录: $ROOT"
echo "日志目录: $LOG_ROOT"
exit "$failed"
