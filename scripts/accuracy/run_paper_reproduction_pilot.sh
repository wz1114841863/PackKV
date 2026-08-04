#!/usr/bin/env bash
set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="NousResearch/Meta-Llama-3.1-8B"
LIMIT="${LIMIT:-200}"

ROOT="eval_logs/paper_reproduction_pilot"
LOG_ROOT="logs/paper_reproduction_pilot"
SUMMARY="${ROOT}/accuracy_summary.csv"

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

  # 已有成功结果时跳过,支持中断后继续.
  if [[ -n "$(find "$output_dir" -name results.json -print -quit 2>/dev/null)" ]]; then
    echo "跳过已有结果: $label"
    return 0
  fi

  if [[ "$no_quant" == "yes" ]]; then
    extra_args+=(--no_quant)
  fi

  echo "=================================================="
  echo "运行: $label"
  echo "模型: $MODEL"
  echo "量化: $quant_method | K=$k_scale | V=$v_scale"
  echo "样本数: $LIMIT"
  echo "=================================================="

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
    --bucket_count 4 \
    --bucket_score_method k_sum \
    --no-high_precision_zero_point \
    "${extra_args[@]}" \
    2>&1 | tee "$log_file"

  local status=${PIPESTATUS[0]}
  if (( status != 0 )); then
    echo "失败: $label (exit=$status)" >&2
  fi
  return "$status"
}

failed=0

# 1. 真正的 FP 基线
run_one \
  fp \
  PackKV \
  0.01 \
  0.01 \
  yes || failed=$((failed + 1))

# 2. 官方代码使用的 .01/.01 代理基线
run_one \
  proxy_001 \
  PackKV \
  0.01 \
  0.01 \
  no || failed=$((failed + 1))

# 3. PackKV K token-wise 粗扫描,V 固定为 .01
for k_scale in 0.04 0.06 0.08; do
  label="k_token_k${k_scale}_v0.01"
  run_one \
    "$label" \
    PackKV \
    "$k_scale" \
    0.01 \
    no || failed=$((failed + 1))
done

# 4. PackKV V token-wise 粗扫描,K 固定为 .01
for v_scale in 0.12 0.18 0.24; do
  label="v_token_k0.01_v${v_scale}"
  run_one \
    "$label" \
    PackKV \
    0.01 \
    "$v_scale" \
    no || failed=$((failed + 1))
done

# 5. KIVI K channel-wise 粗扫描,V 固定为 .01
for k_scale in 0.08 0.12 0.16; do
  label="k_channel_k${k_scale}_v0.01"
  run_one \
    "$label" \
    KIVI \
    "$k_scale" \
    0.01 \
    no || failed=$((failed + 1))
done

echo "=================================================="
echo "开始生成汇总 CSV"
echo "=================================================="

"$PYTHON_BIN" scripts/accuracy/summarize_pa_accuracy.py \
  --root "$ROOT" \
  --output "$SUMMARY"

summary_status=$?
if (( summary_status != 0 )); then
  echo "汇总失败 (exit=$summary_status)" >&2
  failed=$((failed + 1))
fi

echo "=================================================="
echo "实验完成"
echo "失败任务数: $failed"
echo "结果目录: $ROOT"
echo "日志目录: $LOG_ROOT"
echo "汇总文件: $SUMMARY"
echo "=================================================="

exit "$failed"
