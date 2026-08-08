#!/usr/bin/env bash
set -uo pipefail

# 使用已保存的真实高精度 KV Cache，验证固定候选的 BUCKET metadata 与
# K/V bitstream 能逐整数无损恢复。只做审计，不写入 CR CSV。
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-Qwen/Qwen3-4B Qwen/Qwen3-8B NousResearch/Meta-Llama-3.1-8B}"
CTX_LEN="${CTX_LEN:-4096}"
ROUNDTRIP_LAYERS="${ROUNDTRIP_LAYERS:-4}"
ROUNDTRIP_BLOCKS="${ROUNDTRIP_BLOCKS:-1}"
LOG_ROOT="${LOG_ROOT:-logs/real_roundtrip_validation}"

mkdir -p "$LOG_ROOT"
failed=0

for model in $MODELS; do
  model_tag="${model//\//-}"
  log_file="${LOG_ROOT}/${model_tag}_ctx-${CTX_LEN}.log"
  echo "=================================================="
  echo "真实 KV round-trip: $model"
  echo "layers=$ROUNDTRIP_LAYERS blocks/layer=$ROUNDTRIP_BLOCKS"
  echo "=================================================="

  "$PYTHON_BIN" scripts/cr/cr_eval.py \
    -m "$model" \
    --ctx_len "$CTX_LEN" \
    --collect_round 1 \
    --quant_method PackKV \
    --scale_method po2_nearest \
    --k_scale 0.03 \
    --v_scale 0.10 \
    --repack_method BUCKET \
    --block_size 64 \
    --buffer_size 192 \
    --pack_size 16 \
    --bucket_count 4 \
    --bucket_score_method k_sum \
    --no-high_precision_zero_point \
    --roundtrip_only \
    --roundtrip_layers "$ROUNDTRIP_LAYERS" \
    --roundtrip_blocks "$ROUNDTRIP_BLOCKS" \
    2>&1 | tee "$log_file"

  status=${PIPESTATUS[0]}
  if (( status != 0 )); then
    echo "失败: $model (exit=$status)" >&2
    failed=$((failed + 1))
  fi
done

echo "=================================================="
echo "真实 KV round-trip 完成，失败模型数: $failed"
echo "日志目录: $LOG_ROOT"
echo "本脚本不会修改 Global_Macro_Summary 或 Layer_Detail CSV"
echo "=================================================="
exit "$failed"
