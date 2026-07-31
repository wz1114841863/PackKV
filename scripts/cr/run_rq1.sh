#!/usr/bin/env bash

# RQ1: 硬件友好重排能否在不同模型与上下文长度上稳定提高压缩率？
#
# 主实验固定 po2_nearest，仅改变重排方法，以隔离重排贡献：
#   NONE / MEDIAN / BUCKET(k_sum, 4) / BUCKET(k_sum, 8) / GREEDY
#
# 精简验证固定 ctx=4096、continuous，仅比较：
#   NONE / BUCKET(k_sum, 4) / GREEDY
#
# 用法：
#   bash scripts/cr/run_rq1.sh full
#   bash scripts/cr/run_rq1.sh sanity
#   bash scripts/cr/run_rq1.sh all
#
# 可覆盖参数：
#   COLLECT_ROUND=5 PYTHON_BIN=.venv/bin/python bash scripts/cr/run_rq1.sh full
#   MODELS="Qwen/Qwen3-4B NousResearch/Meta-Llama-3-8B" \
#     CTX_LENS="2048 4096" bash scripts/cr/run_rq1.sh full

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-sanity}"
case "$MODE" in
    full|sanity|all) ;;
    *)
        echo "用法: bash scripts/cr/run_rq1.sh {full|sanity|all}" >&2
        exit 2
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "找不到可执行 Python: $PYTHON_BIN" >&2
    echo "请设置 PYTHON_BIN，例如 PYTHON_BIN=/path/to/.venv/bin/python" >&2
    exit 2
fi

read -r -a MODEL_LIST <<< "${MODELS:-Qwen/Qwen3-4B Qwen/Qwen3-8B NousResearch/Meta-Llama-3-8B mistralai/Ministral-8B-Instruct-2410}"
read -r -a CTX_LIST <<< "${CTX_LENS:-2048 4096 8192}"

COLLECT_ROUND="${COLLECT_ROUND:-5}"
SANITY_COLLECT_ROUND="${SANITY_COLLECT_ROUND:-3}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
BUFFER_SIZE="${BUFFER_SIZE:-192}"
PACK_SIZE="${PACK_SIZE:-16}"
K_SCALE="${K_SCALE:-0.1}"
V_SCALE="${V_SCALE:-0.1}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/rq1}"
mkdir -p "$LOG_ROOT/full" "$LOG_ROOT/sanity"

failures=0

run_one() {
    local phase="$1"
    local model="$2"
    local ctx_len="$3"
    local scale_method="$4"
    local repack_method="$5"
    local bucket_count="$6"
    local rounds="$7"
    local model_tag="${model//\//-}"
    local log_file="$LOG_ROOT/$phase/${model_tag}_ctx-${ctx_len}_scale-${scale_method}_repack-${repack_method}"

    if [[ "$repack_method" == "BUCKET" ]]; then
        log_file+="_buckets-${bucket_count}_score-k_sum"
    fi
    log_file+="_rounds-${rounds}.log"

    echo
    echo "运行: phase=$phase model=$model ctx=$ctx_len scale=$scale_method repack=$repack_method buckets=$bucket_count rounds=$rounds"

    "$PYTHON_BIN" scripts/cr/cr_eval.py \
        --model_name "$model" \
        --ctx_len "$ctx_len" \
        --collect_round "$rounds" \
        --quant_method PackKV \
        --repack_method "$repack_method" \
        --block_size "$BLOCK_SIZE" \
        --buffer_size "$BUFFER_SIZE" \
        --pack_size "$PACK_SIZE" \
        --k_scale "$K_SCALE" \
        --v_scale "$V_SCALE" \
        --scale_method "$scale_method" \
        --no-high_precision_zero_point \
        --bucket_count "$bucket_count" \
        --bucket_score_method k_sum \
        2>&1 | tee "$log_file"

    local status=${PIPESTATUS[0]}
    if (( status != 0 )); then
        echo "失败: $model ctx=$ctx_len scale=$scale_method repack=$repack_method (exit=$status)" >&2
        failures=$((failures + 1))
    fi
}

run_full() {
    local model ctx_len spec repack bucket_count
    local repack_specs=(
        "NONE:4"
        "MEDIAN:4"
        "BUCKET:4"
        "BUCKET:8"
        "GREEDY:4"
    )

    for model in "${MODEL_LIST[@]}"; do
        for ctx_len in "${CTX_LIST[@]}"; do
            for spec in "${repack_specs[@]}"; do
                IFS=: read -r repack bucket_count <<< "$spec"
                run_one full "$model" "$ctx_len" po2_nearest "$repack" "$bucket_count" "$COLLECT_ROUND"
            done
        done
    done
}

run_sanity() {
    local model spec repack bucket_count
    local repack_specs=(
        "NONE:4"
        "BUCKET:4"
        "GREEDY:4"
    )

    for model in "${MODEL_LIST[@]}"; do
        for spec in "${repack_specs[@]}"; do
            IFS=: read -r repack bucket_count <<< "$spec"
            run_one sanity "$model" 4096 continuous "$repack" "$bucket_count" "$SANITY_COLLECT_ROUND"
        done
    done
}

if [[ "$MODE" == "full" || "$MODE" == "all" ]]; then
    run_full
fi
if [[ "$MODE" == "sanity" || "$MODE" == "all" ]]; then
    run_sanity
fi

echo
echo "RQ1 运行结束。失败配置数: $failures"
echo "逐层与汇总 CSV: $PROJECT_ROOT/csv_results"
echo "日志目录: $LOG_ROOT"

if (( failures != 0 )); then
    exit 1
fi
