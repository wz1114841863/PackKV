#!/bin/bash

# ==============================================================================
# 大模型基准测试自动评估脚本 (基于 lm_eval)
# 测试基准模型在不同任务上的表现,支持指定模型/任务/输出路径和 Batch Size
# 不对模型进行量化, 压缩等处理
# ==============================================================================

# 设置默认参数
export TRANSFORMERS_VERBOSITY=error
MODEL="Qwen/Qwen3-8B"
TASKS="mmlu,gsm8k"
OUTPUT_DIR="./eval_logs/default_run"
BATCH_SIZE="auto"
LIMIT_CMD="" # 默认跑全量测试

# 帮助文档函数
print_usage() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -m, --model       指定模型路径或名称 (默认: $MODEL)"
    echo "  -t, --tasks       指定测试任务,逗号分隔 (默认: $TASKS)"
    echo "  -o, --output      指定日志输出路径 (默认: $OUTPUT_DIR)"
    echo "  -b, --batch_size  指定 Batch Size (默认: $BATCH_SIZE)"
    echo "  -l, --limit       开启 Debug 模式,限制每个任务跑 N 道题 (例如: -l 10)"
    echo "  -h, --help        显示帮助信息"
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)      MODEL="$2"; shift 2 ;;
        -t|--tasks)      TASKS="$2"; shift 2 ;;
        -o|--output)     OUTPUT_DIR="$2"; shift 2 ;;
        -b|--batch_size) BATCH_SIZE="$2"; shift 2 ;;
        -l|--limit)      LIMIT_CMD="--limit $2"; shift 2 ;;
        -h|--help)       print_usage ;;
        *)               echo "未知参数: $1"; print_usage ;;
    esac
done

# 自动生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 提取任务名 (将逗号替换为下划线,例如 "mmlu,gsm8k" -> "mmlu_gsm8k")
SAFE_TASKS=$(echo $TASKS | tr ',' '_')

# 生成输出根目录
FINAL_OUTPUT_DIR="${OUTPUT_DIR}/${SAFE_TASKS}"

# 4打印当前实验配置
echo "========================================"
echo "开始执行模型评估任务"
echo "模型路径 : $MODEL"
echo "测试任务 : $TASKS"
echo "输出目录 : $FINAL_OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
if [ ! -z "$LIMIT_CMD" ]; then
    echo "调试模式 : 已开启 ($LIMIT_CMD)"
fi
echo "========================================"

# 设置环境变量 (强迫走本地离线缓存,防止网络波动卡死)
# export HF_HUB_OFFLINE=1

# 执行 lm_eval
# 这里写死了 dtype=bfloat16 和 trust_remote_code=True,保证基线纯洁性
lm_eval --model hf \
    --model_args pretrained="${MODEL}",dtype=bfloat16,trust_remote_code=True \
    --tasks "${TASKS}" \
    --device cuda:0 \
    --batch_size "${BATCH_SIZE}" \
    --log_samples \
    --output_path "${FINAL_OUTPUT_DIR}"


# 结束提示
if [ $? -eq 0 ]; then
    echo "Finished! 评估完成!结果已保存至: $FINAL_OUTPUT_DIR"
else
    echo "Error! 评估异常中断,请检查报错信息."
fi
