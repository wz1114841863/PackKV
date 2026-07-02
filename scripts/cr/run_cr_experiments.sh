#!/bin/bash

# ==============================================================================
# PackKV 压缩率 (CR) 自动化网格搜索测试脚本
# ==============================================================================

# 定义需要遍历的参数数组
MODELS=(
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "NousResearch/Meta-Llama-3-8B"
)

QUANT_METHODS=("PackKV" "KIVI")
REPACK_METHODS=("GREEDY" "MEDIAN" "NONE")

# 设定一系列容忍度 (Scale) 来观察压缩率的曲线变化
SCALES=(0.01 0.02 0.05 0.07 0.1 0.2 0.3)

# 固定参数
BLOCK_SIZE=64
PACK_SIZE=16
COLLECT_ROUND=1

echo "开始执行 PackKV 压缩率自动化测试管线..."
echo "所有宏观结果将自动追加至: ./csv_results/Global_Macro_Summary.csv"
echo "=============================================================================="

# 4层嵌套循环遍历所有组合
for MODEL in "${MODELS[@]}"; do
    for QUANT in "${QUANT_METHODS[@]}"; do
        for REPACK in "${REPACK_METHODS[@]}"; do

            # [核心优化]:KIVI 算法没有重排机制,跳过无意义的组合,节省时间
            if [ "$QUANT" == "KIVI" ] && [ "$REPACK" != "NONE" ]; then
                continue
            fi

            for SCALE in "${SCALES[@]}"; do

                echo ""
                echo "------------------------------------------------------------------"
                echo "   当前运行配置:"
                echo "   模型: $MODEL"
                echo "   方法: $QUANT | 重排: $REPACK"
                echo "   K/V Scale: $SCALE | Block/Pack: $BLOCK_SIZE / $PACK_SIZE"
                echo "------------------------------------------------------------------"

                # 调用我们修改好的 Python 脚本
                # 请根据你实际的文件路径调整下面的 cr_eval_cli.py 的路径
                python ./scripts/cr_eval_cli.py \
                    -m "$MODEL" \
                    --quant_method "$QUANT" \
                    --repack_method "$REPACK" \
                    --k_scale "$SCALE" \
                    --v_scale "$SCALE" \
                    --block_size "$BLOCK_SIZE" \
                    --pack_size "$PACK_SIZE" \
                    --collect_round "$COLLECT_ROUND"

                # 如果某次运行崩溃了(比如 OOM),脚本不会退出,而是继续跑下一组
                if [ $? -ne 0 ]; then
                    echo "警告: 该组参数运行失败,已跳过."
                fi

            done
        done
    done
done

echo ""
echo "所有自动化测试执行完毕!"
echo "请打开 ./csv_results/Global_Macro_Summary.csv 查看你的全景汇总表."
