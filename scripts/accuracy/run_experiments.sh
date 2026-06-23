#!/bin/bash

# 自动化消融实验网格搜索脚本
# 实验变量矩阵
MODELS=("Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "NousResearch/Meta-Llama-3-8B")
TASKS=("mmlu" "gsm8k" "hellaswag")
SCALES=("0.01" "0.1" "0.5" "0.9")
QUANT_METHODS=("PackKV" "KIVI")

# 2. 创建主输出目录
TIMESTAMP=$(date +%m%d_%H%M)
BASE_OUT_DIR="./grid_search_logs/exp_${TIMESTAMP}"
mkdir -p "$BASE_OUT_DIR"

echo "================================================="
echo "自动化实验开始运行 | 时间: $(date +'%Y-%m-%d %H:%M:%S')"
echo "所有结果将保存在: $BASE_OUT_DIR"
echo "================================================="

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for quant_method in "${QUANT_METHODS[@]}"; do
            for scale in "${SCALES[@]}"; do

                model_safe_name=$(echo "$model" | tr '/' '_')

                # 创建多级目录: exp_XXX/Qwen3-8B/mmlu/PackKV/scale_0.1
                EXP_DIR="${BASE_OUT_DIR}/${model_safe_name}/${task}/${quant_method}/scale_${scale}"
                mkdir -p "$EXP_DIR"

                echo "[运行中] 模型: $model_safe_name | 任务: $task | 算法: $quant_method | Scale: $scale"

                # ---------------------------------------------------------
                # 第一步:调用你的 accuracy_eval.py 跑模型
                # ---------------------------------------------------------
                python ./scripts/accuracy/pyaccuracy_eval.py \
                    -m "$model" \
                    -t "$task" \
                    -o "$EXP_DIR" \
                    -b "auto" \
                    --quant_method "$quant_method" \
                    --k_scale "$scale" \
                    --v_scale "$scale" \
                    > "${EXP_DIR}/run.log" 2>&1

                if [ $? -eq 0 ]; then
                    echo "[评测完成] 开始调用分析脚本提取结果..."

                    # 寻找生成的 results.json 文件 (LM-Eval通常会生成在输出目录的子文件夹中)
                    # 使用 find 命令来精准定位
                    JSON_FILE=$(find "$EXP_DIR" -name "results.json" | head -n 1)

                    if [ -n "$JSON_FILE" ]; then
                        # 第二步:调用对应的分析脚本 (如 mmlu_result_analy.py)
                        ANALY_SCRIPT="./scripts/accuracy/${task}_result_analy.py"

                        if [ -f "$ANALY_SCRIPT" ]; then
                            echo "  -> 执行分析: python $ANALY_SCRIPT -i $JSON_FILE"
                            python "$ANALY_SCRIPT" -i "$JSON_FILE" >> "${EXP_DIR}/run.log" 2>&1
                        else
                            echo "  -> ⚠️ 未找到分析脚本 $ANALY_SCRIPT,跳过自定义分析." >> "${EXP_DIR}/run.log"
                        fi
                    else
                        echo "  -> ❌ 未找到 results.json,模型可能中途崩溃!" >> "${EXP_DIR}/run.log"
                    fi
                else
                    echo "[❌ 评测出错] 请明早检查日志: ${EXP_DIR}/run.log"
                fi
                echo "-------------------------------------------------"

            done
        done
    done
done

echo "所有实验全部运行完毕!时间: $(date +'%Y-%m-%d %H:%M:%S')"
