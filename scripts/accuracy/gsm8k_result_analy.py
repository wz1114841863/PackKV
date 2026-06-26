#!/usr/bin/env python3
import json
import argparse
import os


def analyze_gsm8k_results(json_path):
    if not os.path.exists(json_path):
        print(f"[Error] 文件不存在: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})

    # 检查日志中是否包含了 gsm8k 的测试结果
    if "gsm8k" not in results:
        print(f"[Warning] 日志中未找到 GSM8K 的成绩.")
        return

    gsm8k_data = results["gsm8k"]

    print(f"\n{'='*55}")
    print(f"GSM8K 推理能力分析 (全提取策略展示)")
    print(f"来源文件: {json_path}")
    print(f"{'-'*55}")

    found_metrics = False

    # 动态遍历并筛选出所有以 exact_match 开头,且不包含 _stderr 的键
    for key, value in gsm8k_data.items():
        if (
            key.startswith("exact_match")
            and "_stderr" not in key
            and isinstance(value, (int, float))
        ):
            # 解析 filter 名称
            if "," in key:
                filter_name = key.split(",")[1]
            else:
                filter_name = "default"

            acc_percentage = value * 100
            print(f"🌟 精确匹配率 (Filter: {filter_name:<16}): {acc_percentage:>6.2f}%")
            found_metrics = True

    if not found_metrics:
        print("[Error] 找到了 GSM8K 数据,但没找到任何 exact_match 指标.")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析 lm_eval 的 GSM8K 结果")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="results.json 的路径"
    )
    args = parser.parse_args()

    analyze_gsm8k_results(args.input)
