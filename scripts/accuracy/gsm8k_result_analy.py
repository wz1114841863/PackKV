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

    # lm_eval 框架在不断升级,我们需要兼容不同版本中精确匹配指标的命名
    score = (
        gsm8k_data.get("exact_match,strict-match")
        or gsm8k_data.get("exact_match,none")
        or gsm8k_data.get("exact_match")
    )

    if score is None:
        print("[Error] 找到了 GSM8K 数据,但没找到 exact_match 指标.")
        return

    acc_percentage = score * 100

    print(f"\n{'='*40}")
    print(f"🧮 GSM8K 推理能力分析")
    print(f"📄 来源文件: {json_path}")
    print(f"{'-'*40}")
    print(f"🌟 精确匹配率 (Exact Match): {acc_percentage:>6.2f}%")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析 lm_eval 的 GSM8K 结果")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="results.json 的路径"
    )
    args = parser.parse_args()

    analyze_gsm8k_results(args.input)
