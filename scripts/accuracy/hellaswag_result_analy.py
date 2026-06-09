#!/usr/bin/env python3
import json
import argparse
import os


def analyze_hellaswag_results(json_path):
    if not os.path.exists(json_path):
        print(f"[Error] 文件不存在: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})

    if "hellaswag" not in results:
        print(f"[Warning] 日志中未找到 HellaSwag 的成绩.")
        return

    hw_data = results["hellaswag"]

    # 提取普通的 acc (仅作参考)
    acc = hw_data.get("acc,none") or hw_data.get("acc")

    # 🌟 提取核心指标 acc_norm
    acc_norm = hw_data.get("acc_norm,none") or hw_data.get("acc_norm")

    if acc_norm is None:
        print("[Error] 找到了 HellaSwag 数据,但没找到 acc_norm 指标.")
        return

    acc_percentage = (acc or 0) * 100
    acc_norm_percentage = acc_norm * 100

    print(f"\n{'='*45}")
    print(f"🛋️  HellaSwag 物理常识与防崩坏测试")
    print(f"📄 来源文件: {json_path}")
    print(f"{'-'*45}")
    print(f"基础准确率 (Acc, 仅参考)   : {acc_percentage:>6.2f}%")
    print(
        f"🌟 归一化准确率 (Acc_Norm)  : {acc_norm_percentage:>6.2f}%  <-- 记录这个数据"
    )
    print(f"{'='*45}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析 lm_eval 的 HellaSwag 结果")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="results.json 的路径"
    )
    args = parser.parse_args()

    analyze_hellaswag_results(args.input)
