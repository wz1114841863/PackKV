#!/usr/bin/env python3
import json
import argparse
import os


def extract_official_mmlu_results(json_path):
    if not os.path.exists(json_path):
        print(f"[Error] 文件不存在: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})

    # lm_eval 官方默认的 group 名称
    target_groups = {
        "mmlu_stem": "STEM",
        "mmlu_humanities": "Humanities",
        "mmlu_social_sciences": "Social Sciences",
        "mmlu_other": "Other",
        "mmlu": "🌟 Overall (总平均分)",
    }

    print(f"\n{'='*40}")
    print(f"📊 MMLU 官方聚合成绩提取 (Micro-average)")
    print(f"📄 来源文件: {json_path}")
    print(f"{'='*40}")

    for group_key, display_name in target_groups.items():
        if group_key in results:
            metrics = results[group_key]
            # 兼容不同版本的 lm_eval
            acc = metrics.get("acc,none") or metrics.get("acc")
            if acc is not None:
                print(f"🔹 {display_name:<20}: {acc * 100:>5.2f}%")
        else:
            print(f"🔹 {display_name:<20}: N/A (未在 JSON 中找到该 Group)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    args = parser.parse_args()
    extract_official_mmlu_results(args.input)
