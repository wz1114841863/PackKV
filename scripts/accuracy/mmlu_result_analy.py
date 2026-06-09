#!/usr/bin/env python3
import json
import argparse
import os

# MMLU 57个子任务的官方四大类映射字典
MMLU_MAPPING = {
    "STEM": [
        "abstract_algebra",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "Social Sciences": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "Other": [
        "anatomy",
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    ],
}


def analyze_mmlu_results(json_path):
    if not os.path.exists(json_path):
        print(f"[Error] 文件不存在: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})

    # 统计容器
    category_scores = {cat: [] for cat in MMLU_MAPPING.keys()}
    global_scores = []

    # 遍历结果,进行归类
    for task_name, metrics in results.items():
        # lm_eval 的任务名可能是 "mmlu_astronomy" 或直接是 "astronomy"
        clean_name = task_name.replace("mmlu_", "")

        acc = metrics.get("acc,none") or metrics.get("acc")  # 兼容不同版本的 lm_eval

        if acc is None:
            continue

        acc_percentage = acc * 100  # 转换为百分比

        # 寻找该任务属于哪个大类
        for category, tasks in MMLU_MAPPING.items():
            if clean_name in tasks:
                category_scores[category].append(acc_percentage)
                global_scores.append(acc_percentage)
                break

    if not global_scores:
        print("[Warning] 在日志中没有找到任何 MMLU 相关的成绩.请检查日志内容.")
        return

    # 打印格式化的结果
    print(f"\n{'='*40}")
    print(f"📊 MMLU 成绩聚合分析报告")
    print(f"📄 来源文件: {json_path}")
    print(f"{'='*40}")

    for category, scores in category_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            print(
                f"🔹 {category:<16} : {avg_score:>5.2f}% (基于 {len(scores):>2} 个子学科)"
            )
        else:
            print(f"🔹 {category:<16} : N/A (未测试)")

    print(f"{'-'*40}")
    overall_avg = sum(global_scores) / len(global_scores)
    print(f"🌟 总平均分 (Overall): {overall_avg:>5.2f}%")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析 lm_eval 的 MMLU 结果")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="results.json 的路径"
    )
    args = parser.parse_args()

    analyze_mmlu_results(args.input)
