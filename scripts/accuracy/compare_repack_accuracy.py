#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


CSV_FIELDS = (
    "Model",
    "Task",
    "Limit",
    "Scale_Method",
    "K_Scale",
    "V_Scale",
    "Bucket_Count",
    "Bucket_Score_Method",
    "None_Strict_Match",
    "Bucket_Strict_Match",
    "Strict_Delta",
    "None_Flexible_Extract",
    "Bucket_Flexible_Extract",
    "Flexible_Delta",
    "Compared_Sample_Keys",
    "Raw_Response_Differences",
    "Filtered_Response_Differences",
    "Exact_Match_Label_Differences",
    "Missing_Sample_Keys",
    "None_Results_JSON",
    "Bucket_Results_JSON",
)


def numeric_metric(metrics, filter_name):
    for key, value in metrics.items():
        metric_name = key.split(",", 1)[0]
        if (
            metric_name == "exact_match"
            and filter_name in key
            and isinstance(value, (int, float))
        ):
            return float(value)
    return None


def load_result(run_dir: Path, task: str):
    result_path = run_dir / "results.json"
    with result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    return result_path, result.get("results", {}).get(task, {})


def load_samples(run_dir: Path, task: str):
    sample_path = run_dir / f"samples_{task}.json"
    if not sample_path.exists():
        return {}
    with sample_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get(task, []) if isinstance(data, dict) else data
    return {
        (str(entry.get("doc_id")), str(entry.get("filter", ""))): entry
        for entry in entries
    }


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def pair_key(config):
    fields = (
        "model",
        "task",
        "limit",
        "batch_size",
        "enable_quant",
        "quant_method",
        "scale_method",
        "k_scale",
        "v_scale",
        "high_precision_zero_point",
        "block_size",
        "buffer_size",
        "pack_size",
        "bucket_count",
        "bucket_score_method",
        "k_error_budget",
        "v_error_budget",
    )
    return tuple(canonical(config.get(field)) for field in fields)


def compare_pair(none_run, bucket_run):
    none_dir, none_config = none_run
    bucket_dir, bucket_config = bucket_run
    task = str(none_config["task"])
    none_result_path, none_metrics = load_result(none_dir, task)
    bucket_result_path, bucket_metrics = load_result(bucket_dir, task)
    none_samples = load_samples(none_dir, task)
    bucket_samples = load_samples(bucket_dir, task)
    common_keys = sorted(set(none_samples) & set(bucket_samples))
    missing_keys = set(none_samples) ^ set(bucket_samples)

    raw_differences = 0
    filtered_differences = 0
    label_differences = 0
    for key in common_keys:
        none_sample = none_samples[key]
        bucket_sample = bucket_samples[key]
        raw_differences += canonical(none_sample.get("resps")) != canonical(
            bucket_sample.get("resps")
        )
        filtered_differences += canonical(
            none_sample.get("filtered_resps")
        ) != canonical(bucket_sample.get("filtered_resps"))
        label_differences += canonical(none_sample.get("exact_match")) != canonical(
            bucket_sample.get("exact_match")
        )

    none_strict = numeric_metric(none_metrics, "strict-match")
    bucket_strict = numeric_metric(bucket_metrics, "strict-match")
    none_flexible = numeric_metric(none_metrics, "flexible-extract")
    bucket_flexible = numeric_metric(bucket_metrics, "flexible-extract")

    def delta(left, right):
        return "" if left is None or right is None else right - left

    return {
        "Model": none_config.get("model", ""),
        "Task": task,
        "Limit": "" if none_config.get("limit") is None else none_config["limit"],
        "Scale_Method": none_config.get("scale_method", ""),
        "K_Scale": none_config.get("k_scale", ""),
        "V_Scale": none_config.get("v_scale", ""),
        "Bucket_Count": bucket_config.get("bucket_count", ""),
        "Bucket_Score_Method": bucket_config.get("bucket_score_method", ""),
        "None_Strict_Match": none_strict if none_strict is not None else "",
        "Bucket_Strict_Match": bucket_strict if bucket_strict is not None else "",
        "Strict_Delta": delta(none_strict, bucket_strict),
        "None_Flexible_Extract": none_flexible if none_flexible is not None else "",
        "Bucket_Flexible_Extract": bucket_flexible if bucket_flexible is not None else "",
        "Flexible_Delta": delta(none_flexible, bucket_flexible),
        "Compared_Sample_Keys": len(common_keys),
        "Raw_Response_Differences": raw_differences,
        "Filtered_Response_Differences": filtered_differences,
        "Exact_Match_Label_Differences": label_differences,
        "Missing_Sample_Keys": len(missing_keys),
        "None_Results_JSON": str(none_result_path),
        "Bucket_Results_JSON": str(bucket_result_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="配对比较 NONE 与 BUCKET 的准确率和逐题输出"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped = defaultdict(lambda: defaultdict(list))
    for config_path in sorted(args.root.rglob("packkv_config.json")):
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        repack_method = config.get("repack_method")
        if repack_method not in {"NONE", "BUCKET"}:
            continue
        grouped[pair_key(config)][repack_method].append((config_path.parent, config))

    rows = []
    problems = []
    for key, methods in grouped.items():
        if len(methods["NONE"]) != 1 or len(methods["BUCKET"]) != 1:
            problems.append(
                f"unpaired/duplicate group: NONE={len(methods['NONE'])}, "
                f"BUCKET={len(methods['BUCKET'])}, key={key}"
            )
            continue
        rows.append(compare_pair(methods["NONE"][0], methods["BUCKET"][0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['Scale_Method']} K={row['K_Scale']} V={row['V_Scale']}: "
            f"strict delta={row['Strict_Delta']}, "
            f"flexible delta={row['Flexible_Delta']}, "
            f"raw/filtered/label differences="
            f"{row['Raw_Response_Differences']}/"
            f"{row['Filtered_Response_Differences']}/"
            f"{row['Exact_Match_Label_Differences']}, "
            f"missing={row['Missing_Sample_Keys']}"
        )
    print(f"配对汇总已保存: {args.output}")
    if problems:
        for problem in problems:
            print(f"警告: {problem}")
        raise SystemExit(2)
    if not rows:
        raise SystemExit("未找到完整的 NONE/BUCKET 配对")


if __name__ == "__main__":
    main()
