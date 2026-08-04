#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


CSV_FIELDS = (
    "Config_Status",
    "Schema_Version",
    "Created_At",
    "Model",
    "Task",
    "Limit",
    "Batch_Size",
    "Enable_Quant",
    "Quant_Method",
    "K_Quant_Mode",
    "V_Quant_Mode",
    "Scale_Method",
    "K_Scale",
    "V_Scale",
    "Repack_Method",
    "High_Precision_Zero_Point",
    "Block_Size",
    "Buffer_Size",
    "Pack_Size",
    "Bucket_Count",
    "Bucket_Score_Method",
    "K_Error_Budget",
    "V_Error_Budget",
    "Effective_Model_Dtype",
    "Python_Version",
    "Torch_Version",
    "Transformers_Version",
    "LM_Eval_Version",
    "Strict_Match",
    "Strict_Match_Stderr",
    "Flexible_Extract",
    "Flexible_Extract_Stderr",
    "Accuracy",
    "Accuracy_Stderr",
    "Results_JSON",
    "PackKV_Config_JSON",
)


def numeric_metric(metrics, prefix, filter_name, stderr=False):
    for key, value in metrics.items():
        metric_name = key.split(",", 1)[0]
        is_stderr = metric_name.endswith("_stderr")
        base_name = metric_name.removesuffix("_stderr")
        if base_name != prefix or filter_name not in key:
            continue
        if is_stderr == stderr and isinstance(value, (int, float)):
            return value
    return ""


def fallback_model(lm_eval_config):
    model = lm_eval_config.get("model")
    if model:
        return str(model)
    model_args = lm_eval_config.get("model_args")
    if not model_args:
        return ""
    return str(model_args).replace("pretrained=", "").split(",")[0]


def load_sidecar(result_path: Path):
    config_path = result_path.with_name("packkv_config.json")
    if not config_path.exists():
        return None, config_path
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"PackKV config must be a JSON object: {config_path}")
    return config, config_path


def build_row(result_path: Path, task, metrics, data, sidecar, sidecar_path):
    lm_eval_config = data.get("config", {})
    config = sidecar or {}
    versions = config.get("software_versions", {})
    model = config.get("model") or fallback_model(lm_eval_config)
    limit = config.get("limit", lm_eval_config.get("limit", ""))
    batch_size = config.get(
        "batch_size", lm_eval_config.get("batch_size", "")
    )
    return {
        "Config_Status": "recorded" if sidecar is not None else "missing",
        "Schema_Version": config.get("schema_version", ""),
        "Created_At": config.get("created_at", ""),
        "Model": model,
        "Task": config.get("task") or task,
        "Limit": "" if limit is None else limit,
        "Batch_Size": batch_size,
        "Enable_Quant": config.get("enable_quant", ""),
        "Quant_Method": config.get("quant_method", ""),
        "K_Quant_Mode": config.get("k_quant_mode", ""),
        "V_Quant_Mode": config.get("v_quant_mode", ""),
        "Scale_Method": config.get("scale_method", ""),
        "K_Scale": config.get("k_scale", ""),
        "V_Scale": config.get("v_scale", ""),
        "Repack_Method": config.get("repack_method", ""),
        "High_Precision_Zero_Point": config.get(
            "high_precision_zero_point", ""
        ),
        "Block_Size": config.get("block_size", ""),
        "Buffer_Size": config.get("buffer_size", ""),
        "Pack_Size": config.get("pack_size", ""),
        "Bucket_Count": config.get("bucket_count", ""),
        "Bucket_Score_Method": config.get("bucket_score_method", ""),
        "K_Error_Budget": config.get("k_error_budget", ""),
        "V_Error_Budget": config.get("v_error_budget", ""),
        "Effective_Model_Dtype": config.get("effective_model_dtype", ""),
        "Python_Version": versions.get("python", ""),
        "Torch_Version": versions.get("torch", ""),
        "Transformers_Version": versions.get("transformers", ""),
        "LM_Eval_Version": versions.get("lm_eval", ""),
        "Strict_Match": numeric_metric(
            metrics, "exact_match", "strict-match"
        ),
        "Strict_Match_Stderr": numeric_metric(
            metrics, "exact_match", "strict-match", stderr=True
        ),
        "Flexible_Extract": numeric_metric(
            metrics, "exact_match", "flexible-extract"
        ),
        "Flexible_Extract_Stderr": numeric_metric(
            metrics, "exact_match", "flexible-extract", stderr=True
        ),
        "Accuracy": numeric_metric(metrics, "acc", "none"),
        "Accuracy_Stderr": numeric_metric(
            metrics, "acc", "none", stderr=True
        ),
        "Results_JSON": str(result_path),
        "PackKV_Config_JSON": str(sidecar_path) if sidecar is not None else "",
    }


def summarize(root: Path, output: Path) -> None:
    rows = []
    for result_path in sorted(root.rglob("results.json")):
        with result_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sidecar, sidecar_path = load_sidecar(result_path)
        for task, metrics in data.get("results", {}).items():
            rows.append(
                build_row(
                    result_path,
                    task,
                    metrics,
                    data,
                    sidecar,
                    sidecar_path,
                )
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    missing = sum(row["Config_Status"] == "missing" for row in rows)
    print(f"汇总 {len(rows)} 条结果至: {output}")
    if missing:
        print(f"警告: {missing} 条旧结果缺少 packkv_config.json,配置列留空")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)
