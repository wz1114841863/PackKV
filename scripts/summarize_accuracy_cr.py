#!/usr/bin/env python3
"""汇总固定 PackKV 候选在三个模型上的精度与压缩率."""

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path


OUTPUT_FIELDS = (
    "Model",
    "Repack_Method",
    "Scale_Method",
    "K_Scale",
    "V_Scale",
    "Strict_Match_Pct",
    "Flexible_Extract_Pct",
    "FP_Strict_Match_Pct",
    "FP_Flexible_Extract_Pct",
    "Strict_Delta_vs_FP_pp",
    "Flexible_Delta_vs_FP_pp",
    "K_Global_CR",
    "V_Global_CR",
    "Overall_Global_CR",
    "Compressed_Bytes",
    "Bucket_CR_Gain_vs_NONE_Pct",
    "Bucket_Bytes_Change_vs_NONE_Pct",
    "Accuracy_Source",
    "CR_Source",
)

DEFAULT_MODELS = (
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "NousResearch/Meta-Llama-3.1-8B",
)


def read_rows(paths):
    rows = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                row["__source"] = str(path)
                rows.append(row)
    return rows


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def close(value, expected):
    parsed = as_float(value)
    return parsed is not None and math.isclose(
        parsed, expected, rel_tol=0.0, abs_tol=1e-9
    )


def false_value(value):
    return str(value).strip().lower() in {"false", "0", "no"}


def full_limit(value):
    return str(value).strip().lower() in {"", "none", "null"}


def latest(rows, timestamp_field, description):
    if not rows:
        raise ValueError(f"缺少结果: {description}")

    def timestamp(row):
        raw = str(row.get(timestamp_field, "")).strip()
        if not raw:
            return datetime.min
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(
                tzinfo=None
            )
        except ValueError:
            return datetime.min

    selected = max(rows, key=timestamp)
    if len(rows) > 1:
        print(f"提示: {description} 有 {len(rows)} 条，采用时间最新的一条")
    return selected


def fixed_accuracy_row(rows, model, repack):
    matches = [
        row
        for row in rows
        if row.get("Model") == model
        and row.get("Task") == "gsm8k"
        and full_limit(row.get("Limit"))
        and str(row.get("Enable_Quant")).strip().lower() in {"true", "1", "yes"}
        and row.get("Quant_Method") == "PackKV"
        and row.get("Scale_Method") == "po2_nearest"
        and row.get("Repack_Method") == repack
        and close(row.get("K_Scale"), 0.03)
        and close(row.get("V_Scale"), 0.10)
        and close(row.get("Block_Size"), 64)
        and close(row.get("Buffer_Size"), 192)
        and close(row.get("Pack_Size"), 16)
        and false_value(row.get("High_Precision_Zero_Point"))
        and (
            repack != "BUCKET"
            or (
                close(row.get("Bucket_Count"), 4)
                and row.get("Bucket_Score_Method") == "k_sum"
            )
        )
    ]
    return latest(matches, "Created_At", f"{model} {repack} accuracy")


def fp_accuracy_row(rows, model):
    matches = [
        row
        for row in rows
        if row.get("Model") == model
        and row.get("Task") == "gsm8k"
        and full_limit(row.get("Limit"))
        and false_value(row.get("Enable_Quant"))
    ]
    return latest(matches, "Created_At", f"{model} FP accuracy")


def fixed_cr_row(rows, model, repack, ctx_len):
    matches = [
        row
        for row in rows
        if row.get("Model") == model
        and close(row.get("Ctx_Len"), ctx_len)
        and row.get("Quant_Method") == "PackKV"
        and row.get("Scale_Method") == "po2_nearest"
        and row.get("Repack_Method") == repack
        and close(row.get("K_Scale"), 0.03)
        and close(row.get("V_Scale"), 0.10)
        and close(row.get("Block_Size"), 64)
        and close(row.get("Buffer_Size"), 192)
        and close(row.get("Pack_Size"), 16)
        and false_value(row.get("High_Precision_Zero_Point"))
        and (
            repack != "BUCKET"
            or (
                close(row.get("Bucket_Count"), 4)
                and row.get("Bucket_Score_Method") == "k_sum"
            )
        )
    ]
    return latest(matches, "Generated_At", f"{model} {repack} CR")


def percent(value):
    parsed = as_float(value)
    return "" if parsed is None else f"{parsed * 100:.6f}"


def number(value, digits=6):
    parsed = as_float(value)
    return "" if parsed is None else f"{parsed:.{digits}f}"


def build_rows(accuracy_rows, cr_rows, models, ctx_len):
    output = []
    for model in models:
        fp = fp_accuracy_row(accuracy_rows, model)
        fp_strict = as_float(fp.get("Strict_Match"))
        fp_flexible = as_float(fp.get("Flexible_Extract"))
        cr_by_repack = {
            repack: fixed_cr_row(cr_rows, model, repack, ctx_len)
            for repack in ("NONE", "BUCKET")
        }
        none_cr = as_float(cr_by_repack["NONE"].get("Overall_Global_CR"))
        none_bytes = sum(
            as_float(cr_by_repack["NONE"].get(field)) or 0.0
            for field in ("K_Compressed_Bytes", "V_Compressed_Bytes")
        )
        for repack in ("NONE", "BUCKET"):
            accuracy = fixed_accuracy_row(accuracy_rows, model, repack)
            cr = cr_by_repack[repack]
            strict = as_float(accuracy.get("Strict_Match"))
            flexible = as_float(accuracy.get("Flexible_Extract"))
            overall_cr = as_float(cr.get("Overall_Global_CR"))
            compressed_bytes = sum(
                as_float(cr.get(field)) or 0.0
                for field in ("K_Compressed_Bytes", "V_Compressed_Bytes")
            )
            cr_gain = (
                (overall_cr / none_cr - 1.0) * 100
                if repack == "BUCKET" and overall_cr is not None and none_cr
                else 0.0
            )
            bytes_change = (
                (compressed_bytes / none_bytes - 1.0) * 100
                if repack == "BUCKET" and none_bytes
                else 0.0
            )
            output.append(
                {
                    "Model": model,
                    "Repack_Method": repack,
                    "Scale_Method": "po2_nearest",
                    "K_Scale": "0.03",
                    "V_Scale": "0.10",
                    "Strict_Match_Pct": percent(strict),
                    "Flexible_Extract_Pct": percent(flexible),
                    "FP_Strict_Match_Pct": percent(fp_strict),
                    "FP_Flexible_Extract_Pct": percent(fp_flexible),
                    "Strict_Delta_vs_FP_pp": number(
                        (strict - fp_strict) * 100
                        if strict is not None and fp_strict is not None
                        else None
                    ),
                    "Flexible_Delta_vs_FP_pp": number(
                        (flexible - fp_flexible) * 100
                        if flexible is not None and fp_flexible is not None
                        else None
                    ),
                    "K_Global_CR": number(cr.get("K_Global_CR")),
                    "V_Global_CR": number(cr.get("V_Global_CR")),
                    "Overall_Global_CR": number(overall_cr),
                    "Compressed_Bytes": str(int(compressed_bytes)),
                    "Bucket_CR_Gain_vs_NONE_Pct": number(cr_gain),
                    "Bucket_Bytes_Change_vs_NONE_Pct": number(bytes_change),
                    "Accuracy_Source": accuracy["__source"],
                    "CR_Source": cr["__source"],
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="联合汇总固定 po2_nearest 0.03/0.10 的三模型精度与 CR"
    )
    parser.add_argument("--accuracy", type=Path, nargs="+", required=True)
    parser.add_argument("--cr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--ctx_len", type=int, default=4096)
    args = parser.parse_args()

    rows = build_rows(
        read_rows(args.accuracy),
        read_rows([args.cr]),
        args.models,
        args.ctx_len,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已生成 {len(rows)} 条三模型联合结果: {args.output}")


if __name__ == "__main__":
    main()
