#!/usr/bin/env python3
"""Join and validate the fixed BRISK-KV S1 software-baseline matrix."""

import argparse
import csv
import json
import sys
from pathlib import Path


ACCURACY_VARIANTS = (
    ("fp", "False", "continuous", "NONE"),
    ("continuous", "True", "continuous", "NONE"),
    ("po2_none", "True", "po2_nearest", "NONE"),
    ("po2_bucket", "True", "po2_nearest", "BUCKET"),
)
CR_VARIANTS = ACCURACY_VARIANTS[1:]
COMMON_EXPECTED = {
    "Quant_Method": "PackKV",
    "K_Scale": "0.03",
    "V_Scale": "0.1",
    "High_Precision_Zero_Point": "False",
    "Block_Size": "64",
    "Buffer_Size": "192",
    "Pack_Size": "16",
    "Bucket_Count": "4",
    "Bucket_Score_Method": "k_sum",
}


def read_csv(path: Path):
    if not path.exists():
        return [], [f"missing file: {path}"]
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle)), []


def canonical(value):
    return "" if value is None else str(value).strip()


def numeric_equal(actual, expected):
    try:
        return abs(float(actual) - float(expected)) < 1e-12
    except (TypeError, ValueError):
        return canonical(actual) == canonical(expected)


def value_matches(row, field, expected):
    actual = row.get(field, "")
    if field in {"K_Scale", "V_Scale"}:
        return numeric_equal(actual, expected)
    return canonical(actual) == canonical(expected)


def format_errors(row, expected):
    return [
        f"{field}: expected={wanted!r} actual={row.get(field, '')!r}"
        for field, wanted in expected.items()
        if not value_matches(row, field, wanted)
    ]


def select_one(rows, expected):
    matched = [row for row in rows if not format_errors(row, expected)]
    if len(matched) == 1:
        return matched[0], None
    if not matched:
        return None, "missing"
    return None, f"duplicate ({len(matched)} rows)"


def main():
    parser = argparse.ArgumentParser(
        description="Join S1 accuracy and CR evidence and reject incomplete rows."
    )
    parser.add_argument("models", nargs="+", help="Exact Hugging Face model ids")
    parser.add_argument("--accuracy-summary", type=Path, required=True)
    parser.add_argument("--cr-summary", type=Path, required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    accuracy_rows, errors = read_csv(args.accuracy_summary)
    cr_rows, cr_errors = read_csv(args.cr_summary)
    errors.extend(cr_errors)
    output_rows = []
    checks = []

    for model in args.models:
        for label, enable_quant, scale_method, repack_method in ACCURACY_VARIANTS:
            accuracy_expected = {
                "Config_Status": "recorded",
                "Model": model,
                "Task": args.task,
                "Enable_Quant": enable_quant,
                "Scale_Method": scale_method,
                "Repack_Method": repack_method,
                "K_Scale": "0.03",
                "V_Scale": "0.1",
                "High_Precision_Zero_Point": "False",
                "Block_Size": "64",
                "Buffer_Size": "192",
                "Pack_Size": "16",
                "Bucket_Count": "4",
                "Bucket_Score_Method": "k_sum",
            }
            # The accuracy sidecar deliberately records no quant method for
            # --no_quant, even though the CLI still receives --quant_method.
            if label != "fp":
                accuracy_expected["Quant_Method"] = "PackKV"
            accuracy_row, accuracy_state = select_one(accuracy_rows, accuracy_expected)
            check = {
                "model": model,
                "variant": label,
                "accuracy": accuracy_state or "ok",
                "cr": "not_applicable" if label == "fp" else "pending",
            }
            if accuracy_state:
                errors.append(f"accuracy {accuracy_state}: {model} {label}")

            cr_row = None
            if label != "fp":
                cr_expected = dict(COMMON_EXPECTED)
                cr_expected.update(
                    {
                        "Suite_ID": args.suite_id,
                        "Model": model,
                        "Ctx_Len": str(args.ctx_len),
                        "Scale_Method": scale_method,
                        "Repack_Method": repack_method,
                    }
                )
                cr_row, cr_state = select_one(cr_rows, cr_expected)
                check["cr"] = cr_state or "ok"
                if cr_state:
                    errors.append(f"CR {cr_state}: {model} {label}")

            checks.append(check)
            output_rows.append(
                {
                    "Model": model,
                    "Variant": label,
                    "Accuracy_Status": check["accuracy"],
                    "CR_Status": check["cr"],
                    "Strict_Match": "" if not accuracy_row else accuracy_row.get("Strict_Match", ""),
                    "Flexible_Extract": "" if not accuracy_row else accuracy_row.get("Flexible_Extract", ""),
                    "Accuracy": "" if not accuracy_row else accuracy_row.get("Accuracy", ""),
                    "Overall_Global_CR": "1.0 (FP reference)" if label == "fp" else ("" if not cr_row else cr_row.get("Overall_Global_CR", "")),
                    "K_Global_CR": "1.0 (FP reference)" if label == "fp" else ("" if not cr_row else cr_row.get("K_Global_CR", "")),
                    "V_Global_CR": "1.0 (FP reference)" if label == "fp" else ("" if not cr_row else cr_row.get("V_Global_CR", "")),
                    "Accuracy_Results_JSON": "" if not accuracy_row else accuracy_row.get("Results_JSON", ""),
                    "Accuracy_Config_JSON": "" if not accuracy_row else accuracy_row.get("PackKV_Config_JSON", ""),
                    "CR_Detailed_Report": "" if not cr_row else cr_row.get("Detailed_Report_Path", ""),
                    "CR_Run_ID": "" if not cr_row else cr_row.get("Run_ID", ""),
                }
            )

    fields = list(output_rows[0]) if output_rows else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "status": "pass" if not errors else "fail",
        "suite_id": args.suite_id,
        "task": args.task,
        "ctx_len": args.ctx_len,
        "models": args.models,
        "expected_accuracy_rows": len(args.models) * len(ACCURACY_VARIANTS),
        "expected_cr_rows": len(args.models) * len(CR_VARIANTS),
        "checks": checks,
        "errors": errors,
        "joint_summary": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"joint summary: {args.output}")
    print(f"check report: {args.report}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
    print("S1 matrix check: PASS")


if __name__ == "__main__":
    main()
