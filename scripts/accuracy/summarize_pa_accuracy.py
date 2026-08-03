#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


KNOWN_LABELS = (
    "fp",
    "continuous",
    "nearest",
    "pa_005_005",
    "pa_010_010",
    "pa_025_010",
    "pa_025_025",
)


def find_label(path: Path) -> str:
    for part in path.parts:
        for label in KNOWN_LABELS:
            if part.endswith(f"_{label}"):
                return label
    return "unknown"


def numeric_metric(metrics, prefix, filter_name):
    for key, value in metrics.items():
        if (
            key.startswith(prefix)
            and filter_name in key
            and "_stderr" not in key
            and isinstance(value, (int, float))
        ):
            return value
    return ""


def summarize(root: Path, output: Path) -> None:
    rows = []
    for result_path in sorted(root.rglob("results.json")):
        with result_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        config = data.get("config", {})
        model_args = str(config.get("model_args", ""))
        model = model_args.replace("pretrained=", "").split(",")[0]
        for task, metrics in data.get("results", {}).items():
            strict = numeric_metric(metrics, "exact_match", "strict-match")
            flexible = numeric_metric(metrics, "exact_match", "flexible-extract")
            if strict == "" and flexible == "":
                strict = numeric_metric(metrics, "acc", "none")
            rows.append(
                {
                    "Model": model,
                    "Task": task,
                    "Config": find_label(result_path.relative_to(root)),
                    "Strict_Match": strict,
                    "Flexible_Extract": flexible,
                    "Limit": config.get("limit", ""),
                    "Results_JSON": str(result_path),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "Model",
                "Task",
                "Config",
                "Strict_Match",
                "Flexible_Extract",
                "Limit",
                "Results_JSON",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"汇总 {len(rows)} 条结果至: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)
