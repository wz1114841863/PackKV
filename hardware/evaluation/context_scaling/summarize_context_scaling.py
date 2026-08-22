#!/usr/bin/env python3
"""Validate and join BRISK-KV S2 cycle and architectural-SRAM scaling data."""

import argparse
import csv
import json
from pathlib import Path


ARCHITECTURES = ("full_v", "jit_v_shared_writer_cg")
BASE_COLUMNS = (
    "architecture", "tokens", "feature_dim", "backpressure",
    "write_cycles", "attention_cycles", "k_active_cycles",
    "qk_active_cycles", "softmax_active_cycles", "av_active_cycles",
    "v_active_cycles", "output_count", "output_checksum",
)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle-summary", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", required=True, help="Comma-separated token counts")
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--backpressure", default="periodic")
    parser.add_argument("--clock-period-ns", type=float, default=2.0)
    parser.add_argument("--frozen-full-cycle", type=Path, required=True)
    parser.add_argument("--frozen-jit-cycle", type=Path, required=True)
    args = parser.parse_args()

    tokens = [int(value.strip()) for value in args.tokens.split(",") if value.strip()]
    if not tokens or len(set(tokens)) != len(tokens):
        raise SystemExit("--tokens must be a non-empty unique list")
    if any(value < 64 or value % 64 for value in tokens):
        raise SystemExit("every S2 token count must be a multiple of 64 and at least 64")

    cycle_rows = read_csv(args.cycle_summary)
    selected = [
        row for row in cycle_rows
        if row.get("architecture") in ARCHITECTURES
        and row.get("backpressure") == args.backpressure
        and int(row.get("feature_dim", 0)) == args.feature_dim
        and int(row.get("tokens", 0)) in tokens
    ]
    index = {(row["architecture"], int(row["tokens"])): row for row in selected}
    expected = {(architecture, token) for architecture in ARCHITECTURES for token in tokens}
    if set(index) != expected:
        raise SystemExit(
            "incomplete cycle matrix: "
            f"missing={sorted(expected - set(index))}, extra={sorted(set(index) - expected)}"
        )

    output_rows = []
    for token in sorted(tokens):
        bits_by_architecture = {}
        manifests = {}
        for architecture in ARCHITECTURES:
            manifest_path = args.rtl_root / architecture / f"t{token}" / "dc_logic" / "manifest.json"
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("architecture") != architecture:
                raise SystemExit(f"architecture mismatch: {manifest_path}")
            if manifest.get("maximum_tokens") != token:
                raise SystemExit(f"token geometry mismatch: {manifest_path}")
            if manifest.get("maximum_feature_dim") != args.feature_dim:
                raise SystemExit(f"feature geometry mismatch: {manifest_path}")
            bits_by_architecture[architecture] = int(manifest["total_inferred_memory_bits"])
            manifests[architecture] = manifest_path

        full_bits = bits_by_architecture["full_v"]
        for architecture in ARCHITECTURES:
            row = index[(architecture, token)]
            result = {column: row.get(column, "") for column in BASE_COLUMNS}
            attention_cycles = int(row["attention_cycles"])
            result.update(
                {
                    "architectural_sram_bits": bits_by_architecture[architecture],
                    "sram_saving_vs_full_pct": f"{(1.0 - bits_by_architecture[architecture] / full_bits) * 100.0:.6f}",
                    "attention_latency_ns": f"{attention_cycles * args.clock_period_ns:.6f}",
                    "effective_context_tokens_per_s": f"{token * 1e9 / (attention_cycles * args.clock_period_ns):.6f}",
                    "manifest": str(manifests[architecture]),
                }
            )
            output_rows.append(result)

    # The retained 1024x128 periodic point must remain cycle-identical.
    regressions = []
    frozen_by_architecture = {
        "full_v": read_csv(args.frozen_full_cycle),
        "jit_v_shared_writer_cg": read_csv(args.frozen_jit_cycle),
    }
    for architecture, rows in frozen_by_architecture.items():
        frozen = [
            row for row in rows
            if row.get("architecture") == architecture
            and row.get("tokens") == "1024"
            and row.get("feature_dim") == str(args.feature_dim)
            and row.get("backpressure") == args.backpressure
        ]
        if len(frozen) != 1:
            raise SystemExit(f"expected one frozen 1024 row for {architecture}")
        current = index.get((architecture, 1024))
        if current is None:
            continue
        checks = ("write_cycles", "attention_cycles", "output_count", "output_checksum")
        mismatched = [field for field in checks if current.get(field) != frozen[0].get(field)]
        regressions.append({"architecture": architecture, "status": "pass" if not mismatched else "fail", "mismatched": mismatched})
        if mismatched:
            raise SystemExit(f"1024 regression for {architecture}: {mismatched}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "context_scaling_summary.csv"
    fields = list(output_rows[0])
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    report_path = args.output_dir / "context_scaling_check.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump({"status": "pass", "tokens": tokens, "regressions": regressions}, handle, indent=2)
        handle.write("\n")
    print(f"S2 context scaling PASS: {summary_path}")


if __name__ == "__main__":
    main()
