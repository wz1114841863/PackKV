#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hardware_dir="$(cd "$script_dir/.." && pwd)"
chisel_dir="$hardware_dir/chisel"
output_dir="${OUTPUT_DIR:-$hardware_dir/evaluation/results/cycle_benchmark}"
feature_dim="${FEATURE_DIM:-8}"
token_counts="${TOKEN_COUNTS:-64,256,1024}"
mkdir -p "$output_dir"

for architecture in full_v jit_v_dual jit_v_shared; do
  (
    cd "$chisel_dir"
    sbt \
      -Dbriskkv.benchmarkArchitecture="$architecture" \
      -Dbriskkv.benchmarkFeatureDim="$feature_dim" \
      -Dbriskkv.benchmarkTokens="$token_counts" \
      -Dbriskkv.benchmarkOutput="$output_dir/$architecture.csv" \
      'testOnly briskkv.BriskKvCycleBenchmarkSpec'
  )
done

python3 - "$output_dir" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for architecture in ("full_v", "jit_v_dual", "jit_v_shared"):
    with (root / f"{architecture}.csv").open(newline="") as handle:
        rows.extend(csv.DictReader(handle))

grouped = {}
for row in rows:
    key = (row["tokens"], row["feature_dim"], row["backpressure"])
    grouped.setdefault(key, []).append(row)
for key, group in grouped.items():
    if len(group) != 3:
        raise SystemExit(f"incomplete architecture group: {key}")
    checksums = {row["output_checksum"] for row in group}
    if len(checksums) != 1:
        raise SystemExit(f"output mismatch across architectures: {key}: {checksums}")

combined = root / "cycle_benchmark_summary.csv"
with combined.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Cycle benchmark PASS: {combined}")
PY
