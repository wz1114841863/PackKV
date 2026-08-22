#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hardware_dir="$(cd "$script_dir/.." && pwd)"
chisel_dir="$hardware_dir/chisel"
output_dir="${OUTPUT_DIR:-$hardware_dir/evaluation/results/cycle_benchmark}"
feature_dim="${FEATURE_DIM:-8}"
token_counts="${TOKEN_COUNTS:-64,256,1024}"
architectures="${ARCHITECTURES:-full_v,jit_v_dual,jit_v_shared}"
backpressure_modes="${BACKPRESSURE_MODES:-none,periodic}"
mkdir -p "$output_dir"

IFS=',' read -r -a architecture_list <<< "$architectures"
for architecture in "${architecture_list[@]}"; do
  (
    cd "$chisel_dir"
    sbt \
      -Dbriskkv.benchmarkArchitecture="$architecture" \
      -Dbriskkv.benchmarkFeatureDim="$feature_dim" \
      -Dbriskkv.benchmarkTokens="$token_counts" \
      -Dbriskkv.benchmarkBackpressure="$backpressure_modes" \
      -Dbriskkv.benchmarkOutput="$output_dir/$architecture.csv" \
      'testOnly briskkv.BriskKvCycleBenchmarkSpec'
  )
done

python3 - "$output_dir" "$architectures" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
architectures = [name.strip() for name in sys.argv[2].split(",") if name.strip()]
rows = []
for architecture in architectures:
    with (root / f"{architecture}.csv").open(newline="") as handle:
        rows.extend(csv.DictReader(handle))

grouped = {}
for row in rows:
    key = (row["tokens"], row["feature_dim"], row["backpressure"])
    grouped.setdefault(key, []).append(row)
for key, group in grouped.items():
    if len(group) != len(architectures):
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
