#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hardware_dir="$(cd "$script_dir/.." && pwd)"

maximum_tokens="${MAXIMUM_TOKENS:-1024}"
maximum_feature_dim="${MAXIMUM_FEATURE_DIM:-128}"
technology_nm="${TECHNOLOGY_NM:-22}"
clock_period_ns="${CLOCK_PERIOD_NS:-2.0}"
maximum_bank_width="${MAXIMUM_BANK_WIDTH:-128}"
config_name="briskkv_attention_t${maximum_tokens}_f${maximum_feature_dim}"
inventory="${MEMORIES_CSV:-$hardware_dir/rtl/generated/$config_name/dc_logic/memories.csv}"
output_dir="${OUTPUT_DIR:-$hardware_dir/evaluation/results/${config_name}_${technology_nm}nm_w${maximum_bank_width}}"

if [[ ! -f "$inventory" ]]; then
  echo "Memory inventory not found: $inventory" >&2
  echo "Run hardware/scripts/generate_attention_rtl.sh first." >&2
  exit 1
fi

python3 "$hardware_dir/evaluation/mem/briskkv_memory_eval.py" \
  --memories-csv "$inventory" \
  --output-dir "$output_dir" \
  --technology-nm "$technology_nm" \
  --clock-period-ns "$clock_period_ns" \
  --maximum-bank-width "$maximum_bank_width"
