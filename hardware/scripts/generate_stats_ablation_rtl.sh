#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hardware_dir="$(cd "$script_dir/.." && pwd)"

maximum_tokens="${MAXIMUM_TOKENS:-1024}"
maximum_feature_dim="${MAXIMUM_FEATURE_DIM:-128}"
scale_lanes="${SCALE_LANES:-4}"
ablation_root="${OUTPUT_ROOT:-$hardware_dir/rtl/generated/stats_ablation_t${maximum_tokens}_f${maximum_feature_dim}_s${scale_lanes}}"

for stats_mode in on off; do
  if [[ "$stats_mode" == "on" ]]; then
    enabled=true
  else
    enabled=false
  fi
  ENABLE_STATS="$enabled" \
  MAXIMUM_TOKENS="$maximum_tokens" \
  MAXIMUM_FEATURE_DIM="$maximum_feature_dim" \
  SCALE_LANES="$scale_lanes" \
  OUTPUT_ROOT="$ablation_root/stats_$stats_mode" \
    bash "$script_dir/generate_attention_rtl.sh"
done

printf 'Stats ablation RTL generated under %s\n' "$ablation_root"
