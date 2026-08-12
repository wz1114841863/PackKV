#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hardware_dir="$(cd "$script_dir/../.." && pwd)"

maximum_tokens="${MAXIMUM_TOKENS:-1024}"
maximum_feature_dim="${MAXIMUM_FEATURE_DIM:-128}"
scale_lanes="${SCALE_LANES:-4}"
rtl_root="${ABLATION_RTL_ROOT:-$hardware_dir/rtl/generated/stats_ablation_t${maximum_tokens}_f${maximum_feature_dim}_s${scale_lanes}}"
report_root="${REPORT_ROOT:-$script_dir/results/stats_ablation_t${maximum_tokens}_f${maximum_feature_dim}_s${scale_lanes}}"
dc_shell_bin="${DC_SHELL:-dc_shell}"

: "${TARGET_LIBRARY:?TARGET_LIBRARY must point to the standard-cell .db library}"

for stats_mode in on off; do
  rtl_dir="$rtl_root/stats_$stats_mode/dc_logic"
  report_dir="$report_root/stats_$stats_mode"
  if [[ ! -f "$rtl_dir/manifest.json" ]]; then
    printf 'Missing stats-%s RTL manifest: %s\n' "$stats_mode" "$rtl_dir/manifest.json" >&2
    exit 1
  fi
  mkdir -p "$report_dir"
  printf 'Running DC stats-%s: RTL=%s REPORT=%s\n' "$stats_mode" "$rtl_dir" "$report_dir"
  RTL_DIR="$rtl_dir" \
  REPORT_DIR="$report_dir" \
  CLOCK_PERIOD="${CLOCK_PERIOD:-2.0}" \
  TARGET_LIBRARY="$TARGET_LIBRARY" \
  LINK_LIBRARY="${LINK_LIBRARY:-$TARGET_LIBRARY}" \
    "$dc_shell_bin" -f "$script_dir/run_dc_logic.tcl" \
      | tee "$report_dir/dc.log"
done

printf 'Stats-on/off DC reports generated under %s\n' "$report_root"
