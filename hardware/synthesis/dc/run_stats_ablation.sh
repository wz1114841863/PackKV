#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dc_tcl="${DC_TCL:-$script_dir/run_dc_logic.tcl}"
report_root="${REPORT_ROOT:-$PWD/dc_stats_ablation_results}"
dc_shell_bin="${DC_SHELL:-dc_shell}"

: "${TARGET_LIBRARY:?TARGET_LIBRARY must point to the standard-cell .db library}"

if [[ -n "${STATS_ON_RTL_DIR:-}" || -n "${STATS_OFF_RTL_DIR:-}" ]]; then
  : "${STATS_ON_RTL_DIR:?Set both STATS_ON_RTL_DIR and STATS_OFF_RTL_DIR}"
  : "${STATS_OFF_RTL_DIR:?Set both STATS_ON_RTL_DIR and STATS_OFF_RTL_DIR}"
  stats_on_rtl="$STATS_ON_RTL_DIR"
  stats_off_rtl="$STATS_OFF_RTL_DIR"
elif [[ -n "${ABLATION_RTL_ROOT:-}" ]]; then
  stats_on_rtl="$ABLATION_RTL_ROOT/stats_on/dc_logic"
  stats_off_rtl="$ABLATION_RTL_ROOT/stats_off/dc_logic"
else
  printf '%s\n' \
    'Set ABLATION_RTL_ROOT, or set both STATS_ON_RTL_DIR and STATS_OFF_RTL_DIR.' >&2
  exit 1
fi

if [[ ! -f "$dc_tcl" ]]; then
  printf 'Missing DC Tcl flow: %s\n' "$dc_tcl" >&2
  printf 'Upload run_dc_logic.tcl or set DC_TCL to its absolute path.\n' >&2
  exit 1
fi

for stats_mode in on off; do
  if [[ "$stats_mode" == "on" ]]; then
    rtl_dir="$stats_on_rtl"
    expected_flag=true
  else
    rtl_dir="$stats_off_rtl"
    expected_flag=false
  fi
  report_dir="$report_root/stats_$stats_mode"
  if [[ ! -f "$rtl_dir/manifest.json" ]]; then
    printf 'Missing stats-%s RTL manifest: %s\n' "$stats_mode" "$rtl_dir/manifest.json" >&2
    exit 1
  fi
  if [[ ! -f "$rtl_dir/memory_modules.tcl" ]]; then
    printf 'Missing stats-%s memory list: %s\n' "$stats_mode" "$rtl_dir/memory_modules.tcl" >&2
    exit 1
  fi
  if ! grep -Eq \
    "\"performance_stats_enabled\"[[:space:]]*:[[:space:]]*$expected_flag" \
    "$rtl_dir/manifest.json"; then
    printf 'Manifest does not identify this as stats-%s: %s\n' \
      "$stats_mode" "$rtl_dir/manifest.json" >&2
    exit 1
  fi
  mkdir -p "$report_dir"
  printf 'Running DC stats-%s: RTL=%s REPORT=%s\n' "$stats_mode" "$rtl_dir" "$report_dir"
  RTL_DIR="$rtl_dir" \
  REPORT_DIR="$report_dir" \
  CLOCK_PERIOD="${CLOCK_PERIOD:-2.0}" \
  TARGET_LIBRARY="$TARGET_LIBRARY" \
  LINK_LIBRARY="${LINK_LIBRARY:-$TARGET_LIBRARY}" \
    "$dc_shell_bin" -f "$dc_tcl" \
      | tee "$report_dir/dc.log"
done

printf 'Stats-on/off DC reports generated under %s\n' "$report_root"
