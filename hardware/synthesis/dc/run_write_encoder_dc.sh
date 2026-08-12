#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dc_tcl="${DC_TCL:-$script_dir/run_dc_logic.tcl}"
dc_shell_bin="${DC_SHELL:-dc_shell}"

: "${RTL_DIR:?RTL_DIR must point to the complete write-side dc_logic directory}"
: "${TARGET_LIBRARY:?TARGET_LIBRARY must point to the standard-cell .db library}"

report_root="${REPORT_ROOT:-$PWD/briskkv_write_dc_results}"
clock_periods="${CLOCK_PERIODS:-1.5 1.2 1.0}"

if [[ ! -f "$dc_tcl" ]]; then
  printf 'Missing DC Tcl flow: %s\n' "$dc_tcl" >&2
  exit 1
fi
if [[ ! -f "$RTL_DIR/manifest.json" ]]; then
  printf 'Missing write-side RTL manifest: %s\n' "$RTL_DIR/manifest.json" >&2
  exit 1
fi
if ! grep -Eq '"top"[[:space:]]*:[[:space:]]*"BriskKvWriteEncoderTop"' \
  "$RTL_DIR/manifest.json"; then
  printf 'Manifest top is not BriskKvWriteEncoderTop: %s\n' \
    "$RTL_DIR/manifest.json" >&2
  exit 1
fi
if ! grep -Eq '"mode"[[:space:]]*:[[:space:]]*"dc_logic"' \
  "$RTL_DIR/manifest.json"; then
  printf 'Manifest is not a dc_logic export: %s\n' "$RTL_DIR/manifest.json" >&2
  exit 1
fi
if [[ ! -f "$RTL_DIR/memory_modules.tcl" ]]; then
  printf 'Missing write-side SRAM black-box list: %s\n' \
    "$RTL_DIR/memory_modules.tcl" >&2
  exit 1
fi

for period in $clock_periods; do
  period_label="${period//./p}ns"
  report_dir="$report_root/period_$period_label"
  mkdir -p "$report_dir"
  printf 'Running write encoder DC: period=%s ns RTL=%s REPORT=%s\n' \
    "$period" "$RTL_DIR" "$report_dir"
  RTL_DIR="$RTL_DIR" \
  TOP=BriskKvWriteEncoderTop \
  REPORT_DIR="$report_dir" \
  CLOCK_PERIOD="$period" \
  TARGET_LIBRARY="$TARGET_LIBRARY" \
  LINK_LIBRARY="${LINK_LIBRARY:-$TARGET_LIBRARY}" \
    "$dc_shell_bin" -f "$dc_tcl" | tee "$report_dir/dc.log"
done

printf 'Write-side DC reports generated under %s\n' "$report_root"
