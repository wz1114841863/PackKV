#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dc_tcl="${DC_TCL:-$script_dir/run_dc_logic.tcl}"
dc_shell_bin="${DC_SHELL:-dc_shell}"

: "${RTL_DIR:?RTL_DIR must point to a complete unified-tile dc_logic directory}"
: "${TARGET_LIBRARY:?TARGET_LIBRARY must point to the standard-cell .db library}"

manifest="$RTL_DIR/manifest.json"
memory_list="$RTL_DIR/memory_modules.tcl"
report_root="${REPORT_ROOT:-$PWD/briskkv_tile_dc_results}"
clock_periods="${CLOCK_PERIODS:-2.0}"
saif_file="${SAIF_FILE:-}"
saif_instance="${SAIF_INSTANCE:-tb_briskkv_jit_v/dut}"

if [[ -n "$saif_file" ]]; then
  if [[ ! -f "$saif_file" ]]; then
    printf 'Missing SAIF_FILE: %s\n' "$saif_file" >&2
    exit 1
  fi
  if ! grep -m 1 -Eq '^[[:space:]]*\(SAIFILE' "$saif_file"; then
    printf 'SAIF_FILE does not contain a SAIFILE header: %s\n' "$saif_file" >&2
    exit 1
  fi
  saif_file="$(realpath "$saif_file")"
  if [[ -z "$saif_instance" ]]; then
    printf 'SAIF_INSTANCE must not be empty when SAIF_FILE is set\n' >&2
    exit 1
  fi
fi

if [[ ! -f "$dc_tcl" ]]; then
  printf 'Missing DC Tcl flow: %s\n' "$dc_tcl" >&2
  exit 1
fi
if [[ ! -f "$manifest" ]]; then
  printf 'Missing tile RTL manifest: %s\n' "$manifest" >&2
  exit 1
fi
if [[ ! -f "$memory_list" ]]; then
  printf 'Missing SRAM black-box list: %s\n' "$memory_list" >&2
  exit 1
fi
if ! grep -Eq '"mode"[[:space:]]*:[[:space:]]*"dc_logic"' "$manifest"; then
  printf 'Manifest is not a dc_logic export: %s\n' "$manifest" >&2
  exit 1
fi

manifest_top="$(sed -n \
  's/^[[:space:]]*"top"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
  "$manifest" | head -n 1)"
if [[ -z "$manifest_top" ]]; then
  printf 'Cannot read top from manifest: %s\n' "$manifest" >&2
  exit 1
fi
case "$manifest_top" in
  BriskKvSingleHeadTileTop|BriskKvJitVSingleHeadTileTop|BriskKvSharedJitVSingleHeadTileTop)
    ;;
  *)
    printf 'Unsupported unified-tile top in manifest: %s\n' "$manifest_top" >&2
    exit 1
    ;;
esac

top_file="$RTL_DIR/$manifest_top.sv"
if [[ ! -f "$top_file" ]]; then
  printf 'Manifest top source is missing: %s\n' "$top_file" >&2
  exit 1
fi
if ! grep -Eq "^module[[:space:]]+$manifest_top([[:space:]]|\\()" "$top_file"; then
  printf 'Expected module declaration was not found in %s: %s\n' \
    "$top_file" "$manifest_top" >&2
  exit 1
fi

for period in $clock_periods; do
  period_label="${period//./p}ns"
  report_dir="$report_root/period_$period_label"
  mkdir -p "$report_dir"
  printf 'Running unified-tile DC: top=%s period=%s ns RTL=%s REPORT=%s\n' \
    "$manifest_top" "$period" "$RTL_DIR" "$report_dir"
  if [[ -n "$saif_file" ]]; then
    printf 'Annotating workload activity: SAIF=%s INSTANCE=%s\n' \
      "$saif_file" "$saif_instance"
  else
    printf 'No SAIF_FILE set; using DC default switching activity\n'
  fi
  RTL_DIR="$RTL_DIR" \
  TOP="$manifest_top" \
  REPORT_DIR="$report_dir" \
  CLOCK_PERIOD="$period" \
  TARGET_LIBRARY="$TARGET_LIBRARY" \
  LINK_LIBRARY="${LINK_LIBRARY:-$TARGET_LIBRARY}" \
  SAIF_FILE="$saif_file" \
  SAIF_INSTANCE="$saif_instance" \
    "$dc_shell_bin" -f "$dc_tcl" | tee "$report_dir/dc.log"
done

printf 'Unified-tile DC reports generated under %s\n' "$report_root"
