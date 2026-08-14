#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${RTL_DIR:?RTL_DIR must point to the generated JIT-V full directory}"

manifest="$RTL_DIR/manifest.json"
filelist="$RTL_DIR/filelist.f"
testbench="${TESTBENCH:-$script_dir/tb_briskkv_jit_v.sv}"
output_dir="${OUTPUT_DIR:-$script_dir/outputs}"
vcs_bin="${VCS:-vcs}"
wave_mode="${WAVE_MODE:-vcd}"

if [[ ! -f "$manifest" || ! -f "$filelist" ]]; then
  printf 'Incomplete full RTL directory: %s\n' "$RTL_DIR" >&2
  exit 1
fi
if ! grep -Eq '"mode"[[:space:]]*:[[:space:]]*"full"' "$manifest"; then
  printf 'VCS requires a full behavioral-memory export: %s\n' "$manifest" >&2
  exit 1
fi
if ! grep -Eq '"top"[[:space:]]*:[[:space:]]*"BriskKvJitVSingleHeadTileTop"' \
  "$manifest"; then
  printf 'Testbench requires the dual JIT-V top: %s\n' "$manifest" >&2
  exit 1
fi
if ! grep -q 'io_attentionProgress_vLaunched' \
  "$RTL_DIR/BriskKvJitVSingleHeadTileTop.sv"; then
  printf 'RTL predates the JIT-V overlap interface; regenerate overlap-v1: %s\n' \
    "$RTL_DIR/BriskKvJitVSingleHeadTileTop.sv" >&2
  exit 1
fi
if [[ ! -f "$testbench" ]]; then
  printf 'Missing VCS testbench: %s\n' "$testbench" >&2
  exit 1
fi
case "$wave_mode" in
  none) wave_arg=() ;;
  vcd)  wave_arg=(+VCD "+WAVE_FILE=$output_dir/jit_v_overlap_64t") ;;
  vpd)  wave_arg=(+VPD "+WAVE_FILE=$output_dir/jit_v_overlap_64t") ;;
  *)
    printf 'Unsupported WAVE_MODE=%s; use none, vcd, or vpd\n' "$wave_mode" >&2
    exit 1
    ;;
esac

mkdir -p "$output_dir"
simv="$output_dir/simv"

(
  cd "$RTL_DIR"
  "$vcs_bin" -full64 -sverilog -timescale=1ns/1ps \
    +define+BRISKKV_VCS \
    -debug_access+all -kdb \
    -f filelist.f "$testbench" \
    -top tb_briskkv_jit_v \
    -o "$simv" \
    -l "$output_dir/compile.log"
)

(
  cd "$script_dir"
  "$simv" "${wave_arg[@]}" -l "$output_dir/simulation.log"
)

if ! grep -q 'BRISK-KV VCS PASS' "$output_dir/simulation.log"; then
  printf 'VCS run did not report PASS; inspect %s\n' \
    "$output_dir/simulation.log" >&2
  exit 1
fi
printf 'VCS validation PASS. Outputs: %s\n' "$output_dir"
