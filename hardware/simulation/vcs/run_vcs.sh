#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${RTL_DIR:?RTL_DIR must point to a generated unified-tile full directory}"

manifest="$RTL_DIR/manifest.json"
filelist="$RTL_DIR/filelist.f"
testbench="${TESTBENCH:-$script_dir/tb_briskkv_jit_v.sv}"
testbench_top="${TESTBENCH_TOP:-tb_briskkv_jit_v}"
output_dir="${OUTPUT_DIR:-$script_dir/outputs}"
vcs_bin="${VCS:-vcs}"
wave_mode="${WAVE_MODE:-vcd}"
wave_basename="${WAVE_BASENAME:-jit_v_overlap_64t}"
activity_phase="${ACTIVITY_PHASE:-combined}"
pass_pattern="${PASS_PATTERN:-BRISK-KV VCS PASS}"
expected_top="${EXPECTED_TOP:-BriskKvJitVSingleHeadTileTop}"
require_overlap_interface="${REQUIRE_OVERLAP_INTERFACE:-true}"
dut_top_define="${DUT_TOP_DEFINE:-}"

case "$require_overlap_interface" in
  true|false) ;;
  *)
    printf 'REQUIRE_OVERLAP_INTERFACE must be true or false, got: %s\n' \
      "$require_overlap_interface" >&2
    exit 1
    ;;
esac

if [[ ! -f "$manifest" || ! -f "$filelist" ]]; then
  printf 'Incomplete full RTL directory: %s\n' "$RTL_DIR" >&2
  exit 1
fi
if ! grep -Eq '"mode"[[:space:]]*:[[:space:]]*"full"' "$manifest"; then
  printf 'VCS requires a full behavioral-memory export: %s\n' "$manifest" >&2
  exit 1
fi
if ! grep -Eq '"top"[[:space:]]*:[[:space:]]*"'"$expected_top"'"' \
  "$manifest"; then
  printf 'Testbench requires top=%s: %s\n' "$expected_top" "$manifest" >&2
  exit 1
fi
top_file="$RTL_DIR/$expected_top.sv"
if [[ ! -f "$top_file" ]]; then
  printf 'Manifest top source is missing: %s\n' "$top_file" >&2
  exit 1
fi
if [[ "$require_overlap_interface" == true ]] && \
   ! grep -q 'io_attentionProgress_vLaunched' "$top_file"; then
  printf 'RTL predates the JIT-V overlap interface; regenerate overlap-v1: %s\n' \
    "$top_file" >&2
  exit 1
fi
if [[ ! -f "$testbench" ]]; then
  printf 'Missing VCS testbench: %s\n' "$testbench" >&2
  exit 1
fi
case "$wave_mode" in
  none) wave_arg=() ;;
  vcd)  wave_arg=(+VCD "+WAVE_FILE=$output_dir/$wave_basename") ;;
  vpd)  wave_arg=(+VPD "+WAVE_FILE=$output_dir/$wave_basename") ;;
  *)
    printf 'Unsupported WAVE_MODE=%s; use none, vcd, or vpd\n' "$wave_mode" >&2
    exit 1
    ;;
esac

mkdir -p "$output_dir"
simv="$output_dir/simv"
compile_dir="$output_dir/csrc"
vcs_defines=(+define+BRISKKV_VCS)
if [[ -n "$dut_top_define" ]]; then
  vcs_defines+=("+define+BRISKKV_DUT_TOP=$dut_top_define")
fi

(
  cd "$RTL_DIR"
  "$vcs_bin" -full64 -sverilog -timescale=1ns/1ps \
    "${vcs_defines[@]}" \
    -debug_access+all -kdb \
    -Mdir="$compile_dir" \
    -LDFLAGS "-Wl,--no-as-needed" \
    -f filelist.f "$testbench" \
    -top "$testbench_top" \
    -o "$simv" \
    -l "$output_dir/compile.log"
)

(
  cd "$script_dir"
  "$simv" "+ACTIVITY_PHASE=$activity_phase" "${wave_arg[@]}" \
    -l "$output_dir/simulation.log"
)

if ! grep -Fq "$pass_pattern" "$output_dir/simulation.log"; then
  printf 'VCS run did not report PASS; inspect %s\n' \
    "$output_dir/simulation.log" >&2
  exit 1
fi
printf 'VCS validation PASS. Outputs: %s\n' "$output_dir"
