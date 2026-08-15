#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
activity_phase="${ACTIVITY_PHASE:-attention}"

case "$activity_phase" in
  attention|write|combined) ;;
  *)
    printf 'Unsupported ACTIVITY_PHASE=%s; use attention, write, or combined\n' \
      "$activity_phase" >&2
    exit 1
    ;;
esac

TESTBENCH="$script_dir/tb_briskkv_jit_v_power_1024.sv" \
TESTBENCH_TOP=tb_briskkv_jit_v_power_1024 \
PASS_PATTERN='BRISK-KV 1024x128 POWER PASS' \
WAVE_BASENAME="jit_v_1024t_128f_${activity_phase}" \
ACTIVITY_PHASE="$activity_phase" \
bash "$script_dir/run_vcs.sh"
