#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
architecture="${ARCHITECTURE:?ARCHITECTURE must be full_v, jit_v_shared, or jit_v_shared_writer_cg}"
activity_phase="${ACTIVITY_PHASE:-attention}"

case "$architecture" in
  full_v)
    expected_top=BriskKvSingleHeadTileTop
    wave_prefix=full_v
    ;;
  jit_v_shared)
    expected_top=BriskKvSharedJitVSingleHeadTileTop
    wave_prefix=jit_v_shared
    ;;
  jit_v_shared_writer_cg)
    expected_top=BriskKvSharedJitVWriterCgSingleHeadTileTop
    wave_prefix=jit_v_shared_writer_cg
    ;;
  *)
    printf 'Unsupported ARCHITECTURE=%s; use full_v, jit_v_shared, or jit_v_shared_writer_cg\n' \
      "$architecture" >&2
    exit 1
    ;;
esac

case "$activity_phase" in
  attention|write|combined) ;;
  *)
    printf 'Unsupported ACTIVITY_PHASE=%s; use attention, write, or combined\n' \
      "$activity_phase" >&2
    exit 1
    ;;
esac

TESTBENCH="$script_dir/tb_briskkv_tile_power_1024.sv" \
TESTBENCH_TOP=tb_briskkv_tile_power_1024 \
EXPECTED_TOP="$expected_top" \
DUT_TOP_DEFINE="$expected_top" \
REQUIRE_OVERLAP_INTERFACE=false \
PASS_PATTERN='BRISK-KV TILE 1024x128 POWER PASS' \
WAVE_BASENAME="${wave_prefix}_1024t_128f_${activity_phase}" \
ACTIVITY_PHASE="$activity_phase" \
bash "$script_dir/run_vcs.sh"
