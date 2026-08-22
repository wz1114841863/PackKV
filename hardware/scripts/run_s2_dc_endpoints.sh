#!/usr/bin/env bash
set -euo pipefail

# Run only selected S2 logic-only DC endpoints. Dynamic power is not a scaling
# result unless each point has a matching validated SAIF.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARDWARE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${RTL_ROOT:?RTL_ROOT must be the run_s2_context_scaling.sh RTL_ROOT}"
: "${TARGET_LIBRARY:?TARGET_LIBRARY must point to the DC .db library}"
DC_POINTS="${DC_POINTS:-4096}"
REPORT_ROOT="${REPORT_ROOT:-$HARDWARE_DIR/synthesis/dc/results/s2_context_scaling_$(date +%Y%m%d_%H%M%S)}"
CLOCK_PERIODS="${CLOCK_PERIODS:-2.0}"

IFS=',' read -r -a POINTS <<< "$DC_POINTS"
for architecture in full_v jit_v_shared_writer_cg; do
  for token in "${POINTS[@]}"; do
    rtl_dir="$RTL_ROOT/$architecture/t$token/dc_logic"
    if [[ ! -f "$rtl_dir/manifest.json" ]]; then
      echo "missing generated dc_logic RTL: $rtl_dir" >&2
      exit 1
    fi
    RTL_DIR="$rtl_dir" TARGET_LIBRARY="$TARGET_LIBRARY" CLOCK_PERIODS="$CLOCK_PERIODS" \
      REPORT_ROOT="$REPORT_ROOT/$architecture/t$token" \
      bash "$HARDWARE_DIR/synthesis/dc/run_tile_dc.sh"
  done
done
printf 'S2 DC endpoint reports: %s\n' "$REPORT_ROOT"
