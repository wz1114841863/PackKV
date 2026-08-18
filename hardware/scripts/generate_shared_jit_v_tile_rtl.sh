#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hardware_dir="$(cd "$script_dir/.." && pwd)"
chisel_dir="$hardware_dir/chisel"

maximum_feature_dim="${MAXIMUM_FEATURE_DIM:-128}"
maximum_tokens="${MAXIMUM_TOKENS:-1024}"
input_bits="${INPUT_BITS:-24}"
scale_lanes="${SCALE_LANES:-4}"
enable_stats="${ENABLE_STATS:-false}"
quant_architecture="${QUANT_ARCHITECTURE:-v1}"
vcs_compatibility="${VCS_COMPATIBILITY:-false}"
writer_clock_gating="${WRITER_CLOCK_GATING:-false}"

case "$writer_clock_gating" in
  true)
    attention_architecture=jit_v_shared_writer_cg
    clock_suffix=_writer_cg
    ;;
  false)
    attention_architecture=jit_v_shared
    clock_suffix=
    ;;
  *)
    printf 'WRITER_CLOCK_GATING must be true or false, got: %s\n' \
      "$writer_clock_gating" >&2
    exit 1
    ;;
esac

config_name="briskkv_jit_v_shared_${quant_architecture}_t${maximum_tokens}_f${maximum_feature_dim}${clock_suffix}"
output_root="${OUTPUT_ROOT:-$hardware_dir/rtl/generated/$config_name}"

case "$vcs_compatibility" in
  true|false) ;;
  *)
    printf 'VCS_COMPATIBILITY must be true or false, got: %s\n' \
      "$vcs_compatibility" >&2
    exit 1
    ;;
esac

if [[ "$output_root" != /* ]]; then
  output_root="$(realpath -m "$output_root")"
fi

generate_variant() {
  local mode="$1"
  local target_dir="$output_root/$mode"
  (
    cd "$chisel_dir"
    sbt "runMain briskkv.GenerateBriskKvSingleHeadTileTop \
      --target-dir $target_dir \
      --maximum-feature-dim $maximum_feature_dim \
      --maximum-tokens $maximum_tokens \
      --input-bits $input_bits \
      --scale-lanes $scale_lanes \
      --enable-stats $enable_stats \
      --quant-architecture $quant_architecture \
      --attention-architecture $attention_architecture \
      --vcs-compatibility $vcs_compatibility \
      --mode $mode"
  )
}

generate_variant full
generate_variant dc_logic

find "$output_root" -maxdepth 2 -type f -print | sort
