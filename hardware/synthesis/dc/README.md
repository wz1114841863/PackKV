# BRISK-KV DC baseline

## Frozen paper-stage result

The retained full-tile result is `shared_writer_cg_v1` at 2.0 ns / 500 MHz:

```text
top:       BriskKvSharedJitVWriterCgSingleHeadTileTop
SAIF:      hardware/simulation/results/2026081801/
DC:        hardware/synthesis/dc/results/2026081801/
DDC hier:  hardware/synthesis/dc/results/2026081802/
hierarchy: tb_briskkv_tile_power_1024/dut
```

It uses TSMC 28 nm HPC+ LVT SSG 0.81 V / 125 C, stats-off RTL,
attention-only activity, ZeroWireload, and 52 architectural SRAM black-box
instances. Setup closes at 2.0 ns; worst hold is about -0.06 ns and there are
no max-transition violations. Cell area is 134594.710815 um^2 and dynamic
power is 15.8115 mW. Exact provenance, coverage, hierarchy attribution, and
claim limits are in `../../docs/PROJECT_STATUS_20260818.md`.

The commands below remain reproduction and ablation instructions. New runs
must use new report roots and may not overwrite any frozen result.

`dc_logic` RTL replaces every architectural Chisel `SyncReadMem` implementation
with a port-compatible bodyless SystemVerilog stub and provides
`memory_modules.tcl` to mark and audit those designs as black boxes. Use Design
Compiler for the logic and memory-interface estimate,
and use `memories.csv` with CACTI or a foundry memory compiler for SRAM
area/energy/latency. The small 2-entry Query replay queue remains logic.

Generate the 1024-token, 128-feature baseline:

```bash
bash hardware/scripts/generate_attention_rtl.sh
```

Generate the independent 1024-token, 128-feature, signed Q12/24-bit write
encoder baseline:

```bash
MAXIMUM_TOKENS=1024 MAXIMUM_FEATURE_DIM=128 ENABLE_STATS=false \
  bash hardware/scripts/generate_write_encoder_rtl.sh
```

This creates `briskkv_write_v1_t1024_f128_i24/full` for RTL simulation and
`briskkv_write_v1_t1024_f128_i24/dc_logic` for
logic-only DC. The write generator scans
the emitted CIRCT memories instead of hard-coding module names, then records
module depth, width, and instance count in `memories.csv` before replacing the
DC variant with bodyless black-box stubs.

The two generated variants are:

```text
hardware/rtl/generated/briskkv_attention_t1024_f128/full/
hardware/rtl/generated/briskkv_attention_t1024_f128/dc_logic/
```

On the DC server, provide the technology library and run:

```bash
export RTL_DIR=/path/to/dc_logic
export TARGET_LIBRARY=/path/to/typical.db
export CLOCK_PERIOD=2.0
export REPORT_DIR=reports_t1024_f128
dc_shell -f hardware/synthesis/dc/run_dc_logic.tcl
```

To annotate logic power with simulation activity, convert the validated VCD to
SAIF, then pass the SAIF file and the DUT hierarchy recorded inside it:

```bash
vcd2saif \
  -input /absolute/path/to/jit_v_overlap_64t.vcd \
  -output /absolute/path/to/jit_v_overlap_64t.saif

export RTL_DIR=/absolute/path/to/vcs-compatible/dc_logic
export TARGET_LIBRARY=/absolute/path/to/standard_cells.db
export SAIF_FILE=/absolute/path/to/jit_v_overlap_64t.saif
export SAIF_INSTANCE=tb_briskkv_jit_v/dut
export REPORT_ROOT=/absolute/path/to/saif_power_results
export CLOCK_PERIODS="2.0"
bash hardware/synthesis/dc/run_tile_dc.sh
```

`SAIF_FILE` is optional. Without it, the flow retains the existing DC default
switching activity. With it, `SAIF_INSTANCE` selects the simulated hierarchy
that corresponds to the current DC top; its default is
`tb_briskkv_jit_v/dut`. Each period directory records:

- `power_activity_source.rpt`: whether SAIF or default activity was used;
- `saif_read.rpt`: `read_saif -verbose` output and mapping diagnostics;
- `saif_annotation_precompile.rpt`: hierarchical annotation before mapping;
- `saif_annotation_postcompile.rpt`: retained annotation after mapping;
- `power.rpt`: logic-only power using the resulting activity.

Inspect the annotation reports before accepting `power.rpt`. A successfully
parsed SAIF with poor hierarchy/name mapping can otherwise leave most objects
on default activity. Architectural SRAMs remain black boxes, so their dynamic
energy still requires access counts plus CACTI or memory-compiler energy.
The 64-token/four-feature VCS trace is suitable only for bringing up this flow.
The accepted workload-power results use architecture-matched 1024-token x
128-feature traces in separate report directories.

For the write encoder, upload the complete generated `dc_logic` directory,
`run_dc_logic.tcl`, and `run_write_encoder_dc.sh`. Absolute paths make the flow
independent of the server repository layout:

```bash
export RTL_DIR=/absolute/path/to/briskkv_write_v1_t1024_f128_i24/dc_logic
export TARGET_LIBRARY=/absolute/path/to/standard_cells.db
export DC_TCL=/absolute/path/to/run_dc_logic.tcl
export REPORT_ROOT=/absolute/path/to/write_encoder_reports
export CLOCK_PERIODS="1.5 1.2 1.0"
bash /absolute/path/to/run_write_encoder_dc.sh
```

To reproduce the v2 quantizer-pipeline ablation instead, generate it with
`QUANT_ARCHITECTURE=v2`; its directory is
`briskkv_write_v2_t1024_f128_i24/dc_logic`. The selected architecture and its
extra parameter cycles are recorded in `manifest.json`, so v1/v2 reports must
not share one report directory.

The v3 leading-one selector is generated with `QUANT_ARCHITECTURE=v3` into
`briskkv_write_v3_t1024_f128_i24/dc_logic`. It retains the v1 single-stage
schedule and is the next optimization candidate. Keep its DC report root
separate from both v1 and v2.

The v4 balanced static threshold tree is generated with
`QUANT_ARCHITECTURE=v4` into
`briskkv_write_v4_t1024_f128_i24/dc_logic`. It retains the v1 schedule and
exact threshold boundary behavior with zero extra parameter cycles. Keep its
DC report root separate from v1, v2, and v3.

The wrapper fixes `TOP=BriskKvWriteEncoderTop`, validates the manifest and
black-box list, and creates one report directory per requested period.

For a matched performance-counter area ablation, first generate both RTL
variants. The DC server does not need the complete repository, but it must
receive both complete `dc_logic` directories, `run_stats_ablation.sh`, and
`run_dc_logic.tcl`. For an arbitrary server-side layout, pass absolute paths:

```bash
export TARGET_LIBRARY=/absolute/path/to/standard_cells.db
export STATS_ON_RTL_DIR=/absolute/path/to/stats_on/dc_logic
export STATS_OFF_RTL_DIR=/absolute/path/to/stats_off/dc_logic
export DC_TCL=/absolute/path/to/run_dc_logic.tcl
export REPORT_ROOT=/absolute/path/to/output_reports
export CLOCK_PERIOD=2.0
bash /absolute/path/to/run_stats_ablation.sh
```

If the generated `stats_on/dc_logic` and `stats_off/dc_logic` hierarchy was
preserved, `ABLATION_RTL_ROOT=/path/to/parent` can replace the two explicit RTL
variables. The shell script no longer derives any path from the repository
layout and validates both manifests and memory black-box lists before invoking
DC.

The two report directories are `stats_on` and `stats_off`. Compare total and
hierarchical area from `area_hier.rpt`; also compare `qor.rpt` and
`timing_setup.rpt` to ensure the area result is not caused by different timing
closure. Both manifests retain the same functional geometry and record the
`performance_stats_enabled` value.

The script creates an isolated per-process WORK library and must produce both
`memory_blackboxes_precompile.rpt` and `memory_blackboxes_postcompile.rpt`.
Each report must contain all five memory module names with a non-zero instance
count. The run aborts if a stub contains a CIRCT `Memory[` behavioral array or
if any black-box instance is missing. A mapping message naming one of the five
memory modules indicates an invalid run and must not be used.

The entire flow is wrapped in Tcl `catch`: a DC/Tcl command error terminates
with `exit 1`, while a complete run terminates explicitly with `exit 0`. This
is required for DC O-2018, which may otherwise continue after an unsupported
command. The SRAM source marker `syn_black_box` establishes black-box status;
the script deliberately does not use the unavailable O-2018 `set_black_box`
command. Additional reports include `units.rpt`, `check_timing.rpt`, setup and
hold timing, clocks, and all constraint violations.

Inside the wrapped Tcl procedure, the target-library list is retained in the
ordinary local variable `target_libraries`; `set_app_var target_library` alone
does not create a readable local `$target_library` variable in DC O-2018.

For a logic-only study, keep the listed references black-boxed and do not
include their area or power in the DC total. Add the CACTI/memory-compiler
totals separately and state this method explicitly in the paper. For a
post-memory-mapping study, replace each listed module with a compatible SRAM
macro wrapper and load its `.db` model instead of black-boxing it.

Run the matching CACTI point from the repository root with:

```bash
bash hardware/scripts/evaluate_cacti.sh
```

Use the same technology assumption as the DC library where CACTI provides it,
and use `memory_summary.csv` for static area/leakage. Dynamic memory energy is
`read_count * read_energy + write_count * write_energy`; the reported
`one_access_each_*` totals are normalization values, not workload energy.

To generate another capacity point:

```bash
MAXIMUM_TOKENS=4096 MAXIMUM_FEATURE_DIM=128 \
  bash hardware/scripts/generate_attention_rtl.sh
```
