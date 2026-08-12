# BRISK-KV DC baseline

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
