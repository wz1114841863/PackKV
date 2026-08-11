# BRISK-KV DC baseline

`dc_logic` RTL places every architectural Chisel `SyncReadMem` in a separate
SystemVerilog module and provides `memory_modules.tcl` to mark those designs as
black boxes. Use Design Compiler for the logic and memory-interface estimate,
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

For a logic-only study, keep the listed references black-boxed and do not
include their area or power in the DC total. Add the CACTI/memory-compiler
totals separately and state this method explicitly in the paper. For a
post-memory-mapping study, replace each listed module with a compatible SRAM
macro wrapper and load its `.db` model instead of black-boxing it.

To generate another capacity point:

```bash
MAXIMUM_TOKENS=4096 MAXIMUM_FEATURE_DIM=128 \
  bash hardware/scripts/generate_attention_rtl.sh
```
