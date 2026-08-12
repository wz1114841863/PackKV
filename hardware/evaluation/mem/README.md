# BRISK-KV CACTI evaluation

This directory evaluates the architectural SRAMs that are black-boxed during
Design Compiler logic synthesis.  It consumes the `memories.csv` generated
with the `dc_logic` SystemVerilog variant; capacities are not copied into a
separate hand-maintained configuration.

## Modeling assumptions

- Each Chisel `SyncReadMem` has one exclusive read port and one exclusive write
  port.  The generated inventory records these port counts explicitly.
- Wide logical words are split into parallel width slices, 128 bits maximum by
  default.  Every slice is active for one logical read or write, so slice
  energies and areas are summed and slice latency is combined with `max()`.
- CACTI byte/line padding is preserved as `modeled_bits`; it is never reported
  as useful logical capacity.
- ECC is disabled because Format v0 and the current RTL do not implement ECC.
- Dynamic access energy is not total workload energy.  Multiply read/write
  energy by access counts from cycle-level evaluation before combining it with
  leakage and DC logic power.
- The bundled technology data covers 22-180 nm.  A sub-22-nm paper point needs
  a foundry SRAM compiler or explicitly documented scaling, not a fabricated
  CACTI input.

The imported CACTI directory did not include an upstream URL, commit ID, build
instructions, or license file.  The evaluator records the executable SHA-256,
but that does not establish provenance or redistribution permission.  Restore
those files from the exact upstream project before publishing or redistributing
the binary.  Until then, the precise CACTI source version is **unable to be
confirmed**.

## Run

First generate the RTL inventory:

```bash
bash hardware/scripts/generate_attention_rtl.sh
```

Then run the 22 nm, 2 ns baseline:

```bash
python3 hardware/evaluation/mem/briskkv_memory_eval.py \
  --memories-csv hardware/rtl/generated/briskkv_attention_t1024_f128/dc_logic/memories.csv \
  --output-dir hardware/evaluation/results/cacti_t1024_f128_22nm \
  --technology-nm 22 \
  --clock-period-ns 2.0 \
  --maximum-bank-width 128
```

Outputs are:

- `memory_summary.csv`: one row per logical RTL memory;
- `bank_detail.csv`: one row per physical width slice;
- `memory_summary.json`: assumptions, per-memory values and totals;
- `cacti_runs/`: exact CACTI input/output files for reproducibility.

For a banking sensitivity study, rerun with widths such as 64, 128 and 256.
Do not select the best result after the fact without reporting the sweep.
