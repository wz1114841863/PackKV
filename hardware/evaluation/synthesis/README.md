# BRISK-KV DC baseline validation

The current accepted DC boundary is no longer the early attention-only
component baseline. The retained full-tile writer-CG reports are under
`hardware/synthesis/dc/results/2026081801/`, with DDC hierarchical power under
`2026081802/`. Use `hardware/docs/PROJECT_STATUS_20260818.md` for the exact
configuration and hashes. This checker remains useful for new runs, which must
use new result directories.

`dc_baseline_report.py` checks whether a Design Compiler run is a usable
logic-only baseline before its area, timing, or power values are compared.

The run is rejected when a mandatory report is absent, the explicit success
marker is missing, or any architectural SRAM black-box instance disappears or
changes count across `compile_ultra`. Metric extraction is best effort because
DC report labels vary by release; an unrecognized metric is emitted as `null`
with a warning and is never replaced by zero.

Run from the repository root after copying the DC output directory back, or run
directly on the synthesis server:

```bash
python3 hardware/evaluation/synthesis/dc_baseline_report.py \
  --report-dir runs/2026081103_output/outputs \
  --dc-log runs/2026081103_output/dc.log \
  --memory-modules hardware/rtl/generated/briskkv_attention_t1024_f128/dc_logic/memory_modules.tcl \
  --output runs/2026081103_output/dc_baseline.json
```

Exit status is `0` only for a structurally valid run and `2` for an invalid
run. Always consult `units.rpt` before combining DC power with CACTI values;
the JSON intentionally retains power in the unit printed by DC.
