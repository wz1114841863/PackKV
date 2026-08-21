# BRISK-KV Artifact Evaluation Index

This is the compact map from paper claims to reproducible artifacts. The
authoritative narrative and claim limits are in
`hardware/docs/PAPER_HANDOFF_20260818.md`.

| Evidence | Primary artifact |
|---|---|
| Algorithm/configuration | `docs/MY_IDEAS.md`, `utils/compute.py`, `utils/config.py` |
| Format specification | `hardware/docs/briskkv_format_v0.md` |
| Python/Chisel golden vectors | `hardware/golden_vectors/`, `hardware/chisel/src/test/resources/golden_vectors/` |
| RTL tests and cycle benchmarks | `hardware/chisel/src/test/scala/briskkv/`, `hardware/evaluation/results/cycle_breakdown/` |
| VCS and SAIF method | `hardware/simulation/vcs/README.md` |
| Frozen writer-CG SAIF | `hardware/simulation/results/2026081801/` |
| Frozen writer-CG DC | `hardware/synthesis/dc/results/2026081801/` |
| Frozen writer-CG hierarchical power | `hardware/synthesis/dc/results/2026081802/` |
| Full-V/dual/shared comparison | `hardware/docs/PROJECT_STATUS_20260817.md` |
| Final retained point | `hardware/docs/PROJECT_STATUS_20260818.md` |

## Reproduction rules

1. Use generated `full/` RTL for VCS and activity generation.
2. Use the matching `dc_logic/` RTL for DC; obtain `TOP` from `manifest.json`.
3. Never reuse SAIF across Full-V, dual, shared, and writer-CG tops.
4. For the generic 1024-token testbench, use
   `SAIF_INSTANCE=tb_briskkv_tile_power_1024/dut`.
5. Keep each experiment in a new directory; archived results are immutable.
6. Report DC logic and CACTI SRAM separately unless technology and accounting
   are explicitly matched.

The archived remote VCS PASS lines for some matched runs were user-confirmed
from the terminal but their `simulation.log` files are not all present locally.
The SAIF and downstream DC reports are locally inspectable; this distinction
must remain visible in the paper artifact description.
