# PackKV / BRISK-KV

This repository starts from the PackKV GPU implementation and develops
**BRISK-KV**, a hardware-oriented KV-cache compression format and fused
single-head Attention tile.

The current hardware optimization boundary is frozen on **2026-08-18**. The
next project phase is paper preparation, not further datapath optimization.

## Start here

- Paper handoff: `hardware/docs/PAPER_HANDOFF_20260818.md`
- Current frozen hardware result: `hardware/docs/PROJECT_STATUS_20260818.md`
- Architecture ablation: `hardware/docs/JIT_V_ABLATION.md`
- Format contract: `hardware/docs/briskkv_format_v0.md`
- Research rationale: `docs/MY_IDEAS.md`
- Artifact-evaluation index: `ae.md`

## Current retained point

```text
name          = shared_writer_cg_v1
top           = BriskKvSharedJitVWriterCgSingleHeadTileTop
architecture  = shared JIT-V + replay_pipe_v1 + phase-gated writer
workload      = 1024 tokens x 128 features, single head
clock         = 2.0 ns / 500 MHz
activity      = attention-only RTL SAIF
PPA scope     = TSMC 28 nm logic, pre-layout, SRAM black boxes
```

Relative to the matched ungated shared design, writer clock gating preserves
278859 attention cycles, reduces average logic dynamic power by 33.12% and
logic-only total energy by about 12.01%, with 0.059% cell-area overhead. The
JIT-V family reduces architectural SRAM capacity by 58.64% relative to Full-V,
with a measured latency trade-off documented in the ablation report.

These are single-head research-tile results. They are not post-layout chip
power, full-system LLM speedup, or a direct measured comparison with PackKV on
a GPU.

## Repository map

- `models/`, `utils/`, `evaluation/`, `scripts/`: PackKV-derived software and
  local algorithm experiments;
- `hardware/chisel/`: synthesizable BRISK-KV implementation and tests;
- `hardware/rtl/generated/`: generated `full/` and `dc_logic/` RTL;
- `hardware/simulation/`: VCS testbenches, scripts, waveforms and SAIF;
- `hardware/synthesis/dc/`: DC scripts and archived reports;
- `hardware/evaluation/`: cycle, CACTI and report-processing utilities.

## Python environment

The historical setup used Python 3.12, PyTorch CUDA 12.1, the repository
requirements, and the local CUDA extension. Prefer the existing `.venv` in
this checkout when it is usable; do not reinstall dependencies merely to read
or organize the paper evidence.
