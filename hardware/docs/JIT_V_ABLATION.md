# BRISK-KV Full-V and JIT-V ablation

Updated through the frozen 2026-08-18 writer-clock-gating result. This file is
the consolidated architecture-ablation record. Detailed provenance and hashes
remain in `PROJECT_STATUS_20260815.md`, `PROJECT_STATUS_20260817.md`, and
`PROJECT_STATUS_20260818.md`.

## Architecture variants

| Variant | Top | V materialization | Decompressors | Status |
|---|---|---|---:|---|
| Full-V | `BriskKvSingleHeadTileTop` | full token-feature SRAM | 2 | latency reference |
| JIT-V dual | `BriskKvJitVSingleHeadTileTop` | packet queues, weights, partial sums | 2 | replay baseline |
| JIT-V shared | `BriskKvSharedJitVSingleHeadTileTop` | same JIT-V buffers | 1 shared | area/power reference |
| Shared writer-CG | `BriskKvSharedJitVWriterCgSingleHeadTileTop` | same as shared | 1 shared | retained point |

JIT-V starts V replay once the first Softmax weight packet is resident. A V
packet leaves the two-entry queue only after its matching weight packet is
available, preventing read-before-write. Pipelined byte-stream replay overlaps
synchronous SRAM response latency with stream consumption. The shared variant
time-multiplexes one K-capable decoder and widens V metadata at its adapter.
The writer-CG variant changes only the writer clock during non-write phases.

## Storage result

At 1024 tokens x 128 features:

| Variant | Architectural SRAM bits | 22 nm CACTI area (mm^2) | CACTI leakage (mW) |
|---|---:|---:|---:|
| Full-V | 4,012,288 | 0.847445 | 50.692293 |
| JIT-V dual/shared/writer-CG | 1,659,392 | 0.343719 | 22.426537 |

JIT-V removes 2,352,896 bits, or 58.64%. The CACTI numbers are storage-only
22 nm estimates. They are not directly additive to the TSMC 28 nm DC logic
results and do not include workload access energy.

## Replay-pipeline ablation

For feature_dim=8 without output backpressure, launching V at the first weight
packet and then applying pipelined replay produced:

| Tokens | Dual before overlap | Dual overlap | Shared before overlap | Shared overlap |
|---:|---:|---:|---:|---:|
| 64 | 1682 | 1673 | 1687 | 1678 |
| 256 | 6368 | 6323 | 6373 | 6328 |
| 1024 | 25112 | 24923 | 25117 | 24928 |

The overlap saves the remaining Softmax-weight production tail but does not
remove serialized V decompression. At the representative 1024 x 128 periodic
workload, replay_pipe_v1 reduced dual attention cycles from 395923 to 278843,
or 29.57%, and logic-only transaction energy by about 28.86% relative to the
frozen pre-replay baseline. Cell area increased by about 0.70% and average
logic power by about 1.01%.

## Matched 1024 x 128 architecture comparison

All points below use a 2.0 ns clock, stats-off PPA RTL, attention-only SAIF,
the same TSMC 28 nm library/corner, periodic output backpressure, and
architectural SRAM black boxes.

| Metric | Full-V | JIT-V dual | JIT-V shared | Shared writer-CG |
|---|---:|---:|---:|---:|
| Attention cycles | 164447 | 278843 | 278859 | 278859 |
| Cell area (um^2) | 145161.574572 | 141438.694767 | 134515.414817 | 134594.710815 |
| Dynamic power (mW) | 26.1802 | 25.8113 | 23.6411 | 15.8115 |
| Leakage power (mW) | 44.0547 | 41.8824 | 39.7520 | 39.9662 |
| Total cell power (mW) | 70.2353 | 67.6944 | 63.3933 | about 55.777 |
| SRAM bits | 4012288 | 1659392 | 1659392 | 1659392 |
| Setup WNS/TNS | 0/0 ns | 0/0 ns | 0/0 ns | 0/0 ns |
| Worst hold | about -0.06 ns | about -0.06 ns | about -0.06 ns | about -0.06 ns |
| Max-transition violations | 0 | 0 | 0 | 0 |

Shared adds 16 attention cycles relative to dual while reducing cell area by
4.89%, dynamic power by 8.41%, and total cell power by 6.35%. Hierarchical
power attributes about 3.416 mW, or 79.4% of the total reduction, to replacing
the dual decoder pair with one decoder plus its metadata adapter.

Full-V remains the latency endpoint and uses 2.418x the JIT-V SRAM bits. The
shared architecture is the area/power-first JIT-V endpoint; dual remains 16
cycles faster and is not described as strictly dominated.

## Writer phase-clock-gating ablation

Attention-only hierarchical power showed an idle writer hotspot: the ungated
shared writer consumed 15.349 mW total, including 7.774 mW internal power,
despite almost zero functional switching. The gated variant holds the writer
clock stationary outside reset, legal write launch, and the active write phase.

Relative to ungated shared:

| Metric | Change |
|---|---:|
| Attention cycles | unchanged at 278859 |
| Cell area | +0.05895% |
| Dynamic power | -33.12% |
| Total cell power/energy | about -12.01% |
| Writer total power | 15.349 -> 7.698 mW |
| Writer internal power | 7.774 -> about 0.000005 mW |

The attention-only SAIF records zero transitions on the writer gated clock,
enable, and latched enable. DC maps the technology-independent gate to one
latch and one AND cell. This is causal evidence for phase-level clock gating,
but not a characterized foundry ICG, CTS, or post-layout result.

## Functional and cycle reproduction

Run architectures in separate simulator processes:

```bash
FEATURE_DIM=8 TOKEN_COUNTS=64,256,1024 \
  bash hardware/scripts/run_cycle_benchmark.sh
```

For the final 1024 x 128 stats-enabled breakdown, use the architecture-specific
cycle benchmark flow and preserve each CSV under a unique result directory.
The matched and writer-CG CSVs are archived under:

```text
hardware/evaluation/results/cycle_breakdown/
  2026081702_matched_replay_pipe_v1/
  2026081703_shared_writer_cg/
```

The benchmark uses identical deterministic K/V/query values, checks output
counts and complete checksums, and records write, QK, Softmax, replay, AV, and
stall milestones. Runs remain sequential because ChiselSim shares a suite work
directory.

## RTL, VCS, SAIF, and DC rules

- Export every architecture into a new directory.
- Use `full/` for VCS and SAIF; use the matching `dc_logic/` for DC.
- Obtain the DC top from `manifest.json`.
- Do not reuse SAIF across architecture tops.
- The generic Full-V/shared/writer-CG power testbench uses
  `tb_briskkv_tile_power_1024/dut`.
- Keep the retained writer-CG artifacts in the paths frozen by
  `PROJECT_STATUS_20260818.md`.

## Interpretation boundary

The ablation demonstrates a single-head storage-latency-logic-power Pareto and
the effects of three bounded hardware changes: replay pipelining, decoder
sharing, and writer clock gating. It does not establish full-chip power,
post-layout timing, complete LLM throughput, or superiority to GPU PackKV.
Architectural SRAM power is outside DC and must be evaluated separately.
