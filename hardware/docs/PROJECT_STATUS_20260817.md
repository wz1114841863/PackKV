# BRISK-KV Hardware Status — 2026-08-17

This document freezes the first bounded hardware optimization after the
`2026081502` dual JIT-V overlap-v1 baseline.  The frozen candidate is named:

```text
JIT-V dual + pipelined byte-stream replay v1
short name: replay_pipe_v1
```

The compression algorithm, BRISK-KV Format v0, write-side quantizer, four
bucket routing, JIT-V accumulator, geometry, clock target, workload and SRAM
black-box policy are unchanged.  Only the resident byte-stream replay
microarchitecture is changed.  Do not overwrite any path recorded below;
future variants must use a new RTL and result directory.

This is an experiment/artifact freeze, not a Git commit.  At the time of this
snapshot the Chisel implementation and its tests remain modified in the
working tree.

## 1. Comparison boundary

The retained pre-optimization baseline remains:

```text
name            = JIT-V dual overlap-v1
SAIF            = hardware/simulation/results/2026081502/
                  jit_v_1024t_128f_attention.saif
DC              = hardware/synthesis/dc/results/2026081502/period_2p0ns/
hier power      = hardware/synthesis/dc/results/2026081503/
                  power_hier_from_2026081502/
attention cycle = 395923
```

The frozen optimized candidate is:

```text
name            = replay_pipe_v1
top             = BriskKvJitVSingleHeadTileTop
architecture    = jit_v_dual
clock           = 2.0 ns / 500 MHz
tokens          = 1024
feature_dim     = 128
activity phase  = attention only
backpressure    = periodic
stats for PPA   = off
memory policy   = architectural SRAM black box; small queues synthesized
attention cycle = 278843
```

Both points use the same TSMC 28 nm HPC+ LVT SSG 0.81 V / 125 C library,
ZeroWireload pre-layout flow, deterministic workload and SAIF hierarchy
`tb_briskkv_jit_v_power_1024/dut`.

## 2. Frozen implementation

Primary source:

```text
hardware/chisel/src/main/scala/briskkv/ReplayByteStreamBuffer.scala
```

The old serialized `readPending`/single response register is replaced by:

- a two-entry `UInt(8.W)` response queue;
- separate issued and delivered read counters;
- up to two outstanding synchronous SRAM reads;
- one delivered byte per cycle after priming when the consumer remains ready;
- preserved byte order, seal/clear/overflow behavior and arbitrary downstream
  backpressure.

The eleven architectural resident byte-stream SRAMs and their capacities are
unchanged.  Eleven 2 x 8-bit response queues add 176 small logic-memory bits:

```text
baseline logic memory = 2 instances / 948 bits
replay_pipe_v1        = 13 instances / 1124 bits
architectural SRAM    = 52 instances / 1659392 bits (unchanged)
```

Frozen generated RTL:

```text
hardware/rtl/generated/
  briskkv_jit_v_dual_v1_t1024_f128_replay_pipe_v1/full/
hardware/rtl/generated/
  briskkv_jit_v_dual_v1_t1024_f128_replay_pipe_v1/dc_logic/
```

Use `full/` only for RTL simulation and activity generation.  Use
`dc_logic/` only for logic synthesis with architectural SRAM black boxes.

## 3. Functional and cycle evidence

Module and tile tests recorded while implementing this candidate:

- `ReplayByteStreamBufferSpec`: 4 tests passed, including one-byte-per-cycle
  delivery and irregular backpressure;
- dual and shared `BriskKvJitVSingleHeadTileTopSpec`: passed in separate runs;
- `BriskKvSingleHeadTileTopSpec`: 2 tests passed;
- 1024 x 128 stats-enabled cycle benchmark: passed with the same complete
  output checksum as the baseline.

Cycle CSV:

```text
hardware/evaluation/results/cycle_breakdown/
  2026081504_pipelined_replay/jit_v_dual.csv
```

Key matched counters:

| Metric | `2026081503_stats_on` baseline | `replay_pipe_v1` |
|---|---:|---:|
| Write cycles | 1174733 | 1174733 |
| Attention cycles | 395923 | 278843 |
| K output values | 131072 | 131072 |
| V output values | 131072 | 131072 |
| V packets | 8192 | 8192 |
| JIT-V MAC operations | 131072 | 131072 |
| Output count | 128 | 128 |
| Full output checksum | identical | identical |

The attention latency reduction is 117080 cycles, or 29.5714%, corresponding
to a 1.41988x attention throughput improvement at the same clock.

The uploaded local archive does not contain the remote VCS `simulation.log`
for the optimized workload.  Therefore an optimized remote `POWER PASS`
cannot be re-read locally from a log.  The local Chisel equivalence result,
the archived optimized SAIF and the subsequent DC reports are locally
verifiable evidence.

## 4. Frozen SAIF

File:

```text
hardware/simulation/results/2026081503/
  jit_v_1024t_128f_attention.saif
```

Validated properties:

- `TIMESCALE 1 ps`;
- `DURATION 557688000` ps;
- hierarchy `tb_briskkv_jit_v_power_1024/dut`;
- 77 clock entries, all with `T0=T1=278844000`, `TX=0`, `TC=557688`;
- 77767 activity entries, all satisfying `T0 + T1 + TX = DURATION`.

The duration uses the same one-clock activity-window boundary convention as
the `2026081502` baseline and agrees with the 278843-cycle attention result.

## 5. Frozen 2.0 ns DC result

Directory:

```text
hardware/synthesis/dc/results/2026081504/period_2p0ns/
```

The DC log confirms the `replay_pipe_v1/dc_logic` RTL, stats disabled, the
same top/library/corner, 52 preserved architectural SRAM instances, successful
completion and SAIF instance `tb_briskkv_jit_v_power_1024/dut`.

SAIF annotation remains comparable to the baseline:

| Postcompile activity | `2026081502` | `replay_pipe_v1` |
|---|---:|---:|
| User-annotated nets | 9.66% | 9.62% |
| Propagated nets | 86.15% | 86.12% |
| Default nets | 4.18% | 4.26% |
| User-annotated ports | 100.00% | 100.00% |
| User-annotated pins | 17.61% | 17.63% |

There is no `PWR-362` zero-annotation error.  The optimized run has 25997
unmatched RTL objects versus 26085 in the baseline and retains the same single
`PWR-19` switching-activity conflict.

Matched PPA result:

| Metric | `2026081502` | `replay_pipe_v1` | Change |
|---|---:|---:|---:|
| Setup WNS/TNS | 0.00/0.00 ns | 0.00/0.00 ns | unchanged |
| Critical path | 1.89 ns | 1.88 ns | -0.01 ns |
| Worst hold | about -0.06 ns | about -0.06 ns | unchanged boundary |
| Hold TNS | about -622.26 ns | about -626.74 ns | -4.48 ns |
| Max-transition violations | 1 | 0 | removed |
| Cell area | 140452.702774 um^2 | 141438.694767 um^2 | +0.7020% |
| Total dynamic power | 25.4163 mW | 25.8113 mW | +1.5541% |
| Leakage power | 41.5986 mW | 41.8824 mW | +0.6822% |
| Total cell power | 67.0155 mW | 67.6944 mW | +1.0130% |

Multiplying matched SAIF power by the attention activity duration gives the
logic-only per-transaction estimate:

```text
dynamic energy: 20.1258 uJ -> 14.3947 uJ (-28.48%)
total cell energy: 53.0661 uJ -> 37.7524 uJ (-28.86%)
area-delay product change: -29.08%
```

## 6. Hierarchical power

Optimized hierarchical report:

```text
hardware/synthesis/dc/results/2026081701/power_hier.rpt
```

`dc_report.log` confirms that this report reloads the `2026081504` DDC.
`power_check.rpt` reproduces 25.8113 mW dynamic power; the 0.0015 mW total
power difference from the original report is report re-propagation/rounding.
The retained SAIF coverage is identical to the original optimized DC run.

| Retained hierarchy | Baseline total | `replay_pipe_v1` total | Change |
|---|---:|---:|---:|
| V decoder | 4.419 mW | 4.438 mW | +0.019 mW |
| K decoder | 4.475 mW | 4.494 mW | +0.019 mW |
| Streaming Softmax | 9.085 mW | 9.093 mW | +0.008 mW |
| QK accumulator | 8.950 mW | 9.027 mW | +0.077 mW |
| JIT-V accumulator | 7.273 mW | 7.353 mW | +0.080 mW |
| Writer | 15.351 mW | 15.349 mW | -0.002 mW |
| Top-local and unlisted logic | about 17.462 mW | about 17.940 mW | +0.478 mW |

Approximately 70.4% of the 0.679 mW total average-power increase is in
top-local/unlisted logic, consistent with the location of the eleven replay
queues and their control.  DC flattened the replay-buffer hierarchy, so this
report cannot attribute the complete residual specifically to the response
queues; exact queue-only power is currently unconfirmed.

Every retained hierarchy nevertheless reduces per-transaction energy because
the activity window is 29.57% shorter.  The total logic-only energy reduction,
not the small average-power increase, is the primary result of this bounded
optimization.

## 7. Artifact hashes

The two RTL tree digests below are SHA-256 over the sorted per-file SHA-256
listing rooted at the corresponding export directory.

```text
089f1c72a14f104dbfa7dd137881a28ed33822a3c47f0e0597b553408c565bc4  ReplayByteStreamBuffer.scala
cf0617df67186b7ed7316704a87d10d7737e36391a46058e2234e88766779066  ReplayByteStreamBufferSpec.scala
408c3ae65b178ad32a2e674cc54ff4a8c15e3f943a4a5cca3d8e347d2cb15fba  BriskKvCycleBenchmarkSpec.scala
32d2a97e8e2917f18804ade3e5ac4551f502621fc3add59ebd25abe5521c2a9a  full RTL tree
0eb0d95acc8bf42a9a4923693a2c035dc9bcf2525736dffbb8066d42ccb5cd3b  dc_logic RTL tree
56966cd301c2f27448e855d02c11fb9b72da66119e5853650f8d787f8a8557f5  full/manifest.json
d582655a761be344bdc761289ca77185eb2c824572d4a8474d925404710457d3  full/filelist.f
126efd5e801d33d1765bab5234256bcfc5b39cf4e7883a81ffc826186bddf8a1  dc_logic/manifest.json
d582655a761be344bdc761289ca77185eb2c824572d4a8474d925404710457d3  dc_logic/filelist.f
484df09d6041de74c7f2d30874d8dd25f12c9fecd8c6b0966fc9f51ec90d7a1f  jit_v_dual.csv
830eddf35f89a611a8938ec958bb11649ab84f13aee3446dc5a11e3f0db05efa  optimized attention SAIF
011cea5e380f67287859fee3e846c509748347efdefcac106fdf63f3b28027cf  power.rpt
dcb61b363bdd90231fdac095efe761722c2476625fbcf57611fbde52001c009f  qor.rpt
6912361468359a0f68e5241c06c525c733ec8f996eec0e0803cc675418ba6238  area_hier.rpt
2980fe532abea52e694df399fbe053bdffe445379cf09bbef3ff9c0dd269799a  saif_annotation_postcompile.rpt
210a65f9ecf068f0bfad613f6e93d33990306dbfbb9b13c73345b9b625f9d70d  power_hier.rpt
```

## 8. Interpretation boundary and next gate

The evidence supports the following statement:

> In the deterministic 1024-token x 128-feature single-head attention
> workload at 500 MHz, pipelined replay reduces attention cycles by 29.57%
> and logic-only transaction energy by approximately 28.86%, at a 0.70%
> standard-cell-area and 1.01% average logic-power cost relative to the frozen
> overlap-v1 baseline.

It does not establish final chip power or end-to-end LLM speedup because:

- architectural SRAM internal dynamic/leakage power is outside DC;
- DC is pre-layout ZeroWireload with no CTS, routing or parasitics;
- activity is RTL SAIF, not gate-level SDF activity;
- the workload is one deterministic single-head transaction;
- multi-head/layer scheduling, DMA/NoC/DRAM and a complete cache manager remain
  outside the implemented tile.

`replay_pipe_v1` is now frozen as the retained dual JIT-V implementation for
the next matched architecture comparisons.  Do not make another replay or
multi-lane change before completing the same-workload Full-V, JIT-V dual and
JIT-V shared comparison.

## 9. Frozen matched Full-V / dual / shared comparison

The matched architecture comparison requested above is complete. All three
points use a 1024-token x 128-feature deterministic single-head workload,
2.0 ns clock, stats-off PPA RTL, attention-only SAIF, the same TSMC 28 nm
library/corner and architectural SRAM black boxes.

Cycle evidence:

```text
hardware/evaluation/results/cycle_breakdown/
  2026081702_matched_replay_pipe_v1/cycle_benchmark_summary.csv
```

Power/PPA evidence:

```text
Full-V DC            hardware/synthesis/dc/results/2026081704/period_2p0ns/
Full-V hier power    hardware/synthesis/dc/results/2026081706/
dual hier power      hardware/synthesis/dc/results/2026081701/
shared DC            hardware/synthesis/dc/results/2026081705/period_2p0ns/
shared hier power    hardware/synthesis/dc/results/2026081707/
```

The Full-V and shared VCS passes are user-confirmed terminal evidence; their
remote `simulation.log` files are not present in the local archive. Their SAIF
and all listed DC reports are locally available and were checked.

| Metric | Full-V | JIT-V dual | JIT-V shared |
|---|---:|---:|---:|
| Attention cycles, periodic | 164447 | 278843 | 278859 |
| Cell area, um^2 | 145161.574572 | 141438.694767 | 134515.414817 |
| Dynamic power, mW | 26.1802 | 25.8113 | 23.6411 |
| Leakage power, mW | 44.0547 | 41.8824 | 39.7520 |
| Total cell power, mW | 70.2353 | 67.6944 | 63.3933 |
| Architectural SRAM bits | 4012288 | 1659392 | 1659392 |
| Setup WNS/TNS | 0.00/0.00 ns | 0.00/0.00 ns | 0.00/0.00 ns |
| Worst hold | about -0.06 ns | about -0.06 ns | about -0.06 ns |
| Max-transition violations | 0 | 0 | 0 |

The shared point adds only 16 attention cycles relative to dual while reducing
cell area by 4.89%, average dynamic power by 8.41% and total cell power by
6.35%. Hierarchical power attributes about 3.416 mW, or 79.4% of the 4.301 mW
total reduction, to replacing the dual decoder pair with one decoder plus the
V metadata adapter. Shared is therefore the retained area/power-first JIT-V
architecture; dual remains 16 cycles faster and is not claimed to be strictly
dominated.

Full-V remains the latency and logic-only transaction-energy endpoint but uses
2.418x the JIT-V SRAM bits. The SRAM black boxes have no modeled internal
dynamic/leakage power in DC; CACTI 22 nm and DC 28 nm results must remain
separate and must not be summed as matched chip PPA.

The attention-only hierarchical reports expose a common idle-write hotspot:
all three variants report about 7.774 mW writer internal power even though
writer switching is essentially zero. In shared, writer total power is
15.349 mW, or 24.2% of total logic power. This motivates the next bounded
phase-level writer clock-gating ablation before any 1/2/4-lane decoder study.

## 10. Shared writer phase-clock-gating candidate

Candidate name:

```text
shared_writer_cg_v1
top = BriskKvSharedJitVWriterCgSingleHeadTileTop
```

This is not yet a frozen PPA result. It is an isolated implementation and
local functional/cycle candidate awaiting remote VCS, attention-only SAIF and
matched 2.0 ns DC.

Implementation:

- `BriskKvClockGate` is a technology-independent low-level latch-and-AND
  glitch-free gate used consistently by Chisel, VCS and DC;
- writer clock enable is asserted for reset, a valid write launch and the
  complete `sWriting` phase;
- the writer clock is held stationary throughout stored/launch/attention
  phases;
- the original `BriskKvSharedJitVSingleHeadTileTop` remains ungated;
- compression, format, decoder, replay, SRAM capacities and attention datapath
  are unchanged.

Local evidence:

- gated Shared small end-to-end Chisel test: passed;
- 1024 x 128 cycle benchmark, none and periodic backpressure: passed;
- write cycles, every attention counter, output count and complete checksum
  are exactly identical to the ungated shared result;
- periodic attention cycles remain 278859;
- exported stats-off full RTL passes Verilator lint.

Cycle CSV:

```text
hardware/evaluation/results/cycle_breakdown/
  2026081703_shared_writer_cg/jit_v_shared_writer_cg.csv
```

Generated RTL, kept separate from the ungated shared baseline:

```text
hardware/rtl/generated/
  briskkv_jit_v_shared_v1_t1024_f128_replay_pipe_v1_writer_cg_vcs2018/full/
hardware/rtl/generated/
  briskkv_jit_v_shared_v1_t1024_f128_replay_pipe_v1_writer_cg_vcs2018/dc_logic/
```

Both manifests report one decompression datapath, `writer_clock_gating=true`,
stats disabled, VCS compatibility enabled, 52 architectural SRAM instances /
1659392 bits and 13 small logic-memory instances / 1124 bits.

Artifact hashes at this candidate boundary:

```text
f186d4d233f9c7e47665eac4487377e0659fd5aa9aac66b95db5ce46ed8628bc  BriskKvClockGate.scala
b2017378048cfdcbdcdbd5af8a5ffd8f0a48b00c5d911c9cb394c06815108ee3  BriskKvJitVSingleHeadTileTop.scala
7f88542c6032a9ca98c367584b71f712ff0e9e22df873b2809041b205aa30b74  jit_v_shared_writer_cg.csv
4219fe6c9b914775c3459c19366e95bfc4225ada210dd6c772ae7569060a3493  full RTL tree
326cdb78bb584804e232718d87090683f334663b30a756b5ed8e2565e5fbc82f  dc_logic RTL tree
```

The next acceptance gate is a user-confirmed remote VCS PASS followed by an
attention-only SAIF using hierarchy `tb_briskkv_tile_power_1024/dut`. The SAIF
must be applied only to the matching writer-CG `dc_logic/` RTL. Compare against
the ungated shared `2026081705` DC and `2026081707` hierarchical power results;
do not overwrite either baseline.
