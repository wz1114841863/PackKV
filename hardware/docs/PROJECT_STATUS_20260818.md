# BRISK-KV Hardware Status — 2026-08-18

> Current hardware freeze. For paper thesis, claim boundaries, table planning,
> and the recommended new-conversation prompt, read
> `PAPER_HANDOFF_20260818.md` first.

This document freezes the retained single-head area/power-first JIT-V point:

```text
JIT-V shared + pipelined byte-stream replay v1 + phase-gated writer
short name: shared_writer_cg_v1
top: BriskKvSharedJitVWriterCgSingleHeadTileTop
```

It supersedes the pending-candidate status in section 10 of
`PROJECT_STATUS_20260817.md`; it does not replace or permit overwriting the
`2026081502` pre-replay baseline, the frozen dual `replay_pipe_v1`, or the
ungated shared comparison point. This is an experiment/artifact freeze, not a
Git commit or physical-design signoff.

## 1. Matched boundary

```text
clock            = 2.0 ns / 500 MHz
tokens           = 1024
feature_dim      = 128
backpressure     = periodic for the primary cycle row
activity phase   = attention only
stats for PPA    = off
technology       = TSMC 28 nm HPC+ LVT, SSG 0.81 V / 125 C
wire model       = ZeroWireload
memory policy    = architectural SRAM black box; small queues synthesized
SAIF hierarchy   = tb_briskkv_tile_power_1024/dut
```

The compression algorithm, BRISK-KV Format v0, write quantizer v1, four-bucket
stable routing, replay pipeline, shared decoder, accumulator, SRAM capacities,
geometry and attention schedule are unchanged from the ungated shared point.
Only the writer clock is disabled outside reset, legal write launch and the
active write phase.

## 2. Functional and cycle evidence

Local Chisel evidence:

- gated shared small end-to-end test passed;
- ungated shared regression passed;
- 1024 x 128 none/periodic cycle benchmark passed;
- every cycle counter, output count and complete checksum matches ungated
  shared exactly;
- stats-off exported full RTL passed Verilator lint.

Cycle CSV:

```text
hardware/evaluation/results/cycle_breakdown/
  2026081703_shared_writer_cg/jit_v_shared_writer_cg.csv
```

| Backpressure | Write cycles | Attention cycles | Output count |
|---|---:|---:|---:|
| none | 1174733 | 278857 | 128 |
| periodic | 1174733 | 278859 | 128 |

The local archive contains the resulting attention-only SAIF but not the
remote VCS `simulation.log`. Therefore the SAIF and downstream DC reports are
locally verifiable, while a remote VCS PASS cannot be independently re-read
from the current archive.

## 3. Frozen RTL and clock-gating implementation

```text
hardware/rtl/generated/
  briskkv_jit_v_shared_v1_t1024_f128_replay_pipe_v1_writer_cg_vcs2018/full/
hardware/rtl/generated/
  briskkv_jit_v_shared_v1_t1024_f128_replay_pipe_v1_writer_cg_vcs2018/dc_logic/
```

Both manifests record:

```text
architecture             = jit_v_shared_writer_cg
writer_clock_gating      = true
decompression_datapaths  = 1
performance_stats        = false
VCS compatibility        = true
architectural SRAM       = 52 instances / 1659392 bits
small logic memory       = 13 instances / 1124 bits
```

`BriskKvClockGate` implements a technology-independent low-phase latch plus
clock AND. DC maps it to one `LNQD1BWP12T30P140LVT` latch and one
`AN2D0BWP12T30P140LVT` AND cell. This is a consistent Chisel/VCS/DC functional
model, not a final foundry ICG/CTS implementation.

## 4. Frozen SAIF

```text
hardware/simulation/results/2026081801/
  jit_v_shared_writer_cg_1024t_128f_attention.saif
```

Validated properties:

- `TIMESCALE 1 ps`;
- `DURATION 557720000` ps;
- hierarchy `tb_briskkv_tile_power_1024/dut`;
- 74031 activity entries and zero `T0 + T1 + TX` duration mismatches;
- source clock `TC=557720`, `TX=0`;
- `writerClock_gate/clockOut`, `enable` and `enableLatched` all have `TC=0`,
  `TX=0` for the entire attention window.

The duration follows the same one-clock activity-window convention as the
ungated shared point and agrees with 278859 periodic attention cycles.

## 5. Frozen 2.0 ns DC and hierarchical power

```text
SAIF-annotated DC: hardware/synthesis/dc/results/2026081801/
DDC hier power:    hardware/synthesis/dc/results/2026081802/
ungated baseline: hardware/synthesis/dc/results/2026081705/period_2p0ns/
ungated hier:     hardware/synthesis/dc/results/2026081707/
```

DC compiled the matching writer-CG `dc_logic/` RTL and read the matching SAIF
at `tb_briskkv_tile_power_1024/dut`. It completed successfully with all 52
architectural SRAM instances preserved. Postcompile SAIF annotation is 9.28%
user / 86.71% propagated / 4.01% default for nets, with 100% annotated ports.
There is no `PWR-362`; the 25214 unmatched objects and one `PWR-19` conflict
match the ungated shared boundary.

| Metric | Ungated shared | `shared_writer_cg_v1` | Change |
|---|---:|---:|---:|
| Attention cycles | 278859 | 278859 | unchanged |
| Critical path | 1.89 ns | 1.89 ns | unchanged |
| Setup WNS/TNS | 0.00/0.00 ns | 0.00/0.00 ns | unchanged |
| Worst hold | about -0.06 ns | about -0.06 ns | unchanged boundary |
| Hold TNS | about -566.16 ns | about -566.26 ns | -0.10 ns |
| Max-transition violations | 0 | 0 | unchanged |
| Cell area | 134515.414817 um^2 | 134594.710815 um^2 | +0.05895% |
| Internal power | 23.3221 mW | 15.5158 mW | -33.47% |
| Switching power | 0.31896 mW | 0.29566 mW | -7.30% |
| Total dynamic power | 23.6411 mW | 15.8115 mW | -33.12% |
| Leakage power | 39.7520 mW | 39.9662 mW | +0.54% |
| Total cell power | 63.3933 mW | about 55.777 mW | -12.01% |

The DDC reload reproduces total power within about 0.0002 mW and retains the
same SAIF coverage.

Hierarchical attribution:

```text
writer total:    15.349 mW -> 7.698 mW
writer internal:  7.774 mW -> about 0.000005 mW
```

Writer leakage remains and rises slightly with mapping variation. The roughly
7.651 mW writer total-power reduction accounts for essentially all of the
roughly 7.616 mW whole-design reduction; small changes elsewhere offset about
0.035 mW. The activity and hierarchy therefore support causal attribution to
phase-level writer clock gating rather than SAIF mismatch.

At the unchanged activity duration, logic-only attention energy is:

```text
dynamic energy:   13.185 uJ -> 8.818 uJ  (-33.12%)
total cell energy: 35.356 uJ -> 31.108 uJ (-12.01%)
```

## 6. Artifact hashes

```text
7f88542c6032a9ca98c367584b71f712ff0e9e22df873b2809041b205aa30b74  jit_v_shared_writer_cg.csv
a89c514102ea3cbaae663d450e47a11919728d374bc426b2255376f0f8de61ac  attention SAIF
e758ec0507b86552e7fecf02591a247beb1e90bd6e6f7ad5711802d9536b5187  qor.rpt
436167d1a82ad10d627206736ed79dccc8811e292fbc393d443b87cd93b3d09b  power.rpt
49d54e1e96762a5327a6a6ee825f86afe9b6754c2e7b107f9708571047d5a4b3  area_hier.rpt
4c1e4ebe8a48d1512d2e7175ad3f9869760ebb3dfb83debb5f25059d9b541b55  saif_annotation_postcompile.rpt
69fb2d1ff9c3ada1075f61066c3c5cc27263df7b2845150682fadc3094e528bc  writer-CG DDC
ede36d6ef6831fe59013d0983dc337f48fc3505afed8b03be3954285fb3b997b  power_hier.rpt
23b7e534f648bb138a7073eeafe6d43f2611c282985e52a7cce51270896183b6  power_check.rpt
```

## 7. Interpretation boundary and retained point

The supported result is:

> In the deterministic 1024-token x 128-feature single-head attention
> workload at 500 MHz, phase-gating the idle write encoder preserves the
> 278859-cycle shared-JIT-V attention latency while reducing average logic
> dynamic power by 33.12% and logic-only total transaction energy by about
> 12.01%, at a 0.059% standard-cell-area cost relative to ungated shared.

`shared_writer_cg_v1` is the retained single-head area/power-first JIT-V point.
Do not overwrite its RTL, cycle CSV, SAIF, DC or DDC directories.

This result does not establish final chip power or full LLM speedup because:

- architectural SRAM internal dynamic/leakage power is outside DC;
- the flow is pre-layout ZeroWireload with ideal clocks and no CTS/parasitics;
- the gate is not yet replaced by a characterized foundry ICG;
- the activity window is attention-only, not write or combined transaction;
- the workload is one deterministic single-head transaction;
- multi-head/layer scheduling, DMA/NoC/DRAM and cache-window management remain
  outside the implemented tile.

Write-only, combined-phase, extra workload and physical-clock experiments are
deferred supporting experiments for paper preparation; they are not required
to keep changing the current single-head RTL boundary.

The next phase is documentation and paper construction. Begin with a
claim-evidence-limitation matrix and an outline; do not reopen datapath or
multi-lane optimization unless a concrete paper gap requires a new experiment.
