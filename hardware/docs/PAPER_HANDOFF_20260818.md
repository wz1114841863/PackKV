# BRISK-KV paper handoff — 2026-08-18

Documentation updated on 2026-08-20; the hardware evidence boundary remains
the 2026-08-18 freeze.

This is the recommended first document for a new paper-writing conversation.
It summarizes the research thesis, frozen hardware result, evidence boundary,
paper structure, and remaining work. Exact report details and hashes are in
`PROJECT_STATUS_20260818.md`.

## 1. One-sentence thesis

BRISK-KV constrains PackKV-style compression into hardware-friendly
power-of-two quantization, stable four-bucket repacking, and an explicit
dynamic bit-packed stream format, then fuses write-side compression, resident
compressed storage, decompression, and Attention in a single-head tile whose
Full-V/JIT-V variants expose a measurable storage-latency-power Pareto.

The contribution is the **algorithm-format-architecture co-design and
cross-layer evidence**, not the isolated novelty of powers of two or bucket
sorting.

## 2. Frozen algorithm and format

```text
scale method       = po2_nearest
K relative scale   = 0.03
V relative scale   = 0.10
zero point         = integer
repacking          = stable BUCKET
bucket count       = 4
bucket score       = k_sum
block size         = 64 tokens
pack size          = 16 tokens
software recent window = 192 tokens
```

BRISK-KV Format v0 uses eleven resident component streams: five K streams,
five V streams, and one bucket-count stream. K/V and token-wise quantization
metadata share one stable permutation. The format is a component-stream
contract; a complete DMA/container header is not implemented.

## 3. Frozen retained hardware point

```text
short name      = shared_writer_cg_v1
top             = BriskKvSharedJitVWriterCgSingleHeadTileTop
architecture    = shared JIT-V + replay_pipe_v1 + phase-gated writer
tokens/features = 1024 x 128
clock           = 2.0 ns / 500 MHz
activity        = attention-only, periodic output backpressure
technology      = TSMC 28 nm HPC+ LVT, SSG 0.81 V / 125 C
flow            = RTL SAIF -> DC Ultra, ZeroWireload
memory policy   = architectural SRAM black boxes
```

The implementation contains the complete single-head research path from raw
Q12 K/V write input through compression, eleven stored byte streams,
decompression, QK, fixed-point scaling, streaming Softmax, AV, and Q6 output.
It is not a multi-head/multi-layer accelerator or a full memory system.

## 4. Matched architecture evidence

All rows use the same 1024 x 128 deterministic single-head workload, 2.0 ns
clock, stats-off RTL, attention-only SAIF, library/corner, and SRAM-black-box
policy.

| Metric | Full-V | JIT-V dual | JIT-V shared | Shared writer-CG |
|---|---:|---:|---:|---:|
| Attention cycles | 164447 | 278843 | 278859 | 278859 |
| Cell area (um^2) | 145161.574572 | 141438.694767 | 134515.414817 | 134594.710815 |
| Dynamic power (mW) | 26.1802 | 25.8113 | 23.6411 | 15.8115 |
| Leakage power (mW) | 44.0547 | 41.8824 | 39.7520 | 39.9662 |
| Total cell power (mW) | 70.2353 | 67.6944 | 63.3933 | about 55.777 |
| Architectural SRAM bits | 4012288 | 1659392 | 1659392 | 1659392 |
| Setup WNS/TNS | 0/0 ns | 0/0 ns | 0/0 ns | 0/0 ns |
| Worst hold | about -0.06 ns | about -0.06 ns | about -0.06 ns | about -0.06 ns |
| Max-transition violations | 0 | 0 | 0 | 0 |

Interpretation:

- JIT-V removes 2,352,896 SRAM bits, or 58.64%, relative to Full-V, but the
  retained shared implementation takes about 69.6% more attention cycles.
- Sharing the decompressor adds only 16 cycles relative to dual while reducing
  cell area by 4.89%, dynamic power by 8.41%, and total cell power by 6.35%.
- Phase-gating the writer preserves shared latency, reduces dynamic power by
  33.12% and logic-only energy by about 12.01%, at 0.059% area overhead.
- Full-V remains the latency endpoint. Shared writer-CG is the retained
  area/power-first JIT-V endpoint. Neither is claimed universally optimal.

The writer-CG mapped clock gate is a latch-plus-AND model, not a characterized
foundry ICG or CTS result.

## 5. Evidence chain and artifact map

| Layer | Evidence | Location |
|---|---|---|
| Algorithm/CR | three-model local summaries and software path | `docs/MY_IDEAS.md`, `utils/`, `evaluation/` |
| Format | normative fields and validation rules | `hardware/docs/briskkv_format_v0.md` |
| Golden reference | deterministic Python vectors | `hardware/golden_vectors/` |
| RTL correctness | ScalaTest/ChiselSim exact and backpressure tests | `hardware/chisel/src/test/scala/briskkv/` |
| Cycles | matched 1024 x 128 CSVs | `hardware/evaluation/results/cycle_breakdown/` |
| Full RTL | generated behavioral-memory RTL | `hardware/rtl/generated/*/full/` |
| Activity | architecture-matched attention-only SAIF | `hardware/simulation/results/` |
| Logic PPA | DC reports with SRAM black boxes | `hardware/synthesis/dc/results/` |
| SRAM capacity/model | manifests, `memories.csv`, CACTI | `hardware/evaluation/mem/` |

Current retained artifact roots:

```text
cycle: hardware/evaluation/results/cycle_breakdown/2026081703_shared_writer_cg/
SAIF:  hardware/simulation/results/2026081801/
DC:    hardware/synthesis/dc/results/2026081801/
DDC:   hardware/synthesis/dc/results/2026081802/
```

## 6. Claims the current evidence supports

1. A hardware-oriented restriction of PackKV-style compression can be
   expressed as a synthesizable, explicitly validated component-stream format.
2. The repository implements and verifies an end-to-end single-head
   compression-to-Attention tile, not only isolated codecs.
3. JIT-V trades attention latency for a 58.64% reduction in architectural SRAM
   bits at the evaluated 1024 x 128 geometry.
4. Decoder sharing provides an area/power improvement with negligible
   additional latency relative to the dual JIT-V implementation.
5. Phase-aware clock gating removes an attention-phase idle-writer hotspot
   with unchanged cycle behavior and small area overhead.
6. Python, Chisel, VCS, SAIF, DC, and CACTI form a cross-layer validation
   methodology, subject to the stated technology and modeling limits.

## 7. Claims the paper must not make

- BRISK-KV is not yet a complete multi-head/multi-layer LLM accelerator.
- No matched experiment currently proves end-to-end speedup or energy savings
  over the GPU PackKV implementation.
- DC power excludes architectural SRAM internal power and is pre-layout.
- CACTI 22 nm and DC 28 nm results are separate contexts and are not directly
  additive.
- One deterministic attention workload is not a distributional power study.
- Attention-only activity is not whole-request or write-plus-attention power.
- Current algorithm parameters are retained candidates, not universal optima.

## 8. Recommended paper structure

1. **Motivation and problem** — GPU-oriented irregularity versus predictable
   edge-accelerator dataflow and SRAM pressure.
2. **Background and gap** — PackKV flow, what is retained, and what BRISK-KV
   changes for hardware.
3. **Algorithm-format co-design** — power-of-two scales, stable four buckets,
   metadata, bit packing, and decode invariants.
4. **Architecture** — writer, eleven compressed streams, replay/decompressor,
   QK/Softmax/AV, Full-V and JIT-V variants.
5. **Optimizations** — replay pipeline, decoder sharing, phase-level writer
   clock gating.
6. **Methodology** — software accuracy/CR, golden vectors, Chisel/VCS, cycle
   benchmark, SAIF/DC, SRAM model, and all boundaries.
7. **Results** — accuracy/CR, storage, cycle, logic PPA, hierarchical power,
   and Pareto analysis.
8. **Limitations and future system integration**.

Suggested core figures/tables:

- PackKV-to-BRISK-KV algorithm/format/architecture mapping;
- single-head tile block diagram and phase timeline;
- Full-V versus JIT-V storage organization;
- matched architecture PPA/cycle table;
- ablation waterfall: baseline -> replay -> shared -> writer gating;
- accuracy-versus-compression table/plot with explicit FP baseline;
- claim/evidence/limitation table.

## 9. Work to do in the paper phase

Required before a polished submission:

- reconstruct exact software experiment provenance and include the true FP and
  PackKV baselines in the accuracy/CR tables;
- define comparison rules for GPU PackKV and related KV-compression hardware;
- produce figures and tables directly from archived CSV/report values;
- write a methodology section that separates logic, SRAM, cycles, and power;
- state the single-head/system-scaling assumptions explicitly;
- turn artifact paths and hashes into a reproducibility appendix.

Useful but deferred supporting experiments:

- write-only and combined-phase SAIF;
- additional token lengths, feature dimensions, workloads, and real-model
  traces;
- matched SRAM technology or foundry memory compiler results;
- foundry ICG, P&R/CTS and gate-level activity;
- multi-head scheduling and DMA/NoC/cache-manager prototypes.

Do not block the start of writing on these deferred experiments. First build a
claim-evidence matrix and paper outline from the frozen boundary; add only
experiments that close a specific reviewer-facing gap.

## 10. Recommended opening prompt for a new conversation

```text
请先阅读 AGENTS.md、hardware/docs/PAPER_HANDOFF_20260818.md、
hardware/docs/PROJECT_STATUS_20260818.md、hardware/docs/JIT_V_ABLATION.md
和 docs/MY_IDEAS.md，并检查 git status。当前硬件边界已经冻结，不修改算法
或 RTL。请先为 BRISK-KV 论文建立 claim-evidence-limitation 矩阵和详细提纲，
所有数值必须指向归档证据，无法确认的内容明确标注。
```
