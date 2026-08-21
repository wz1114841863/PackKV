# BRISK-KV Paper Claim-Evidence-Limitation Matrix and Outline

Date: 2026-08-20

This document is a paper-preparation working draft built from the frozen
2026-08-18 single-head hardware boundary. It does not reopen algorithm or RTL
optimization. All numeric claims below must remain tied to archived evidence;
claims without current evidence are explicitly marked `无法确认`.

## 1. Evidence Boundary

Frozen retained point:

- Short name: `shared_writer_cg_v1`
- Top: `BriskKvSharedJitVWriterCgSingleHeadTileTop`
- Workload: 1024 tokens x 128 features, deterministic single-head attention
- Clock: 2.0 ns / 500 MHz
- Activity: attention-only SAIF, periodic output backpressure for primary PPA
- Synthesis: TSMC 28 nm HPC+ LVT, SSG 0.81 V / 125 C, DC Ultra, ZeroWireload
- Memory policy: architectural SRAM black boxes; small logic memories synthesized

Primary archived roots:

- Cycle CSVs: `hardware/evaluation/results/cycle_breakdown/2026081702_matched_replay_pipe_v1/` and `hardware/evaluation/results/cycle_breakdown/2026081703_shared_writer_cg/`
- Retained SAIF: `hardware/simulation/results/2026081801/jit_v_shared_writer_cg_1024t_128f_attention.saif`
- Retained DC reports: `hardware/synthesis/dc/results/2026081801/`
- Retained DDC/hierarchical power: `hardware/synthesis/dc/results/2026081802/`
- Retained RTL manifests: `hardware/rtl/generated/briskkv_jit_v_shared_v1_t1024_f128_replay_pipe_v1_writer_cg_vcs2018/{full,dc_logic}/manifest.json`
- Format contract: `hardware/docs/briskkv_format_v0.md`
- Software rationale and current summaries: `docs/MY_IDEAS.md`

## 2. Claim-Evidence-Limitation Matrix

| Claim | Evidence to cite | Supported paper wording | Limitation / forbidden extension |
|---|---|---|---|
| BRISK-KV is a hardware-constrained PackKV-style co-design, not just a codec tweak. | Algorithm parameters and thesis in `hardware/docs/PAPER_HANDOFF_20260818.md`; rationale in `docs/MY_IDEAS.md`; normative format in `hardware/docs/briskkv_format_v0.md`. | "BRISK-KV constrains PackKV-style compression into power-of-two quantization, stable four-bucket repacking, and an explicit component-stream format, then fuses it with a single-head Attention tile." | Do not claim novelty for power-of-two scaling or bucket sorting alone. Do not claim the current work is a complete multi-head/multi-layer accelerator. |
| Format v0 is explicit and hardware-oriented. | `hardware/docs/briskkv_format_v0.md`: 64-token blocks, 16-token packs, stable four-bucket FIFO, eleven component streams, fixed metadata widths, dynamic per-pack payload widths, validation/reject rules. | "Format v0 defines eleven independently aligned resident component streams with shared K/V token permutation and bounded metadata fields." | It is not a monolithic file format, DMA descriptor, or external-memory container. Stream lengths and tensor shape are supplied out of band. |
| The software algorithm point has measured local accuracy/CR summaries on three models. | `docs/MY_IDEAS.md` section "Existing software evidence": Qwen3-4B, Qwen3-8B, Llama-3.1-8B rows. | "In current local summaries, four-bucket repacking improves overall CR by 7.03%, 7.02%, and 6.07% versus NONE for Qwen3-4B, Qwen3-8B, and Llama-3.1-8B respectively, with no meaningful NONE/BUCKET accuracy movement in those summaries." | `无法确认`: accuracy relative to a true FP baseline, because exact FP provenance is not present in the summary. Do not use these rows as final paper tables until experiment directories and baselines are reconstructed. |
| The RTL implements an end-to-end single-head path from write-side compression to Attention output. | Retained top manifest reports `stored_component_streams=11`, `maximum_tokens=1024`, `maximum_feature_dim=128`, `writer_clock_gating=true`, `decompression_datapaths=1`, `total_inferred_memory_bits=1659392` in `hardware/rtl/generated/.../dc_logic/manifest.json`; implementation modules are in generated `full/` and `dc_logic/`. | "The implementation covers raw Q12 K/V write input, quantization, stable bucket routing, bit-packed resident streams, replay/decompression, QK, fixed-point scaling, streaming Softmax, AV, and Q6 output for one head." | Do not infer broader system correctness from module existence alone. Multi-head scheduling, DMA/NoC/DRAM, recent-window lifetime management, and a complete memory system are outside the RTL. |
| Full-V/JIT-V exposes a storage-latency Pareto at 1024 x 128. | Cycle CSV: `hardware/evaluation/results/cycle_breakdown/2026081702_matched_replay_pipe_v1/cycle_benchmark_summary.csv`; storage values in `hardware/evaluation/mem/README.md` and `hardware/docs/JIT_V_ABLATION.md`. | "At 1024 x 128, JIT-V reduces architectural SRAM from 4,012,288 bits to 1,659,392 bits, removing 2,352,896 bits / 58.64%, while retained shared JIT-V attention cycles increase from Full-V 164,447 to 278,859 under periodic backpressure." | This is a single geometry and deterministic workload. CACTI storage-only 22 nm estimates are not additive with 28 nm DC logic PPA. |
| Replay-pipe v1 materially reduces JIT-V latency versus the pre-replay baseline. | `hardware/docs/JIT_V_ABLATION.md` replay-pipeline ablation and provenance in `hardware/docs/PROJECT_STATUS_20260817.md`; matched post-replay cycle CSV in `hardware/evaluation/results/cycle_breakdown/2026081702_matched_replay_pipe_v1/cycle_benchmark_summary.csv`. | "At 1024 x 128 periodic, replay_pipe_v1 reduces dual JIT-V attention cycles from 395,923 to 278,843, a 29.57% reduction." | Pre-replay evidence is documented in older status files; paper table should cite the archived baseline paths before submission. This does not remove serialized V decompression. |
| Decoder sharing is an area/power-first JIT-V optimization with negligible latency cost. | Matched comparison in `hardware/evaluation/results/cycle_breakdown/2026081702_matched_replay_pipe_v1/cycle_benchmark_summary.csv`; PPA values summarized in `hardware/docs/JIT_V_ABLATION.md` and `hardware/docs/PROJECT_STATUS_20260817.md`. | "Shared JIT-V adds 16 cycles relative to dual JIT-V at 1024 x 128 periodic, while reducing cell area by 4.89%, dynamic power by 8.41%, and total cell power by 6.35%." | Shared is not strictly dominant: dual remains 16 cycles faster. The power result is logic-only, attention-only, pre-layout DC. |
| Phase-aware writer clock gating removes an attention-phase idle-writer hotspot. | Retained cycle CSV: `hardware/evaluation/results/cycle_breakdown/2026081703_shared_writer_cg/jit_v_shared_writer_cg.csv`; retained SAIF: `hardware/simulation/results/2026081801/jit_v_shared_writer_cg_1024t_128f_attention.saif`; DC reports: `hardware/synthesis/dc/results/2026081801/{qor.rpt,power.rpt,saif_annotation_postcompile.rpt}`; hierarchical power: `hardware/synthesis/dc/results/2026081802/power_hier.rpt`. | "Relative to ungated shared, writer-CG preserves 278,859 attention cycles and reduces total dynamic power from 23.6411 mW to 15.8115 mW (-33.12%), with cell area increasing from 134,515.414817 um^2 to 134,594.710815 um^2 (+0.05895%)." | The gate is a technology-independent latch-plus-AND mapped model, not a characterized foundry ICG or CTS/post-layout result. Activity is attention-only. |
| Retained writer-CG meets the frozen setup and transition boundary at 2.0 ns. | `hardware/synthesis/dc/results/2026081801/qor.rpt`: critical path 1.89 ns, setup WNS/TNS 0/0, max-transition violations 0, worst hold about -0.06 ns. | "The retained point compiles at the 2.0 ns target with 0 setup WNS/TNS and no max-transition violations under the stated DC boundary." | Hold remains a pre-layout minimum-delay boundary: worst hold about -0.06 ns and hold TNS about -566.26 ns. Do not describe this as physical timing signoff. |
| Retained writer-CG logic-only power and energy are quantified. | `hardware/synthesis/dc/results/2026081801/power.rpt`: dynamic 15.8115 mW, leakage 39.9662 mW, total 55.7771 mW; `hardware/docs/PROJECT_STATUS_20260818.md`: activity duration 557,720,000 ps and energy calculations. | "At the unchanged attention activity duration, dynamic energy drops from 13.185 uJ to 8.818 uJ and total cell energy from 35.356 uJ to 31.108 uJ." | These energies exclude architectural SRAM internal energy and non-attention phases. Do not report them as full chip or full request energy. |
| Hierarchical power supports causal attribution of the writer-CG reduction. | `hardware/synthesis/dc/results/2026081802/power_hier.rpt`: writer total 7.698 mW and near-zero writer internal power; `hardware/docs/PROJECT_STATUS_20260818.md`: ungated writer 15.349 mW total and 7.774 mW internal. | "The writer hierarchy drops from 15.349 mW to 7.698 mW, and writer internal power drops from 7.774 mW to about 0.000005 mW, accounting for essentially all whole-design power reduction." | The ungated comparison comes from archived previous reports/status. This supports attention-phase gating attribution, not a full workload average. |
| The validation stack is cross-layer. | Artifact map in `hardware/docs/PAPER_HANDOFF_20260818.md`; golden vectors in `hardware/golden_vectors/`; Chisel tests in `hardware/chisel/src/test/scala/briskkv/`; VCS/SAIF/DC paths above. | "BRISK-KV is validated across Python, deterministic vectors, Chisel/ScalaTest, cycle CSVs, SAIF, DC logic reports, hierarchical power, and SRAM modeling." | Some remote VCS logs were not copied back for local reread. The local archive verifies SAIF and downstream DC, but independent reread of the remote VCS PASS is unavailable for the retained SAIF run. |
| End-to-end GPU PackKV speedup/energy benefit is not currently supported. | Explicit boundary in `hardware/docs/PAPER_HANDOFF_20260818.md`, `hardware/docs/PROJECT_STATUS_20260818.md`, and `docs/MY_IDEAS.md`. | `无法确认`: no matched experiment currently proves end-to-end speedup or energy saving over GPU PackKV. | Do not include headline speedup/energy-over-GPU claims. The paper can compare methodology and constraints, but final quantitative GPU comparison needs new matched evidence. |
| Full-system accelerator claims are not currently supported. | Explicit boundary in `AGENTS.md`, `hardware/docs/PAPER_HANDOFF_20260818.md`, `hardware/docs/PROJECT_STATUS_20260818.md`, and `docs/MY_IDEAS.md`. | `无法确认`: multi-head/multi-layer request scheduling, DMA/NoC/cache management, DRAM traffic, and system-level throughput/energy. | Frame the artifact as a reusable single-head research tile and a component-stream format. |

## 3. Numeric Evidence Index

| Number | Meaning | Direct evidence |
|---:|---|---|
| 64 tokens | Format v0 base block size | `hardware/docs/briskkv_format_v0.md` |
| 16 tokens | Format v0 pack size | `hardware/docs/briskkv_format_v0.md` |
| 192 tokens | Software recent high-precision window, outside encoded streams | `hardware/docs/briskkv_format_v0.md`; `docs/MY_IDEAS.md` |
| 11 streams | Resident component-stream count | `hardware/docs/PAPER_HANDOFF_20260818.md`; `hardware/rtl/generated/.../dc_logic/manifest.json` |
| 1024 x 128 | Frozen evaluated geometry | `hardware/docs/PROJECT_STATUS_20260818.md`; retained manifests |
| 2.0 ns / 500 MHz | Frozen clock target | `hardware/docs/PROJECT_STATUS_20260818.md`; `hardware/synthesis/dc/results/2026081801/qor.rpt` |
| 4,012,288 bits | Full-V architectural SRAM bits | `hardware/evaluation/mem/README.md`; `hardware/docs/JIT_V_ABLATION.md` |
| 1,659,392 bits | JIT-V architectural SRAM bits | `hardware/evaluation/mem/README.md`; retained manifest |
| 2,352,896 bits / 58.64% | JIT-V SRAM reduction relative to Full-V | `hardware/evaluation/mem/README.md`; `hardware/docs/JIT_V_ABLATION.md` |
| 164,447 cycles | Full-V periodic attention cycles | `hardware/evaluation/results/cycle_breakdown/2026081702_matched_replay_pipe_v1/cycle_benchmark_summary.csv` |
| 278,843 cycles | Dual JIT-V periodic attention cycles | `hardware/evaluation/results/cycle_breakdown/2026081702_matched_replay_pipe_v1/cycle_benchmark_summary.csv` |
| 278,859 cycles | Shared and writer-CG periodic attention cycles | `hardware/evaluation/results/cycle_breakdown/2026081702_matched_replay_pipe_v1/cycle_benchmark_summary.csv`; `hardware/evaluation/results/cycle_breakdown/2026081703_shared_writer_cg/jit_v_shared_writer_cg.csv` |
| 134,594.710815 um^2 | Retained writer-CG cell area | `hardware/synthesis/dc/results/2026081801/qor.rpt` |
| 15.8115 mW | Retained writer-CG dynamic power | `hardware/synthesis/dc/results/2026081801/power.rpt` |
| 39.9662 mW | Retained writer-CG leakage power | `hardware/synthesis/dc/results/2026081801/power.rpt` |
| 55.7771 mW | Retained writer-CG total cell power | `hardware/synthesis/dc/results/2026081801/power.rpt` |
| 1.89 ns | Retained writer-CG critical path | `hardware/synthesis/dc/results/2026081801/qor.rpt` |
| 0/0 ns | Retained writer-CG setup WNS/TNS | `hardware/synthesis/dc/results/2026081801/qor.rpt` |
| about -0.06 ns | Retained writer-CG worst hold boundary | `hardware/synthesis/dc/results/2026081801/qor.rpt` |
| 0 | Retained writer-CG max-transition violations | `hardware/synthesis/dc/results/2026081801/qor.rpt` |
| 557,720,000 ps | Retained attention-only SAIF duration | `hardware/simulation/results/2026081801/jit_v_shared_writer_cg_1024t_128f_attention.saif`; summarized in `hardware/docs/PROJECT_STATUS_20260818.md` |
| 13.185 uJ -> 8.818 uJ | Logic-only dynamic attention energy, ungated shared to writer-CG | `hardware/docs/PROJECT_STATUS_20260818.md` derived from archived DC power and SAIF duration |
| 35.356 uJ -> 31.108 uJ | Logic-only total cell attention energy, ungated shared to writer-CG | `hardware/docs/PROJECT_STATUS_20260818.md` derived from archived DC power and SAIF duration |
| 15.349 mW -> 7.698 mW | Writer hierarchy total power, ungated shared to writer-CG | `hardware/docs/PROJECT_STATUS_20260818.md`; `hardware/synthesis/dc/results/2026081802/power_hier.rpt` |
| 7.774 mW -> about 0.000005 mW | Writer internal power, ungated shared to writer-CG | `hardware/docs/PROJECT_STATUS_20260818.md`; `hardware/synthesis/dc/results/2026081802/power_hier.rpt` |
| 7.03%, 7.02%, 6.07% | BUCKET overall CR gain vs NONE for Qwen3-4B, Qwen3-8B, Llama-3.1-8B | `docs/MY_IDEAS.md` |

## 4. Detailed Paper Outline

### Abstract

- Problem: GPU-oriented KV compression often leaves irregular metadata,
  ordering, and decompression behavior that are awkward for small SRAM-bound
  accelerators.
- Approach: BRISK-KV constrains PackKV-style compression into power-of-two
  scales, stable four-bucket repacking, and a component-stream format, then
  integrates write compression, resident compressed storage, replay
  decompression, and single-head Attention.
- Main measured result: cite only the frozen 1024 x 128 single-head Pareto and
  writer-CG logic-only result.
- Boundary sentence: pre-layout logic-only, attention-only, single-head; SRAM
  internal power and full-system scheduling are outside the current evidence.

### 1. Introduction

- KV cache compression is attractive for long-context inference because SRAM
  and memory traffic scale with token count and feature dimension.
- PackKV motivates computation-aware compression, but a hardware tile needs
  bounded widths, predictable routing, explicit stream metadata, and simple
  dequantization.
- Thesis: with modest algorithm restrictions, compressed KV streams can remain
  resident and feed Attention directly, exposing a controllable
  storage-latency-power Pareto.
- Contributions:
  - hardware-constrained PackKV-style algorithm point;
  - BRISK-KV Format v0 with explicit stream and validation contract;
  - single-head write-store-decompress-Attention tile;
  - Full-V/JIT-V/replay/shared/writer-CG ablation;
  - cross-layer validation stack.
- Non-goals in introduction: no full LLM accelerator, no post-layout signoff,
  no GPU speedup/energy claim.

### 2. Background and Motivation

- KV cache layout and why full V materialization dominates storage in a
  straightforward single-head tile.
- PackKV ingredients retained conceptually: quantization, repacking, bit
  packing, append-oriented cache, computation-aware decompression.
- Hardware friction points:
  - arbitrary scale multiply/dequantization;
  - global/greedy repacking and permutation management;
  - irregular payload widths and metadata alignment;
  - decompression timing relative to Softmax and AV;
  - idle write hardware during attention.
- State the research question exactly: can a restricted format and fused tile
  reduce SRAM capacity and unnecessary movement with measurable costs?

### 3. Algorithm-Format Co-Design

- Frozen algorithm point:
  - `po2_nearest` scale;
  - K relative scale 0.03, V relative scale 0.10;
  - integer zero point;
  - stable four-bucket `BUCKET` by `k_sum`;
  - 64-token block, 16-token pack;
  - 192-token recent high-precision software window outside encoded streams.
- Explain shift-based dequantization and why the power-of-two restriction
  matters for hardware.
- Stable bucket routing:
  - K/V q values and K/V metadata share one token permutation;
  - bucket counts delimit storage order;
  - no stored 64-entry permutation;
  - decode Attention is valid by K/V shared permutation invariance.
- Dynamic bit-packing:
  - per-pack/feature min, width, and payload;
  - independent K and V streams;
  - zero-width constant packs.
- Format v0 validation:
  - fixed field widths and observed ranges;
  - overflow/reject behavior rather than silent truncation;
  - explicit statement that Format v0 is not a complete container.
- Software evidence subsection:
  - include three-model local CR/accuracy summary only as current evidence;
  - mark FP baseline reconstruction as required before final camera-ready
    accuracy claims.

### 4. Single-Head BRISK-KV Architecture

- Top-level dataflow:
  - raw Q12 K/V input;
  - write quantizers;
  - token join and bucket router;
  - pack transpose and dynamic bit-pack encoders;
  - eleven resident compressed byte streams;
  - replay buffers and stream decoders;
  - QK, fixed-point scale, streaming Softmax;
  - Full-V or JIT-V AV;
  - rounded/saturated Q6 output.
- Storage organization:
  - Full-V stores full dequantized V;
  - JIT-V stores packets, weights, partial sums, and compressed V streams.
- Explain the two Attention endpoints:
  - Full-V: latency reference;
  - JIT-V: storage reduction with serialized/replayed V decompression.
- Clarify single-head scope and external interfaces.

### 5. Microarchitecture Optimizations

- Replay-pipe v1:
  - starts V replay once Softmax weight packet is resident;
  - overlaps synchronous SRAM response with stream consumption;
  - cite 395,923 -> 278,843 dual JIT-V periodic attention-cycle reduction.
- Shared decoder:
  - replaces dual decoders with one K-capable decoder plus V metadata adapter;
  - cite 278,843 -> 278,859 cycle cost, 4.89% area reduction, 8.41% dynamic
    power reduction, 6.35% total cell-power reduction.
- Writer phase clock gating:
  - writer active only reset/write launch/write phase;
  - cite unchanged 278,859 cycles, 23.6411 -> 15.8115 mW dynamic, and
    134,515.414817 -> 134,594.710815 um^2 cell area.
- Explicitly state that the mapped gate is latch-plus-AND, not foundry ICG.

### 6. Methodology

- Software:
  - report model/task provenance once reconstructed;
  - include NONE/BUCKET/continuous/FP baselines with exact directories and
    hashes.
- Golden vectors:
  - deterministic vectors under `hardware/golden_vectors/`;
  - round-trip and boundary tests.
- RTL verification:
  - Chisel/ScalaTest exact output and backpressure tests;
  - cycle CSV output count and checksum equality.
- RTL export:
  - `full/` RTL for functional VCS/SAIF;
  - matching `dc_logic/` RTL for synthesis;
  - top always read from `manifest.json`.
- Cycle benchmark:
  - identical deterministic K/V/query values;
  - none and periodic output backpressure;
  - record write, QK, Softmax, replay, AV, stalls, output count, checksum.
- Activity and PPA:
  - attention-only SAIF at `tb_briskkv_tile_power_1024/dut`;
  - DC Ultra, 2.0 ns, TSMC 28 nm HPC+ LVT SSG 0.81 V / 125 C;
  - architectural SRAM black boxes;
  - small queues synthesized;
  - report SAIF annotation coverage and warnings.
- SRAM modeling:
  - use bit capacity and CACTI storage-only context separately;
  - do not add 22 nm CACTI numbers to 28 nm logic DC numbers.

### 7. Results

- Accuracy and compression:
  - table with FP, continuous PackKV, NONE, BUCKET once provenance is recovered;
  - current draft can include the three-model BUCKET vs NONE CR gain as
    preliminary archived local summary, with FP baseline marked missing.
- Storage:
  - Full-V 4,012,288 bits vs JIT-V 1,659,392 bits;
  - 2,352,896-bit / 58.64% reduction.
- Latency:
  - Full-V 164,447 cycles;
  - JIT-V dual 278,843 cycles;
  - shared and writer-CG 278,859 cycles;
  - show none/periodic rows if space allows.
- Logic PPA:
  - matched architecture table: area, dynamic, leakage, total cell power,
    SRAM bits, setup/hold/transition boundary.
- Ablation waterfall:
  - pre-replay -> replay_pipe_v1;
  - dual -> shared;
  - shared -> writer-CG.
- Hierarchical power:
  - writer 15.349 -> 7.698 mW total;
  - writer internal 7.774 -> about 0.000005 mW;
  - explain why this supports gating attribution.
- Pareto discussion:
  - Full-V is latency endpoint;
  - shared writer-CG is retained area/power-first endpoint;
  - dual is still 16 cycles faster than shared and should not be called
    dominated.

### 8. Related Work

- KV cache quantization/compression systems.
- GPU-oriented PackKV and related software cache approaches.
- Hardware attention accelerators and near-memory/cache-oriented attention.
- Bit-serial/bit-packed compression and decompression datapaths.
- Clock gating and phase-aware accelerator design.
- Comparison rule: describe related work qualitatively unless matched
  technology/workload evidence is available.

### 9. Limitations and Future Work

- Current artifact is a single-head tile, not a complete LLM accelerator.
- Attention-only activity is not write-plus-attention or whole-request power.
- DC result is pre-layout, ZeroWireload, ideal-clock logic-only PPA.
- Architectural SRAM internal dynamic/leakage power is excluded from DC.
- 22 nm CACTI storage estimates are separate from 28 nm DC logic.
- Remote VCS PASS logs are not fully archived for independent reread in all
  cases.
- One deterministic 1024 x 128 workload is not a workload distribution.
- `无法确认`: GPU PackKV speedup/energy advantage.
- `无法确认`: system-level throughput, DRAM traffic, DMA/NoC/cache manager
  behavior, multi-head/layer scheduling.
- Future work:
  - write-only and combined-phase SAIF;
  - additional token lengths and feature dimensions;
  - real-model traces;
  - matched SRAM technology or memory compiler results;
  - foundry ICG, P&R/CTS, gate-level activity;
  - full-system integration.

### 10. Reproducibility Appendix

- Artifact hashes from `hardware/docs/PROJECT_STATUS_20260818.md`.
- Exact paths for cycle CSVs, generated RTL manifests, SAIF, DC reports, and
  DDC reports.
- Command recipes from `hardware/docs/JIT_V_ABLATION.md` and project READMEs.
- Table-generation scripts should read archived CSV/report values directly
  rather than copying numbers by hand.

## 5. Reviewer-Facing Gaps To Close Before Submission

| Gap | Current status | Action |
|---|---|---|
| True FP and PackKV software baselines | `无法确认` from current summary | Reconstruct exact experiment directories and hashes; report FP, continuous PackKV, NONE, BUCKET under a fixed comparison rule. |
| GPU PackKV speed/energy comparison | `无法确认` | Either add a matched experiment or avoid quantitative GPU speedup/energy claims. |
| Full request power | `无法确认` | Add write-only and combined write+attention SAIF if this becomes a claimed result. |
| SRAM energy/power in matched technology | `无法确认` | Keep CACTI 22 nm separate, or obtain matched memory compiler/technology evidence. |
| Physical signoff | `无法确认` | Foundry ICG, P&R/CTS/parasitics, and gate-level activity are future work unless required by reviewer scope. |
| Workload distribution | `无法确认` | Add more token lengths/features/traces only if the paper needs generalization beyond the deterministic 1024 x 128 point. |
