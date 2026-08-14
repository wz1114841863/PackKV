# PackKV / BRISK-KV project status — 2026-08-14

This document is the handoff point for a new conversation. It records the
current research decisions, verified results, code boundaries, open issues,
and immediate commands. Read `AGENTS.md` first, then this file, then the
referenced implementation files. Do not infer that an experiment has finished
unless it is explicitly marked verified below.

## 1. Research objective

The project starts from the PackKV paper and official implementation, then
develops a hardware-oriented KV-cache compression and attention accelerator.
The intended paper story is joint algorithm–format–architecture optimization:

1. replace arbitrary floating-point quantization scales with hardware-friendly
   power-of-two scales;
2. replace expensive global/greedy repacking with fixed four-bucket stable
   routing;
3. define a compact byte-stream format with explicit metadata;
4. implement lossless pack/unpack and fixed-point decompression/attention in
   Chisel;
5. evaluate logic with Design Compiler and memories separately with CACTI;
6. reduce the large dequantized-V storage using JIT-V and quantify its
   area/latency trade-off.

The hardware name is **BRISK-KV**. It is intentionally separate from PackKV:
PackKV is the source algorithm/baseline; BRISK-KV is the hardware-oriented
format and accelerator developed in this repository.

## 2. Algorithm decisions and software status

### 2.1 Selected candidate

The current retained algorithm point is:

```text
quantization scale : po2_nearest
K scale threshold : 0.03
V scale threshold : 0.10
zero point         : integer / high_precision_zero_point=false
repacking          : BUCKET
bucket count       : 4
bucket score       : k_sum
block size         : 64 tokens
pack size          : 16 tokens
recent buffer      : 192 tokens in the software PackKV experiments
```

`continuous` remains the PackKV baseline and must not be presented as a local
innovation. `po2_floor` often lowers offline error only because it can require
an additional magnitude bit; it is not a like-for-like four-bit candidate.
`po2_ceil` can improve packing but has larger quantization error. The tested
packing-aware nearest/ceil selector was not retained because its measured
accuracy/CR trade-off did not beat plain conservative `po2_nearest`.

Primary code:

- `utils/compute.py`: scale methods, quantization, bucket repacking, metadata,
  bit packing, packing-aware reference, round-trip helpers;
- `models/cache/packkv_quant.py`: online cache path and configuration dispatch;
- `utils/config.py`: serialized PackKV configuration;
- `evaluation/evaluation.py`: accuracy/CR/metadata/round-trip evaluation;
- `scripts/cr/cr_eval.py`: compression-rate CLI and CSV export;
- `scripts/accuracy/accuracy_eval.py`: lm-eval accuracy entry point.

### 2.2 Verified software correctness

The following have been verified in prior work:

- quantize/dequantize consistency tests for continuous and power-of-two modes;
- bucket boundary, empty-bucket, stable permutation, K/V shared-permutation,
  metadata serialize/deserialize tests;
- real bit-pack/unpack and bucket-metadata round trips;
- compact power-of-two metadata round trips;
- full software unit suite: 25 tests passed at the recorded checkpoint;
- real extracted-KV round-trip audit on Qwen3-4B, Qwen3-8B and
  NousResearch/Meta-Llama-3.1-8B, four sampled layers per model; later 8192
  runs checked two blocks per sampled layer.

Current test files include:

```text
tests/test_quantization_correctness.py
tests/test_bucket_repacking.py
tests/test_packing_aware_quant.py
tests/test_hardware_format_profile.py
tests/test_golden_vectors.py
tests/test_joint_summary.py
```

### 2.3 Three-model accuracy and compression result

The current joint summary was generated from full GSM8K accuracy and the v9 CR
table using `po2_nearest`, K=0.03, V=0.10. The exact stored summary is
`Three_Model_Accuracy_CR_Summary.csv` in the user's experiment archive.

| Model | Repack | Strict (%) | Flexible (%) | Overall CR | Bucket CR gain vs NONE |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | NONE | 71.1903 | 80.1365 | 3.5409x | — |
| Qwen3-4B | BUCKET | 71.2661 | 80.2881 | 3.7898x | 7.03% |
| Qwen3-8B | NONE | 87.1114 | 84.4579 | 3.5295x | — |
| Qwen3-8B | BUCKET | 87.1873 | 84.6096 | 3.7772x | 7.02% |
| Llama-3.1-8B | NONE | 50.3412 | 50.2654 | 3.6223x | — |
| Llama-3.1-8B | BUCKET | 50.2654 | 50.1895 | 3.8423x | 6.07% |

Interpretation:

- bucket=4 with `k_sum` consistently reduces encoded bytes by about 5.7–6.6%
  and improves overall CR by about 6–7%;
- NONE/BUCKET accuracy differences are within roughly 0.15 percentage points,
  supporting the claim that repacking is lossless/permutation-equivalent in
  the tested decode path;
- the summary's `FP_Baseline_Status` is `missing`. Therefore these rows do not
  by themselves establish the exact accuracy delta relative to unquantized FP.
  Use the separate paper-reproduction accuracy records when reporting that
  comparison.

The full Qwen3-4B packing-aware decision run recorded:

```text
plain nearest      strict=71.2661%, flexible=80.2881%
packing-aware K=.05/V=.00 strict=70.2805%, flexible=78.6202%
```

This is why packing-aware selection is currently an ablation, not the default.

## 3. BRISK-KV Format v0

The normative format is `hardware/docs/briskkv_format_v0.md`.

Frozen architectural choices:

- 64-token block;
- 16-token pack;
- four stable `k_sum` buckets;
- the same token permutation for K and V;
- power-of-two exponent metadata and integer zero point;
- LSB-first compact fixed-width metadata;
- dynamic per-pack payload width;
- three stored bucket counts per block; the fourth is reconstructed from 64;
- eleven resident byte streams: five K, five V, one bucket-count stream.

The Python golden encoder/decoder and deterministic directed/random vectors are
in:

```text
utils/golden_vectors.py
scripts/hardware/export_golden_vectors.py
hardware/golden_vectors/
```

The exported vectors are the functional contract for Chisel modules. Format
fields, bit ordering and metadata widths must not be changed without updating
both Python and hardware tests.

## 4. Hardware architecture implemented

The Chisel project is under `hardware/chisel/`. The following single-head path
is functionally present:

```text
raw K/V Q12 input
  -> write-side power-of-two quantization
  -> K/V token join and stable k_sum four-bucket routing
  -> 16-token transpose
  -> dynamic bit-pack encoding + compact metadata
  -> eleven resident byte-stream SRAMs
  -> metadata decode + dynamic unpack + shift dequantization
  -> 16-lane QK accumulation
  -> fixed-point scaling + stable streaming Softmax
  -> V replay / JIT-V
  -> 16-lane Softmax×V accumulation
  -> Q21 round/saturate to Q6
  -> attention output and tagged completion
```

Important implemented modules:

- write/format: `KvWriteQuantizer`, `KvTokenJoinBucketRouter`,
  `KvPackTransposeBuffer`, `DynamicBitPackEncoder`,
  `CompactKvMetadataEncoder`, `BucketCountEncoder`,
  `BriskKvWriteEncoderTop`;
- read/decompress: `FixedWidthFieldUnpacker`, `DynamicBitUnpacker`,
  `CompactMetadataDecoder`, `Po2FixedPointDequantizer`,
  `BufferedPackMetadataDequantizer`, `DecompressionPipelineController`;
- compute: `QueryReplayBuffer`, `QkDotProductAccumulator`,
  `AttentionScaleUnit`, `StreamingSoftmax`, `SoftmaxVAccumulator`,
  `AvOutputQuantizer`;
- integration: `BriskKvSingleHeadTileTop`,
  `BriskKvJitVSingleHeadTileTop`,
  `BriskKvSharedJitVSingleHeadTileTop`.

Write quantizer variants v2–v4 were evaluated as PPA ablations. None produced a
compelling improvement over v1. **v1 remains the retained write-side design.**
Do not silently replace it with v2, v3 or v4.

## 5. Full-V and JIT-V variants

Three comparable tops are retained:

| Variant | Top | V storage | Decompressors |
| --- | --- | --- | ---: |
| Full-V | `BriskKvSingleHeadTileTop` | complete dequantized token×feature V SRAM | 2 |
| JIT-V dual | `BriskKvJitVSingleHeadTileTop` | 2-packet queue + weight/partial-sum SRAM | 2 |
| JIT-V shared | `BriskKvSharedJitVSingleHeadTileTop` | same JIT buffers | 1 shared |

JIT-V removes the complete dequantized-V SRAM. The latest **overlap-v1** starts
V replay when the first Softmax weight packet is committed, continues loading
later weights while V decode/AV accumulation proceeds, and permits a V packet
to leave the two-entry queue only when `packIndex < weightLoadIndex`. This
prevents read-before-write of weight SRAM.

Implementation evidence:

- `hardware/chisel/src/main/scala/briskkv/JitVAccumulator.scala`;
- `hardware/chisel/src/main/scala/briskkv/BriskKvJitVAttentionTop.scala`;
- `hardware/docs/JIT_V_ABLATION.md`.

### 5.1 Cycle result

For feature_dim=8 without output backpressure:

| Tokens | Full-V | JIT dual before overlap | JIT dual overlap-v1 | JIT shared overlap-v1 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1179 | 1682 | 1673 | 1678 |
| 256 | 4443 | 6368 | 6323 | 6328 |
| 1024 | 17499 | 25112 | 24923 | 24928 |

The overlap saves 9/45/189 cycles, exactly the interval from first to last
Softmax weight packet. Output checksums are identical across architectures.

JIT-V remains about 42.4% slower than Full-V at 1024 tokens. The current trace
shows about 9118 cycles from first to last V value and zero JIT accumulator
input stalls, so the remaining performance bottleneck is primarily V
decompression throughput, not the AV accumulator.

### 5.2 Memory result

At maximum_tokens=1024 and maximum_feature_dim=128:

```text
Full-V external SRAM bits : 4,012,288
JIT-V external SRAM bits  : 1,659,392
reduction                 : 2,352,896 bits / 58.64%
```

The recorded CACTI 22 nm, 2.0 ns, maximum-128-bit banking result is:

| Variant | SRAM area | Leakage |
| --- | ---: | ---: |
| Full-V | 0.847445 mm² | 50.692293 mW |
| JIT-V dual/shared | 0.343719 mm² | 22.426537 mW |

This is a storage-only estimate. It is not yet a workload energy number and it
uses a 22 nm CACTI assumption while current DC uses a TSMC 28 nm standard-cell
library. Do not add these numbers without explicitly stating the methodology
or rerunning a matched memory technology point.

## 6. Existing DC results and synthesis method

Current DC environment:

```text
tool         : Synopsys DC Ultra O-2018.06-SP1
library      : TSMC 28 nm HPC+ LVT, SSG 0.81 V, 125 C
main period  : 2.0 ns / 500 MHz
stats        : disabled for main PPA
memory       : architectural SRAMs black-boxed
memory PPA   : evaluated separately with CACTI
```

The following 2.0 ns numbers are **pre-overlap-v1 baselines**:

| Run | Variant | Logic area | DC default-activity dynamic power |
| --- | --- | ---: | ---: |
| `2026081302` | Full-V unified tile | 144051.77 | 26.3578 mW |
| `2026081303` | JIT-V dual | 140752.25 | 27.1185 mW |
| `2026081304` | JIT-V shared | 133878.53 | 24.8698 mW |

The shared decoder reduced logic area by about 4.9% relative to JIT dual and
by about 7.1% relative to Full-V at that point. These power reports do not yet
use workload SAIF, so they are preliminary synthesis estimates.

Use two RTL exports from the same Chisel source/configuration:

- `full/`: behavioral memories; functional RTL simulation/VCS;
- `dc_logic/`: bodyless SRAM black boxes; DC logic-only PPA.

Relevant flow files:

```text
hardware/scripts/generate_jit_v_tile_rtl.sh
hardware/scripts/generate_shared_jit_v_tile_rtl.sh
hardware/synthesis/dc/run_dc_logic.tcl
hardware/synthesis/dc/run_tile_dc.sh
```

`run_tile_dc.sh` reads the top from `manifest.json`. `run_dc_logic.tcl` also
reads the manifest and rejects an external `TOP` mismatch. The top-existence
check must remain after `elaborate`, not after `analyze`, because DC O-2018 does
not expose the design collection at the earlier point.

## 7. Current work in progress

As of 2026-08-14:

1. The user has launched remote DC synthesis for
   `briskkv_jit_v_dual_overlap_v1/dc_logic`.
2. The requested sweep is `2.0 1.5 1.2 1.0 0.9 0.8` ns.
3. The intended report root is the remote
   `runs/2026081305_output/outputs` directory.
4. Results have not yet been returned to the local workspace. Therefore the
   overlap-v1 logic area, power and timing are **not yet confirmed**.
5. The matching shared overlap-v1 DC sweep should be run after or alongside it
   using a separate report root.

Do not overwrite the old `2026081302`, `2026081303`, or `2026081304` reports;
they are the comparison baselines.

## 8. VCS infrastructure just added

Files:

```text
hardware/simulation/vcs/tb_briskkv_jit_v.sv
hardware/simulation/vcs/run_vcs.sh
hardware/simulation/vcs/README.md
```

The testbench uses the dual overlap-v1 `full/` RTL and performs a real
64-token, four-feature end-to-end transaction:

- raw K/V write through encoding and resident behavioral SRAM;
- query input and Attention execution;
- periodic output backpressure;
- exact output check `[0, 24, 24, 64]`;
- tag/error checks;
- assertion that V launches before all four weight packets arrive.

Local Verilator execution passed:

```text
BRISK-KV VCS PASS: write_cycles=2747 attention_cycles=756 overlap=1
```

Remote VCS has not yet been run. Use:

```bash
export RTL_DIR=/absolute/path/briskkv_jit_v_dual_overlap_v1/full
export OUTPUT_DIR=/absolute/path/simulation/vcs/outputs
export WAVE_MODE=vcd   # vcd, vpd, or none
bash /absolute/path/simulation/vcs/run_vcs.sh
```

The script rejects `dc_logic`, old pre-overlap RTL, and a shared top. For a
short debug trace use VCD/VPD. For later power work, add a representative
1024-token SAIF run instead of dumping the entire design to VCD.

## 9. Known limitations

The present hardware is a research single-head tile, not a complete LLM SoC:

- single head; no multi-head or multi-layer scheduler;
- one resident compressed transaction at a time;
- write and attention are phase separated;
- no external DMA, NoC or DRAM controller;
- no causal-prefill scheduler; current permutation argument is for decode;
- no foundry SRAM macro integration or post-layout result;
- DC currently excludes SRAM area/power and CACTI is reported separately;
- DC power has not yet been annotated with VCS/SAIF activity;
- VCS testbench currently targets only dual JIT-V and a 64-token smoke case;
- the software recent window is part of PackKV evaluation but is not a
  multi-window hardware cache manager in this tile.

These limitations must be stated rather than implied away. The current paper
scope should emphasize a verified single-head codec/attention tile and PPA
trade-off, with multi-head/layer scheduling as scaling analysis or future work
unless it is subsequently implemented.

## 10. Immediate next steps

Perform these in order:

1. collect the remote dual overlap-v1 DC reports (`qor.rpt`,
   `timing_setup.rpt`, `area_hier.rpt`, `power.rpt`, `references.rpt`,
   `check_design.rpt`, memory audits and `dc.log`);
2. validate completion and compare against `2026081303` at identical periods;
3. run and collect the shared overlap-v1 DC sweep in a separate directory;
4. run the VCS full-RTL 64-token test and inspect `vLaunched`,
   `weightLoadIndex`, V decoder valid/ready and JIT queue activity;
5. establish SAIF generation for representative power annotation;
6. freeze overlap-v1 as a named baseline;
7. optimize the remaining V bottleneck, first with bounded V prefetch and then
   with a 2-lane V decoder; retain 1-lane/2-lane/4-lane as an area–latency
   ablation rather than replacing the baseline without comparison.

Do not immediately implement multi-head scheduling. First finish the matched
Full-V/JIT-dual/JIT-shared PPA and activity comparison, because it determines
whether the JIT-V area–latency trade-off is strong enough to retain.

## 11. Workspace caution

At this snapshot, the working tree includes the intended new/modified files:

```text
hardware/synthesis/dc/run_dc_logic.tcl
hardware/synthesis/dc/run_tile_dc.sh
hardware/simulation/vcs/
```

There is also an unrelated deleted `.metals/metals.lock.db` entry. Do not
restore, delete, or commit it without first confirming the user's intent. Do
not use destructive Git commands; earlier experiment and hardware changes
belong to the user.

## 12. Suggested opening prompt for the next conversation

```text
请先阅读 AGENTS.md、hardware/docs/PROJECT_STATUS_20260814.md、
hardware/docs/JIT_V_ABLATION.md，并检查当前 git status。本轮继续 BRISK-KV
硬件工作。当前 dual JIT-V overlap-v1 的 DC 综合正在/已经完成；请先验证新上传
的综合报告并与 2026081303 基线做同周期比较，不要覆盖旧结果，也不要修改算法
或继续做多 lane 优化，直到确认当前 PPA 边界。
```
