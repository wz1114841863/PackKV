# BRISK-KV 论文主线与下一阶段研发/实验计划

日期：2026-08-20

用途：本文档用于指导后续 Codex 对 BRISK-KV 工程进行修改、补实验和整理论文证据。当前阶段原则是：**停止继续扩展“完整 LLM 加速器”功能，转入 paper-driven validation and scaling 阶段**。

---

## 1. 当前论文定位

### 1.1 推荐定位

BRISK-KV 不应被写成“PackKV 的硬件版”，也不应以“PackKV 不适合专用硬件”为出发点。

更合适的定位是：

> **面向 resource-constrained LLM inference 的 compressed-resident KV Attention tile。通过将 PackKV-style KV compression 约束为有界、规则、可流式消费的硬件格式，使 KV Cache 在片上存储期间保持压缩状态，并在 Attention 中按需解压，从而降低 SRAM 容量需求，并形成可控的 storage–latency–power tradeoff。**

推荐论文关键词：

- resource-constrained LLM inference
- SRAM-constrained attention accelerator
- compressed-resident KV
- hardware-constrained compression
- algorithm–format–architecture co-design
- storage–latency–power Pareto

“Edge”可以作为应用背景，但当前不要把系统描述成完整 edge LLM accelerator，因为现有证据并不覆盖完整 SoC、DMA/NoC、DRAM、multi-head/multi-layer scheduling 或端到端 token/s。

### 1.2 一句话 thesis

> **BRISK-KV demonstrates that, by constraining PackKV-style compression into a bounded hardware format, KV cache can remain compressed in on-chip storage and be consumed directly by Attention, substantially reducing SRAM capacity while exposing a controllable latency–area–power tradeoff.**

中文：

> **BRISK-KV 表明，通过将 PackKV 式 KV 压缩约束为有界、规则的硬件格式，可以使 KV Cache 在片上存储期间保持压缩状态并直接参与 Attention 计算，从而显著降低 SRAM 容量需求，并形成可控的延迟—面积—功耗权衡。**

---

## 2. 论文要解决的核心问题

### 2.1 背景问题

长上下文 LLM 推理中，KV Cache 随 context length 增长，持续增加：

- 存储容量需求；
- 片上/片外带宽压力；
- Attention 数据搬移开销；
- 能量开销。

PackKV 已证明 compression + computation-aware decompression 在高性能 GPU 上具有价值，但 resource-constrained accelerator 的设计目标不同：

- SRAM 容量更严格；
- datapath 需要更简单；
- metadata 和控制需要有界；
- 数据流需要规则和可预测；
- energy 与 idle hardware 更重要。

因此，不需要论证“GPU 不好”，而应强调：

> **不同 deployment regime 对 KV compression 的优化目标不同。**

### 2.2 核心 Research Question

推荐论文明确提出：

> **Can KV cache remain compressed throughout on-chip storage and be consumed directly by Attention, while keeping decompression hardware simple enough for a resource-constrained accelerator?**

中文：

> **能否让 KV Cache 从写入片上 SRAM 到参与 Attention 计算期间始终保持压缩状态，并通过简单、规则的按需解压数据通路直接消费压缩数据？**

这个问题比“如何做一个 PackKV accelerator”更准确，也更符合当前硬件实现边界。

---

## 3. 文章主旨与技术主线

整篇论文建议围绕三层结构组织。

### 3.1 第一层：Hardware-Constrained Compression Format

目标不是追求最大 compression ratio，而是把压缩算法限制为可直接映射到硬件的数据表示。

当前冻结算法/格式点包括：

- power-of-two (`po2_nearest`) quantization scale；
- integer zero point；
- stable four-bucket repacking；
- K/V 共用 token permutation；
- 64-token block；
- 16-token pack；
- dynamic per-pack bit packing；
- 11 个 resident component streams；
- bounded metadata / validation rules；
- recent high-precision window 仍属于软件策略，不进入当前硬件 encoded streams。

这里不要把“2^k quantization”或“bucket sorting”单独包装成主要算法创新。

更合适的表述是：

> **BRISK-KV reformulates PackKV-style compression into a bounded hardware contract.**

算法限制的意义：

- arbitrary scale → power-of-two scale → shift-oriented dequantization；
- general/data-dependent repacking → stable four-bucket routing；
- implicit compressed representation → explicit component streams；
- unconstrained metadata → bounded fields + validation/reject behavior；
- shared K/V token permutation → 无需保存完整 64-entry permutation。

核心观点：

> **Compression ratio alone is not the correct optimization objective for an SRAM-constrained accelerator.**

可以将设计目标理解为：

`maximize useful compression`，同时满足：

- bounded metadata；
- bounded routing；
- simple dequantization；
- streaming decode；
- direct Attention consumption。

### 3.2 第二层：Compressed-Resident Attention Architecture

这是论文最核心的 architecture contribution。

比较两个 endpoint：

**Full-V**

```text
compressed V
    |
 decoder
    |
dequantized V SRAM
    |
    AV
```

**JIT-V**

```text
compressed V SRAM
    |
 replay/decompress
    |
    AV
```

JIT-V 的核心价值不是 decoder 本身，而是：

> **eliminate full dequantized-V materialization**

冻结 1024×128 单-head 结果：

- Full-V architectural SRAM：4,012,288 bits；
- JIT-V architectural SRAM：1,659,392 bits；
- 减少 2,352,896 bits；
- SRAM reduction：58.64%；
- Full-V periodic attention：164,447 cycles；
- retained shared/writer-CG JIT-V：278,859 cycles。

因此论文应主动把它写成：

> **Storage–Latency Pareto**

而不是试图证明 JIT-V 在所有指标上优于 Full-V。

- Full-V：latency-oriented endpoint；
- JIT-V：SRAM-oriented endpoint。

### 3.3 第三层：Microarchitecture Recovers the Cost

在 JIT-V 形成 storage saving 后，通过微架构优化降低其 latency/area/power 成本。

#### Replay pipeline

- pre-replay dual JIT-V：395,923 cycles；
- replay_pipe_v1 dual JIT-V：278,843 cycles；
- latency reduction：29.57%。

#### Shared decoder

- dual JIT-V：278,843 cycles；
- shared JIT-V：278,859 cycles；
- 仅增加 16 cycles；
- cell area：-4.89%；
- dynamic power：-8.41%；
- total cell power：-6.35%。

注意：dual 仍然快 16 cycles，不能称为被 shared 严格 dominated。

#### Writer phase clock gating

- shared ungated：278,859 cycles；
- writer-CG：278,859 cycles；
- dynamic power：23.6411 mW → 15.8115 mW（-33.12%）；
- cell area：134,515.414817 µm² → 134,594.710815 µm²（+0.05895%）。

其作用应描述为 attention-phase idle writer power reduction，而不是完整 request 功耗降低，直到 combined-phase SAIF 完成。

---

## 4. 推荐论文贡献点

建议最终收敛为四条贡献，不要把很多实现细节拆成独立 contribution。

### Contribution 1 — Hardware-constrained KV compression format

将 PackKV-style compression 约束为 power-of-two quantization、stable bucket repacking 和 explicit component streams，形成可综合、可验证、可流式消费的压缩 KV 硬件契约。

### Contribution 2 — Compressed-resident Attention tile

设计单-head write–store–decompress–Attention tile，使压缩 K/V streams 可驻留片上 SRAM，并通过 replay decompression 直接进入 QK/Softmax/AV，避免完整 V materialization。

### Contribution 3 — Storage–latency–power co-design

通过 Full-V/JIT-V、replay pipeline、decoder sharing、phase-aware clock gating 展示并优化 storage–latency–area–power Pareto。

### Contribution 4 — Cross-layer validation

从软件压缩行为、deterministic/real-model vectors、Chisel/RTL、cycle CSV、SAIF、DC synthesis 到 SRAM capacity modeling 形成跨层验证链。

---

## 5. 当前硬件开发边界

### 5.1 建议停止继续扩展的方向

当前不建议继续开发：

- 完整 multi-head scheduler；
- multi-layer Transformer scheduling；
- DMA；
- NoC；
- DRAM controller；
- full LLM accelerator；
- tokenizer / model execution pipeline；
- 为了与 A100 比绝对 token/s 而构建完整系统。

原因：这些功能会把论文问题从 compressed-resident KV Attention 扩展成完整 LLM accelerator，开发成本高，并会引入新的 reviewer 要求，包括 weight storage、GEMM、FFN、LayerNorm、RoPE、system scheduling、DRAM traffic、end-to-end token/s 与 whole-chip energy。

### 5.2 推荐硬件终点

当前项目的合理终点应定义为：

> **A parameterized, trace-validated, single-head compressed-KV Attention tile.**

至少满足：

1. 完整 write → compressed storage → replay/decode → QK → Softmax → AV 路径；
2. 能用 real-model trace 验证；
3. 能展示多个 context length 的 scaling；
4. 有可信的 logic PPA、activity power 和 SRAM capacity tradeoff。

---

## 6. 下一阶段工作优先级

下一阶段原则：**停止增加主要功能，优先补齐论文证据宽度。**

### Priority 0：保持冻结 retained point 不被破坏

在开展后续实验前：

- 保留 `shared_writer_cg_v1` 作为 frozen retained point；
- 不覆盖已有 cycle CSV、SAIF、DC、DDC、manifest 和 format contract；
- 新实验必须写入新的 timestamped/result directories；
- 表格数据优先由脚本直接读取 archived reports/CSV，而不是手工复制；
- 新改动必须能与 frozen point 做回归比较。

---

### Priority 1：投稿前强烈建议完成

#### P1-A. 重建完整软件算法基线

目标：回答“为了硬件约束算法后，到底牺牲/获得了什么？”

统一模型、benchmark、recent-window 和评测规则，至少比较：

1. FP baseline；
2. Original/continuous PackKV；
3. PO2 + NONE；
4. PO2 + BUCKET。

报告：

- accuracy；
- K compression ratio；
- V compression ratio；
- overall compression ratio；
- 必要时 metadata overhead。

当前已有 Qwen3-4B、Qwen3-8B、Llama-3.1-8B 的 local summary，BUCKET 相对 NONE overall CR 分别提升 7.03%、7.02%、6.07%，且 summary 中 NONE/BUCKET accuracy 无明显变化。但真实 FP provenance 尚未恢复，因此当前数字只能作为 preliminary evidence，不能直接作为最终主表。

**Codex 任务重点：**

- 找回/重构原始实验目录与配置；
- 固定并记录 model revision、dataset、seed、quant 参数、recent window；
- 生成统一 CSV/JSON summary；
- 输出可自动生成 paper table 的脚本。

#### P1-B. Context-length scaling

这是当前硬件证据最需要补的部分。

至少测试：

- 256；
- 512；
- 1024；
- 2048；
- 4096 tokens（若 RTL/仿真成本允许）。

不要求每个点都重新做完整 PPA，但至少需要：

- architectural SRAM bits；
- write cycles；
- QK cycles；
- Softmax cycles；
- replay/decompression cycles；
- AV cycles；
- total attention cycles；
- output count/checksum；
- effective throughput（可由固定时钟和工作量推导）。

目标图：

1. `Context Length -> SRAM Footprint`
   - Full-V
   - JIT-V

2. `Context Length -> Attention Latency / Cycles`
   - Full-V
   - JIT-V variants

3. 可选：`Context Length -> SRAM Saving (%)`

重点不是单一 1024-token 数字，而是证明 architecture tradeoff 随 context length 的 scaling 行为。

**Codex 任务重点：**

- 参数化 workload generation；
- 自动批量生成/运行 cycle benchmark；
- 统一 summary CSV；
- regression 确保 1024×128 retained point 数值不变。

#### P1-C. Real-model trace RTL validation

目标：证明硬件不是只对 deterministic synthetic workload 有效。

建议从已有软件实验模型中选择 1–2 个：

- Llama-3.1-8B；
- Qwen3-4B 或 Qwen3-8B。

从真实模型 inference 中导出：

- K；
- V；
- Q；
- quantization/repacking/packing 后的 stream；
- software golden Attention output。

然后：

- 喂入 RTL；
- 比较 stream round-trip；
- 比较 QK/Softmax/AV/final output；
- 明确 fixed-point rounding/saturation 误差规则；
- 保存 trace provenance 和 checksum。

不要求完成完整 LLM RTL inference，也不要求 multi-head/multi-layer execution。

**Codex 任务重点：**

- 增加 trace export/import 工具；
- 生成 deterministic reproducible real-model golden vectors；
- 复用当前 testbench；
- 生成 pass/fail 和误差 summary。

#### P1-D. Combined-phase power evaluation

当前 writer-CG 的 -33.12% dynamic power 仅针对 attention-only SAIF。

需要至少补：

1. write-only SAIF；
2. attention-only SAIF（已有 retained point）；
3. combined write + attention SAIF。

报告：

- SAIF duration；
- annotation coverage；
- dynamic/internal/switching/leakage/total cell power；
- energy；
- writer hierarchy power；
- whole-design power。

这样可以区分：

- writer-CG 对 attention phase 的收益；
- 对完整 write+attention operation 的真实平均收益。

注意仍需明确：

- logic-only；
- pre-layout；
- SRAM internal power 不包含在 DC logic power 内。

**Codex 任务重点：**

- 增加 phase-selectable power testbench；
- 自动生成 SAIF；
- 自动运行 DC power；
- 自动汇总 phase-level energy 表。

---

### Priority 2：推荐完成，有较高论文收益

#### P2-A. Feature-dimension sensitivity

当前冻结点为 head_dim = 128。

若 RTL 参数化较容易，建议补：

- D=64；
- D=128；
- 可选 D=256。

报告：

- cycle scaling；
- logic area；
- SRAM bits；
- critical path；
- throughput。

目的：证明架构不依赖恰好 128 这一固定 geometry。

#### P2-B. 简化的 multi-head scalability study

不建议开发完整 multi-head scheduler。

如果单-head tile 可直接复制，可以做：

- 1 head；
- 2 heads；
- 4 heads。

只研究：

- synthesis area scaling；
- frequency；
- logic power；
- SRAM capacity；
- 理论并行 throughput。

明确不覆盖：

- shared memory arbitration；
- NoC；
- DRAM bandwidth contention；
- multi-head scheduling。

如果复制 wrapper 会引入较多新控制逻辑，则此实验可以不做。

#### P2-C. Matched-technology SRAM evidence

当前 22 nm CACTI storage-only estimate 与 28 nm DC logic PPA 必须分开报告，不能相加。

如果资源允许，可以进一步获取：

- 与 28 nm 更接近的 SRAM compiler / macro estimate；
- 或统一技术节点的 SRAM area/power estimate。

这样可以增强 whole-tile storage discussion，但不是当前论文成立的必要条件。

---

### Priority 3：可选/高成本，不建议成为当前主线

#### P3-A. P&R / CTS / foundry ICG

可以提升证据等级，但开发成本较高。

当前 DC 结果必须保持以下边界：

- pre-layout；
- ZeroWireload；
- ideal clock；
- writer gate 为 latch-plus-AND mapped model；
- 不是 foundry characterized ICG；
- hold 不属于 physical signoff。

若没有明确投稿要求，不必为了当前论文强行进入完整后端。

#### P3-B. GPU PackKV matched speed/energy comparison

当前没有 matched experiment 支持端到端 GPU PackKV speedup 或 energy advantage。

因此：

- 暂时不要将“比 A100 更快/更省电”作为 headline；
- PackKV GPU 作为背景和 high-performance reference 即可；
- 只有在能构造严格 matched workload、相同 compressed-KV operation、明确 energy boundary 时再加入 quantitative comparison。

若后续做 GPU 对比，优先比较：

- effective compressed-KV processing throughput；
- memory traffic；
- energy efficiency；
- resource-normalized efficiency。

不要把“绝对 GB/s 打败 A100”作为目标。

---

## 7. 推荐实验矩阵

### 7.1 软件层

| Dimension | Baseline / Variant | Required Metrics |
|---|---|---|
| Algorithm | FP | Accuracy |
| Algorithm | Original/continuous PackKV | Accuracy, K/V/overall CR |
| Algorithm | PO2 + NONE | Accuracy, K/V/overall CR |
| Algorithm | PO2 + BUCKET | Accuracy, K/V/overall CR |
| Model | Llama-3.1-8B | same |
| Model | Qwen3-4B | same |
| Model | Qwen3-8B | same |

### 7.2 RTL functional/cycle 层

| Dimension | Suggested Points | Metrics |
|---|---|---|
| Context length | 256/512/1024/2048/4096 | phase cycles, total cycles, checksum |
| Feature dim | 64/128/(256) | cycles, SRAM, area if synthesized |
| Data source | deterministic | exact regression |
| Data source | real-model trace | output error/checksum |
| Architecture | Full-V | SRAM, cycles |
| Architecture | JIT-V dual | SRAM, cycles |
| Architecture | JIT-V shared | SRAM, cycles |
| Architecture | writer-CG | cycles, power |

### 7.3 Power/PPA 层

| Experiment | Required Outputs |
|---|---|
| Attention-only SAIF | retained comparison |
| Write-only SAIF | writer active behavior |
| Combined-phase SAIF | request-level logic-only energy |
| DC QoR | area, critical path, setup/transition |
| DC power | dynamic, leakage, total cell |
| Hierarchical power | writer/decompress/attention attribution |
| SRAM capacity | architectural bits, separated from DC logic |

---

## 8. 论文结果部分推荐组织

### 8.1 Algorithm / Format Viability

先证明 hardware restrictions 没有破坏算法：

`FP -> PackKV -> PO2+NONE -> PO2+BUCKET`

主指标：Accuracy + Compression Ratio。

### 8.2 Storage–Latency Pareto

这是主结果。

建议画散点图：

- x-axis：Attention cycles / latency；
- y-axis：Architectural SRAM bits；
- points：Full-V、dual JIT-V、shared JIT-V、writer-CG。

突出：

- Full-V = latency endpoint；
- JIT-V = storage endpoint。

### 8.3 Context-Length Scaling

画：

- context length vs SRAM；
- context length vs latency；
- 可选 context length vs effective throughput。

### 8.4 Microarchitecture Ablation

建议 waterfall：

`pre-replay -> replay_pipe_v1 -> shared -> writer-CG`

分别展示 latency、area、dynamic power。

### 8.5 Logic PPA

报告 retained point：

- cell area：134,594.710815 µm²；
- dynamic power：15.8115 mW；
- leakage：39.9662 mW；
- total cell power：55.7771 mW；
- critical path：1.89 ns；
- target：2.0 ns / 500 MHz；
- setup WNS/TNS：0/0；
- max-transition violations：0；
- worst hold：约 -0.06 ns，仅作为 pre-layout boundary。

必须标注：logic-only、pre-layout、SRAM power excluded。

### 8.6 Power Attribution

展示 writer-CG 前后：

- writer total：15.349 → 7.698 mW；
- writer internal：7.774 → ~0.000005 mW。

结合 combined-phase SAIF 后，再讨论完整操作能量收益。

---

## 9. 论文中应避免的表述

在现有证据下，不要写：

- “BRISK-KV is a complete LLM accelerator.”
- “BRISK-KV supports full multi-head/multi-layer inference.”
- “BRISK-KV achieves end-to-end speedup over PackKV GPU.”
- “BRISK-KV is more energy-efficient than A100/RTX PackKV.”
- “writer clock gating reduces whole-request power by 33.12%.”
- “28 nm logic PPA + 22 nm CACTI SRAM = total chip PPA.”
- “the design meets physical timing signoff.”
- “PO2 quantization or bucket sorting alone is the primary novelty.”

推荐写：

- “single-head research tile”；
- “compressed-resident KV Attention”；
- “resource-constrained inference target”；
- “pre-layout logic-only PPA”；
- “architectural SRAM capacity”；
- “attention-phase dynamic power”；
- “storage–latency–power Pareto”。

---

## 10. Codex 后续修改原则

### 10.1 总原则

后续 Codex 工作应围绕“补论文证据”而不是“继续增加系统功能”。

修改前应先读取：

- `AGENTS.md`
- `docs/MY_IDEAS.md`
- `hardware/docs/PAPER_HANDOFF_20260818.md`
- `hardware/docs/PROJECT_STATUS_20260818.md`
- `hardware/docs/JIT_V_ABLATION.md`
- `hardware/docs/briskkv_format_v0.md`
- 本文档

### 10.2 每个新任务必须回答

1. 这个改动补的是哪个 paper claim？
2. 新结果与哪个 frozen baseline 对比？
3. 输出文件和 provenance 放在哪里？
4. 是否改变 format v0 或 retained architecture？
5. 是否需要新的 regression？
6. 结果是否可以自动生成 paper table/figure？

如果一个改动不能明显增强论文 claim-evidence chain，优先不要做。

### 10.3 结果管理

所有新实验：

- 使用 timestamped directory；
- 保存配置；
- 保存 model/dataset/RTL manifest/hash；
- 保存原始 CSV/report；
- 生成 machine-readable summary；
- 不覆盖 frozen retained evidence；
- 所有 derived percentage 尽量由脚本计算。

---

## 11. 推荐下一阶段执行顺序

建议 Codex 按以下顺序推进：

### Stage 1 — Software Baseline Reconstruction

完成：FP / PackKV / PO2+NONE / PO2+BUCKET 的统一 accuracy + CR 对比。

完成标准：可以直接生成论文算法表。

### Stage 2 — Context-Length Scaling Infrastructure

完成参数化 workload、批量 cycle benchmark 和 summary 脚本。

完成标准：可以自动生成 SRAM/latency scaling 图的数据。

### Stage 3 — Real-Model Trace Validation

至少完成一个真实模型 trace 的 software → stream → RTL → output 验证。

完成标准：有可复现实验目录、golden output 和误差/checksum summary。

### Stage 4 — Power Phase Completion

完成 write-only + attention-only + combined-phase SAIF/DC power。

完成标准：可以区分 attention-phase gating benefit 与 combined-operation logic energy。

### Stage 5 — Optional Generalization

根据时间选择：

- D=64/128 sensitivity；
- 1/2/4 head replication synthesis；
- matched SRAM evidence。

不要在以上工作完成前进入 DMA/NoC/full accelerator 开发。

---

## 12. 最终论文证据链目标

最终论文应形成下面这条完整链：

```text
FP
 |
Original PackKV
 |
PO2 + NONE
 |
PO2 + BUCKET
 |
Format v0
 |
Compressed resident streams
 |
Full-V <---- storage/latency comparison ----> JIT-V
                                             |
                                      replay pipeline
                                             |
                                       shared decoder
                                             |
                                      writer clock gating
                                             |
                         RTL + real-model trace + SAIF + DC
                                             |
                           Storage–Latency–Power Pareto
```

论文是否成熟，不再以“实现了多少完整 LLM 模块”为判断标准，而以这条 claim–evidence chain 是否闭合为判断标准。

