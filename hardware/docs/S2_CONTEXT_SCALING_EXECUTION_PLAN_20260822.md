# BRISK-KV S2 Context-Scaling Execution Plan

Date: 2026-08-22

S2 extends evidence collection only. It must not modify the frozen compression
algorithm, RTL source, or archived retained-result directories.

## 1. Required Cycle and SRAM Matrix

Run every point below at 128 features:

| Token count | Full-V | retained shared writer-CG JIT-V | Required evidence |
|---:|---|---|---|
| 256 | yes | yes | cycle CSV, generated `dc_logic` manifest, architectural SRAM bits |
| 512 | yes | yes | cycle CSV, generated `dc_logic` manifest, architectural SRAM bits |
| 1024 | yes | yes | cycle CSV, generated `dc_logic` manifest, regression against frozen cycle rows |
| 2048 | yes | yes | cycle CSV, generated `dc_logic` manifest, architectural SRAM bits |
| 4096 | yes | yes | cycle CSV, generated `dc_logic` manifest, architectural SRAM bits |

The primary comparison uses periodic output backpressure, matching the retained
1024-token cycle point. The runner also emits no-backpressure rows as a
functional/cycle sanity control. Each row verifies output count and checksum
agreement across Full-V and JIT-V.

Architectural SRAM capacity is read from each generated `dc_logic/manifest.json`.
This is the required S2 storage metric. CACTI is optional and remains a
separate 22 nm memory-modeling result; it must not be added to 28 nm DC logic
PPA as a single chip PPA number.

The cycle CSV records stage-local active counters for K decompression, QK,
Softmax, AV/JIT-V, and V decompression. They can overlap, so they must not be
summed into total attention cycles. Total attention latency is the measured
`attention_cycles` field.

## 2. DC Endpoint Policy

Do not run DC for all ten geometry/architecture combinations.

| Point | DC action | Rationale |
|---|---|---|
| 1024, retained writer-CG JIT-V | reuse frozen archive | Existing matched 2.0 ns retained point is the anchor; do not overwrite or rerun it. |
| 4096, Full-V | run logic-only DC at 2.0 ns | Tests the large-context endpoint for the latency-oriented architecture. |
| 4096, retained writer-CG JIT-V | run logic-only DC at 2.0 ns | Tests the large-context endpoint for the SRAM-oriented retained architecture. |
| 256/512/2048 | no DC by default | Cycle/SRAM scaling is the S2 claim; intermediate DC runs are optional only if an address-width discontinuity is observed. |

The two new 4096 DC runs support logic area, timing, and black-box SRAM
interface reporting only. Default-activity `power.rpt` is not a context-scaling
power claim. Dynamic power across token counts requires a geometry-matched VCS
workload and a separately audited SAIF for every reported point.

## 3. Commands

Run cycle and architectural SRAM collection locally or on the simulation host:

```bash
RUN_TAG=s2_context_scaling_20260822_full \
bash hardware/scripts/run_s2_context_scaling.sh
```

This writes fresh results under:

```text
hardware/evaluation/results/context_scaling/<RUN_TAG>/
hardware/rtl/generated/context_scaling/<RUN_TAG>/
```

After collection, run only the 4096 DC endpoints on the DC host:

```bash
RTL_ROOT=/absolute/path/to/hardware/rtl/generated/context_scaling/s2_context_scaling_20260822_full \
TARGET_LIBRARY=/absolute/path/to/standard_cells.db \
DC_POINTS=4096 \
REPORT_ROOT=/absolute/path/to/s2_context_scaling_dc_20260822_full \
bash hardware/scripts/run_s2_dc_endpoints.sh
```

Before accepting the result, require `context_scaling_check.json` to pass and
confirm its 1024 periodic Full-V and writer-CG rows match the frozen cycle
archives for write cycles, attention cycles, output count, and checksum.
