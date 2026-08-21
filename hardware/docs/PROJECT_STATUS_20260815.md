# BRISK-KV Hardware Status — 2026-08-15

> Historical pre-replay baseline. It remains immutable comparison evidence,
> but the current retained architecture is documented in
> `PROJECT_STATUS_20260818.md`; start paper work from
> `PAPER_HANDOFF_20260818.md`.

This document is the continuation of `PROJECT_STATUS_20260814.md`. It records
the verified 1024-token VCS/SAIF/DC power boundary before any further hardware
optimization. Do not overwrite the referenced result directories.

## 1. Frozen workload and RTL boundary

The recorded workload exercises the dual-decoder JIT-V overlap-v1 full RTL at
the maximum generated geometry:

- top: `BriskKvJitVSingleHeadTileTop`;
- clock: 2.0 ns / 500 MHz;
- tokens: 1024, represented as 16 complete 64-token blocks;
- feature dimension: 128;
- activity window: attention only;
- deterministic stimulus: the K/V and query pattern shared with
  `BriskKvCycleBenchmarkSpec`;
- downstream output readiness: periodic backpressure;
- no Chisel algorithm change was made for this power experiment.

The 64-token exact-output smoke test remains the compact functional golden.
The separate 1024 x 128 test checks transaction geometry, protocol completion,
result status, output order/count, overlap and a deterministic checksum.

## 2. Remote VCS result

The user confirmed that the remote VCS O-2018.09-SP2 run reported:

```text
BRISK-KV 1024x128 POWER PASS
```

The locally validated deterministic reference for this testbench is:

```text
write_cycles=1174731
attention_cycles=395923
overlap=1
checksum=6216823619359318016
activity_duration_ns=791848
```

The remote `simulation.log` is not present in the local result archive, so the
PASS is user-confirmed terminal evidence rather than a locally re-read log.
The archived SAIF duration and clock activity exactly match the expected
attention window.

Relevant flow files:

```text
hardware/simulation/vcs/tb_briskkv_jit_v_power_1024.sv
hardware/simulation/vcs/run_vcs_power_1024.sh
hardware/simulation/vcs/run_vcs.sh
```

## 3. Archived SAIF

File:

```text
hardware/simulation/results/2026081502/jit_v_1024t_128f_attention.saif
```

SHA-256:

```text
141317c102071122a1d42351a391ffec7ac477596efbf2f318b9504340390eee
```

Validated properties:

- `TIMESCALE 1 ps`;
- `DURATION 791848000` ps;
- hierarchy `tb_briskkv_jit_v_power_1024/dut`;
- 66 clock entries, all with `T0=395924000`, `T1=395924000`, `TX=0`,
  `TC=791848`;
- 77175 activity entries in total;
- aggregate TX fraction is about 15.46%; full-duration X values are mainly
  associated with behavioral-memory read data and inactive logic.

This is a phase-specific RTL activity trace, not a full transaction-average
trace.

## 4. Archived 2.0 ns DC result

Directory:

```text
hardware/synthesis/dc/results/2026081502/period_2p0ns
```

DC used:

```text
saif_file=.../jit_v_1024t_128f_attention.saif
saif_instance=tb_briskkv_jit_v_power_1024/dut
```

SAIF annotation coverage:

| Stage | User nets | User ports | User pins | Propagated nets | Default nets |
|---|---:|---:|---:|---:|---:|
| Precompile | 24.00% | 100.00% | 25.62% | 0.00% | 0.10% |
| Postcompile | 9.66% | 100.00% | 17.61% | 86.15% | 4.18% |

`read_saif` reported 26085 RTL objects not found. These are associated with
optimization/name changes and architectural memory boundaries. There is no
`PWR-362` zero-annotation error in the accepted run.

Logic-only DC power at the 2.0 ns corner:

| Metric | No-SAIF control (`2026081501`) | 1024 x 128 SAIF (`2026081502`) | Change |
|---|---:|---:|---:|
| Cell internal | 25.9407 mW | 25.1848 mW | -2.91% |
| Net switching | 1.0958 mW | 0.2315 mW | -78.87% |
| Total dynamic | 27.0364 mW | 25.4163 mW | -5.99% |
| Cell leakage | 40.1263 mW | 41.5986 mW | +3.67% |
| Total cell power | 67.1654 mW | 67.0155 mW | -0.22% |

The mapped netlists for `2026081501`, `2026081502`, and the earlier 64-token
SAIF run are identical after removing the generated date header. Their common
normalized SHA-256 is:

```text
e011ba270535280a4b70c123260eba34ba3f2954b46b76126306f740d0c782b8
```

The different leakage values therefore reflect state-dependent leakage under
different annotated signal probabilities, not a structural netlist change.

Report hashes:

```text
power.rpt: 064a112d5fa71f65a5a6ca3febd4ca321b9815f8cb40d02a936236651a9f6ec1
qor.rpt: 7ff2bb8f8f1d80ed7d1d8f4690ff69bb4e248812e5e276e9b45217148d8352a8
saif_annotation_postcompile.rpt: cbb5d53b10024dee9e4e88a9f34f2ffd93ec600b7fb33737383eb3fe1b8a627e
```

## 5. Timing and area boundary

The accepted 2.0 ns result retains the previously observed synthesis boundary:

- setup WNS/TNS: 0.00 ns / 0.00 ns, no setup violating paths;
- worst hold violation: about -0.06 ns;
- hold TNS: about -622.27 ns over 14451 paths;
- cell/design area: 140452.702774 um^2;
- one max-transition violating net, about 0.26 ns actual versus 0.25 ns
  required;
- no max-capacitance violations.

The hold and transition findings are DC pre-layout limitations and are not new
failures introduced by the workload or SAIF.

## 6. Interpretation boundary

The `2026081502` result is accepted as the current representative
attention-only RTL-SAIF logic-power baseline. It is sufficient to close the
present VCS/SAIF/DC validation step and to evaluate the next bounded hardware
optimization.

It is not final silicon power signoff because:

- architectural SRAM area and internal power remain outside the DC logic
  total and must be combined with the separate memory model;
- the result is RTL activity mapped through synthesis, not gate-level SDF
  activity;
- the design uses ZeroWireload and has no placement/clock-tree/parasitics;
- 4.18% of postcompile nets retain default activity and some RTL activity is
  unknown;
- the workload is one deterministic single-head attention transaction, not a
  model/task distribution.

Do not compare the new attention-only result directly with the old 64-token
whole-transaction SAIF as an optimization delta. Their context length and
activity windows differ.

## 7. Next-step gate

The following items are now complete:

1. remote full-RTL VCS smoke and 1024 x 128 representative workload;
2. valid phase-specific SAIF generation and hierarchy mapping;
3. 2.0 ns SAIF-annotated DC power estimation;
4. frozen pre-optimization PPA/activity boundary.

The next design change should be bounded and compared against this exact
baseline without changing the compression algorithm or overwriting old
results. Candidate work should remain focused on the known JIT-V latency and
decoder/accumulator bottleneck; multi-head scheduling is still outside the
current tile scope.

## 8. 2026-08-17 continuation

The first bounded post-baseline optimization has now been completed and frozen
as `JIT-V dual + pipelined byte-stream replay v1`.  Its generated RTL, matched
1024 x 128 cycle result, attention-only SAIF, 2.0 ns DC PPA and hierarchical
power evidence are recorded in:

```text
hardware/docs/PROJECT_STATUS_20260817.md
```

`2026081502` remains the immutable pre-optimization baseline.  Use the
2026-08-17 continuation for the retained replay implementation and do not
overwrite either result set.
