# BRISK-KV VCS full-RTL validation

This testbench uses the generated `full/` RTL, including behavioral SRAM
implementations. It must not be compiled against the DC `dc_logic/` export.

Export the current overlap RTL, then run on a machine with VCS:

```bash
OUTPUT_ROOT=hardware/rtl/generated/briskkv_jit_v_dual_overlap_v1_vcs2018 \
VCS_COMPATIBILITY=true \
ENABLE_STATS=false \
bash hardware/scripts/generate_jit_v_tile_rtl.sh

export RTL_DIR=/absolute/path/to/briskkv_jit_v_dual_overlap_v1_vcs2018/full
export OUTPUT_DIR=$PWD/hardware/simulation/vcs/outputs
export WAVE_MODE=vcd
bash hardware/simulation/vcs/run_vcs.sh
```

`VCS_COMPATIBILITY=true` asks CIRCT not to emit block-local `automatic logic`
variables that are mis-simulated by the VCS O-2018.09 toolchain. Keep this
export separate from the RTL used for the recorded DC PPA results.

`WAVE_MODE` accepts:

- `vcd`: generate `jit_v_overlap_64t.vcd` for portable waveform inspection;
- `vpd`: generate `jit_v_overlap_64t.vpd` for DVE;
- `none`: run the same functional and overlap checks without waveform dumping.

The test uses a 2.0 ns clock, 64 tokens and four features. It checks the exact
attention output `[0, 24, 24, 64]`, the result tag/error status, output
backpressure handling, and observes V launch before all four Softmax weight
packets have entered the JIT-V accumulator.

Open the VPD result with:

```bash
dve -vpd hardware/simulation/vcs/outputs/jit_v_overlap_64t.vpd
```

For VCD, use DVE's VCD import or another waveform viewer such as GTKWave.

## Representative 1024-token power workload

Keep the exact-value 64-token smoke test above as the functional gate. After
it passes, run the separate 1024-token x 128-feature workload. It uses the same
deterministic K/V and query pattern as `BriskKvCycleBenchmarkSpec`, checks the
complete protocol/result path, and starts waveform collection only at the
selected activity phase.

The recommended primary report is the attention phase at the same 2.0 ns clock
used for DC:

```bash
export RTL_DIR=/absolute/path/to/briskkv_jit_v_dual_overlap_v1_vcs2018/full
export OUTPUT_DIR=/absolute/path/to/runs/20260815_power_attention/outputs
export WAVE_MODE=vcd
export ACTIVITY_PHASE=attention
bash /absolute/path/to/hardware/simulation/vcs/run_vcs_power_1024.sh
```

Convert the resulting activity-only VCD:

```bash
vcd2saif \
  -input  /absolute/path/to/jit_v_1024t_128f_attention.vcd \
  -output /absolute/path/to/jit_v_1024t_128f_attention.saif
```

The PASS line prints `activity_duration_ns`. Before DC, do not assume that a
phase-gated VCD was normalized correctly by the installed `vcd2saif`. Confirm:

- the SAIF hierarchy is `tb_briskkv_jit_v_power_1024/dut`;
- SAIF `DURATION` equals `activity_duration_ns * 1000` for a 1 ps SAIF;
- the DUT clock has approximately half T0 and half T1, with TX equal to zero.

If duration includes the earlier cache-fill interval or clock TX is nonzero,
stop: that SAIF is not valid for phase-only power. In that case retain the VCD
and check the installed VCS/vcd2saif help for a supported time-window or direct
SAIF option before proceeding. Once these checks pass, use:

```bash
export SAIF_FILE=/absolute/path/to/jit_v_1024t_128f_attention.saif
export SAIF_INSTANCE=tb_briskkv_jit_v_power_1024/dut
export CLOCK_PERIODS=2.0
export REPORT_ROOT=/absolute/path/to/new_saif_power_results
bash /absolute/path/to/hardware/synthesis/dc/run_tile_dc.sh
```

Do not overwrite the 64-token VCD/SAIF or any previous DC result directory.
For attribution, repeat into separate output directories with
`ACTIVITY_PHASE=write` and optionally `ACTIVITY_PHASE=combined`. The combined
window is useful for transaction-average energy, but it is not a substitute
for the attention-only power result because the long cache-fill phase changes
the phase weighting.
