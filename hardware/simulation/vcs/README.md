# BRISK-KV VCS full-RTL validation

This testbench uses the generated `full/` RTL, including behavioral SRAM
implementations. It must not be compiled against the DC `dc_logic/` export.

Export the current overlap RTL, then run on a machine with VCS:

```bash
export RTL_DIR=/absolute/path/to/briskkv_jit_v_dual_overlap_v1/full
export OUTPUT_DIR=$PWD/hardware/simulation/vcs/outputs
export WAVE_MODE=vcd
bash hardware/simulation/vcs/run_vcs.sh
```

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
