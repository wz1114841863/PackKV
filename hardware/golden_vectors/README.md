# BRISK-KV Golden Vectors

These vectors remain the normative Python-to-Chisel transport for Format v0.
They support the frozen single-head implementation but do not by themselves
establish VCS/DC correctness; the complete evidence chain is indexed in
`../docs/PAPER_HANDOFF_20260818.md`.

Run from the repository root:

```bash
python scripts/hardware/export_golden_vectors.py
```

Generated data is written to `hardware/golden_vectors/generated/` and is not
tracked by Git. Each case contains:

- `input_*.bin`: byte-per-element encoder inputs in original token order;
- bucket, compact-metadata, and K/V bit-packed streams: decoder inputs;
- `expected_*.bin`: repacked or dequantized hardware outputs;
- `descriptor.json`: shapes, field widths, stream descriptors, byte lengths,
  SHA-256 checksums, and validation status.

`manifest.json` lists all generated cases. These files implement the named
component streams in `hardware/docs/briskkv_format_v0.md`; they are a testbench
transport format, not a final DMA/container header.

The small deterministic regression copy used directly by SBT is stored in
`hardware/chisel/src/test/resources/golden_vectors/` and is tracked so that
`sbt test` does not require Python or PyTorch.
