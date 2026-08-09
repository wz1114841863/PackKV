# BRISK-KV Chisel

This directory contains the synthesizable Chisel reference implementation of
the BRISK-KV streaming codec. It is intentionally isolated from the Python
algorithm implementation in the repository root.

The normative component-stream contract is documented in
[`../docs/briskkv_format_v0.md`](../docs/briskkv_format_v0.md).

## Requirements

- JDK 17 or newer
- SBT
- Verilator for ChiselSim tests

## Commands

Run from this directory:

```bash
sbt test
```

The initial test suite checks the frozen Format v0 parameters. Subsequent
modules must add golden-vector tests against the Python encoder before they are
treated as implemented.

## Source layout

```text
src/main/scala/briskkv/   Chisel modules and shared parameters
src/test/scala/briskkv/   ScalaTest/ChiselSim verification
```

SBT is the only supported build entry point for the first implementation. This
avoids dependency and plugin drift between multiple build systems.
