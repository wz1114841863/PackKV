package briskkv

import chisel3._

/** Technology-independent, glitch-free clock gate for bounded power studies.
  *
  * The enable is transparent only while the source clock is low. Foundry
  * integration may replace this module with the characterized ICG cell, but
  * keeping a synthesizable model here makes Chisel, VCS and DC use the same
  * functional clock-gating semantics.
  */
class BriskKvClockGate extends ExtModule {
  val clockIn = IO(Input(Clock()))
  val enable = IO(Input(Bool()))
  val clockOut = IO(Output(Clock()))

  setInline(
    "BriskKvClockGate.sv",
    """module BriskKvClockGate(
      |  input  wire clockIn,
      |  input  wire enable,
      |  output wire clockOut
      |);
      |  reg enableLatched;
      |  always_latch begin
      |    if (!clockIn)
      |      enableLatched = enable;
      |  end
      |  assign clockOut = clockIn & enableLatched;
      |endmodule
      |""".stripMargin
  )
}
