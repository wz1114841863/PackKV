package briskkv

import chisel3._
import chisel3.util._

class IterativeUnsignedDividerIO(numeratorBits: Int, denominatorBits: Int)
    extends Bundle {
  val start = Input(Bool())
  val numerator = Input(UInt(numeratorBits.W))
  val denominator = Input(UInt(denominatorBits.W))
  val quotient = Output(UInt(numeratorBits.W))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
}

/** Exact radix-2 restoring divider.
  *
  * One quotient bit is produced per cycle, replacing a wide combinational `/`
  * with a subtract/compare path whose width is bounded by the denominator.
  */
class IterativeUnsignedDivider(numeratorBits: Int, denominatorBits: Int)
    extends Module {
  require(numeratorBits >= 2)
  require(denominatorBits >= 1)

  private val iterationBits = math.max(1, log2Ceil(numeratorBits))
  val io = IO(new IterativeUnsignedDividerIO(numeratorBits, denominatorBits))

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val numeratorReg = RegInit(0.U(numeratorBits.W))
  val denominatorReg = RegInit(0.U(denominatorBits.W))
  val remainderReg = RegInit(0.U((denominatorBits + 1).W))
  val quotientReg = RegInit(0.U(numeratorBits.W))
  val resultReg = RegInit(0.U(numeratorBits.W))
  val iteration = RegInit(0.U(iterationBits.W))

  io.quotient := resultReg
  io.busy := active
  io.done := doneReg
  io.error := errorReg
  doneReg := false.B

  val shiftedRemainder = Cat(
    remainderReg(denominatorBits - 1, 0),
    numeratorReg(numeratorBits - 1)
  )
  val denominatorExtended = Cat(0.U(1.W), denominatorReg)
  val canSubtract = shiftedRemainder >= denominatorExtended
  val nextRemainder = Mux(
    canSubtract,
    shiftedRemainder - denominatorExtended,
    shiftedRemainder
  )
  val nextQuotient = Cat(
    quotientReg(numeratorBits - 2, 0),
    canSubtract
  )

  when(io.start && !active) {
    errorReg := io.denominator === 0.U
    numeratorReg := io.numerator
    denominatorReg := io.denominator
    remainderReg := 0.U
    quotientReg := 0.U
    resultReg := 0.U
    iteration := 0.U
    when(io.denominator === 0.U) {
      active := false.B
      doneReg := true.B
    }.otherwise {
      active := true.B
    }
  }.elsewhen(io.start && active) {
    errorReg := true.B
  }.elsewhen(active) {
    numeratorReg := Cat(numeratorReg(numeratorBits - 2, 0), 0.U(1.W))
    remainderReg := nextRemainder
    quotientReg := nextQuotient
    when(iteration === (numeratorBits - 1).U) {
      resultReg := nextQuotient
      active := false.B
      doneReg := true.B
    }.otherwise {
      iteration := iteration + 1.U
    }
  }
}
