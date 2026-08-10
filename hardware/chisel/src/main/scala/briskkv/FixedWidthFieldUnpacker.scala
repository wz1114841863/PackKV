package briskkv

import chisel3._
import chisel3.util._

class FixedWidthFieldUnpackerIO(fieldBits: Int, countBits: Int) extends Bundle {
  val start = Input(Bool())
  val fieldCount = Input(UInt(countBits.W))
  val in = Flipped(Decoupled(UInt(8.W)))
  val out = Decoupled(UInt(fieldBits.W))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
}

/** LSB-first byte-stream to fixed-width field unpacker.
  *
  * A transaction starts with a one-cycle `start` pulse and a non-zero
  * `fieldCount`. The module then consumes exactly ceil(fieldCount * fieldBits / 8)
  * bytes. Input and output are independently backpressured. Unused high bits in
  * the final byte must be zero.
  */
class FixedWidthFieldUnpacker(fieldBits: Int, countBits: Int = 32) extends Module {
  require(fieldBits >= 1 && fieldBits <= 8)
  require(countBits >= 2)

  private val reservoirBits = 16
  private val bitCountBits = log2Ceil(reservoirBits + 1)

  val io = IO(new FixedWidthFieldUnpackerIO(fieldBits, countBits))

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val reservoir = RegInit(0.U(reservoirBits.W))
  val bitCount = RegInit(0.U(bitCountBits.W))
  val fieldsRemaining = RegInit(0.U(countBits.W))
  val bytesRemaining = RegInit(0.U(countBits.W))

  io.busy := active
  io.done := doneReg
  io.error := errorReg
  doneReg := false.B

  io.out.valid := active && fieldsRemaining =/= 0.U && bitCount >= fieldBits.U
  io.out.bits := reservoir(fieldBits - 1, 0)
  io.in.ready := active && bytesRemaining =/= 0.U &&
    bitCount <= (reservoirBits - 8).U

  val inFire = io.in.valid && io.in.ready
  val outFire = io.out.valid && io.out.ready
  val shiftedInput = (io.in.bits << bitCount)(reservoirBits - 1, 0)
  val countAfterOutput = bitCount - fieldBits.U
  val shiftedInputAfterOutput =
    (io.in.bits << countAfterOutput)(reservoirBits - 1, 0)
  val reservoirAfterOutput = reservoir >> fieldBits
  val reservoirAfterBoth = reservoirAfterOutput | shiftedInputAfterOutput

  when(io.start && !active) {
    reservoir := 0.U
    bitCount := 0.U
    fieldsRemaining := io.fieldCount
    errorReg := false.B
    when(io.fieldCount === 0.U) {
      active := false.B
      doneReg := true.B
      errorReg := true.B
      bytesRemaining := 0.U
    }.otherwise {
      val totalBits = io.fieldCount * fieldBits.U
      bytesRemaining := (totalBits + 7.U) >> 3
      active := true.B
    }
  }.elsewhen(io.start && active) {
    // Transactions may not be restarted while bytes are still in flight.
    errorReg := true.B
  }.elsewhen(active) {
    when(inFire && outFire) {
      reservoir := reservoirAfterBoth
      bitCount := bitCount + (8 - fieldBits).U
    }.elsewhen(inFire) {
      reservoir := reservoir | shiftedInput
      bitCount := bitCount + 8.U
    }.elsewhen(outFire) {
      reservoir := reservoirAfterOutput
      bitCount := countAfterOutput
    }

    when(inFire) {
      bytesRemaining := bytesRemaining - 1.U
    }
    when(outFire) {
      fieldsRemaining := fieldsRemaining - 1.U
      when(fieldsRemaining === 1.U) {
        val residual = Mux(inFire, reservoirAfterBoth, reservoirAfterOutput)
        val remainingBytes = bytesRemaining - inFire.asUInt
        active := false.B
        doneReg := true.B
        // Any residual one is a non-zero byte-alignment padding bit.
        when(residual.orR || remainingBytes =/= 0.U) {
          errorReg := true.B
        }
      }
    }
  }
}
