package briskkv

import chisel3._
import chisel3.util._

class CompactMetadataDecoderIO(outputBits: Int, countBits: Int) extends Bundle {
  val start = Input(Bool())
  val fieldCount = Input(UInt(countBits.W))
  val in = Flipped(Decoupled(UInt(8.W)))
  val out = Decoupled(SInt(outputBits.W))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
}

/** Decoder for one signed compact-metadata component stream.
  *
  * Instantiate with fieldBits=7 for K zero points, fieldBits=5 for V zero
  * points, and fieldBits=4 for either exponent stream. Values are sign-extended
  * to `outputBits` while preserving Decoupled backpressure.
  */
class CompactMetadataDecoder(
  fieldBits: Int,
  outputBits: Int = 8,
  countBits: Int = 32
) extends Module {
  require(fieldBits >= 2 && fieldBits <= 8)
  require(outputBits >= fieldBits)

  val io = IO(new CompactMetadataDecoderIO(outputBits, countBits))

  private val unpacker = Module(new FixedWidthFieldUnpacker(fieldBits, countBits))
  unpacker.io.start := io.start
  unpacker.io.fieldCount := io.fieldCount
  unpacker.io.in.valid := io.in.valid
  unpacker.io.in.bits := io.in.bits
  io.in.ready := unpacker.io.in.ready

  io.out.valid := unpacker.io.out.valid
  unpacker.io.out.ready := io.out.ready
  io.out.bits := Cat(
    Fill(outputBits - fieldBits, unpacker.io.out.bits(fieldBits - 1)),
    unpacker.io.out.bits
  ).asSInt

  io.busy := unpacker.io.busy
  io.done := unpacker.io.done
  io.error := unpacker.io.error
}
