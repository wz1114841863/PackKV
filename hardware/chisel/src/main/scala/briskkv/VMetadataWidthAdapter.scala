package briskkv

import chisel3._
import chisel3.util._

class VMetadataWidthAdapterIO(countBits: Int) extends Bundle {
  val start = Input(Bool())
  val descriptorCount = Input(UInt(countBits.W))
  val tokenCount = Input(UInt(countBits.W))
  val minimumIn = Flipped(Decoupled(UInt(8.W)))
  val zeroPointIn = Flipped(Decoupled(UInt(8.W)))
  val minimumOut = Decoupled(UInt(8.W))
  val zeroPointOut = Decoupled(UInt(8.W))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
}

/** Lossless Format-v0 V-to-K metadata width adapter.
  *
  * A single K-capable decompressor can decode both streams after V's unsigned
  * 4-bit q minima are zero-extended and its signed 5-bit zero points are
  * sign-extended, then repacked as 6-bit and 7-bit fields respectively. Width,
  * payload and exponent streams already have compatible representations and
  * bypass this adapter in the shared-decoder top.
  */
class VMetadataWidthAdapter(
  countBits: Int = 32,
  enableStats: Boolean = false
) extends Module {
  val io = IO(new VMetadataWidthAdapterIO(countBits))

  val minimumUnpacker = Module(new FixedWidthFieldUnpacker(4, countBits))
  val minimumPacker = Module(
    new FixedWidthFieldPacker(6, countBits, enableStats = enableStats)
  )
  val zeroUnpacker = Module(new FixedWidthFieldUnpacker(5, countBits))
  val zeroPacker = Module(
    new FixedWidthFieldPacker(7, countBits, enableStats = enableStats)
  )
  val minimumDoneSeen = RegInit(false.B)
  val zeroDoneSeen = RegInit(false.B)

  minimumUnpacker.io.start := io.start
  minimumUnpacker.io.fieldCount := io.descriptorCount
  minimumUnpacker.io.in <> io.minimumIn
  minimumPacker.io.start := io.start
  minimumPacker.io.fieldCount := io.descriptorCount
  minimumPacker.io.in.valid := minimumUnpacker.io.out.valid
  minimumPacker.io.in.bits := minimumUnpacker.io.out.bits
  minimumUnpacker.io.out.ready := minimumPacker.io.in.ready
  io.minimumOut <> minimumPacker.io.out

  zeroUnpacker.io.start := io.start
  zeroUnpacker.io.fieldCount := io.tokenCount
  zeroUnpacker.io.in <> io.zeroPointIn
  zeroPacker.io.start := io.start
  zeroPacker.io.fieldCount := io.tokenCount
  zeroPacker.io.in.valid := zeroUnpacker.io.out.valid
  zeroPacker.io.in.bits := Cat(
    Fill(2, zeroUnpacker.io.out.bits(4)),
    zeroUnpacker.io.out.bits
  )
  zeroUnpacker.io.out.ready := zeroPacker.io.in.ready
  io.zeroPointOut <> zeroPacker.io.out

  when(io.start) {
    minimumDoneSeen := false.B
    zeroDoneSeen := false.B
  }.otherwise {
    when(minimumPacker.io.done) { minimumDoneSeen := true.B }
    when(zeroPacker.io.done) { zeroDoneSeen := true.B }
  }
  val minimumComplete = minimumDoneSeen || minimumPacker.io.done
  val zeroComplete = zeroDoneSeen || zeroPacker.io.done

  io.busy := minimumUnpacker.io.busy || minimumPacker.io.busy ||
    zeroUnpacker.io.busy || zeroPacker.io.busy
  io.done := minimumComplete && zeroComplete
  io.error := minimumUnpacker.io.error || minimumPacker.io.error ||
    zeroUnpacker.io.error || zeroPacker.io.error
}
