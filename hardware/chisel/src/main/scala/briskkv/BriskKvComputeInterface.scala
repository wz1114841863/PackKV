package briskkv

import chisel3._
import chisel3.util._

class BriskKvComputeProgress(countBits: Int) extends Bundle {
  val decompression = new DualKvDecompressionProgress(countBits)
  val kPacketizer = new AttentionPacketizerStats
  val vPacketizer = new AttentionPacketizerStats
}

class BriskKvComputeInterfaceIO(
  outputBits: Int,
  countBits: Int,
  tagBits: Int,
  packTokens: Int,
  bucketCountBits: Int
) extends Bundle {
  val command = Flipped(
    Decoupled(new DualKvDecompressionCommand(countBits, tagBits))
  )
  val kMinimumIn = Flipped(Decoupled(UInt(8.W)))
  val kWidthIn = Flipped(Decoupled(UInt(8.W)))
  val kPayloadIn = Flipped(Decoupled(UInt(8.W)))
  val kZeroPointIn = Flipped(Decoupled(UInt(8.W)))
  val kExponentIn = Flipped(Decoupled(UInt(8.W)))
  val vMinimumIn = Flipped(Decoupled(UInt(8.W)))
  val vWidthIn = Flipped(Decoupled(UInt(8.W)))
  val vPayloadIn = Flipped(Decoupled(UInt(8.W)))
  val vZeroPointIn = Flipped(Decoupled(UInt(8.W)))
  val vExponentIn = Flipped(Decoupled(UInt(8.W)))
  val bucketCountIn = Flipped(Decoupled(UInt(8.W)))
  val kFeatureOut = Decoupled(
    new AttentionFeaturePacket(outputBits, packTokens, countBits)
  )
  val vFeatureOut = Decoupled(
    new AttentionFeaturePacket(outputBits, packTokens, countBits)
  )
  val bucketOut = Decoupled(new BucketCountRecord(bucketCountBits, countBits))
  val result = Decoupled(new DualKvDecompressionResult(countBits, tagBits))
  val busy = Output(Bool())
  val progress = Output(new BriskKvComputeProgress(countBits))
}

/** Compute-facing wrapper that packetizes scalar K/V streams into 16 lanes. */
class BriskKvComputeInterface(
  outputBits: Int = 18,
  countBits: Int = 32,
  tagBits: Int = 16,
  useBufferedMetadata: Boolean = true
) extends Module {
  private val params = BriskKvFormatV0.params
  private val bucketCountBits = log2Ceil(params.blockTokens + 1)

  val io = IO(
    new BriskKvComputeInterfaceIO(
      outputBits,
      countBits,
      tagBits,
      params.packTokens,
      bucketCountBits
    )
  )
  val decompressor = Module(
    new DualKvDecompressionController(
      outputBits = outputBits,
      countBits = countBits,
      tagBits = tagBits,
      useBufferedMetadata = useBufferedMetadata
    )
  )
  val kPacketizer = Module(
    new AttentionFeaturePacketizer(outputBits, countBits = countBits)
  )
  val vPacketizer = Module(
    new AttentionFeaturePacketizer(outputBits, countBits = countBits)
  )

  decompressor.io.command <> io.command
  decompressor.io.kMinimumIn <> io.kMinimumIn
  decompressor.io.kWidthIn <> io.kWidthIn
  decompressor.io.kPayloadIn <> io.kPayloadIn
  decompressor.io.kZeroPointIn <> io.kZeroPointIn
  decompressor.io.kExponentIn <> io.kExponentIn
  decompressor.io.vMinimumIn <> io.vMinimumIn
  decompressor.io.vWidthIn <> io.vWidthIn
  decompressor.io.vPayloadIn <> io.vPayloadIn
  decompressor.io.vZeroPointIn <> io.vZeroPointIn
  decompressor.io.vExponentIn <> io.vExponentIn
  decompressor.io.bucketCountIn <> io.bucketCountIn
  io.bucketOut <> decompressor.io.bucketOut

  kPacketizer.io.in <> decompressor.io.kOut
  vPacketizer.io.in <> decompressor.io.vOut
  io.kFeatureOut <> kPacketizer.io.out
  io.vFeatureOut <> vPacketizer.io.out

  val commandFire = io.command.valid && io.command.ready
  val requestedPackCount =
    (io.command.bits.tokenCount + (params.packTokens - 1).U) >>
      log2Ceil(params.packTokens)
  val geometryValid = io.command.bits.tokenCount =/= 0.U &&
    io.command.bits.featureDim =/= 0.U &&
    io.command.bits.descriptorCount ===
      requestedPackCount * io.command.bits.featureDim
  val packetizerStart = commandFire && geometryValid
  kPacketizer.io.start := packetizerStart
  kPacketizer.io.tokenCount := io.command.bits.tokenCount
  kPacketizer.io.featureDim := io.command.bits.featureDim
  kPacketizer.io.descriptorCount := io.command.bits.descriptorCount
  vPacketizer.io.start := packetizerStart
  vPacketizer.io.tokenCount := io.command.bits.tokenCount
  vPacketizer.io.featureDim := io.command.bits.featureDim
  vPacketizer.io.descriptorCount := io.command.bits.descriptorCount

  val kPacketsDone = RegInit(false.B)
  val vPacketsDone = RegInit(false.B)
  when(commandFire) {
    kPacketsDone := !geometryValid
    vPacketsDone := !geometryValid
  }.otherwise {
    when(kPacketizer.io.done) {
      kPacketsDone := true.B
    }
    when(vPacketizer.io.done) {
      vPacketsDone := true.B
    }
  }

  val packetizersDone = kPacketsDone && vPacketsDone
  io.result.valid := decompressor.io.result.valid && packetizersDone
  decompressor.io.result.ready := io.result.ready && packetizersDone
  io.result.bits := decompressor.io.result.bits
  io.result.bits.error := decompressor.io.result.bits.error ||
    kPacketizer.io.error || vPacketizer.io.error

  io.busy := decompressor.io.busy || kPacketizer.io.busy || vPacketizer.io.busy
  io.progress.decompression := decompressor.io.progress
  io.progress.kPacketizer := kPacketizer.io.stats
  io.progress.vPacketizer := vPacketizer.io.stats
}
