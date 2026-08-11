package briskkv

import chisel3._
import chisel3.util._

class BriskKvDecompressQkProgress(countBits: Int) extends Bundle {
  val compute = new BriskKvComputeProgress(countBits)
  val qk = new QkComputePipelineStats
}

class BriskKvDecompressQkTopIO(
  outputBits: Int,
  accumulatorBits: Int,
  countBits: Int,
  tagBits: Int,
  packTokens: Int,
  bucketCountBits: Int
) extends Bundle {
  val command = Flipped(
    Decoupled(new DualKvDecompressionCommand(countBits, tagBits))
  )
  val queryLoadIn = Flipped(Decoupled(SInt(outputBits.W)))
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
  val qkLogitsOut = Decoupled(
    new QkLogitPacket(accumulatorBits, packTokens, countBits)
  )
  val vFeatureOut = Decoupled(
    new AttentionFeaturePacket(outputBits, packTokens, countBits)
  )
  val bucketOut = Decoupled(new BucketCountRecord(bucketCountBits, countBits))
  val result = Decoupled(new DualKvDecompressionResult(countBits, tagBits))
  val busy = Output(Bool())
  val queryLoaded = Output(Bool())
  val progress = Output(new BriskKvDecompressQkProgress(countBits))
}

/** Full Format v0 decompression-to-QK integration top.
  *
  * K is consumed internally by the query-replay QK pipeline. V and bucket
  * records remain independent outputs for the later softmax/AV path.
  */
class BriskKvDecompressQkTop(
  outputBits: Int = 18,
  accumulatorBits: Int = 44,
  countBits: Int = 32,
  tagBits: Int = 16,
  maximumFeatureDim: Int = 256,
  useBufferedMetadata: Boolean = true
) extends Module {
  private val params = BriskKvFormatV0.params
  private val bucketCountBits = log2Ceil(params.blockTokens + 1)

  val io = IO(
    new BriskKvDecompressQkTopIO(
      outputBits,
      accumulatorBits,
      countBits,
      tagBits,
      params.packTokens,
      bucketCountBits
    )
  )

  val compute = Module(
    new BriskKvComputeInterface(
      outputBits = outputBits,
      countBits = countBits,
      tagBits = tagBits,
      useBufferedMetadata = useBufferedMetadata
    )
  )
  val qk = Module(
    new QkComputePipeline(
      valueBits = outputBits,
      accumulatorBits = accumulatorBits,
      packTokens = params.packTokens,
      blockTokens = params.blockTokens,
      maximumFeatureDim = maximumFeatureDim,
      countBits = countBits
    )
  )

  val commandActive = RegInit(false.B)
  val qkDone = RegInit(false.B)
  val invalidResultValid = RegInit(false.B)
  val invalidCommand = RegInit(
    0.U.asTypeOf(new DualKvDecompressionCommand(countBits, tagBits))
  )

  val topIdle = !commandActive && !invalidResultValid
  val featureDimSupported = io.command.bits.featureDim <= maximumFeatureDim.U
  compute.io.command.valid :=
    io.command.valid && topIdle && featureDimSupported
  compute.io.command.bits := io.command.bits
  io.command.ready := topIdle && Mux(
    featureDimSupported,
    compute.io.command.ready,
    true.B
  )
  val commandFire = io.command.valid && io.command.ready
  val forwardedCommandFire = commandFire && featureDimSupported
  val rejectedCommandFire = commandFire && !featureDimSupported

  qk.io.start := compute.io.acceptedGeometryValid
  qk.io.featureDim := io.command.bits.featureDim
  qk.io.packCount := compute.io.acceptedPackCount
  qk.io.queryLoadIn <> io.queryLoadIn
  qk.io.keyIn <> compute.io.kFeatureOut
  io.qkLogitsOut <> qk.io.out

  compute.io.kMinimumIn <> io.kMinimumIn
  compute.io.kWidthIn <> io.kWidthIn
  compute.io.kPayloadIn <> io.kPayloadIn
  compute.io.kZeroPointIn <> io.kZeroPointIn
  compute.io.kExponentIn <> io.kExponentIn
  compute.io.vMinimumIn <> io.vMinimumIn
  compute.io.vWidthIn <> io.vWidthIn
  compute.io.vPayloadIn <> io.vPayloadIn
  compute.io.vZeroPointIn <> io.vZeroPointIn
  compute.io.vExponentIn <> io.vExponentIn
  compute.io.bucketCountIn <> io.bucketCountIn
  io.vFeatureOut <> compute.io.vFeatureOut
  io.bucketOut <> compute.io.bucketOut

  when(forwardedCommandFire) {
    commandActive := true.B
    qkDone := !compute.io.acceptedGeometryValid
  }.elsewhen(qk.io.done) {
    qkDone := true.B
  }
  when(rejectedCommandFire) {
    invalidResultValid := true.B
    invalidCommand := io.command.bits
  }

  val normalResultValid = commandActive && compute.io.result.valid && qkDone
  io.result.valid := invalidResultValid || normalResultValid
  io.result.bits := 0.U.asTypeOf(
    new DualKvDecompressionResult(countBits, tagBits)
  )
  when(invalidResultValid) {
    io.result.bits.tag := invalidCommand.tag
    io.result.bits.error := true.B
    io.result.bits.tokenCount := invalidCommand.tokenCount
    io.result.bits.packCount :=
      (invalidCommand.tokenCount + (params.packTokens - 1).U) >>
        log2Ceil(params.packTokens)
    io.result.bits.blockCount :=
      (invalidCommand.tokenCount + (params.blockTokens - 1).U) >>
        log2Ceil(params.blockTokens)
    io.result.bits.descriptorCount := invalidCommand.descriptorCount
  }.otherwise {
    io.result.bits := compute.io.result.bits
    io.result.bits.error := compute.io.result.bits.error || qk.io.error
  }

  compute.io.result.ready :=
    !invalidResultValid && commandActive && qkDone && io.result.ready
  val resultFire = io.result.valid && io.result.ready
  when(resultFire && invalidResultValid) {
    invalidResultValid := false.B
  }
  when(resultFire && !invalidResultValid) {
    commandActive := false.B
    qkDone := false.B
  }

  io.busy := commandActive || invalidResultValid || compute.io.busy || qk.io.busy
  io.queryLoaded := qk.io.queryLoaded
  io.progress.compute := compute.io.progress
  io.progress.qk := qk.io.stats
}
