package briskkv

import chisel3._
import chisel3.util._

class BriskKvAttentionProgress(countBits: Int) extends Bundle {
  val decompressionQk = new BriskKvDecompressQkProgress(countBits)
  val scaleSoftmax = new QkScaleSoftmaxStats
  val softmaxV = new SoftmaxVComputeStats
  val outputQuantizer = new AvOutputQuantizerStats
}

class BriskKvAttentionTopIO(
  valueBits: Int,
  outputBits: Int,
  countBits: Int,
  tagBits: Int,
  bucketCountBits: Int
) extends Bundle {
  private val packTokens = BriskKvFormatV0.params.packTokens
  val command = Flipped(
    Decoupled(new DualKvDecompressionCommand(countBits, tagBits))
  )
  val queryLoadIn = Flipped(Decoupled(SInt(valueBits.W)))
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
  val bucketOut = Decoupled(new BucketCountRecord(bucketCountBits, countBits))
  val attentionOut = Decoupled(new AttentionOutputFeature(outputBits, countBits))
  val result = Decoupled(new DualKvDecompressionResult(countBits, tagBits))
  val busy = Output(Bool())
  val queryLoaded = Output(Bool())
  val vLoaded = Output(Bool())
  val progress = Output(new BriskKvAttentionProgress(countBits))
}

/** Complete BRISK-KV Format v0 decode-time attention reference top. */
class BriskKvAttentionTop(
  valueBits: Int = 18,
  qkAccumulatorBits: Int = 44,
  avAccumulatorBits: Int = 50,
  outputBits: Int = 18,
  countBits: Int = 32,
  tagBits: Int = 16,
  maximumFeatureDim: Int = 256,
  maximumTokens: Int = 16384,
  useBufferedMetadata: Boolean = true
) extends Module {
  private val params = BriskKvFormatV0.params
  private val bucketCountBits = log2Ceil(params.blockTokens + 1)

  val io = IO(
    new BriskKvAttentionTopIO(
      valueBits,
      outputBits,
      countBits,
      tagBits,
      bucketCountBits
    )
  )

  val decompressionQk = Module(
    new BriskKvDecompressQkTop(
      outputBits = valueBits,
      accumulatorBits = qkAccumulatorBits,
      countBits = countBits,
      tagBits = tagBits,
      maximumFeatureDim = maximumFeatureDim,
      maximumTokens = maximumTokens,
      useBufferedMetadata = useBufferedMetadata
    )
  )
  val scaleSoftmax = Module(
    new QkScaleSoftmaxPipeline(
      logitBits = qkAccumulatorBits,
      packTokens = params.packTokens,
      maximumFeatureDim = maximumFeatureDim,
      maximumTokens = maximumTokens,
      countBits = countBits
    )
  )
  val softmaxV = Module(
    new SoftmaxVComputePipeline(
      valueBits = valueBits,
      accumulatorBits = avAccumulatorBits,
      packTokens = params.packTokens,
      maximumFeatureDim = maximumFeatureDim,
      maximumTokens = maximumTokens,
      countBits = countBits
    )
  )
  val outputQuantizer = Module(
    new AvOutputQuantizer(
      accumulatorBits = avAccumulatorBits,
      outputBits = outputBits,
      maximumFeatureDim = maximumFeatureDim,
      countBits = countBits
    )
  )

  decompressionQk.io.command <> io.command
  decompressionQk.io.queryLoadIn <> io.queryLoadIn
  decompressionQk.io.kMinimumIn <> io.kMinimumIn
  decompressionQk.io.kWidthIn <> io.kWidthIn
  decompressionQk.io.kPayloadIn <> io.kPayloadIn
  decompressionQk.io.kZeroPointIn <> io.kZeroPointIn
  decompressionQk.io.kExponentIn <> io.kExponentIn
  decompressionQk.io.vMinimumIn <> io.vMinimumIn
  decompressionQk.io.vWidthIn <> io.vWidthIn
  decompressionQk.io.vPayloadIn <> io.vPayloadIn
  decompressionQk.io.vZeroPointIn <> io.vZeroPointIn
  decompressionQk.io.vExponentIn <> io.vExponentIn
  decompressionQk.io.bucketCountIn <> io.bucketCountIn
  io.bucketOut <> decompressionQk.io.bucketOut

  val downstreamStart = decompressionQk.io.acceptedGeometryValid
  scaleSoftmax.io.start := downstreamStart
  scaleSoftmax.io.featureDim := io.command.bits.featureDim
  scaleSoftmax.io.tokenCount := io.command.bits.tokenCount
  scaleSoftmax.io.in <> decompressionQk.io.qkLogitsOut

  softmaxV.io.start := downstreamStart
  softmaxV.io.featureDim := io.command.bits.featureDim
  softmaxV.io.tokenCount := io.command.bits.tokenCount
  softmaxV.io.vIn <> decompressionQk.io.vFeatureOut
  softmaxV.io.weightIn <> scaleSoftmax.io.out

  outputQuantizer.io.start := downstreamStart
  outputQuantizer.io.featureDim := io.command.bits.featureDim
  outputQuantizer.io.in <> softmaxV.io.out
  io.attentionOut <> outputQuantizer.io.out

  val commandActive = RegInit(false.B)
  val attentionDone = RegInit(false.B)
  when(decompressionQk.io.commandAccepted) {
    commandActive := true.B
    attentionDone := !decompressionQk.io.acceptedGeometryValid
  }.elsewhen(outputQuantizer.io.done) {
    attentionDone := true.B
  }

  io.result.valid := commandActive &&
    decompressionQk.io.result.valid && attentionDone
  io.result.bits := decompressionQk.io.result.bits
  io.result.bits.error := decompressionQk.io.result.bits.error ||
    scaleSoftmax.io.error || softmaxV.io.error || outputQuantizer.io.error
  decompressionQk.io.result.ready :=
    commandActive && attentionDone && io.result.ready
  when(io.result.valid && io.result.ready) {
    commandActive := false.B
    attentionDone := false.B
  }

  io.busy := commandActive || decompressionQk.io.busy ||
    scaleSoftmax.io.busy || softmaxV.io.busy || outputQuantizer.io.busy
  io.queryLoaded := decompressionQk.io.queryLoaded
  io.vLoaded := softmaxV.io.vLoaded
  io.progress.decompressionQk := decompressionQk.io.progress
  io.progress.scaleSoftmax := scaleSoftmax.io.stats
  io.progress.softmaxV := softmaxV.io.stats
  io.progress.outputQuantizer := outputQuantizer.io.stats
}
