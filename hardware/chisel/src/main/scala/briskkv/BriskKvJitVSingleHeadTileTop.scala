package briskkv

import chisel3._
import chisel3.util._

class BriskKvJitVSingleHeadTileTopIO(
  inputBits: Int,
  valueBits: Int,
  outputBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int,
  bucketCountBits: Int,
  streamCount: Int
) extends Bundle {
  val writeStart = Input(Bool())
  val writeReady = Output(Bool())
  val featureDim = Input(UInt(countBits.W))
  val blockCount = Input(UInt(countBits.W))
  val firstBlockIndex = Input(UInt(countBits.W))
  val writeIn = Flipped(
    Decoupled(new BriskKvWriteInput(inputBits, countBits, tagBits, tokenIndexBits))
  )
  val attentionStart = Input(Bool())
  val attentionReady = Output(Bool())
  val attentionTag = Input(UInt(tagBits.W))
  val queryIn = Flipped(Decoupled(SInt(valueBits.W)))
  val bucketOut = Decoupled(new BucketCountRecord(bucketCountBits, countBits))
  val attentionOut = Decoupled(new AttentionOutputFeature(outputBits, countBits))
  val result = Decoupled(new DualKvDecompressionResult(countBits, tagBits))
  val writeDone = Output(Bool())
  val encodedReady = Output(Bool())
  val busy = Output(Bool())
  val error = Output(Bool())
  val storedLengths = Output(Vec(streamCount, UInt(countBits.W)))
  val writeStats = Output(new BriskKvWriteEncoderStats)
  val attentionProgress = Output(new BriskKvJitVAttentionProgress(countBits))
}

/** Complete write/store/read/attention tile with dual-decoder JIT-V compute. */
class BriskKvJitVSingleHeadTileTop(
  inputBits: Int = 24,
  inputFractionalBits: Int = 12,
  valueBits: Int = 18,
  qkAccumulatorBits: Int = 44,
  avAccumulatorBits: Int = 50,
  outputBits: Int = 18,
  countBits: Int = 32,
  tagBits: Int = 16,
  maximumFeatureDim: Int = 128,
  maximumTokens: Int = 1024,
  scaleLanes: Int = 4,
  useBufferedMetadata: Boolean = true,
  enableStats: Boolean = true,
  quantParameterArchitecture: QuantParameterArchitecture =
    QuantParameterArchitecture.V1SingleStage,
  sharedDecompressor: Boolean = false
) extends Module {
  private val params = BriskKvFormatV0.params
  private val streamCount = 11
  private val tokenIndexBits = log2Ceil(params.blockTokens)
  private val bucketCountBits = log2Ceil(params.blockTokens + 1)
  private val maximumBlocks = maximumTokens / params.blockTokens
  private val maximumDescriptors =
    (maximumTokens / params.packTokens) * maximumFeatureDim

  require(maximumTokens >= params.blockTokens)
  require(maximumTokens % params.blockTokens == 0)
  require(maximumFeatureDim >= 2 && isPow2(maximumFeatureDim))
  require(inputFractionalBits == 12)

  private def bytesForBits(items: Long, bits: Int): Int =
    math.max(1, ((items * bits + 7L) / 8L).toInt)
  private val streamCapacities = IndexedSeq(
    bytesForBits(maximumDescriptors, params.kQuantBits),
    bytesForBits(maximumDescriptors, params.packWidthBits),
    bytesForBits(maximumTokens.toLong * maximumFeatureDim, params.kQuantBits),
    bytesForBits(maximumTokens, params.kZeroBits),
    bytesForBits(maximumTokens, params.exponentBits),
    bytesForBits(maximumDescriptors, params.vQuantBits),
    bytesForBits(maximumDescriptors, params.packWidthBits),
    bytesForBits(maximumTokens.toLong * maximumFeatureDim, params.vQuantBits),
    bytesForBits(maximumTokens, params.vZeroBits),
    bytesForBits(maximumTokens, params.exponentBits),
    math.max(1, maximumBlocks * params.bucketHeaderBytesPerBlock)
  )

  val io = IO(
    new BriskKvJitVSingleHeadTileTopIO(
      inputBits,
      valueBits,
      outputBits,
      countBits,
      tagBits,
      tokenIndexBits,
      bucketCountBits,
      streamCount
    )
  )
  val writer = Module(
    new BriskKvWriteEncoderTop(
      inputBits = inputBits,
      inputFractionalBits = inputFractionalBits,
      maximumFeatureDim = maximumFeatureDim,
      maximumTokens = maximumTokens,
      countBits = countBits,
      tagBits = tagBits,
      enableStats = enableStats,
      quantParameterArchitecture = quantParameterArchitecture
    )
  )
  val attention = Module(
    new BriskKvJitVAttentionTop(
      valueBits = valueBits,
      qkAccumulatorBits = qkAccumulatorBits,
      avAccumulatorBits = avAccumulatorBits,
      outputBits = outputBits,
      countBits = countBits,
      tagBits = tagBits,
      maximumFeatureDim = maximumFeatureDim,
      maximumTokens = maximumTokens,
      scaleLanes = scaleLanes,
      useBufferedMetadata = useBufferedMetadata,
      enableStats = enableStats,
      sharedDecompressor = sharedDecompressor
    )
  )
  val stores = streamCapacities.map(capacity =>
    Module(new ReplayByteStreamBuffer(capacity, countBits))
  )

  private val Seq(sIdle, sWriting, sSealCheck, sStored, sLaunch, sAttention) =
    Enum(6)
  val state = RegInit(sIdle)
  val featureDimReg = RegInit(0.U(countBits.W))
  val tokenCountReg = RegInit(0.U(countBits.W))
  val descriptorCountReg = RegInit(0.U(countBits.W))
  val attentionTagReg = RegInit(0.U(tagBits.W))
  val writeDoneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)

  val geometryValid = io.featureDim =/= 0.U &&
    io.featureDim <= maximumFeatureDim.U && !io.featureDim(0) &&
    io.blockCount =/= 0.U && io.blockCount <= maximumBlocks.U &&
    io.firstBlockIndex === 0.U
  val writeAccept = io.writeStart && (state === sIdle || state === sStored)
  val validWriteAccept = writeAccept && geometryValid
  val sealStores = writer.io.done && state === sWriting
  val anyOverflow = stores.map(_.io.overflow).reduce(_ || _)

  writer.io.start := validWriteAccept
  writer.io.featureDim := io.featureDim
  writer.io.blockCount := io.blockCount
  writer.io.firstBlockIndex := io.firstBlockIndex
  writer.io.in <> io.writeIn
  val writerStreams = IndexedSeq(
    writer.io.kMinimumOut,
    writer.io.kWidthOut,
    writer.io.kPayloadOut,
    writer.io.kZeroOut,
    writer.io.kExponentOut,
    writer.io.vMinimumOut,
    writer.io.vWidthOut,
    writer.io.vPayloadOut,
    writer.io.vZeroOut,
    writer.io.vExponentOut,
    writer.io.bucketCountOut
  )
  val attentionStreams = IndexedSeq(
    attention.io.kMinimumIn,
    attention.io.kWidthIn,
    attention.io.kPayloadIn,
    attention.io.kZeroPointIn,
    attention.io.kExponentIn,
    attention.io.vMinimumIn,
    attention.io.vWidthIn,
    attention.io.vPayloadIn,
    attention.io.vZeroPointIn,
    attention.io.vExponentIn,
    attention.io.bucketCountIn
  )
  stores.indices.foreach { index =>
    stores(index).io.clear := writeAccept
    stores(index).io.seal := sealStores
    stores(index).io.readStart := (if (index < 5) {
      attention.io.kStreamStart
    } else if (index < 10) {
      attention.io.vStreamStart
    } else {
      attention.io.bucketStreamStart
    })
    stores(index).io.writeIn <> writerStreams(index)
    attentionStreams(index) <> stores(index).io.readOut
    io.storedLengths(index) := stores(index).io.length
  }

  val attentionCommandFire = attention.io.command.valid && attention.io.command.ready
  attention.io.command.valid := state === sLaunch
  attention.io.command.bits.tag := attentionTagReg
  attention.io.command.bits.tokenCount := tokenCountReg
  attention.io.command.bits.featureDim := featureDimReg
  attention.io.command.bits.descriptorCount := descriptorCountReg
  attention.io.command.bits.kPayloadByteCount := stores(2).io.length
  attention.io.command.bits.vPayloadByteCount := stores(7).io.length
  attention.io.queryLoadIn <> io.queryIn
  io.bucketOut <> attention.io.bucketOut
  io.attentionOut <> attention.io.attentionOut
  io.result <> attention.io.result

  writeDoneReg := false.B
  when(writeAccept) {
    errorReg := !geometryValid
    writeDoneReg := !geometryValid
    when(geometryValid) {
      featureDimReg := io.featureDim
      tokenCountReg := io.blockCount << log2Ceil(params.blockTokens)
      descriptorCountReg :=
        (io.blockCount * io.featureDim) <<
          log2Ceil(params.blockTokens / params.packTokens)
      state := sWriting
    }
  }.elsewhen(state === sWriting && writer.io.done) {
    when(writer.io.error) {
      errorReg := true.B
      writeDoneReg := true.B
      state := sIdle
    }.otherwise { state := sSealCheck }
  }.elsewhen(state === sSealCheck) {
    writeDoneReg := true.B
    when(anyOverflow) {
      errorReg := true.B
      state := sIdle
    }.otherwise { state := sStored }
  }.elsewhen(state === sStored && io.attentionStart && !io.writeStart) {
    attentionTagReg := io.attentionTag
    state := sLaunch
  }.elsewhen(state === sLaunch && attentionCommandFire) {
    state := sAttention
  }.elsewhen(state === sAttention && io.result.valid && io.result.ready) {
    when(io.result.bits.error) {
      errorReg := true.B
      state := sIdle
    }.otherwise { state := sStored }
  }

  io.writeReady := state === sIdle || state === sStored
  io.attentionReady := state === sStored
  io.writeDone := writeDoneReg
  io.encodedReady := state === sStored
  io.busy := (state =/= sIdle && state =/= sStored) ||
    writer.io.busy || attention.io.busy
  io.error := errorReg || writer.io.error || anyOverflow
  io.writeStats := writer.io.stats
  io.attentionProgress := attention.io.progress
}

/** Distinct synthesis top for the area-first shared-decoder ablation. */
class BriskKvSharedJitVSingleHeadTileTop(
  inputBits: Int = 24,
  inputFractionalBits: Int = 12,
  valueBits: Int = 18,
  qkAccumulatorBits: Int = 44,
  avAccumulatorBits: Int = 50,
  outputBits: Int = 18,
  countBits: Int = 32,
  tagBits: Int = 16,
  maximumFeatureDim: Int = 128,
  maximumTokens: Int = 1024,
  scaleLanes: Int = 4,
  useBufferedMetadata: Boolean = true,
  enableStats: Boolean = true,
  quantParameterArchitecture: QuantParameterArchitecture =
    QuantParameterArchitecture.V1SingleStage
) extends BriskKvJitVSingleHeadTileTop(
      inputBits,
      inputFractionalBits,
      valueBits,
      qkAccumulatorBits,
      avAccumulatorBits,
      outputBits,
      countBits,
      tagBits,
      maximumFeatureDim,
      maximumTokens,
      scaleLanes,
      useBufferedMetadata,
      enableStats,
      quantParameterArchitecture,
      sharedDecompressor = true
    )
