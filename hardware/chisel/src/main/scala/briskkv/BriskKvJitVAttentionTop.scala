package briskkv

import chisel3._
import chisel3.util._

class BriskKvJitVAttentionProgress(countBits: Int) extends Bundle {
  val kDecompression = new DecompressionProgress(countBits)
  val vDecompression = new DecompressionProgress(countBits)
  val kPacketizer = new AttentionPacketizerStats
  val vPacketizer = new AttentionPacketizerStats
  val qk = new QkComputePipelineStats
  val scaleSoftmax = new QkScaleSoftmaxStats
  val jitV = new JitVAccumulatorStats
  val outputQuantizer = new AvOutputQuantizerStats
  val bucketRecords = UInt(countBits.W)
  val vLaunched = Bool()
}

class BriskKvJitVAttentionTopIO(
  valueBits: Int,
  outputBits: Int,
  countBits: Int,
  tagBits: Int,
  bucketCountBits: Int
) extends Bundle {
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
  val kStreamStart = Output(Bool())
  val vStreamStart = Output(Bool())
  val bucketStreamStart = Output(Bool())
  val busy = Output(Bool())
  val queryLoaded = Output(Bool())
  val weightsLoaded = Output(Bool())
  val progress = Output(new BriskKvJitVAttentionProgress(countBits))
}

/** Selectable dual/shared-decoder JIT-V attention path.
  *
  * K and bucket streams start with the attention command. V remains compressed
  * until the first Softmax weight packet has entered JitVAccumulator, then V
  * decode overlaps the remaining weight production. The default retains two
  * decoder datapaths for the storage-focused ablation; sharedDecompressor
  * selects the area-first follow-up.
  */
class BriskKvJitVAttentionTop(
  valueBits: Int = 18,
  qkAccumulatorBits: Int = 44,
  avAccumulatorBits: Int = 50,
  outputBits: Int = 18,
  countBits: Int = 32,
  tagBits: Int = 16,
  maximumFeatureDim: Int = 256,
  maximumTokens: Int = 16384,
  scaleLanes: Int = 4,
  useBufferedMetadata: Boolean = true,
  enableStats: Boolean = true,
  sharedDecompressor: Boolean = false
) extends Module {
  private val params = BriskKvFormatV0.params
  private val bucketCountBits = log2Ceil(params.blockTokens + 1)

  val io = IO(
    new BriskKvJitVAttentionTopIO(
      valueBits,
      outputBits,
      countBits,
      tagBits,
      bucketCountBits
    )
  )

  val decompression: JitVDecompressionPairBase = if (sharedDecompressor) {
    Module(
      new SharedJitVDecompressionPair(
        valueBits,
        countBits,
        tagBits,
        useBufferedMetadata,
        enableStats
      )
    )
  } else {
    Module(
      new DualJitVDecompressionPair(
        valueBits,
        countBits,
        tagBits,
        useBufferedMetadata,
        enableStats
      )
    )
  }
  val kPacketizer = Module(
    new AttentionFeaturePacketizer(valueBits, countBits = countBits, enableStats = enableStats)
  )
  val vPacketizer = Module(
    new AttentionFeaturePacketizer(valueBits, countBits = countBits, enableStats = enableStats)
  )
  val qk = Module(
    new QkComputePipeline(
      valueBits = valueBits,
      accumulatorBits = qkAccumulatorBits,
      maximumFeatureDim = maximumFeatureDim,
      countBits = countBits,
      enableStats = enableStats
    )
  )
  val scaleSoftmax = Module(
    new QkScaleSoftmaxPipeline(
      logitBits = qkAccumulatorBits,
      scaleLanes = scaleLanes,
      maximumFeatureDim = maximumFeatureDim,
      maximumTokens = maximumTokens,
      countBits = countBits,
      enableStats = enableStats
    )
  )
  val jitV = Module(
    new JitVAccumulator(
      valueBits = valueBits,
      accumulatorBits = avAccumulatorBits,
      maximumFeatureDim = maximumFeatureDim,
      maximumTokens = maximumTokens,
      countBits = countBits,
      enableStats = enableStats
    )
  )
  val outputQuantizer = Module(
    new AvOutputQuantizer(
      accumulatorBits = avAccumulatorBits,
      outputBits = outputBits,
      maximumFeatureDim = maximumFeatureDim,
      countBits = countBits,
      enableStats = enableStats
    )
  )
  val bucketDecoder = Module(new BucketCountDecoder(blockIndexBits = countBits))

  private val Seq(sIdle, sLaunchK, sRunning, sResponse) = Enum(4)
  val state = RegInit(sIdle)
  val commandReg = Reg(
    new DualKvDecompressionCommand(countBits, tagBits)
  )
  val packCountReg = RegInit(0.U(countBits.W))
  val blockCountReg = RegInit(0.U(countBits.W))
  val commandError = RegInit(false.B)
  val vLaunched = RegInit(false.B)
  val kResultSeen = RegInit(false.B)
  val vResultSeen = RegInit(false.B)
  val bucketDoneSeen = RegInit(false.B)
  val outputDoneSeen = RegInit(false.B)
  val bucketRecords = RegInit(0.U(countBits.W))
  val kResultReg = RegInit(0.U.asTypeOf(new DecompressionResult(countBits, tagBits)))
  val vResultReg = RegInit(0.U.asTypeOf(new DecompressionResult(countBits, tagBits)))

  io.command.ready := state === sIdle
  val commandFire = io.command.valid && io.command.ready
  val requestedPackCount =
    (io.command.bits.tokenCount + (params.packTokens - 1).U) >>
      log2Ceil(params.packTokens)
  val requestedBlockCount =
    (io.command.bits.tokenCount + (params.blockTokens - 1).U) >>
      log2Ceil(params.blockTokens)
  val geometryValid = io.command.bits.tokenCount =/= 0.U &&
    io.command.bits.tokenCount <= maximumTokens.U &&
    io.command.bits.featureDim =/= 0.U &&
    io.command.bits.featureDim <= maximumFeatureDim.U &&
    io.command.bits.descriptorCount ===
      requestedPackCount * io.command.bits.featureDim

  val launchK = state === sLaunchK && decompression.io.kReady
  val launchV = state === sRunning && !vLaunched && jitV.io.vLaunchReady &&
    decompression.io.vReady
  io.kStreamStart := launchK
  io.bucketStreamStart := launchK
  io.vStreamStart := launchV

  decompression.io.kStart := launchK
  decompression.io.vStart := launchV
  decompression.io.tag := commandReg.tag
  decompression.io.tokenCount := commandReg.tokenCount
  decompression.io.featureDim := commandReg.featureDim
  decompression.io.descriptorCount := commandReg.descriptorCount
  decompression.io.kPayloadByteCount := commandReg.kPayloadByteCount
  decompression.io.vPayloadByteCount := commandReg.vPayloadByteCount
  decompression.io.kMinimumIn <> io.kMinimumIn
  decompression.io.kWidthIn <> io.kWidthIn
  decompression.io.kPayloadIn <> io.kPayloadIn
  decompression.io.kZeroPointIn <> io.kZeroPointIn
  decompression.io.kExponentIn <> io.kExponentIn
  decompression.io.vMinimumIn <> io.vMinimumIn
  decompression.io.vWidthIn <> io.vWidthIn
  decompression.io.vPayloadIn <> io.vPayloadIn
  decompression.io.vZeroPointIn <> io.vZeroPointIn
  decompression.io.vExponentIn <> io.vExponentIn

  kPacketizer.io.start := launchK
  kPacketizer.io.tokenCount := commandReg.tokenCount
  kPacketizer.io.featureDim := commandReg.featureDim
  kPacketizer.io.descriptorCount := commandReg.descriptorCount
  kPacketizer.io.in <> decompression.io.kOut
  vPacketizer.io.start := launchV
  vPacketizer.io.tokenCount := commandReg.tokenCount
  vPacketizer.io.featureDim := commandReg.featureDim
  vPacketizer.io.descriptorCount := commandReg.descriptorCount
  vPacketizer.io.in <> decompression.io.vOut

  qk.io.start := launchK
  qk.io.featureDim := commandReg.featureDim
  qk.io.packCount := packCountReg
  qk.io.queryLoadIn <> io.queryLoadIn
  qk.io.keyIn <> kPacketizer.io.out
  scaleSoftmax.io.start := launchK
  scaleSoftmax.io.featureDim := commandReg.featureDim
  scaleSoftmax.io.tokenCount := commandReg.tokenCount
  scaleSoftmax.io.in <> qk.io.out
  jitV.io.start := launchK
  jitV.io.featureDim := commandReg.featureDim
  jitV.io.tokenCount := commandReg.tokenCount
  jitV.io.weightIn <> scaleSoftmax.io.out
  jitV.io.vIn <> vPacketizer.io.out
  outputQuantizer.io.start := launchK
  outputQuantizer.io.featureDim := commandReg.featureDim
  outputQuantizer.io.in <> jitV.io.out
  io.attentionOut <> outputQuantizer.io.out

  bucketDecoder.io.start := launchK
  bucketDecoder.io.blockCount := blockCountReg
  bucketDecoder.io.in <> io.bucketCountIn
  io.bucketOut <> bucketDecoder.io.out
  val bucketFire = io.bucketOut.valid && io.bucketOut.ready

  decompression.io.kResult.ready := state === sRunning && !kResultSeen
  decompression.io.vResult.ready := state === sRunning && !vResultSeen
  val kResultFire = decompression.io.kResult.valid && decompression.io.kResult.ready
  val vResultFire = decompression.io.vResult.valid && decompression.io.vResult.ready

  switch(state) {
    is(sIdle) {
      when(commandFire) {
        commandReg := io.command.bits
        packCountReg := requestedPackCount
        blockCountReg := requestedBlockCount
        commandError := !geometryValid
        vLaunched := false.B
        kResultSeen := false.B
        vResultSeen := false.B
        bucketDoneSeen := false.B
        outputDoneSeen := false.B
        bucketRecords := 0.U
        kResultReg := 0.U.asTypeOf(new DecompressionResult(countBits, tagBits))
        vResultReg := 0.U.asTypeOf(new DecompressionResult(countBits, tagBits))
        state := Mux(geometryValid, sLaunchK, sResponse)
      }
    }
    is(sLaunchK) {
      when(launchK) { state := sRunning }
    }
    is(sRunning) {
      when(launchV) { vLaunched := true.B }
      when(bucketFire) { bucketRecords := bucketRecords + 1.U }
      when(bucketDecoder.io.done) { bucketDoneSeen := true.B }
      when(outputQuantizer.io.done) { outputDoneSeen := true.B }
      when(kResultFire) {
        kResultReg := decompression.io.kResult.bits
        kResultSeen := true.B
      }
      when(vResultFire) {
        vResultReg := decompression.io.vResult.bits
        vResultSeen := true.B
      }
      val allDone = (kResultSeen || kResultFire) &&
        (vResultSeen || vResultFire) &&
        (bucketDoneSeen || bucketDecoder.io.done) &&
        (outputDoneSeen || outputQuantizer.io.done)
      when(allDone) {
        when(kResultFire) { kResultReg := decompression.io.kResult.bits }
        when(vResultFire) { vResultReg := decompression.io.vResult.bits }
        state := sResponse
      }
    }
    is(sResponse) {
      when(io.result.valid && io.result.ready) { state := sIdle }
    }
  }

  io.result.valid := state === sResponse
  io.result.bits := 0.U.asTypeOf(new DualKvDecompressionResult(countBits, tagBits))
  io.result.bits.tag := commandReg.tag
  io.result.bits.tokenCount := commandReg.tokenCount
  io.result.bits.packCount := packCountReg
  io.result.bits.blockCount := blockCountReg
  io.result.bits.descriptorCount := commandReg.descriptorCount
  io.result.bits.bucketRecords := bucketRecords
  io.result.bits.error := commandError || kResultReg.error || vResultReg.error ||
    decompression.io.error || bucketDecoder.io.error ||
    kPacketizer.io.error || vPacketizer.io.error ||
    qk.io.error || scaleSoftmax.io.error || jitV.io.error ||
    outputQuantizer.io.error || bucketRecords =/= blockCountReg
  if (enableStats) {
    io.result.bits.kStats := kResultReg.stats
    io.result.bits.vStats := vResultReg.stats
  } else {
    io.result.bits.kStats := 0.U.asTypeOf(new DequantizerPerformanceStats)
    io.result.bits.vStats := 0.U.asTypeOf(new DequantizerPerformanceStats)
  }

  io.busy := state =/= sIdle || decompression.io.busy ||
    qk.io.busy || scaleSoftmax.io.busy || jitV.io.busy || outputQuantizer.io.busy
  io.queryLoaded := qk.io.queryLoaded
  io.weightsLoaded := jitV.io.weightsLoaded
  io.progress.kDecompression := decompression.io.kProgress
  io.progress.vDecompression := decompression.io.vProgress
  io.progress.kPacketizer := kPacketizer.io.stats
  io.progress.vPacketizer := vPacketizer.io.stats
  io.progress.qk := qk.io.stats
  io.progress.scaleSoftmax := scaleSoftmax.io.stats
  io.progress.jitV := jitV.io.stats
  io.progress.outputQuantizer := outputQuantizer.io.stats
  io.progress.bucketRecords := bucketRecords
  io.progress.vLaunched := vLaunched || launchV
}
