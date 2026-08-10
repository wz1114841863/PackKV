package briskkv

import chisel3._
import chisel3.util._

class DualKvDecompressionCommand(countBits: Int, tagBits: Int) extends Bundle {
  val tag = UInt(tagBits.W)
  val tokenCount = UInt(countBits.W)
  val featureDim = UInt(countBits.W)
  val descriptorCount = UInt(countBits.W)
  val kPayloadByteCount = UInt(countBits.W)
  val vPayloadByteCount = UInt(countBits.W)
}

class DualKvDecompressionResult(countBits: Int, tagBits: Int) extends Bundle {
  val tag = UInt(tagBits.W)
  val error = Bool()
  val tokenCount = UInt(countBits.W)
  val packCount = UInt(countBits.W)
  val blockCount = UInt(countBits.W)
  val descriptorCount = UInt(countBits.W)
  val bucketRecords = UInt(countBits.W)
  val kStats = new DequantizerPerformanceStats
  val vStats = new DequantizerPerformanceStats
}

class DualKvDecompressionProgress(countBits: Int) extends Bundle {
  val k = new DecompressionProgress(countBits)
  val v = new DecompressionProgress(countBits)
  val bucketRecords = UInt(countBits.W)
}

class DualKvDecompressionControllerIO(
  outputBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int,
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
  val kOut = Decoupled(
    new FixedPointDequantizedValue(outputBits, countBits, tokenIndexBits)
  )
  val vOut = Decoupled(
    new FixedPointDequantizedValue(outputBits, countBits, tokenIndexBits)
  )
  val bucketOut = Decoupled(new BucketCountRecord(bucketCountBits, countBits))
  val result = Decoupled(new DualKvDecompressionResult(countBits, tagBits))
  val busy = Output(Bool())
  val progress = Output(new DualKvDecompressionProgress(countBits))
}

/** Shared lifecycle controller for K, V, and bucket-count decode streams. */
class DualKvDecompressionController(
  outputBits: Int = 18,
  countBits: Int = 32,
  tagBits: Int = 16,
  useBufferedMetadata: Boolean = true
) extends Module {
  private val params = BriskKvFormatV0.params
  private val tokenIndexBits = log2Ceil(params.packTokens)
  private val bucketCountBits = log2Ceil(params.blockTokens + 1)
  private object DualState extends ChiselEnum {
    val idle, launch, running, response = Value
  }

  val io = IO(
    new DualKvDecompressionControllerIO(
      outputBits,
      countBits,
      tagBits,
      tokenIndexBits,
      bucketCountBits
    )
  )
  val kController = Module(
    new DecompressionPipelineController(
      codeValueBits = params.kQuantBits,
      zeroPointBits = params.kZeroBits,
      outputBits = outputBits,
      countBits = countBits,
      tagBits = tagBits,
      useBufferedMetadata = useBufferedMetadata
    )
  )
  val vController = Module(
    new DecompressionPipelineController(
      codeValueBits = params.vQuantBits,
      zeroPointBits = params.vZeroBits,
      outputBits = outputBits,
      countBits = countBits,
      tagBits = tagBits,
      useBufferedMetadata = useBufferedMetadata
    )
  )
  val bucketDecoder = Module(new BucketCountDecoder(blockIndexBits = countBits))

  private val state = RegInit(DualState.idle)
  val commandReg = Reg(new DualKvDecompressionCommand(countBits, tagBits))
  val packCountReg = RegInit(0.U(countBits.W))
  val blockCountReg = RegInit(0.U(countBits.W))
  val bucketRecords = RegInit(0.U(countBits.W))
  val commandLaunched = RegInit(false.B)
  val kResultSeen = RegInit(false.B)
  val vResultSeen = RegInit(false.B)
  val bucketDoneSeen = RegInit(false.B)
  val kResultReg = RegInit(0.U.asTypeOf(new DecompressionResult(countBits, tagBits)))
  val vResultReg = RegInit(0.U.asTypeOf(new DecompressionResult(countBits, tagBits)))
  val resultError = RegInit(false.B)

  io.command.ready := state === DualState.idle
  io.busy := state === DualState.launch || state === DualState.running

  val childrenReady = kController.io.command.ready &&
    vController.io.command.ready
  val launchFire = state === DualState.launch && childrenReady
  kController.io.command.valid := launchFire
  vController.io.command.valid := launchFire
  kController.io.command.bits.tag := commandReg.tag
  kController.io.command.bits.tokenCount := commandReg.tokenCount
  kController.io.command.bits.featureDim := commandReg.featureDim
  kController.io.command.bits.descriptorCount := commandReg.descriptorCount
  kController.io.command.bits.payloadByteCount := commandReg.kPayloadByteCount
  vController.io.command.bits.tag := commandReg.tag
  vController.io.command.bits.tokenCount := commandReg.tokenCount
  vController.io.command.bits.featureDim := commandReg.featureDim
  vController.io.command.bits.descriptorCount := commandReg.descriptorCount
  vController.io.command.bits.payloadByteCount := commandReg.vPayloadByteCount

  bucketDecoder.io.start := launchFire
  bucketDecoder.io.blockCount := blockCountReg
  bucketDecoder.io.in <> io.bucketCountIn
  io.bucketOut <> bucketDecoder.io.out

  kController.io.minimumIn <> io.kMinimumIn
  kController.io.widthIn <> io.kWidthIn
  kController.io.payloadIn <> io.kPayloadIn
  kController.io.zeroPointIn <> io.kZeroPointIn
  kController.io.exponentIn <> io.kExponentIn
  io.kOut <> kController.io.out
  vController.io.minimumIn <> io.vMinimumIn
  vController.io.widthIn <> io.vWidthIn
  vController.io.payloadIn <> io.vPayloadIn
  vController.io.zeroPointIn <> io.vZeroPointIn
  vController.io.exponentIn <> io.vExponentIn
  io.vOut <> vController.io.out

  kController.io.result.ready := state === DualState.running && !kResultSeen
  vController.io.result.ready := state === DualState.running && !vResultSeen
  val kResultFire = kController.io.result.valid && kController.io.result.ready
  val vResultFire = vController.io.result.valid && vController.io.result.ready
  val bucketOutputFire = io.bucketOut.valid && io.bucketOut.ready

  val zeroProgress = 0.U.asTypeOf(new DecompressionProgress(countBits))
  io.progress.k := Mux(commandLaunched, kController.io.progress, zeroProgress)
  io.progress.v := Mux(commandLaunched, vController.io.progress, zeroProgress)
  io.progress.bucketRecords := bucketRecords

  io.result.valid := state === DualState.response
  io.result.bits.tag := commandReg.tag
  io.result.bits.error := resultError
  io.result.bits.tokenCount := commandReg.tokenCount
  io.result.bits.packCount := packCountReg
  io.result.bits.blockCount := blockCountReg
  io.result.bits.descriptorCount := commandReg.descriptorCount
  io.result.bits.bucketRecords := bucketRecords
  io.result.bits.kStats := kResultReg.stats
  io.result.bits.vStats := vResultReg.stats

  val commandFire = io.command.valid && io.command.ready

  switch(state) {
    is(DualState.idle) {
      when(commandFire) {
        val requestedPackCount =
          (io.command.bits.tokenCount + (params.packTokens - 1).U) >>
            log2Ceil(params.packTokens)
        val requestedBlockCount =
          (io.command.bits.tokenCount + (params.blockTokens - 1).U) >>
            log2Ceil(params.blockTokens)
        val expectedDescriptorCount =
          requestedPackCount * io.command.bits.featureDim
        val commandValid = io.command.bits.tokenCount =/= 0.U &&
          io.command.bits.featureDim =/= 0.U &&
          io.command.bits.descriptorCount === expectedDescriptorCount

        commandReg := io.command.bits
        packCountReg := requestedPackCount
        blockCountReg := requestedBlockCount
        bucketRecords := 0.U
        commandLaunched := false.B
        kResultSeen := false.B
        vResultSeen := false.B
        bucketDoneSeen := false.B
        kResultReg := 0.U.asTypeOf(new DecompressionResult(countBits, tagBits))
        vResultReg := 0.U.asTypeOf(new DecompressionResult(countBits, tagBits))
        resultError := !commandValid
        state := Mux(commandValid, DualState.launch, DualState.response)
      }
    }

    is(DualState.launch) {
      when(launchFire) {
        commandLaunched := true.B
        state := DualState.running
      }
    }

    is(DualState.running) {
      when(bucketOutputFire) {
        bucketRecords := bucketRecords + 1.U
      }
      when(kResultFire) {
        kResultReg := kController.io.result.bits
        kResultSeen := true.B
      }
      when(vResultFire) {
        vResultReg := vController.io.result.bits
        vResultSeen := true.B
      }
      when(bucketDecoder.io.done) {
        bucketDoneSeen := true.B
      }

      val allDone = (kResultSeen || kResultFire) &&
        (vResultSeen || vResultFire) &&
        (bucketDoneSeen || bucketDecoder.io.done)
      when(allDone) {
        val selectedKResult = Mux(kResultFire, kController.io.result.bits, kResultReg)
        val selectedVResult = Mux(vResultFire, vController.io.result.bits, vResultReg)
        val completedBucketRecords = bucketRecords + bucketOutputFire.asUInt
        kResultReg := selectedKResult
        vResultReg := selectedVResult
        resultError := selectedKResult.error || selectedVResult.error ||
          bucketDecoder.io.error ||
          selectedKResult.tag =/= commandReg.tag ||
          selectedVResult.tag =/= commandReg.tag ||
          selectedKResult.tokenCount =/= commandReg.tokenCount ||
          selectedVResult.tokenCount =/= commandReg.tokenCount ||
          completedBucketRecords =/= blockCountReg
        state := DualState.response
      }
    }

    is(DualState.response) {
      when(io.result.valid && io.result.ready) {
        state := DualState.idle
      }
    }
  }
}
