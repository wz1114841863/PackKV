package briskkv

import chisel3._
import chisel3.util._

class BriskKvWriteInput(
  inputBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int
) extends Bundle {
  val kFixedRaw = SInt(inputBits.W)
  val vFixedRaw = SInt(inputBits.W)
  val tokenTag = UInt(tagBits.W)
  val blockIndex = UInt(countBits.W)
  val tokenIndex = UInt(tokenIndexBits.W)
  val featureIndex = UInt(countBits.W)
  val lastFeature = Bool()
  val last = Bool()
}

class BriskKvWriteEncoderStats extends Bundle {
  val activeCycles = UInt(64.W)
  val inputPairs = UInt(64.W)
  val completedTokens = UInt(64.W)
  val completedBlocks = UInt(64.W)
  val sourceWaitCycles = UInt(64.W)
  val rejectedTransactions = UInt(64.W)
}

class BriskKvWriteEncoderTopIO(
  inputBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val blockCount = Input(UInt(countBits.W))
  val firstBlockIndex = Input(UInt(countBits.W))
  val in = Flipped(
    Decoupled(
      new BriskKvWriteInput(inputBits, countBits, tagBits, tokenIndexBits)
    )
  )

  val kMinimumOut = Decoupled(UInt(8.W))
  val kWidthOut = Decoupled(UInt(8.W))
  val kPayloadOut = Decoupled(UInt(8.W))
  val vMinimumOut = Decoupled(UInt(8.W))
  val vWidthOut = Decoupled(UInt(8.W))
  val vPayloadOut = Decoupled(UInt(8.W))
  val kZeroOut = Decoupled(UInt(8.W))
  val kExponentOut = Decoupled(UInt(8.W))
  val vZeroOut = Decoupled(UInt(8.W))
  val vExponentOut = Decoupled(UInt(8.W))
  val bucketCountOut = Decoupled(UInt(8.W))

  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new BriskKvWriteEncoderStats)
}

/** Unified Format v0 write-side K/V compression pipeline.
  *
  * One transaction contains `blockCount` complete 64-token blocks. Input is
  * block-major, token-major, feature-major and K/V values are accepted
  * atomically. Each token is quantized independently; both quantized streams
  * then share one stable k_sum bucket permutation. Routed q values are
  * transposed and dynamically bit-packed, while routed metadata and bucket
  * occupancies are serialized into their compact component streams.
  *
  * The current q component encoder is block-scoped. Its fields therefore
  * concatenate without intermediate padding only for an even feature dimension;
  * unsupported geometry is rejected at command acceptance. Metadata encoders
  * span the complete multi-block transaction. Any runtime child error is fatal
  * for the active transaction and requires reset because the existing child
  * interfaces do not provide an abort handshake.
  */
class BriskKvWriteEncoderTop(
  inputBits: Int = 24,
  inputFractionalBits: Int = 12,
  maximumFeatureDim: Int = 256,
  metadataBits: Int = 8,
  countBits: Int = 32,
  tagBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  private val format = BriskKvFormatV0.params
  private val tokenIndexBits = log2Ceil(format.blockTokens)

  require(maximumFeatureDim >= 2 && isPow2(maximumFeatureDim))
  require(inputFractionalBits == 12)

  val io = IO(
    new BriskKvWriteEncoderTopIO(
      inputBits,
      countBits,
      tagBits,
      tokenIndexBits
    )
  )

  val kQuantizer = Module(
    new KvWriteQuantizer(
      isKey = true,
      inputBits = inputBits,
      inputFractionalBits = inputFractionalBits,
      maximumFeatureDim = maximumFeatureDim,
      metadataBits = metadataBits,
      countBits = countBits,
      tagBits = tagBits,
      enableStats = enableStats
    )
  )
  val vQuantizer = Module(
    new KvWriteQuantizer(
      isKey = false,
      inputBits = inputBits,
      inputFractionalBits = inputFractionalBits,
      maximumFeatureDim = maximumFeatureDim,
      metadataBits = metadataBits,
      countBits = countBits,
      tagBits = tagBits,
      enableStats = enableStats
    )
  )
  val router = Module(
    new KvTokenJoinBucketRouter(
      maximumFeatureDim,
      metadataBits,
      countBits,
      tagBits,
      enableStats
    )
  )
  val qEncoder = Module(
    new KvPackTransposeBitPackEncoder(
      maximumFeatureDim,
      countBits,
      tagBits,
      enableStats
    )
  )
  val metadataEncoder = Module(
    new CompactKvMetadataEncoder(
      metadataBits,
      countBits,
      tagBits,
      enableStats
    )
  )
  val bucketEncoder = Module(
    new BucketCountEncoder(countBits, enableStats)
  )

  private val Seq(
    sIdle,
    sBlockStart,
    sTokenStart,
    sTokenInput,
    sTokenDrain,
    sBlockDrain,
    sFault
  ) = Enum(7)
  val state = RegInit(sIdle)
  val featureDimReg = RegInit(0.U(countBits.W))
  val blocksRemaining = RegInit(0.U(countBits.W))
  val blockIndexReg = RegInit(0.U(countBits.W))
  val tokenIndexReg = RegInit(0.U(tokenIndexBits.W))
  val tokenTagReg = RegInit(0.U(tagBits.W))
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val kQuantizerDone = RegInit(false.B)
  val vQuantizerDone = RegInit(false.B)
  val routerDone = RegInit(false.B)
  val qEncoderDone = RegInit(false.B)
  val metadataEncoderDone = RegInit(false.B)
  val bucketEncoderDone = RegInit(false.B)

  val activeCycles = RegInit(0.U(64.W))
  val inputPairs = RegInit(0.U(64.W))
  val completedTokens = RegInit(0.U(64.W))
  val completedBlocks = RegInit(0.U(64.W))
  val sourceWaitCycles = RegInit(0.U(64.W))
  val rejectedTransactions = RegInit(0.U(64.W))

  private val maximumCount = (BigInt(1) << countBits) - 1
  private val maximumBlockCount = maximumCount >> tokenIndexBits
  val finalBlockIndexWide = io.firstBlockIndex +& io.blockCount - 1.U
  val commandValid = io.featureDim =/= 0.U &&
    io.featureDim <= maximumFeatureDim.U &&
    !io.featureDim(0) &&
    io.blockCount =/= 0.U &&
    io.blockCount <= maximumBlockCount.U &&
    finalBlockIndexWide <= maximumCount.U
  val finalBlock = blocksRemaining === 1.U
  val finalToken = tokenIndexReg === (format.blockTokens - 1).U
  val blockStart = state === sBlockStart
  val tokenStart = state === sTokenStart && io.in.valid
  val transactionStart = io.start && state === sIdle && commandValid

  router.io.start := blockStart
  router.io.featureDim := featureDimReg
  router.io.blockIndex := blockIndexReg
  router.io.blockLast := finalBlock
  qEncoder.io.start := blockStart
  qEncoder.io.featureDim := featureDimReg
  qEncoder.io.blockIndex := blockIndexReg
  metadataEncoder.io.start := transactionStart
  metadataEncoder.io.parameterCount :=
    (io.blockCount << tokenIndexBits)(countBits - 1, 0)
  metadataEncoder.io.firstBlockIndex := io.firstBlockIndex
  bucketEncoder.io.start := transactionStart
  bucketEncoder.io.blockCount := io.blockCount
  bucketEncoder.io.firstBlockIndex := io.firstBlockIndex

  kQuantizer.io.start := tokenStart
  kQuantizer.io.featureDim := featureDimReg
  kQuantizer.io.tokenTag := io.in.bits.tokenTag
  vQuantizer.io.start := tokenStart
  vQuantizer.io.featureDim := featureDimReg
  vQuantizer.io.tokenTag := io.in.bits.tokenTag

  val acceptingInput = state === sTokenInput
  val bothQuantizersReady = kQuantizer.io.in.ready && vQuantizer.io.in.ready
  io.in.ready := acceptingInput && bothQuantizersReady
  kQuantizer.io.in.valid := acceptingInput && io.in.valid &&
    vQuantizer.io.in.ready
  vQuantizer.io.in.valid := acceptingInput && io.in.valid &&
    kQuantizer.io.in.ready
  kQuantizer.io.in.bits.fixedRaw := io.in.bits.kFixedRaw
  kQuantizer.io.in.bits.featureIndex := io.in.bits.featureIndex
  kQuantizer.io.in.bits.last := io.in.bits.lastFeature
  vQuantizer.io.in.bits.fixedRaw := io.in.bits.vFixedRaw
  vQuantizer.io.in.bits.featureIndex := io.in.bits.featureIndex
  vQuantizer.io.in.bits.last := io.in.bits.lastFeature

  router.io.kMetadataIn <> kQuantizer.io.metadataOut
  router.io.vMetadataIn <> vQuantizer.io.metadataOut
  router.io.kQIn <> kQuantizer.io.qOut
  router.io.vQIn <> vQuantizer.io.qOut
  metadataEncoder.io.in <> router.io.metadataOut
  bucketEncoder.io.in <> router.io.bucketCountsOut
  qEncoder.io.in <> router.io.qOut

  io.kMinimumOut <> qEncoder.io.kMinimumOut
  io.kWidthOut <> qEncoder.io.kWidthOut
  io.kPayloadOut <> qEncoder.io.kPayloadOut
  io.vMinimumOut <> qEncoder.io.vMinimumOut
  io.vWidthOut <> qEncoder.io.vWidthOut
  io.vPayloadOut <> qEncoder.io.vPayloadOut
  io.kZeroOut <> metadataEncoder.io.kZeroOut
  io.kExponentOut <> metadataEncoder.io.kExponentOut
  io.vZeroOut <> metadataEncoder.io.vZeroOut
  io.vExponentOut <> metadataEncoder.io.vExponentOut
  io.bucketCountOut <> bucketEncoder.io.out

  val inputFire = io.in.valid && io.in.ready
  val expectedLastFeature = io.in.bits.featureIndex === featureDimReg - 1.U
  val expectedLastTransaction = finalBlock && finalToken && expectedLastFeature
  val inputOrderValid = io.in.bits.tokenTag === tokenTagReg &&
    io.in.bits.blockIndex === blockIndexReg &&
    io.in.bits.tokenIndex === tokenIndexReg &&
    io.in.bits.lastFeature === expectedLastFeature &&
    io.in.bits.last === expectedLastTransaction
  val anyChildError = kQuantizer.io.error || vQuantizer.io.error ||
    router.io.error || qEncoder.io.error || metadataEncoder.io.error ||
    bucketEncoder.io.error

  doneReg := false.B
  when(io.start && state === sIdle) {
    featureDimReg := io.featureDim
    blocksRemaining := io.blockCount
    blockIndexReg := io.firstBlockIndex
    tokenIndexReg := 0.U
    errorReg := !commandValid
    doneReg := !commandValid
    kQuantizerDone := false.B
    vQuantizerDone := false.B
    routerDone := false.B
    qEncoderDone := false.B
    metadataEncoderDone := false.B
    bucketEncoderDone := false.B
    activeCycles := 0.U
    inputPairs := 0.U
    completedTokens := 0.U
    completedBlocks := 0.U
    sourceWaitCycles := 0.U
    rejectedTransactions := Mux(commandValid, 0.U, 1.U)
    state := Mux(commandValid, sBlockStart, sIdle)
  }.elsewhen(io.start) {
    errorReg := true.B
  }.otherwise {
    when(state =/= sIdle) { activeCycles := activeCycles + 1.U }
    when((state === sTokenStart || state === sTokenInput) && !io.in.valid) {
      sourceWaitCycles := sourceWaitCycles + 1.U
    }

    when(blockStart) {
      routerDone := false.B
      qEncoderDone := false.B
      tokenIndexReg := 0.U
      state := sTokenStart
    }

    when(tokenStart) {
      tokenTagReg := io.in.bits.tokenTag
      kQuantizerDone := false.B
      vQuantizerDone := false.B
      when(
        io.in.bits.blockIndex =/= blockIndexReg ||
          io.in.bits.tokenIndex =/= tokenIndexReg ||
          io.in.bits.featureIndex =/= 0.U
      ) {
        errorReg := true.B
      }
      state := sTokenInput
    }

    when(inputFire) {
      inputPairs := inputPairs + 1.U
      when(!inputOrderValid) { errorReg := true.B }
      when(expectedLastFeature) { state := sTokenDrain }
    }

    when(kQuantizer.io.done) { kQuantizerDone := true.B }
    when(vQuantizer.io.done) { vQuantizerDone := true.B }
    val bothQuantizersDone =
      (kQuantizerDone || kQuantizer.io.done) &&
        (vQuantizerDone || vQuantizer.io.done)
    when(state === sTokenDrain && bothQuantizersDone) {
      completedTokens := completedTokens + 1.U
      when(finalToken) {
        state := sBlockDrain
      }.otherwise {
        tokenIndexReg := tokenIndexReg + 1.U
        state := sTokenStart
      }
    }

    when(router.io.done) { routerDone := true.B }
    when(qEncoder.io.done) { qEncoderDone := true.B }
    when(metadataEncoder.io.done) { metadataEncoderDone := true.B }
    when(bucketEncoder.io.done) { bucketEncoderDone := true.B }
    val blockCoreDone =
      (routerDone || router.io.done) &&
        (qEncoderDone || qEncoder.io.done)
    val transactionMetadataDone =
      (metadataEncoderDone || metadataEncoder.io.done) &&
        (bucketEncoderDone || bucketEncoder.io.done)
    when(
      state === sBlockDrain && blockCoreDone &&
        (!finalBlock || transactionMetadataDone)
    ) {
      completedBlocks := completedBlocks + 1.U
      when(finalBlock) {
        state := sIdle
        doneReg := true.B
      }.otherwise {
        blocksRemaining := blocksRemaining - 1.U
        blockIndexReg := blockIndexReg + 1.U
        state := sBlockStart
      }
    }

    when(state =/= sIdle && anyChildError) {
      errorReg := true.B
      rejectedTransactions := 1.U
      doneReg := state =/= sFault
      state := sFault
    }
  }

  io.busy := state =/= sIdle
  io.done := doneReg
  io.error := errorReg || anyChildError
  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.inputPairs := Mux(enableStats.B, inputPairs, 0.U)
  io.stats.completedTokens := Mux(enableStats.B, completedTokens, 0.U)
  io.stats.completedBlocks := Mux(enableStats.B, completedBlocks, 0.U)
  io.stats.sourceWaitCycles := Mux(enableStats.B, sourceWaitCycles, 0.U)
  io.stats.rejectedTransactions :=
    Mux(enableStats.B, rejectedTransactions, 0.U)
}
