package briskkv

import chisel3._
import chisel3.util._

class KvPackFeatureVector(
  kBits: Int,
  vBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val kValues = Vec(packTokens, UInt(kBits.W))
  val vValues = Vec(packTokens, UInt(vBits.W))
  val descriptorIndex = UInt(countBits.W)
  val packIndex = UInt(countBits.W)
  val featureIndex = UInt(countBits.W)
  val last = Bool()
}

class KvPackTransposeStats extends Bundle {
  val activeCycles = UInt(64.W)
  val inputValues = UInt(64.W)
  val outputDescriptors = UInt(64.W)
  val sourceWaitCycles = UInt(64.W)
  val sinkStallCycles = UInt(64.W)
}

class KvPackTransposeBufferIO(
  kBits: Int,
  vBits: Int,
  packTokens: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int,
  bucketIdBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val blockIndex = Input(UInt(countBits.W))
  val in = Flipped(
    Decoupled(
      new RoutedKvFeatureValue(
        kBits,
        vBits,
        countBits,
        tagBits,
        tokenIndexBits,
        bucketIdBits
      )
    )
  )
  val out = Decoupled(
    new KvPackFeatureVector(kBits, vBits, packTokens, countBits)
  )
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new KvPackTransposeStats)
}

/** Converts routed token-major K/V q into pack/feature/token order.
  *
  * Sixteen independent token banks allow all lanes of one feature descriptor
  * to be read in parallel. The implementation buffers one pack, emits every
  * feature, then reuses the banks for the next pack.
  */
class KvPackTransposeBuffer(
  maximumFeatureDim: Int = 256,
  countBits: Int = 32,
  tagBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  private val format = BriskKvFormatV0.params
  private val packTokens = format.packTokens
  private val blockTokens = format.blockTokens
  private val tokenIndexBits = log2Ceil(blockTokens)
  private val packLaneBits = log2Ceil(packTokens)
  private val packCount = blockTokens / packTokens

  require(packTokens == 16)
  require(blockTokens == 64)
  require(maximumFeatureDim > 0)

  val io = IO(
    new KvPackTransposeBufferIO(
      format.kQuantBits,
      format.vQuantBits,
      packTokens,
      countBits,
      tagBits,
      tokenIndexBits,
      format.bucketIdBits
    )
  )

  private val sIdle :: sCollect :: sEmit :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val kBanks = Seq.fill(packTokens)(
    SyncReadMem(maximumFeatureDim, UInt(format.kQuantBits.W))
  )
  val vBanks = Seq.fill(packTokens)(
    SyncReadMem(maximumFeatureDim, UInt(format.vQuantBits.W))
  )
  val featureDimReg = RegInit(0.U(countBits.W))
  val blockIndexReg = RegInit(0.U(countBits.W))
  val expectedTokenIndex = RegInit(0.U(tokenIndexBits.W))
  val expectedFeatureIndex = RegInit(0.U(countBits.W))
  val currentPackIndex = RegInit(0.U(countBits.W))
  val emitFeatureIndex = RegInit(0.U(countBits.W))
  val emitDescriptorIndex = RegInit(0.U(countBits.W))
  val outputValid = RegInit(false.B)
  val outputReg = Reg(
    new KvPackFeatureVector(
      format.kQuantBits,
      format.vQuantBits,
      packTokens,
      countBits
    )
  )
  val readOutstanding = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)

  val activeCycles = RegInit(0.U(64.W))
  val inputValues = RegInit(0.U(64.W))
  val outputDescriptors = RegInit(0.U(64.W))
  val sourceWaitCycles = RegInit(0.U(64.W))
  val sinkStallCycles = RegInit(0.U(64.W))

  io.busy := state =/= sIdle || outputValid || readOutstanding
  io.done := doneReg
  io.error := errorReg
  doneReg := false.B
  io.in.ready := state === sCollect
  io.out.valid := outputValid
  io.out.bits := outputReg
  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.inputValues := Mux(enableStats.B, inputValues, 0.U)
  io.stats.outputDescriptors := Mux(enableStats.B, outputDescriptors, 0.U)
  io.stats.sourceWaitCycles := Mux(enableStats.B, sourceWaitCycles, 0.U)
  io.stats.sinkStallCycles := Mux(enableStats.B, sinkStallCycles, 0.U)

  val inputFire = io.in.valid && io.in.ready
  val outputFire = io.out.valid && io.out.ready
  val inputLane = io.in.bits.routedTokenIndex(packLaneBits - 1, 0)
  for (lane <- 0 until packTokens) {
    when(inputFire && inputLane === lane.U) {
      kBanks(lane).write(io.in.bits.featureIndex, io.in.bits.kQ)
      vBanks(lane).write(io.in.bits.featureIndex, io.in.bits.vQ)
    }
  }

  val issueRead = state === sEmit && !outputValid && !readOutstanding
  val kReadValues = kBanks.map(_.read(emitFeatureIndex, issueRead))
  val vReadValues = vBanks.map(_.read(emitFeatureIndex, issueRead))
  val readResponseValid = RegNext(issueRead, false.B)
  when(readResponseValid) {
    outputReg.kValues.zip(kReadValues).foreach { case (out, value) => out := value }
    outputReg.vValues.zip(vReadValues).foreach { case (out, value) => out := value }
    outputReg.descriptorIndex := emitDescriptorIndex
    outputReg.packIndex := currentPackIndex
    outputReg.featureIndex := emitFeatureIndex
    outputReg.last := currentPackIndex === (packCount - 1).U &&
      emitFeatureIndex === featureDimReg - 1.U
    outputValid := true.B
    readOutstanding := false.B
  }
  when(issueRead) {
    readOutstanding := true.B
  }

  when(io.start && state === sIdle && !outputValid && !readOutstanding) {
    val commandValid = io.featureDim =/= 0.U &&
      io.featureDim <= maximumFeatureDim.U
    featureDimReg := io.featureDim
    blockIndexReg := io.blockIndex
    expectedTokenIndex := 0.U
    expectedFeatureIndex := 0.U
    currentPackIndex := 0.U
    emitFeatureIndex := 0.U
    emitDescriptorIndex := 0.U
    outputValid := false.B
    readOutstanding := false.B
    errorReg := !commandValid
    doneReg := !commandValid
    activeCycles := 0.U
    inputValues := 0.U
    outputDescriptors := 0.U
    sourceWaitCycles := 0.U
    sinkStallCycles := 0.U
    state := Mux(commandValid, sCollect, sIdle)
  }.elsewhen(io.start) {
    errorReg := true.B
  }.otherwise {
    when(state =/= sIdle || outputValid || readOutstanding) {
      activeCycles := activeCycles + 1.U
    }
    when(state === sCollect && !io.in.valid) {
      sourceWaitCycles := sourceWaitCycles + 1.U
    }
    when(io.out.valid && !io.out.ready) {
      sinkStallCycles := sinkStallCycles + 1.U
    }

    when(inputFire) {
      val expectedLastFeature = expectedFeatureIndex === featureDimReg - 1.U
      val expectedLast = expectedTokenIndex === (blockTokens - 1).U &&
        expectedLastFeature
      when(
        io.in.bits.routedTokenIndex =/= expectedTokenIndex ||
          io.in.bits.featureIndex =/= expectedFeatureIndex ||
          io.in.bits.blockIndex =/= blockIndexReg ||
          io.in.bits.lastFeature =/= expectedLastFeature ||
          io.in.bits.last =/= expectedLast
      ) {
        errorReg := true.B
      }
      inputValues := inputValues + 1.U
      when(expectedLastFeature) {
        expectedFeatureIndex := 0.U
        expectedTokenIndex := expectedTokenIndex + 1.U
        when(inputLane === (packTokens - 1).U) {
          emitFeatureIndex := 0.U
          state := sEmit
        }
      }.otherwise {
        expectedFeatureIndex := expectedFeatureIndex + 1.U
      }
    }

    when(outputFire) {
      outputValid := false.B
      outputDescriptors := outputDescriptors + 1.U
      emitDescriptorIndex := emitDescriptorIndex + 1.U
      when(outputReg.last) {
        doneReg := true.B
        state := sIdle
      }.elsewhen(emitFeatureIndex === featureDimReg - 1.U) {
        currentPackIndex := currentPackIndex + 1.U
        state := sCollect
      }.otherwise {
        emitFeatureIndex := emitFeatureIndex + 1.U
      }
    }
  }
}
