package briskkv

import chisel3._
import chisel3.util._

class QueryFeature(valueBits: Int, countBits: Int) extends Bundle {
  val value = SInt(valueBits.W)
  val featureIndex = UInt(countBits.W)
  val packIndex = UInt(countBits.W)
  val last = Bool()
}

class QkLogitPacket(
  accumulatorBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val logits = Vec(packTokens, SInt(accumulatorBits.W))
  val validTokens = UInt(log2Ceil(packTokens + 1).W)
  val packIndex = UInt(countBits.W)
  val blockIndex = UInt(countBits.W)
  val packWithinBlock = UInt(
    math.max(1, log2Ceil(BriskKvFormatV0.params.blockTokens / packTokens)).W
  )
  val last = Bool()
}

class QkAccumulatorStats extends Bundle {
  val activeCycles = UInt(64.W)
  val inputPackets = UInt(64.W)
  val outputPackets = UInt(64.W)
  val macOperations = UInt(64.W)
  val queryWaitCycles = UInt(64.W)
  val keyWaitCycles = UInt(64.W)
  val downstreamStallCycles = UInt(64.W)
}

class QkDotProductAccumulatorIO(
  valueBits: Int,
  accumulatorBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val queryIn = Flipped(Decoupled(new QueryFeature(valueBits, countBits)))
  val keyIn = Flipped(
    Decoupled(new AttentionFeaturePacket(valueBits, packTokens, countBits))
  )
  val out = Decoupled(new QkLogitPacket(accumulatorBits, packTokens, countBits))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new QkAccumulatorStats)
}

/** Computes one query against a feature-major stream of 16-token K packets.
  *
  * Query and K values use the same fixed-point convention. With the Format v0
  * Q6 input, each exact product and the accumulated logits have 12 implied
  * fractional bits. Scaling by 1/sqrt(featureDim) is intentionally left to the
  * following attention stage.
  */
class QkDotProductAccumulator(
  valueBits: Int = 18,
  accumulatorBits: Int = 44,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  blockTokens: Int = BriskKvFormatV0.params.blockTokens,
  maximumFeatureDim: Int = 256,
  countBits: Int = 32
) extends Module {
  require(valueBits >= 2)
  require(maximumFeatureDim > 0)
  require(
    accumulatorBits >= 2 * valueBits + log2Ceil(maximumFeatureDim),
    "accumulator must hold the worst-case sum without overflow"
  )
  require(packTokens > 0 && isPow2(packTokens))
  require(blockTokens >= packTokens && blockTokens % packTokens == 0)

  private val validTokenBits = log2Ceil(packTokens + 1)
  private val packsPerBlock = blockTokens / packTokens
  private val packWithinBlockBits = math.max(1, log2Ceil(packsPerBlock))

  val io = IO(
    new QkDotProductAccumulatorIO(
      valueBits,
      accumulatorBits,
      packTokens,
      countBits
    )
  )

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val featureDimReg = RegInit(0.U(countBits.W))
  val expectedFeatureIndex = RegInit(0.U(countBits.W))
  val expectedPackIndex = RegInit(0.U(countBits.W))
  val expectedDescriptorIndex = RegInit(0.U(countBits.W))
  val packValidTokens = RegInit(0.U(validTokenBits.W))
  val accumulators = RegInit(VecInit(Seq.fill(packTokens)(0.S(accumulatorBits.W))))

  val outputValid = RegInit(false.B)
  val outputLogits = Reg(Vec(packTokens, SInt(accumulatorBits.W)))
  val outputValidTokens = RegInit(0.U(validTokenBits.W))
  val outputPackIndex = RegInit(0.U(countBits.W))
  val outputBlockIndex = RegInit(0.U(countBits.W))
  val outputPackWithinBlock = RegInit(0.U(packWithinBlockBits.W))
  val outputLast = RegInit(false.B)

  val activeCycles = RegInit(0.U(64.W))
  val inputPackets = RegInit(0.U(64.W))
  val outputPackets = RegInit(0.U(64.W))
  val macOperations = RegInit(0.U(64.W))
  val queryWaitCycles = RegInit(0.U(64.W))
  val keyWaitCycles = RegInit(0.U(64.W))
  val downstreamStallCycles = RegInit(0.U(64.W))

  io.busy := active || outputValid
  io.done := doneReg
  io.error := errorReg
  io.stats.activeCycles := activeCycles
  io.stats.inputPackets := inputPackets
  io.stats.outputPackets := outputPackets
  io.stats.macOperations := macOperations
  io.stats.queryWaitCycles := queryWaitCycles
  io.stats.keyWaitCycles := keyWaitCycles
  io.stats.downstreamStallCycles := downstreamStallCycles
  doneReg := false.B

  io.out.valid := outputValid
  io.out.bits.logits := outputLogits
  io.out.bits.validTokens := outputValidTokens
  io.out.bits.packIndex := outputPackIndex
  io.out.bits.blockIndex := outputBlockIndex
  io.out.bits.packWithinBlock := outputPackWithinBlock
  io.out.bits.last := outputLast

  val outputFire = io.out.valid && io.out.ready
  val canAccept = active && (!outputValid || io.out.ready)
  io.queryIn.ready := canAccept && io.keyIn.valid
  io.keyIn.ready := canAccept && io.queryIn.valid
  val inputFire = io.queryIn.valid && io.queryIn.ready

  val finalFeature = expectedFeatureIndex === featureDimReg - 1.U
  val firstFeature = expectedFeatureIndex === 0.U
  val expectedBlockIndex = expectedPackIndex / packsPerBlock.U
  val expectedPackWithinBlock = expectedPackIndex % packsPerBlock.U
  when(io.start && !active && !outputValid) {
    val parametersValid =
      io.featureDim =/= 0.U && io.featureDim <= maximumFeatureDim.U
    active := parametersValid
    doneReg := !parametersValid
    errorReg := !parametersValid
    featureDimReg := io.featureDim
    expectedFeatureIndex := 0.U
    expectedPackIndex := 0.U
    expectedDescriptorIndex := 0.U
    packValidTokens := 0.U
    for (lane <- 0 until packTokens) {
      accumulators(lane) := 0.S
    }
    activeCycles := 0.U
    inputPackets := 0.U
    outputPackets := 0.U
    macOperations := 0.U
    queryWaitCycles := 0.U
    keyWaitCycles := 0.U
    downstreamStallCycles := 0.U
  }.elsewhen(io.start && (active || outputValid)) {
    errorReg := true.B
  }.otherwise {
    when(active || outputValid) {
      activeCycles := activeCycles + 1.U
    }
    when(canAccept && io.keyIn.valid && !io.queryIn.valid) {
      queryWaitCycles := queryWaitCycles + 1.U
    }
    when(canAccept && io.queryIn.valid && !io.keyIn.valid) {
      keyWaitCycles := keyWaitCycles + 1.U
    }
    when(io.out.valid && !io.out.ready) {
      downstreamStallCycles := downstreamStallCycles + 1.U
    }
    when(outputFire) {
      outputValid := false.B
      outputPackets := outputPackets + 1.U
      when(outputLast) {
        doneReg := true.B
      }
    }

    when(inputFire) {
      inputPackets := inputPackets + 1.U
      macOperations := macOperations + io.keyIn.bits.validTokens
      expectedDescriptorIndex := expectedDescriptorIndex + 1.U

      val indicesValid =
        io.queryIn.bits.featureIndex === expectedFeatureIndex &&
          io.queryIn.bits.packIndex === expectedPackIndex &&
          io.keyIn.bits.featureIndex === expectedFeatureIndex &&
          io.keyIn.bits.packIndex === expectedPackIndex &&
          io.keyIn.bits.descriptorIndex === expectedDescriptorIndex &&
          io.keyIn.bits.blockIndex === expectedBlockIndex &&
          io.keyIn.bits.packWithinBlock === expectedPackWithinBlock
      val validTokensValid = io.keyIn.bits.validTokens =/= 0.U &&
        io.keyIn.bits.validTokens <= packTokens.U &&
        (firstFeature || io.keyIn.bits.validTokens === packValidTokens)
      val lastValid =
        io.queryIn.bits.last === io.keyIn.bits.last &&
          (!io.keyIn.bits.last || finalFeature)
      when(!indicesValid || !validTokensValid || !lastValid) {
        errorReg := true.B
      }
      when(firstFeature) {
        packValidTokens := io.keyIn.bits.validTokens
      }

      for (lane <- 0 until packTokens) {
        val product = io.queryIn.bits.value * io.keyIn.bits.values(lane)
        val productWide = Wire(SInt(accumulatorBits.W))
        productWide := product
        val accumulated = Mux(firstFeature, productWide, accumulators(lane) + productWide)
        val laneValid = lane.U < io.keyIn.bits.validTokens
        val nextValue = Mux(laneValid, accumulated, 0.S(accumulatorBits.W))
        accumulators(lane) := nextValue
        when(finalFeature) {
          outputLogits(lane) := nextValue
        }
      }

      when(finalFeature) {
        outputValid := true.B
        outputValidTokens := io.keyIn.bits.validTokens
        outputPackIndex := io.keyIn.bits.packIndex
        outputBlockIndex := io.keyIn.bits.blockIndex
        outputPackWithinBlock := io.keyIn.bits.packWithinBlock
        outputLast := io.keyIn.bits.last
        expectedFeatureIndex := 0.U
        expectedPackIndex := expectedPackIndex + 1.U
        when(io.keyIn.bits.last) {
          active := false.B
        }
      }.otherwise {
        expectedFeatureIndex := expectedFeatureIndex + 1.U
      }
    }
  }
}
