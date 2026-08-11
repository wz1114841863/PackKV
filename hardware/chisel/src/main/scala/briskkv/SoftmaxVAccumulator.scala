package briskkv

import chisel3._
import chisel3.util._

class AvFeatureResult(
  accumulatorBits: Int,
  countBits: Int
) extends Bundle {
  val value = SInt(accumulatorBits.W)
  val featureIndex = UInt(countBits.W)
  val last = Bool()
}

class SoftmaxVAccumulatorStats extends Bundle {
  val activeCycles = UInt(64.W)
  val loadedWeightPackets = UInt(64.W)
  val vReadRequests = UInt(64.W)
  val vReadResponses = UInt(64.W)
  val outputFeatures = UInt(64.W)
  val macOperations = UInt(64.W)
  val vLoadWaitCycles = UInt(64.W)
  val vResponseWaitCycles = UInt(64.W)
  val downstreamStallCycles = UInt(64.W)
}

class SoftmaxVAccumulatorIO(
  valueBits: Int,
  weightBits: Int,
  accumulatorBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val tokenCount = Input(UInt(countBits.W))
  val weightIn = Flipped(
    Decoupled(new AttentionWeightPacket(weightBits, packTokens, countBits))
  )
  val vLoaded = Input(Bool())
  val vReadRequest = Decoupled(new VPacketReadRequest(countBits))
  val vReadResponse = Flipped(
    Decoupled(new AttentionFeaturePacket(valueBits, packTokens, countBits))
  )
  val out = Decoupled(new AvFeatureResult(accumulatorBits, countBits))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new SoftmaxVAccumulatorStats)
}

/** Replays one stored Softmax weight packet against every V feature.
  *
  * Each compute step performs 16 parallel unsigned-weight by signed-V
  * multiplies and reduces them into one exact Q21 partial sum. Partial sums are
  * accumulated across packs before one feature result is emitted.
  */
class SoftmaxVAccumulator(
  valueBits: Int = 18,
  valueFractionalBits: Int = 6,
  weightBits: Int = 16,
  weightFractionalBits: Int = 15,
  accumulatorBits: Int = 50,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  blockTokens: Int = BriskKvFormatV0.params.blockTokens,
  maximumFeatureDim: Int = 256,
  maximumTokens: Int = 16384,
  countBits: Int = 32
) extends Module {
  require(valueBits >= 2 && weightBits > weightFractionalBits)
  require(valueFractionalBits > 0 && weightFractionalBits > 0)
  require(packTokens > 0 && isPow2(packTokens))
  require(blockTokens >= packTokens && blockTokens % packTokens == 0)
  require(maximumFeatureDim > 0 && maximumTokens >= packTokens)
  require(
    accumulatorBits >= valueBits + weightBits + log2Ceil(maximumTokens),
    "AV accumulator must hold the worst-case token sum"
  )

  private val packShift = log2Ceil(packTokens)
  private val maximumPacks = (maximumTokens + packTokens - 1) / packTokens
  private val packAddressBits = math.max(1, log2Ceil(maximumPacks))
  private val packsPerBlock = blockTokens / packTokens
  private val productBits = valueBits + weightBits + 1

  val io = IO(
    new SoftmaxVAccumulatorIO(
      valueBits,
      weightBits,
      accumulatorBits,
      packTokens,
      countBits
    )
  )

  val weightMemory = SyncReadMem(
    maximumPacks,
    new AttentionWeightPacket(weightBits, packTokens, countBits)
  )

  val Seq(
    sIdle,
    sLoadWeights,
    sWaitForV,
    sReadRequest,
    sCaptureWeight,
    sWaitForVResponse,
    sOutputHold
  ) = Enum(7)
  val state = RegInit(sIdle)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val featureDimReg = RegInit(0.U(countBits.W))
  val packCountReg = RegInit(0.U(countBits.W))
  val finalValidTokensReg = RegInit(0.U(log2Ceil(packTokens + 1).W))
  val weightLoadIndex = RegInit(0.U(countBits.W))
  val computeFeatureIndex = RegInit(0.U(countBits.W))
  val computePackIndex = RegInit(0.U(countBits.W))
  val featureAccumulator = RegInit(0.S(accumulatorBits.W))
  val weightPacketReg = Reg(
    new AttentionWeightPacket(weightBits, packTokens, countBits)
  )
  val outputReg = Reg(new AvFeatureResult(accumulatorBits, countBits))

  val activeCycles = RegInit(0.U(64.W))
  val loadedWeightPackets = RegInit(0.U(64.W))
  val vReadRequests = RegInit(0.U(64.W))
  val vReadResponses = RegInit(0.U(64.W))
  val outputFeatures = RegInit(0.U(64.W))
  val macOperations = RegInit(0.U(64.W))
  val vLoadWaitCycles = RegInit(0.U(64.W))
  val vResponseWaitCycles = RegInit(0.U(64.W))
  val downstreamStallCycles = RegInit(0.U(64.W))

  doneReg := false.B
  io.busy := state =/= sIdle
  io.done := doneReg
  io.error := errorReg
  io.stats.activeCycles := activeCycles
  io.stats.loadedWeightPackets := loadedWeightPackets
  io.stats.vReadRequests := vReadRequests
  io.stats.vReadResponses := vReadResponses
  io.stats.outputFeatures := outputFeatures
  io.stats.macOperations := macOperations
  io.stats.vLoadWaitCycles := vLoadWaitCycles
  io.stats.vResponseWaitCycles := vResponseWaitCycles
  io.stats.downstreamStallCycles := downstreamStallCycles

  io.weightIn.ready := state === sLoadWeights
  io.vReadRequest.valid := state === sReadRequest
  io.vReadRequest.bits.packIndex := computePackIndex
  io.vReadRequest.bits.featureIndex := computeFeatureIndex
  io.vReadResponse.ready := state === sWaitForVResponse
  io.out.valid := state === sOutputHold
  io.out.bits := outputReg

  val weightFire = io.weightIn.valid && io.weightIn.ready
  val readRequestFire = io.vReadRequest.valid && io.vReadRequest.ready
  val readResponseFire = io.vReadResponse.valid && io.vReadResponse.ready
  val outputFire = io.out.valid && io.out.ready
  val weightReadData = weightMemory.read(
    computePackIndex(packAddressBits - 1, 0),
    readRequestFire
  )

  when(io.start && state === sIdle) {
    val parametersValid = io.featureDim =/= 0.U &&
      io.featureDim <= maximumFeatureDim.U &&
      io.tokenCount =/= 0.U && io.tokenCount <= maximumTokens.U
    state := Mux(parametersValid, sLoadWeights, sIdle)
    doneReg := !parametersValid
    errorReg := !parametersValid
    featureDimReg := io.featureDim
    packCountReg := (io.tokenCount + (packTokens - 1).U) >> packShift
    val remainder = io.tokenCount & (packTokens - 1).U
    finalValidTokensReg := Mux(remainder === 0.U, packTokens.U, remainder)
    weightLoadIndex := 0.U
    computeFeatureIndex := 0.U
    computePackIndex := 0.U
    featureAccumulator := 0.S
    activeCycles := 0.U
    loadedWeightPackets := 0.U
    vReadRequests := 0.U
    vReadResponses := 0.U
    outputFeatures := 0.U
    macOperations := 0.U
    vLoadWaitCycles := 0.U
    vResponseWaitCycles := 0.U
    downstreamStallCycles := 0.U
  }.elsewhen(io.start && state =/= sIdle) {
    errorReg := true.B
  }.otherwise {
    when(state =/= sIdle) {
      activeCycles := activeCycles + 1.U
    }
    when(state === sWaitForV && !io.vLoaded) {
      vLoadWaitCycles := vLoadWaitCycles + 1.U
    }
    when(state === sWaitForVResponse && !io.vReadResponse.valid) {
      vResponseWaitCycles := vResponseWaitCycles + 1.U
    }
    when(io.out.valid && !io.out.ready) {
      downstreamStallCycles := downstreamStallCycles + 1.U
    }

    switch(state) {
      is(sIdle) {}

      is(sLoadWeights) {
        when(weightFire) {
          val expectedLast = weightLoadIndex === packCountReg - 1.U
          val expectedValidTokens = Mux(
            expectedLast,
            finalValidTokensReg,
            packTokens.U
          )
          val packetValid = io.weightIn.bits.packIndex === weightLoadIndex &&
            io.weightIn.bits.blockIndex === weightLoadIndex / packsPerBlock.U &&
            io.weightIn.bits.packWithinBlock === weightLoadIndex % packsPerBlock.U &&
            io.weightIn.bits.validTokens === expectedValidTokens &&
            io.weightIn.bits.last === expectedLast
          when(!packetValid) {
            errorReg := true.B
          }
          weightMemory.write(
            weightLoadIndex(packAddressBits - 1, 0),
            io.weightIn.bits
          )
          loadedWeightPackets := loadedWeightPackets + 1.U
          weightLoadIndex := weightLoadIndex + 1.U
          when(expectedLast) {
            state := Mux(io.vLoaded, sReadRequest, sWaitForV)
          }
        }
      }

      is(sWaitForV) {
        when(io.vLoaded) {
          state := sReadRequest
        }
      }

      is(sReadRequest) {
        when(readRequestFire) {
          vReadRequests := vReadRequests + 1.U
          state := sCaptureWeight
        }
      }

      is(sCaptureWeight) {
        weightPacketReg := weightReadData
        state := sWaitForVResponse
      }

      is(sWaitForVResponse) {
        when(readResponseFire) {
          val expectedValidTokens = Mux(
            computePackIndex === packCountReg - 1.U,
            finalValidTokensReg,
            packTokens.U
          )
          val aligned = weightPacketReg.packIndex === computePackIndex &&
            io.vReadResponse.bits.packIndex === computePackIndex &&
            io.vReadResponse.bits.featureIndex === computeFeatureIndex &&
            weightPacketReg.validTokens === expectedValidTokens &&
            io.vReadResponse.bits.validTokens === expectedValidTokens
          when(!aligned) {
            errorReg := true.B
          }

          val laneProducts = Wire(Vec(packTokens, SInt(productBits.W)))
          for (lane <- 0 until packTokens) {
            val positiveWeight = Cat(0.U(1.W), weightPacketReg.weights(lane)).asSInt
            val product = positiveWeight * io.vReadResponse.bits.values(lane)
            laneProducts(lane) := Mux(
              lane.U < expectedValidTokens,
              product,
              0.S(productBits.W)
            )
          }
          val packetSum = laneProducts.reduce(_ +& _)
          val extendedPacketSum = Wire(SInt(accumulatorBits.W))
          extendedPacketSum := packetSum
          val nextAccumulator = Mux(
            computePackIndex === 0.U,
            extendedPacketSum,
            featureAccumulator + extendedPacketSum
          )
          featureAccumulator := nextAccumulator
          vReadResponses := vReadResponses + 1.U
          macOperations := macOperations + expectedValidTokens

          when(computePackIndex === packCountReg - 1.U) {
            outputReg.value := nextAccumulator
            outputReg.featureIndex := computeFeatureIndex
            outputReg.last := computeFeatureIndex === featureDimReg - 1.U
            state := sOutputHold
          }.otherwise {
            computePackIndex := computePackIndex + 1.U
            state := sReadRequest
          }
        }
      }

      is(sOutputHold) {
        when(outputFire) {
          outputFeatures := outputFeatures + 1.U
          when(outputReg.last) {
            state := sIdle
            doneReg := true.B
          }.otherwise {
            computeFeatureIndex := computeFeatureIndex + 1.U
            computePackIndex := 0.U
            featureAccumulator := 0.S
            state := sReadRequest
          }
        }
      }
    }
  }
}
