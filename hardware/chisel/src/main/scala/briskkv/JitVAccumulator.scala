package briskkv

import chisel3._
import chisel3.util._

class JitVAccumulatorStats extends Bundle {
  val activeCycles = UInt(64.W)
  val loadedWeightPackets = UInt(64.W)
  val acceptedVPackets = UInt(64.W)
  val outputFeatures = UInt(64.W)
  val macOperations = UInt(64.W)
  val inputStallCycles = UInt(64.W)
  val downstreamStallCycles = UInt(64.W)
}

class JitVAccumulatorIO(
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
  val vIn = Flipped(
    Decoupled(new AttentionFeaturePacket(valueBits, packTokens, countBits))
  )
  val out = Decoupled(new AvFeatureResult(accumulatorBits, countBits))
  val vLaunchReady = Output(Bool())
  val weightsLoaded = Output(Bool())
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new JitVAccumulatorStats)
}

/** Pack-major JIT-V compute engine.
  *
  * Softmax weights are small and are retained once per token pack. V decode
  * may start as soon as the first weight packet is committed. Later weights
  * and V packets overlap; the two-entry elastic queue backpressures V only
  * when its head packet references a weight that has not arrived yet. This is
  * the only dequantized-V storage. Per-feature partial sums are retained in a
  * narrow accumulator SRAM, eliminating the full token-by-feature V SRAM.
  */
class JitVAccumulator(
  valueBits: Int = 18,
  weightBits: Int = 16,
  accumulatorBits: Int = 50,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  blockTokens: Int = BriskKvFormatV0.params.blockTokens,
  maximumFeatureDim: Int = 256,
  maximumTokens: Int = 16384,
  countBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  require(packTokens > 0 && isPow2(packTokens))
  require(blockTokens >= packTokens && blockTokens % packTokens == 0)
  require(maximumFeatureDim > 0 && maximumTokens >= packTokens)

  private val maximumPacks = (maximumTokens + packTokens - 1) / packTokens
  private val packAddressBits = math.max(1, log2Ceil(maximumPacks))
  private val featureAddressBits = math.max(1, log2Ceil(maximumFeatureDim))
  private val productBits = valueBits + weightBits + 1
  private val packsPerBlock = blockTokens / packTokens

  val io = IO(
    new JitVAccumulatorIO(
      valueBits,
      weightBits,
      accumulatorBits,
      packTokens,
      countBits
    )
  )

  val weightMemory = SyncReadMem(
    maximumPacks,
    new StoredAttentionWeights(weightBits, packTokens)
  )
  val partialSumMemory = SyncReadMem(maximumFeatureDim, SInt(accumulatorBits.W))
  val vQueue = Module(
    new Queue(
      new AttentionFeaturePacket(valueBits, packTokens, countBits),
      entries = 2,
      pipe = false,
      flow = false
    )
  )
  vQueue.io.enq <> io.vIn

  private val Seq(sIdle, sWaitFirstWeight, sFetchV, sAccumulate, sOutput) =
    Enum(5)
  val state = RegInit(sIdle)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val featureDimReg = RegInit(0.U(countBits.W))
  val packCountReg = RegInit(0.U(countBits.W))
  val finalValidTokensReg = RegInit(0.U(log2Ceil(packTokens + 1).W))
  val weightLoadIndex = RegInit(0.U(countBits.W))
  val expectedPackIndex = RegInit(0.U(countBits.W))
  val expectedFeatureIndex = RegInit(0.U(countBits.W))
  val packetReg = Reg(
    new AttentionFeaturePacket(valueBits, packTokens, countBits)
  )
  val outputReg = Reg(new AvFeatureResult(accumulatorBits, countBits))

  val activeCycles = RegInit(0.U(64.W))
  val loadedWeightPackets = RegInit(0.U(64.W))
  val acceptedVPackets = RegInit(0.U(64.W))
  val outputFeatures = RegInit(0.U(64.W))
  val macOperations = RegInit(0.U(64.W))
  val inputStallCycles = RegInit(0.U(64.W))
  val downstreamStallCycles = RegInit(0.U(64.W))

  doneReg := false.B
  io.busy := state =/= sIdle
  io.done := doneReg
  io.error := errorReg
  io.vLaunchReady := state =/= sIdle && weightLoadIndex =/= 0.U
  io.weightsLoaded := state =/= sIdle && weightLoadIndex === packCountReg
  io.weightIn.ready := state =/= sIdle && weightLoadIndex < packCountReg
  val headWeightAvailable = vQueue.io.deq.valid &&
    vQueue.io.deq.bits.packIndex < weightLoadIndex
  vQueue.io.deq.ready := state === sFetchV && headWeightAvailable
  io.out.valid := state === sOutput
  io.out.bits := outputReg

  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.loadedWeightPackets := Mux(enableStats.B, loadedWeightPackets, 0.U)
  io.stats.acceptedVPackets := Mux(enableStats.B, acceptedVPackets, 0.U)
  io.stats.outputFeatures := Mux(enableStats.B, outputFeatures, 0.U)
  io.stats.macOperations := Mux(enableStats.B, macOperations, 0.U)
  io.stats.inputStallCycles := Mux(enableStats.B, inputStallCycles, 0.U)
  io.stats.downstreamStallCycles := Mux(
    enableStats.B,
    downstreamStallCycles,
    0.U
  )

  val weightFire = io.weightIn.valid && io.weightIn.ready
  val vPacketFire = vQueue.io.deq.valid && vQueue.io.deq.ready
  val outputFire = io.out.valid && io.out.ready
  val weightReadData = weightMemory.read(
    vQueue.io.deq.bits.packIndex(packAddressBits - 1, 0),
    vPacketFire
  )
  val partialSumReadData = partialSumMemory.read(
    vQueue.io.deq.bits.featureIndex(featureAddressBits - 1, 0),
    vPacketFire && vQueue.io.deq.bits.packIndex =/= 0.U
  )

  val laneProducts = Wire(Vec(packTokens, SInt(productBits.W)))
  for (lane <- 0 until packTokens) {
    val positiveWeight = Cat(0.U(1.W), weightReadData.weights(lane)).asSInt
    laneProducts(lane) := Mux(
      lane.U < packetReg.validTokens,
      positiveWeight * packetReg.values(lane),
      0.S(productBits.W)
    )
  }
  val packetSum = laneProducts.reduce(_ +& _)
  val extendedPacketSum = Wire(SInt(accumulatorBits.W))
  extendedPacketSum := packetSum
  val nextAccumulator = Mux(
    packetReg.packIndex === 0.U,
    extendedPacketSum,
    partialSumReadData + extendedPacketSum
  )

  when(io.start && state === sIdle) {
    val valid = io.featureDim =/= 0.U &&
      io.featureDim <= maximumFeatureDim.U && io.tokenCount =/= 0.U &&
      io.tokenCount <= maximumTokens.U
    featureDimReg := io.featureDim
    packCountReg := (io.tokenCount + (packTokens - 1).U) >> log2Ceil(packTokens)
    val remainder = io.tokenCount & (packTokens - 1).U
    finalValidTokensReg := Mux(remainder === 0.U, packTokens.U, remainder)
    weightLoadIndex := 0.U
    expectedPackIndex := 0.U
    expectedFeatureIndex := 0.U
    errorReg := !valid
    doneReg := !valid
    state := Mux(valid, sWaitFirstWeight, sIdle)
    activeCycles := 0.U
    loadedWeightPackets := 0.U
    acceptedVPackets := 0.U
    outputFeatures := 0.U
    macOperations := 0.U
    inputStallCycles := 0.U
    downstreamStallCycles := 0.U
  }.elsewhen(io.start && state =/= sIdle) {
    errorReg := true.B
  }.otherwise {
    when(state =/= sIdle) { activeCycles := activeCycles + 1.U }
    when(io.vIn.valid && !io.vIn.ready) {
      inputStallCycles := inputStallCycles + 1.U
    }
    when(io.out.valid && !io.out.ready) {
      downstreamStallCycles := downstreamStallCycles + 1.U
    }

    // Weight loading remains active throughout V fetch/accumulate/output.
    // A strict index comparison at the V queue head guarantees that a
    // SyncReadMem location is never read in the cycle in which it is written.
    when(weightFire) {
      val expectedLast = weightLoadIndex === packCountReg - 1.U
      val expectedValidTokens = Mux(
        expectedLast,
        finalValidTokensReg,
        packTokens.U
      )
      when(
        io.weightIn.bits.packIndex =/= weightLoadIndex ||
          io.weightIn.bits.validTokens =/= expectedValidTokens ||
          io.weightIn.bits.last =/= expectedLast
      ) { errorReg := true.B }
      val stored = Wire(new StoredAttentionWeights(weightBits, packTokens))
      stored.weights := io.weightIn.bits.weights
      weightMemory.write(weightLoadIndex(packAddressBits - 1, 0), stored)
      loadedWeightPackets := loadedWeightPackets + 1.U
      weightLoadIndex := weightLoadIndex + 1.U
    }

    switch(state) {
      is(sIdle) {}
      is(sWaitFirstWeight) {
        when(weightFire) {
          state := sFetchV
        }
      }
      is(sFetchV) {
        when(vPacketFire) {
          val finalPack = expectedPackIndex === packCountReg - 1.U
          val expectedValidTokens = Mux(
            finalPack,
            finalValidTokensReg,
            packTokens.U
          )
          when(
            vQueue.io.deq.bits.packIndex =/= expectedPackIndex ||
              vQueue.io.deq.bits.featureIndex =/= expectedFeatureIndex ||
              vQueue.io.deq.bits.validTokens =/= expectedValidTokens ||
              vQueue.io.deq.bits.blockIndex =/=
                expectedPackIndex / packsPerBlock.U ||
              vQueue.io.deq.bits.packWithinBlock =/=
                expectedPackIndex % packsPerBlock.U
          ) { errorReg := true.B }
          packetReg := vQueue.io.deq.bits
          acceptedVPackets := acceptedVPackets + 1.U
          state := sAccumulate
        }
      }
      is(sAccumulate) {
        macOperations := macOperations + packetReg.validTokens
        val finalPack = packetReg.packIndex === packCountReg - 1.U
        val finalFeature = packetReg.featureIndex === featureDimReg - 1.U
        when(finalPack) {
          outputReg.value := nextAccumulator
          outputReg.featureIndex := packetReg.featureIndex
          outputReg.last := finalFeature
          state := sOutput
        }.otherwise {
          partialSumMemory.write(
            packetReg.featureIndex(featureAddressBits - 1, 0),
            nextAccumulator
          )
          when(finalFeature) {
            expectedPackIndex := expectedPackIndex + 1.U
            expectedFeatureIndex := 0.U
          }.otherwise {
            expectedFeatureIndex := expectedFeatureIndex + 1.U
          }
          state := sFetchV
        }
      }
      is(sOutput) {
        when(outputFire) {
          outputFeatures := outputFeatures + 1.U
          when(outputReg.last) {
            doneReg := true.B
            state := sIdle
          }.otherwise {
            expectedFeatureIndex := expectedFeatureIndex + 1.U
            state := sFetchV
          }
        }
      }
    }
  }
}
