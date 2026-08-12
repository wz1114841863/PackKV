package briskkv

import chisel3._
import chisel3.util._

class SoftmaxVComputeStats extends Bundle {
  val vBuffer = new VPacketBufferStats
  val accumulator = new SoftmaxVAccumulatorStats
}

class SoftmaxVComputePipelineIO(
  valueBits: Int,
  weightBits: Int,
  accumulatorBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val tokenCount = Input(UInt(countBits.W))
  val vIn = Flipped(
    Decoupled(new AttentionFeaturePacket(valueBits, packTokens, countBits))
  )
  val weightIn = Flipped(
    Decoupled(new AttentionWeightPacket(weightBits, packTokens, countBits))
  )
  val out = Decoupled(new AvFeatureResult(accumulatorBits, countBits))
  val vLoaded = Output(Bool())
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new SoftmaxVComputeStats)
}

/** Buffers V while QK/Softmax run, then computes the exact fixed-point AV. */
class SoftmaxVComputePipeline(
  valueBits: Int = 18,
  weightBits: Int = 16,
  accumulatorBits: Int = 50,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  maximumFeatureDim: Int = 256,
  maximumTokens: Int = 16384,
  countBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  val io = IO(
    new SoftmaxVComputePipelineIO(
      valueBits,
      weightBits,
      accumulatorBits,
      packTokens,
      countBits
    )
  )

  val vBuffer = Module(
    new VPacketBuffer(
      valueBits = valueBits,
      packTokens = packTokens,
      maximumFeatureDim = maximumFeatureDim,
      maximumTokens = maximumTokens,
      countBits = countBits,
      enableStats = enableStats
    )
  )
  val accumulator = Module(
    new SoftmaxVAccumulator(
      valueBits = valueBits,
      weightBits = weightBits,
      accumulatorBits = accumulatorBits,
      packTokens = packTokens,
      maximumFeatureDim = maximumFeatureDim,
      maximumTokens = maximumTokens,
      countBits = countBits,
      enableStats = enableStats
    )
  )

  vBuffer.io.start := io.start
  vBuffer.io.finish := accumulator.io.done
  vBuffer.io.featureDim := io.featureDim
  vBuffer.io.tokenCount := io.tokenCount
  vBuffer.io.loadIn <> io.vIn

  accumulator.io.start := io.start
  accumulator.io.featureDim := io.featureDim
  accumulator.io.tokenCount := io.tokenCount
  accumulator.io.weightIn <> io.weightIn
  accumulator.io.vLoaded := vBuffer.io.loaded
  accumulator.io.vReadRequest <> vBuffer.io.readRequest
  accumulator.io.vReadResponse <> vBuffer.io.readResponse
  io.out <> accumulator.io.out

  io.vLoaded := vBuffer.io.loaded
  io.busy := vBuffer.io.busy || accumulator.io.busy
  io.done := accumulator.io.done
  io.error := vBuffer.io.error || accumulator.io.error
  io.stats.vBuffer := vBuffer.io.stats
  io.stats.accumulator := accumulator.io.stats
}
