package briskkv

import chisel3._
import chisel3.util._

class QkScaleSoftmaxStats extends Bundle {
  val scaling = new AttentionScaleStats
  val softmax = new StreamingSoftmaxStats
}

class QkScaleSoftmaxPipelineIO(
  logitBits: Int,
  weightBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val tokenCount = Input(UInt(countBits.W))
  val in = Flipped(Decoupled(new QkLogitPacket(logitBits, packTokens, countBits)))
  val out = Decoupled(new AttentionWeightPacket(weightBits, packTokens, countBits))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val scaleMultiplier = Output(UInt(19.W))
  val stats = Output(new QkScaleSoftmaxStats)
}

/** Composes attention scaling and sequence-wide stable softmax. */
class QkScaleSoftmaxPipeline(
  logitBits: Int = 44,
  weightBits: Int = 16,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  maximumFeatureDim: Int = 256,
  maximumTokens: Int = 16384,
  countBits: Int = 32
) extends Module {
  val io = IO(
    new QkScaleSoftmaxPipelineIO(
      logitBits,
      weightBits,
      packTokens,
      countBits
    )
  )

  val scaler = Module(
    new AttentionScaleUnit(
      inputBits = logitBits,
      outputBits = logitBits,
      packTokens = packTokens,
      maximumFeatureDim = maximumFeatureDim,
      maximumTokens = maximumTokens,
      countBits = countBits
    )
  )
  val softmax = Module(
    new StreamingSoftmax(
      logitBits = logitBits,
      weightBits = weightBits,
      packTokens = packTokens,
      maximumTokens = maximumTokens,
      countBits = countBits
    )
  )

  scaler.io.start := io.start
  scaler.io.featureDim := io.featureDim
  scaler.io.tokenCount := io.tokenCount
  scaler.io.in <> io.in
  softmax.io.start := io.start
  softmax.io.tokenCount := io.tokenCount
  softmax.io.in <> scaler.io.out
  io.out <> softmax.io.out

  io.busy := scaler.io.busy || softmax.io.busy
  io.done := softmax.io.done
  io.error := scaler.io.error || softmax.io.error
  io.scaleMultiplier := scaler.io.scaleMultiplier
  io.stats.scaling := scaler.io.stats
  io.stats.softmax := softmax.io.stats
}
