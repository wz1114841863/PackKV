package briskkv

import chisel3._
import chisel3.util._

class QkComputePipelineStats extends Bundle {
  val queryReplay = new QueryReplayStats
  val accumulator = new QkAccumulatorStats
}

class QkComputePipelineIO(
  valueBits: Int,
  accumulatorBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val packCount = Input(UInt(countBits.W))
  val queryLoadIn = Flipped(Decoupled(SInt(valueBits.W)))
  val keyIn = Flipped(
    Decoupled(new AttentionFeaturePacket(valueBits, packTokens, countBits))
  )
  val out = Decoupled(new QkLogitPacket(accumulatorBits, packTokens, countBits))
  val busy = Output(Bool())
  val queryLoaded = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new QkComputePipelineStats)
}

/** Joins one stored query vector with the decompressed K feature stream. */
class QkComputePipeline(
  valueBits: Int = 18,
  accumulatorBits: Int = 44,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  blockTokens: Int = BriskKvFormatV0.params.blockTokens,
  maximumFeatureDim: Int = 256,
  countBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  val io = IO(
    new QkComputePipelineIO(
      valueBits,
      accumulatorBits,
      packTokens,
      countBits
    )
  )

  val queryReplay = Module(
    new QueryReplayBuffer(
      valueBits = valueBits,
      maximumFeatureDim = maximumFeatureDim,
      countBits = countBits,
      enableStats = enableStats
    )
  )
  val accumulator = Module(
    new QkDotProductAccumulator(
      valueBits = valueBits,
      accumulatorBits = accumulatorBits,
      packTokens = packTokens,
      blockTokens = blockTokens,
      maximumFeatureDim = maximumFeatureDim,
      countBits = countBits,
      enableStats = enableStats
    )
  )

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val replayDone = RegInit(false.B)
  doneReg := false.B

  val parametersValid = io.featureDim =/= 0.U &&
    io.featureDim <= maximumFeatureDim.U && io.packCount =/= 0.U
  val subStart = io.start && !active && parametersValid

  queryReplay.io.start := subStart
  queryReplay.io.featureDim := io.featureDim
  queryReplay.io.packCount := io.packCount
  queryReplay.io.loadIn <> io.queryLoadIn
  accumulator.io.start := subStart
  accumulator.io.featureDim := io.featureDim
  accumulator.io.queryIn <> queryReplay.io.out
  accumulator.io.keyIn <> io.keyIn
  io.out <> accumulator.io.out

  when(io.start && !active) {
    active := parametersValid
    doneReg := !parametersValid
    errorReg := !parametersValid
    replayDone := false.B
  }.elsewhen(io.start && active) {
    errorReg := true.B
  }.otherwise {
    when(queryReplay.io.done) {
      replayDone := true.B
    }
    when(queryReplay.io.error || accumulator.io.error) {
      errorReg := true.B
    }
    when(accumulator.io.done) {
      active := false.B
      doneReg := true.B
      when(!replayDone && !queryReplay.io.done) {
        errorReg := true.B
      }
    }
  }

  io.busy := active || queryReplay.io.busy || accumulator.io.busy
  io.queryLoaded := queryReplay.io.loaded
  io.done := doneReg
  io.error := errorReg || queryReplay.io.error || accumulator.io.error
  io.stats.queryReplay := queryReplay.io.stats
  io.stats.accumulator := accumulator.io.stats
}
