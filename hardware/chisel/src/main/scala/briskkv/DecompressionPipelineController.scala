package briskkv

import chisel3._
import chisel3.util._

class DecompressionCommand(countBits: Int, tagBits: Int) extends Bundle {
  val tag = UInt(tagBits.W)
  val tokenCount = UInt(countBits.W)
  val featureDim = UInt(countBits.W)
  val descriptorCount = UInt(countBits.W)
  val payloadByteCount = UInt(countBits.W)
}

class DecompressionResult(countBits: Int, tagBits: Int) extends Bundle {
  val tag = UInt(tagBits.W)
  val error = Bool()
  val tokenCount = UInt(countBits.W)
  val packCount = UInt(countBits.W)
  val blockCount = UInt(countBits.W)
  val descriptorCount = UInt(countBits.W)
  val stats = new DequantizerPerformanceStats
}

class DecompressionProgress(countBits: Int) extends Bundle {
  val completedValues = UInt(64.W)
  val completedDescriptors = UInt(countBits.W)
  val completedPacks = UInt(countBits.W)
  val completedBlocks = UInt(countBits.W)
}

class DecompressionPipelineControllerIO(
  outputBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int
) extends Bundle {
  val command = Flipped(Decoupled(new DecompressionCommand(countBits, tagBits)))
  val minimumIn = Flipped(Decoupled(UInt(8.W)))
  val widthIn = Flipped(Decoupled(UInt(8.W)))
  val payloadIn = Flipped(Decoupled(UInt(8.W)))
  val zeroPointIn = Flipped(Decoupled(UInt(8.W)))
  val exponentIn = Flipped(Decoupled(UInt(8.W)))
  val out = Decoupled(
    new FixedPointDequantizedValue(outputBits, countBits, tokenIndexBits)
  )
  val result = Decoupled(new DecompressionResult(countBits, tagBits))
  val busy = Output(Bool())
  val progress = Output(new DecompressionProgress(countBits))
}

/** Command-driven controller around one K or V decompression pipeline. */
class DecompressionPipelineController(
  codeValueBits: Int,
  zeroPointBits: Int,
  exponentBits: Int = BriskKvFormatV0.params.exponentBits,
  encodeLengthBits: Int = BriskKvFormatV0.params.packWidthBits,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  blockTokens: Int = BriskKvFormatV0.params.blockTokens,
  outputBits: Int = 18,
  countBits: Int = 32,
  tagBits: Int = 16,
  useBufferedMetadata: Boolean = true,
  enableStats: Boolean = true
) extends Module {
  require(packTokens > 0 && isPow2(packTokens))
  require(blockTokens >= packTokens && isPow2(blockTokens))
  require(blockTokens % packTokens == 0)

  private val tokenIndexBits = log2Ceil(packTokens)
  private val packsPerBlock = blockTokens / packTokens
  private val packsWithinBlockBits = math.max(1, log2Ceil(packsPerBlock))
  private object ControllerState extends ChiselEnum {
    val idle, launch, running, response = Value
  }

  val io = IO(
    new DecompressionPipelineControllerIO(
      outputBits,
      countBits,
      tagBits,
      tokenIndexBits
    )
  )
  val engine = Module(
    new KvStreamDequantizer(
      codeValueBits = codeValueBits,
      zeroPointBits = zeroPointBits,
      exponentBits = exponentBits,
      encodeLengthBits = encodeLengthBits,
      packTokens = packTokens,
      outputBits = outputBits,
      countBits = countBits,
      useBufferedMetadata = useBufferedMetadata,
      enableStats = enableStats
    )
  )

  private val state = RegInit(ControllerState.idle)
  val commandReg = Reg(new DecompressionCommand(countBits, tagBits))
  val packCountReg = RegInit(0.U(countBits.W))
  val blockCountReg = RegInit(0.U(countBits.W))
  val currentValidTokens = RegInit(0.U((tokenIndexBits + 1).W))
  val tokensRemaining = RegInit(0.U(countBits.W))
  val featureWithinPack = RegInit(0.U(countBits.W))
  val packWithinBlock = RegInit(0.U(packsWithinBlockBits.W))
  val completedValues = RegInit(0.U(64.W))
  val completedDescriptors = RegInit(0.U(countBits.W))
  val completedPacks = RegInit(0.U(countBits.W))
  val completedBlocks = RegInit(0.U(countBits.W))
  val resultError = RegInit(false.B)
  val resultStats = RegInit(0.U.asTypeOf(new DequantizerPerformanceStats))

  io.command.ready := state === ControllerState.idle
  io.busy := state === ControllerState.launch || state === ControllerState.running
  io.progress.completedValues := completedValues
  io.progress.completedDescriptors := completedDescriptors
  io.progress.completedPacks := completedPacks
  io.progress.completedBlocks := completedBlocks

  engine.io.start := state === ControllerState.launch
  engine.io.tokenCount := commandReg.tokenCount
  engine.io.descriptorCount := commandReg.descriptorCount
  engine.io.featureDim := commandReg.featureDim
  engine.io.payloadByteCount := commandReg.payloadByteCount
  engine.io.minimumIn <> io.minimumIn
  engine.io.widthIn <> io.widthIn
  engine.io.payloadIn <> io.payloadIn
  engine.io.zeroPointIn <> io.zeroPointIn
  engine.io.exponentIn <> io.exponentIn
  io.out <> engine.io.out

  io.result.valid := state === ControllerState.response
  io.result.bits.tag := commandReg.tag
  io.result.bits.error := resultError
  io.result.bits.tokenCount := commandReg.tokenCount
  io.result.bits.packCount := packCountReg
  io.result.bits.blockCount := blockCountReg
  io.result.bits.descriptorCount := commandReg.descriptorCount
  if (enableStats) {
    io.result.bits.stats := resultStats
  } else {
    io.result.bits.stats := 0.U.asTypeOf(new DequantizerPerformanceStats)
  }

  val commandFire = io.command.valid && io.command.ready
  val outputFire = io.out.valid && io.out.ready
  val descriptorFinished =
    io.out.bits.tokenIndex === currentValidTokens - 1.U
  val packFinished = descriptorFinished &&
    featureWithinPack === commandReg.featureDim - 1.U

  switch(state) {
    is(ControllerState.idle) {
      when(commandFire) {
        val requestedPackCount =
          (io.command.bits.tokenCount + (packTokens - 1).U) >> log2Ceil(packTokens)
        val requestedBlockCount =
          (io.command.bits.tokenCount + (blockTokens - 1).U) >>
            log2Ceil(blockTokens)
        val expectedDescriptorCount =
          requestedPackCount * io.command.bits.featureDim
        val commandValid = io.command.bits.tokenCount =/= 0.U &&
          io.command.bits.featureDim =/= 0.U &&
          io.command.bits.descriptorCount === expectedDescriptorCount

        commandReg := io.command.bits
        packCountReg := requestedPackCount
        blockCountReg := requestedBlockCount
        currentValidTokens := Mux(
          io.command.bits.tokenCount >= packTokens.U,
          packTokens.U,
          io.command.bits.tokenCount
        )
        tokensRemaining := io.command.bits.tokenCount
        featureWithinPack := 0.U
        packWithinBlock := 0.U
        completedValues := 0.U
        completedDescriptors := 0.U
        completedPacks := 0.U
        completedBlocks := 0.U
        resultError := !commandValid
        if (enableStats) {
          resultStats := 0.U.asTypeOf(new DequantizerPerformanceStats)
        }
        state := Mux(
          commandValid,
          ControllerState.launch,
          ControllerState.response
        )
      }
    }

    is(ControllerState.launch) {
      state := ControllerState.running
    }

    is(ControllerState.running) {
      when(outputFire) {
        completedValues := completedValues + 1.U
        when(descriptorFinished) {
          completedDescriptors := completedDescriptors + 1.U
          when(packFinished) {
            val remainingAfterPack = tokensRemaining - currentValidTokens
            completedPacks := completedPacks + 1.U
            tokensRemaining := remainingAfterPack
            currentValidTokens := Mux(
              remainingAfterPack >= packTokens.U,
              packTokens.U,
              remainingAfterPack
            )
            featureWithinPack := 0.U
            when(packWithinBlock === (packsPerBlock - 1).U || io.out.bits.last) {
              completedBlocks := completedBlocks + 1.U
              packWithinBlock := 0.U
            }.otherwise {
              packWithinBlock := packWithinBlock + 1.U
            }
          }.otherwise {
            featureWithinPack := featureWithinPack + 1.U
          }
        }
      }

      when(engine.io.done) {
        // The engine validates every descriptor/token index and only asserts
        // done after the final descriptor.  Reuse those structural counters
        // here instead of rebuilding tokenCount * featureDim on the result
        // error path.  tokensRemaining also covers a partial final pack.
        val completionMismatch =
          completedDescriptors =/= commandReg.descriptorCount ||
            tokensRemaining =/= 0.U || completedPacks =/= packCountReg ||
            completedBlocks =/= blockCountReg
        resultError := engine.io.error ||
          completionMismatch
        if (enableStats) {
          resultStats := engine.io.stats
        }
        state := ControllerState.response
      }
    }

    is(ControllerState.response) {
      when(io.result.valid && io.result.ready) {
        state := ControllerState.idle
      }
    }
  }
}
