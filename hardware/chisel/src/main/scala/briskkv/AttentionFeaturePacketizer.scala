package briskkv

import chisel3._
import chisel3.util._

class AttentionFeaturePacket(
  valueBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val values = Vec(packTokens, SInt(valueBits.W))
  val validTokens = UInt(log2Ceil(packTokens + 1).W)
  val descriptorIndex = UInt(countBits.W)
  val packIndex = UInt(countBits.W)
  val featureIndex = UInt(countBits.W)
  val blockIndex = UInt(countBits.W)
  val packWithinBlock = UInt(
    math.max(1, log2Ceil(BriskKvFormatV0.params.blockTokens / packTokens)).W
  )
  val last = Bool()
}

class AttentionPacketizerStats extends Bundle {
  val inputValues = UInt(64.W)
  val outputPackets = UInt(64.W)
  val downstreamStallCycles = UInt(64.W)
}

class AttentionFeaturePacketizerIO(
  valueBits: Int,
  packTokens: Int,
  countBits: Int,
  tokenIndexBits: Int
) extends Bundle {
  val start = Input(Bool())
  val tokenCount = Input(UInt(countBits.W))
  val featureDim = Input(UInt(countBits.W))
  val descriptorCount = Input(UInt(countBits.W))
  val in = Flipped(
    Decoupled(
      new FixedPointDequantizedValue(valueBits, countBits, tokenIndexBits)
    )
  )
  val out = Decoupled(new AttentionFeaturePacket(valueBits, packTokens, countBits))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new AttentionPacketizerStats)
}

/** Converts scalar feature-major dequantized values into token-lane packets. */
class AttentionFeaturePacketizer(
  valueBits: Int = 18,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  blockTokens: Int = BriskKvFormatV0.params.blockTokens,
  countBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  require(packTokens > 0 && isPow2(packTokens))
  require(blockTokens >= packTokens && blockTokens % packTokens == 0)

  private val tokenIndexBits = log2Ceil(packTokens)
  private val validTokenBits = log2Ceil(packTokens + 1)
  private val packsPerBlock = blockTokens / packTokens
  private val packsWithinBlockBits = math.max(1, log2Ceil(packsPerBlock))

  val io = IO(
    new AttentionFeaturePacketizerIO(
      valueBits,
      packTokens,
      countBits,
      tokenIndexBits
    )
  )

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val packetValid = RegInit(false.B)
  val packetValues = Reg(Vec(packTokens, SInt(valueBits.W)))
  val packetValidTokens = RegInit(0.U(validTokenBits.W))
  val packetDescriptorIndex = RegInit(0.U(countBits.W))
  val packetPackIndex = RegInit(0.U(countBits.W))
  val packetFeatureIndex = RegInit(0.U(countBits.W))
  val packetBlockIndex = RegInit(0.U(countBits.W))
  val packetPackWithinBlock = RegInit(0.U(packsWithinBlockBits.W))
  val packetLast = RegInit(false.B)

  val currentValidTokens = RegInit(0.U(validTokenBits.W))
  val tokensRemaining = RegInit(0.U(countBits.W))
  val expectedDescriptorIndex = RegInit(0.U(countBits.W))
  val expectedTokenIndex = RegInit(0.U(tokenIndexBits.W))
  val packIndex = RegInit(0.U(countBits.W))
  val featureIndex = RegInit(0.U(countBits.W))
  val blockIndex = RegInit(0.U(countBits.W))
  val packWithinBlock = RegInit(0.U(packsWithinBlockBits.W))
  val featureDimReg = RegInit(0.U(countBits.W))
  val descriptorCountReg = RegInit(0.U(countBits.W))

  val inputValues = RegInit(0.U(64.W))
  val outputPackets = RegInit(0.U(64.W))
  val downstreamStallCycles = RegInit(0.U(64.W))

  io.busy := active || packetValid
  io.done := doneReg
  io.error := errorReg
  io.stats.inputValues := Mux(enableStats.B, inputValues, 0.U)
  io.stats.outputPackets := Mux(enableStats.B, outputPackets, 0.U)
  io.stats.downstreamStallCycles := Mux(enableStats.B, downstreamStallCycles, 0.U)
  doneReg := false.B

  io.out.valid := packetValid
  io.out.bits.values := packetValues
  io.out.bits.validTokens := packetValidTokens
  io.out.bits.descriptorIndex := packetDescriptorIndex
  io.out.bits.packIndex := packetPackIndex
  io.out.bits.featureIndex := packetFeatureIndex
  io.out.bits.blockIndex := packetBlockIndex
  io.out.bits.packWithinBlock := packetPackWithinBlock
  io.out.bits.last := packetLast

  io.in.ready := active && (!packetValid || io.out.ready)
  val inputFire = io.in.valid && io.in.ready
  val outputFire = io.out.valid && io.out.ready
  val descriptorFinished = io.in.bits.tokenIndex === currentValidTokens - 1.U
  val packFinished = descriptorFinished && featureIndex === featureDimReg - 1.U

  when(io.start && !active && !packetValid) {
    val parametersValid = io.tokenCount =/= 0.U && io.featureDim =/= 0.U &&
      io.descriptorCount =/= 0.U
    active := parametersValid
    doneReg := !parametersValid
    errorReg := !parametersValid
    packetValid := false.B
    currentValidTokens := Mux(
      io.tokenCount >= packTokens.U,
      packTokens.U,
      io.tokenCount
    )
    tokensRemaining := io.tokenCount
    expectedDescriptorIndex := 0.U
    expectedTokenIndex := 0.U
    packIndex := 0.U
    featureIndex := 0.U
    blockIndex := 0.U
    packWithinBlock := 0.U
    featureDimReg := io.featureDim
    descriptorCountReg := io.descriptorCount
    inputValues := 0.U
    outputPackets := 0.U
    downstreamStallCycles := 0.U
  }.elsewhen(io.start && (active || packetValid)) {
    errorReg := true.B
  }.otherwise {
    when(io.out.valid && !io.out.ready) {
      downstreamStallCycles := downstreamStallCycles + 1.U
    }
    when(outputFire) {
      packetValid := false.B
      outputPackets := outputPackets + 1.U
      when(packetLast) {
        doneReg := true.B
      }
    }

    when(inputFire) {
      inputValues := inputValues + 1.U
      when(io.in.bits.tokenIndex === 0.U) {
        for (lane <- 0 until packTokens) {
          packetValues(lane) := 0.S
        }
      }
      packetValues(io.in.bits.tokenIndex) := io.in.bits.fixedRaw

      when(
        io.in.bits.descriptorIndex =/= expectedDescriptorIndex ||
          io.in.bits.tokenIndex =/= expectedTokenIndex ||
          io.in.bits.last =/= (
            io.in.bits.descriptorIndex === descriptorCountReg - 1.U &&
              io.in.bits.tokenIndex === currentValidTokens - 1.U
          )
      ) {
        errorReg := true.B
      }

      when(descriptorFinished) {
        packetValid := true.B
        packetValidTokens := currentValidTokens
        packetDescriptorIndex := expectedDescriptorIndex
        packetPackIndex := packIndex
        packetFeatureIndex := featureIndex
        packetBlockIndex := blockIndex
        packetPackWithinBlock := packWithinBlock
        packetLast := io.in.bits.last
        expectedDescriptorIndex := expectedDescriptorIndex + 1.U
        expectedTokenIndex := 0.U

        when(packFinished) {
          val remainingAfterPack = tokensRemaining - currentValidTokens
          tokensRemaining := remainingAfterPack
          currentValidTokens := Mux(
            remainingAfterPack >= packTokens.U,
            packTokens.U,
            remainingAfterPack
          )
          featureIndex := 0.U
          packIndex := packIndex + 1.U
          when(packWithinBlock === (packsPerBlock - 1).U) {
            packWithinBlock := 0.U
            blockIndex := blockIndex + 1.U
          }.otherwise {
            packWithinBlock := packWithinBlock + 1.U
          }
        }.otherwise {
          featureIndex := featureIndex + 1.U
        }
        when(io.in.bits.last) {
          active := false.B
        }
      }.otherwise {
        expectedTokenIndex := expectedTokenIndex + 1.U
      }
    }
  }
}
