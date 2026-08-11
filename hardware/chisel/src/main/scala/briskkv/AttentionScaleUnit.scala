package briskkv

import chisel3._
import chisel3.util._

class AttentionScaleStats extends Bundle {
  val activeCycles = UInt(64.W)
  val inputPackets = UInt(64.W)
  val outputPackets = UInt(64.W)
  val downstreamStallCycles = UInt(64.W)
}

class AttentionScaleUnitIO(
  inputBits: Int,
  outputBits: Int,
  packTokens: Int,
  countBits: Int,
  scaleBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val tokenCount = Input(UInt(countBits.W))
  val in = Flipped(Decoupled(new QkLogitPacket(inputBits, packTokens, countBits)))
  val out = Decoupled(new QkLogitPacket(outputBits, packTokens, countBits))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val scaleMultiplier = Output(UInt(scaleBits.W))
  val stats = Output(new AttentionScaleStats)
}

/** Multiplies Q12 QK logits by a Q18 lookup of 1/sqrt(featureDim).
  *
  * The lookup removes a runtime square-root/divide unit. Products are rounded
  * symmetrically to the nearest Q12 value and packet metadata is preserved.
  */
class AttentionScaleUnit(
  inputBits: Int = 44,
  outputBits: Int = 44,
  inputFractionalBits: Int = 12,
  scaleFractionalBits: Int = 18,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  blockTokens: Int = BriskKvFormatV0.params.blockTokens,
  maximumFeatureDim: Int = 256,
  maximumTokens: Int = 16384,
  countBits: Int = 32
) extends Module {
  require(inputBits >= 2 && outputBits >= 2)
  require(inputFractionalBits > 0 && scaleFractionalBits > 0)
  require(outputBits >= inputBits, "attention scaling must not narrow logits")
  require(packTokens > 0 && isPow2(packTokens))
  require(blockTokens >= packTokens && blockTokens % packTokens == 0)
  require(maximumFeatureDim > 0 && maximumTokens >= packTokens)

  private val scaleBits = scaleFractionalBits + 1
  private val scaleIndexBits = math.max(1, log2Ceil(maximumFeatureDim + 1))
  private val packsPerBlock = blockTokens / packTokens
  private val packShift = log2Ceil(packTokens)
  private val validTokenBits = log2Ceil(packTokens + 1)
  val io = IO(
    new AttentionScaleUnitIO(
      inputBits,
      outputBits,
      packTokens,
      countBits,
      scaleBits
    )
  )

  val scaleRom = VecInit((0 to maximumFeatureDim).map { dim =>
    val value =
      if (dim == 0) 0L
      else Math.round((1L << scaleFractionalBits).toDouble / Math.sqrt(dim.toDouble))
    value.U(scaleBits.W)
  })

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val packCountReg = RegInit(0.U(countBits.W))
  val expectedPackIndex = RegInit(0.U(countBits.W))
  val scaleReg = RegInit(0.U(scaleBits.W))

  val outputValid = RegInit(false.B)
  val outputPacket = Reg(
    new QkLogitPacket(outputBits, packTokens, countBits)
  )

  val activeCycles = RegInit(0.U(64.W))
  val inputPackets = RegInit(0.U(64.W))
  val outputPackets = RegInit(0.U(64.W))
  val downstreamStallCycles = RegInit(0.U(64.W))

  doneReg := false.B
  io.busy := active || outputValid
  io.done := doneReg
  io.error := errorReg
  io.scaleMultiplier := scaleReg
  io.stats.activeCycles := activeCycles
  io.stats.inputPackets := inputPackets
  io.stats.outputPackets := outputPackets
  io.stats.downstreamStallCycles := downstreamStallCycles

  io.out.valid := outputValid
  io.out.bits := outputPacket
  val canAccept = active && (!outputValid || io.out.ready)
  io.in.ready := canAccept
  val inputFire = io.in.valid && io.in.ready
  val outputFire = io.out.valid && io.out.ready

  when(io.start && !active && !outputValid) {
    val parametersValid = io.featureDim =/= 0.U &&
      io.featureDim <= maximumFeatureDim.U &&
      io.tokenCount =/= 0.U && io.tokenCount <= maximumTokens.U
    active := parametersValid
    doneReg := !parametersValid
    errorReg := !parametersValid
    packCountReg := (io.tokenCount + (packTokens - 1).U) >> packShift
    expectedPackIndex := 0.U
    val safeFeatureIndex = Mux(
      io.featureDim <= maximumFeatureDim.U,
      io.featureDim,
      0.U
    )
    scaleReg := scaleRom(safeFeatureIndex(scaleIndexBits - 1, 0))
    outputValid := false.B
    activeCycles := 0.U
    inputPackets := 0.U
    outputPackets := 0.U
    downstreamStallCycles := 0.U
  }.elsewhen(io.start && (active || outputValid)) {
    errorReg := true.B
  }.otherwise {
    when(active || outputValid) {
      activeCycles := activeCycles + 1.U
    }
    when(io.out.valid && !io.out.ready) {
      downstreamStallCycles := downstreamStallCycles + 1.U
    }
    when(outputFire) {
      outputValid := false.B
      outputPackets := outputPackets + 1.U
      when(outputPacket.last) {
        doneReg := true.B
      }
    }

    when(inputFire) {
      val expectedLast = expectedPackIndex === packCountReg - 1.U
      val expectedBlock = expectedPackIndex / packsPerBlock.U
      val expectedWithinBlock = expectedPackIndex % packsPerBlock.U
      val packetValid = io.in.bits.packIndex === expectedPackIndex &&
        io.in.bits.blockIndex === expectedBlock &&
        io.in.bits.packWithinBlock === expectedWithinBlock &&
        io.in.bits.validTokens =/= 0.U &&
        io.in.bits.validTokens <= packTokens.U &&
        io.in.bits.last === expectedLast &&
        (expectedLast || io.in.bits.validTokens === packTokens.U)
      when(!packetValid) {
        errorReg := true.B
      }

      outputPacket.validTokens := io.in.bits.validTokens
      outputPacket.packIndex := io.in.bits.packIndex
      outputPacket.blockIndex := io.in.bits.blockIndex
      outputPacket.packWithinBlock := io.in.bits.packWithinBlock
      outputPacket.last := expectedLast
      for (lane <- 0 until packTokens) {
        val product = io.in.bits.logits(lane) * scaleReg.asSInt
        val negative = product < 0.S
        val magnitude = Mux(negative, (-product).asUInt, product.asUInt)
        val roundedMagnitude =
          (magnitude + (BigInt(1) << (scaleFractionalBits - 1)).U) >>
            scaleFractionalBits
        val roundedSigned = Mux(
          negative,
          -roundedMagnitude.asSInt,
          roundedMagnitude.asSInt
        )
        val laneValid = lane.U < io.in.bits.validTokens
        outputPacket.logits(lane) := Mux(
          laneValid,
          roundedSigned.asSInt.pad(outputBits),
          0.S(outputBits.W)
        )
      }
      outputValid := true.B
      inputPackets := inputPackets + 1.U
      expectedPackIndex := expectedPackIndex + 1.U
      when(expectedLast) {
        active := false.B
      }
    }
  }
}
