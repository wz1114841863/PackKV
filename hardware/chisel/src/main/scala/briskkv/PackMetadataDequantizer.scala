package briskkv

import chisel3._
import chisel3.util._

class FixedPointDequantizedValue(
  outputBits: Int,
  descriptorIndexBits: Int,
  tokenIndexBits: Int
) extends Bundle {
  val fixedRaw = SInt(outputBits.W)
  val descriptorIndex = UInt(descriptorIndexBits.W)
  val tokenIndex = UInt(tokenIndexBits.W)
  val last = Bool()
}

class DequantizerPerformanceStats extends Bundle {
  val activeCycles = UInt(64.W)
  val outputValues = UInt(64.W)
  val metadataStallCycles = UInt(64.W)
  val downstreamStallCycles = UInt(64.W)
}

class PackMetadataDequantizerIO(
  outputBits: Int,
  metadataBits: Int,
  descriptorIndexBits: Int,
  tokenIndexBits: Int,
  countBits: Int
) extends Bundle {
  val start = Input(Bool())
  val tokenCount = Input(UInt(countBits.W))
  val descriptorCount = Input(UInt(descriptorIndexBits.W))
  val featureDim = Input(UInt(descriptorIndexBits.W))
  val qIn = Flipped(
    Decoupled(
      new DynamicUnpackedValue(8, descriptorIndexBits, tokenIndexBits)
    )
  )
  val zeroIn = Flipped(Decoupled(SInt(metadataBits.W)))
  val exponentIn = Flipped(Decoupled(SInt(metadataBits.W)))
  val out = Decoupled(
    new FixedPointDequantizedValue(
      outputBits,
      descriptorIndexBits,
      tokenIndexBits
    )
  )
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new DequantizerPerformanceStats)
}

abstract class PackMetadataDequantizerBase(
  outputBits: Int,
  metadataBits: Int,
  descriptorIndexBits: Int,
  packTokens: Int,
  countBits: Int
) extends Module {
  protected val tokenIndexBits: Int = log2Ceil(packTokens)
  val io = IO(
    new PackMetadataDequantizerIO(
      outputBits,
      metadataBits,
      descriptorIndexBits,
      tokenIndexBits,
      countBits
    )
  )
}

/** Reuses one token-wise zero/exponent record across every feature in a pack.
  *
  * Metadata arrives token-major, while the bit-packed q stream is
  * pack-major/feature-major/token-major. A 16-entry buffer bridges the two
  * orders without storing a complete layer.
  */
class PackMetadataDequantizer(
  codeValueBits: Int,
  zeroPointBits: Int,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  metadataBits: Int = 8,
  outputBits: Int = 18,
  descriptorIndexBits: Int = 32,
  countBits: Int = 32
) extends PackMetadataDequantizerBase(
      outputBits,
      metadataBits,
      descriptorIndexBits,
      packTokens,
      countBits
    ) {
  require(packTokens > 0 && isPow2(packTokens))
  require(packTokens <= 256)

  private val bufferedCountBits = log2Ceil(packTokens + 1)

  val dequantizer = Module(
    new Po2FixedPointDequantizer(
      codeValueBits = codeValueBits,
      zeroPointBits = zeroPointBits,
      metadataBits = metadataBits,
      outputBits = outputBits
    )
  )

  val zeroBuffer = Reg(Vec(packTokens, SInt(metadataBits.W)))
  val exponentBuffer = Reg(Vec(packTokens, SInt(metadataBits.W)))
  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val metadataReady = RegInit(false.B)
  val metadataLoaded = RegInit(0.U(bufferedCountBits.W))
  val currentValidTokens = RegInit(0.U(bufferedCountBits.W))
  val tokensRemaining = RegInit(0.U(countBits.W))
  val descriptorCountReg = RegInit(0.U(descriptorIndexBits.W))
  val featureDimReg = RegInit(0.U(descriptorIndexBits.W))
  val descriptorsWithinPack = RegInit(0.U(descriptorIndexBits.W))
  val expectedDescriptorIndex = RegInit(0.U(descriptorIndexBits.W))
  val expectedTokenIndex = RegInit(0.U(tokenIndexBits.W))
  val activeCycles = RegInit(0.U(64.W))
  val outputValues = RegInit(0.U(64.W))
  val metadataStallCycles = RegInit(0.U(64.W))
  val downstreamStallCycles = RegInit(0.U(64.W))

  io.busy := active
  io.done := doneReg
  io.error := errorReg
  io.stats.activeCycles := activeCycles
  io.stats.outputValues := outputValues
  io.stats.metadataStallCycles := metadataStallCycles
  io.stats.downstreamStallCycles := downstreamStallCycles
  doneReg := false.B

  val metadataCanLoad = active && !metadataReady
  val metadataPairValid = io.zeroIn.valid && io.exponentIn.valid
  io.zeroIn.ready := metadataCanLoad && io.exponentIn.valid
  io.exponentIn.ready := metadataCanLoad && io.zeroIn.valid
  val metadataFire = metadataCanLoad && metadataPairValid

  val tokenIsPadding = io.qIn.bits.tokenIndex >= currentValidTokens
  val selectedZero = zeroBuffer(io.qIn.bits.tokenIndex)
  val selectedExponent = exponentBuffer(io.qIn.bits.tokenIndex)
  dequantizer.io.in.valid := active && metadataReady && io.qIn.valid &&
    !tokenIsPadding
  dequantizer.io.in.bits.q :=
    io.qIn.bits.value.asUInt(codeValueBits - 1, 0)
  dequantizer.io.in.bits.zeroPoint := selectedZero
  dequantizer.io.in.bits.exponent := selectedExponent

  io.qIn.ready := active && metadataReady && Mux(
    tokenIsPadding,
    true.B,
    dequantizer.io.in.ready
  )
  dequantizer.io.out.ready := io.out.ready
  io.out.valid := dequantizer.io.out.valid
  io.out.bits.fixedRaw := dequantizer.io.out.bits.fixedRaw
  io.out.bits.descriptorIndex := io.qIn.bits.descriptorIndex
  io.out.bits.tokenIndex := io.qIn.bits.tokenIndex
  io.out.bits.last :=
    io.qIn.bits.descriptorIndex === descriptorCountReg - 1.U &&
      io.qIn.bits.tokenIndex === currentValidTokens - 1.U

  val qFire = io.qIn.valid && io.qIn.ready
  val outputFire = io.out.valid && io.out.ready
  val descriptorFinished = io.qIn.bits.tokenIndex === (packTokens - 1).U
  val packFinished = descriptorFinished &&
    descriptorsWithinPack === featureDimReg - 1.U

  when(io.start && !active) {
    active := io.tokenCount =/= 0.U && io.descriptorCount =/= 0.U &&
      io.featureDim =/= 0.U
    doneReg := io.tokenCount === 0.U || io.descriptorCount === 0.U ||
      io.featureDim === 0.U
    errorReg := io.tokenCount === 0.U || io.descriptorCount === 0.U ||
      io.featureDim === 0.U
    metadataReady := false.B
    metadataLoaded := 0.U
    currentValidTokens := Mux(
      io.tokenCount >= packTokens.U,
      packTokens.U,
      io.tokenCount
    )
    tokensRemaining := io.tokenCount
    descriptorCountReg := io.descriptorCount
    featureDimReg := io.featureDim
    descriptorsWithinPack := 0.U
    expectedDescriptorIndex := 0.U
    expectedTokenIndex := 0.U
    activeCycles := 0.U
    outputValues := 0.U
    metadataStallCycles := 0.U
    downstreamStallCycles := 0.U
  }.elsewhen(io.start && active) {
    errorReg := true.B
  }.elsewhen(active) {
    activeCycles := activeCycles + 1.U
    when(io.qIn.valid && !metadataReady) {
      metadataStallCycles := metadataStallCycles + 1.U
    }
    when(io.out.valid && !io.out.ready) {
      downstreamStallCycles := downstreamStallCycles + 1.U
    }
    when(outputFire) {
      outputValues := outputValues + 1.U
    }

    when(metadataFire) {
      val metadataWriteIndex = metadataLoaded(tokenIndexBits - 1, 0)
      zeroBuffer(metadataWriteIndex) := io.zeroIn.bits
      exponentBuffer(metadataWriteIndex) := io.exponentIn.bits
      when(metadataLoaded === currentValidTokens - 1.U) {
        metadataReady := true.B
        metadataLoaded := 0.U
      }.otherwise {
        metadataLoaded := metadataLoaded + 1.U
      }
    }

    when(outputFire && dequantizer.io.out.bits.error) {
      errorReg := true.B
    }

    when(qFire) {
      when(
        io.qIn.bits.descriptorIndex =/= expectedDescriptorIndex ||
          io.qIn.bits.tokenIndex =/= expectedTokenIndex ||
          io.qIn.bits.last =/= (
            io.qIn.bits.descriptorIndex === descriptorCountReg - 1.U &&
              io.qIn.bits.tokenIndex === (packTokens - 1).U
          )
      ) {
        errorReg := true.B
      }

      when(descriptorFinished) {
        expectedTokenIndex := 0.U
        expectedDescriptorIndex := expectedDescriptorIndex + 1.U
        when(packFinished) {
          descriptorsWithinPack := 0.U
        }.otherwise {
          descriptorsWithinPack := descriptorsWithinPack + 1.U
        }
      }.otherwise {
        expectedTokenIndex := expectedTokenIndex + 1.U
      }

      when(io.qIn.bits.last) {
        active := false.B
        doneReg := true.B
        when(!packFinished || tokensRemaining =/= currentValidTokens) {
          errorReg := true.B
        }
      }.elsewhen(packFinished) {
        val remainingAfterPack = tokensRemaining - currentValidTokens
        tokensRemaining := remainingAfterPack
        currentValidTokens := Mux(
          remainingAfterPack >= packTokens.U,
          packTokens.U,
          remainingAfterPack
        )
        metadataReady := false.B
        metadataLoaded := 0.U
      }
    }
  }
}
