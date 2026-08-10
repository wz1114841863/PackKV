package briskkv

import chisel3._
import chisel3.util._

/** Ping-pong buffered variant of PackMetadataDequantizer.
  *
  * One 16-token bank serves the current q pack while the other accepts the
  * next pack's zero/exponent records. Banks exchange roles at a pack boundary.
  */
class BufferedPackMetadataDequantizer(
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

  val zeroBanks = Reg(Vec(2, Vec(packTokens, SInt(metadataBits.W))))
  val exponentBanks = Reg(Vec(2, Vec(packTokens, SInt(metadataBits.W))))
  val bankReady = RegInit(VecInit(false.B, false.B))
  val bankTokenCount = Reg(Vec(2, UInt(bufferedCountBits.W)))
  val readBank = RegInit(false.B)
  val loadBank = RegInit(false.B)
  val loadCount = RegInit(0.U(bufferedCountBits.W))
  val loadTarget = RegInit(0.U(bufferedCountBits.W))
  val metadataTokensRemaining = RegInit(0.U(countBits.W))

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val descriptorCountReg = RegInit(0.U(descriptorIndexBits.W))
  val featureDimReg = RegInit(0.U(descriptorIndexBits.W))
  val descriptorsWithinPack = RegInit(0.U(descriptorIndexBits.W))
  val expectedDescriptorIndex = RegInit(0.U(descriptorIndexBits.W))
  val expectedTokenIndex = RegInit(0.U(tokenIndexBits.W))
  val tokensRemaining = RegInit(0.U(countBits.W))

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

  val currentBankReady = bankReady(readBank)
  val currentValidTokens = bankTokenCount(readBank)
  val metadataCanLoad = active && metadataTokensRemaining =/= 0.U &&
    !bankReady(loadBank)
  val metadataPairValid = io.zeroIn.valid && io.exponentIn.valid
  io.zeroIn.ready := metadataCanLoad && io.exponentIn.valid
  io.exponentIn.ready := metadataCanLoad && io.zeroIn.valid
  val metadataFire = metadataCanLoad && metadataPairValid

  val tokenIsPadding = io.qIn.bits.tokenIndex >= currentValidTokens
  val selectedZero = zeroBanks(readBank)(io.qIn.bits.tokenIndex)
  val selectedExponent = exponentBanks(readBank)(io.qIn.bits.tokenIndex)
  dequantizer.io.in.valid := active && currentBankReady && io.qIn.valid &&
    !tokenIsPadding
  dequantizer.io.in.bits.q :=
    io.qIn.bits.value.asUInt(codeValueBits - 1, 0)
  dequantizer.io.in.bits.zeroPoint := selectedZero
  dequantizer.io.in.bits.exponent := selectedExponent

  io.qIn.ready := active && currentBankReady && Mux(
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
    val parametersValid = io.tokenCount =/= 0.U &&
      io.descriptorCount =/= 0.U && io.featureDim =/= 0.U
    active := parametersValid
    doneReg := !parametersValid
    errorReg := !parametersValid
    bankReady(0) := false.B
    bankReady(1) := false.B
    bankTokenCount(0) := 0.U
    bankTokenCount(1) := 0.U
    readBank := false.B
    loadBank := false.B
    loadCount := 0.U
    loadTarget := Mux(
      io.tokenCount >= packTokens.U,
      packTokens.U,
      io.tokenCount
    )
    metadataTokensRemaining := io.tokenCount
    descriptorCountReg := io.descriptorCount
    featureDimReg := io.featureDim
    descriptorsWithinPack := 0.U
    expectedDescriptorIndex := 0.U
    expectedTokenIndex := 0.U
    tokensRemaining := io.tokenCount
    activeCycles := 0.U
    outputValues := 0.U
    metadataStallCycles := 0.U
    downstreamStallCycles := 0.U
  }.elsewhen(io.start && active) {
    errorReg := true.B
  }.elsewhen(active) {
    activeCycles := activeCycles + 1.U
    when(io.qIn.valid && !currentBankReady) {
      metadataStallCycles := metadataStallCycles + 1.U
    }
    when(io.out.valid && !io.out.ready) {
      downstreamStallCycles := downstreamStallCycles + 1.U
    }
    when(outputFire) {
      outputValues := outputValues + 1.U
    }

    when(metadataFire) {
      val metadataWriteIndex = loadCount(tokenIndexBits - 1, 0)
      zeroBanks(loadBank)(metadataWriteIndex) := io.zeroIn.bits
      exponentBanks(loadBank)(metadataWriteIndex) := io.exponentIn.bits
      when(loadCount === loadTarget - 1.U) {
        val remainingAfterLoad = metadataTokensRemaining - loadTarget
        bankReady(loadBank) := true.B
        bankTokenCount(loadBank) := loadTarget
        metadataTokensRemaining := remainingAfterLoad
        loadBank := !loadBank
        loadCount := 0.U
        loadTarget := Mux(
          remainingAfterLoad >= packTokens.U,
          packTokens.U,
          remainingAfterLoad
        )
      }.otherwise {
        loadCount := loadCount + 1.U
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
        bankReady(readBank) := false.B
        when(
          !packFinished || tokensRemaining =/= currentValidTokens ||
            metadataTokensRemaining =/= 0.U
        ) {
          errorReg := true.B
        }
      }.elsewhen(packFinished) {
        val remainingAfterPack = tokensRemaining - currentValidTokens
        tokensRemaining := remainingAfterPack
        bankReady(readBank) := false.B
        readBank := !readBank
      }
    }
  }
}
