package briskkv

import chisel3._
import chisel3.util._

class KvWriteQuantizerInput(inputBits: Int, countBits: Int) extends Bundle {
  val fixedRaw = SInt(inputBits.W)
  val featureIndex = UInt(countBits.W)
  val last = Bool()
}

class KvWriteQuantizedValue(
  codeValueBits: Int,
  countBits: Int,
  tagBits: Int
) extends Bundle {
  val q = UInt(codeValueBits.W)
  val tokenTag = UInt(tagBits.W)
  val featureIndex = UInt(countBits.W)
  val last = Bool()
}

class KvWriteQuantMetadata(metadataBits: Int, tagBits: Int) extends Bundle {
  val zeroPoint = SInt(metadataBits.W)
  val exponent = SInt(metadataBits.W)
  val tokenTag = UInt(tagBits.W)
}

class KvWriteQuantizerStats extends Bundle {
  val activeCycles = UInt(64.W)
  val inputValues = UInt(64.W)
  val outputValues = UInt(64.W)
  val sourceStallCycles = UInt(64.W)
  val metadataStallCycles = UInt(64.W)
  val outputStallCycles = UInt(64.W)
  val rejectedTokens = UInt(64.W)
}

class KvWriteQuantizerIO(
  inputBits: Int,
  codeValueBits: Int,
  metadataBits: Int,
  countBits: Int,
  tagBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val tokenTag = Input(UInt(tagBits.W))
  val in = Flipped(Decoupled(new KvWriteQuantizerInput(inputBits, countBits)))
  val metadataOut = Decoupled(new KvWriteQuantMetadata(metadataBits, tagBits))
  val qOut = Decoupled(
    new KvWriteQuantizedValue(codeValueBits, countBits, tagBits)
  )
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new KvWriteQuantizerStats)
}

/** Buffered token-wise nearest-power-of-two quantizer for Format v0 writes.
  *
  * Input values are signed Q12 fixed-point numbers; `inputBits` controls the
  * signed integer range. One complete token vector is buffered so its
  * minimum and maximum can select the token-wise scale. For a selected scale
  * `2^exponent`, the emitted fields exactly implement the integer-zero-point
  * software contract:
  *
  *   zero = roundEven(minimum / 2^exponent)
  *   q     = roundEven(value / 2^exponent) - zero
  *
  * K uses a relative scale of 3/100 and V uses 1/10. Scale selection is a bank
  * of constant threshold comparisons, while value quantization is a rounded
  * power-of-two shift. No runtime divider, logarithm, or floating-point unit
  * is inferred.
  *
  * Metadata is emitted before q values. If the exponent, zero point, or q range
  * does not fit the frozen Format v0 profile, the transaction completes with
  * `error` and emits neither metadata nor q. This makes overflow explicit
  * instead of silently truncating or saturating an encoded token.
  */
class KvWriteQuantizer(
  isKey: Boolean,
  inputBits: Int = 24,
  inputFractionalBits: Int = 12,
  maximumFeatureDim: Int = 256,
  minimumExponent: Int = -6,
  maximumExponent: Int = 4,
  metadataBits: Int = 8,
  countBits: Int = 32,
  tagBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  private val format = BriskKvFormatV0.params
  private val codeValueBits = if (isKey) format.kQuantBits else format.vQuantBits
  private val zeroPointBits = if (isKey) format.kZeroBits else format.vZeroBits
  private val maximumShift = inputFractionalBits + maximumExponent
  private val roundedBits = inputBits + 2

  require(inputBits >= 3)
  require(inputFractionalBits == 12)
  require(maximumFeatureDim > 0)
  require(minimumExponent == -6 && maximumExponent == 4)
  require(metadataBits >= format.exponentBits)
  require(metadataBits >= zeroPointBits)
  require(maximumShift >= 0)

  val io = IO(
    new KvWriteQuantizerIO(
      inputBits,
      codeValueBits,
      metadataBits,
      countBits,
      tagBits
    )
  )

  private val sIdle :: sCollect :: sPrepare :: sMetadata :: sEmit :: Nil =
    Enum(5)
  val state = RegInit(sIdle)
  val valueMemory = SyncReadMem(maximumFeatureDim, SInt(inputBits.W))
  val featureDimReg = RegInit(0.U(countBits.W))
  val tokenTagReg = RegInit(0.U(tagBits.W))
  val collectIndex = RegInit(0.U(countBits.W))
  val emitIndex = RegInit(0.U(countBits.W))
  val minimumReg = Reg(SInt(inputBits.W))
  val maximumReg = Reg(SInt(inputBits.W))
  val zeroPointReg = RegInit(0.S(metadataBits.W))
  val exponentReg = RegInit(0.S(metadataBits.W))
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val qOutputValid = RegInit(false.B)
  val qOutputReg = Reg(
    new KvWriteQuantizedValue(codeValueBits, countBits, tagBits)
  )
  val readOutstanding = RegInit(false.B)

  val activeCycles = RegInit(0.U(64.W))
  val inputValues = RegInit(0.U(64.W))
  val outputValues = RegInit(0.U(64.W))
  val sourceStallCycles = RegInit(0.U(64.W))
  val metadataStallCycles = RegInit(0.U(64.W))
  val outputStallCycles = RegInit(0.U(64.W))
  val rejectedTokens = RegInit(0.U(64.W))

  io.busy := state =/= sIdle || qOutputValid || readOutstanding
  io.done := doneReg
  io.error := errorReg
  doneReg := false.B

  io.in.ready := state === sCollect
  io.metadataOut.valid := state === sMetadata
  io.metadataOut.bits.zeroPoint := zeroPointReg
  io.metadataOut.bits.exponent := exponentReg
  io.metadataOut.bits.tokenTag := tokenTagReg
  io.qOut.valid := qOutputValid
  io.qOut.bits := qOutputReg

  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.inputValues := Mux(enableStats.B, inputValues, 0.U)
  io.stats.outputValues := Mux(enableStats.B, outputValues, 0.U)
  io.stats.sourceStallCycles := Mux(enableStats.B, sourceStallCycles, 0.U)
  io.stats.metadataStallCycles := Mux(enableStats.B, metadataStallCycles, 0.U)
  io.stats.outputStallCycles := Mux(enableStats.B, outputStallCycles, 0.U)
  io.stats.rejectedTokens := Mux(enableStats.B, rejectedTokens, 0.U)

  /** Signed round-to-nearest-even division by a runtime power of two. */
  private def roundShiftEven(value: SInt, shift: UInt): SInt = {
    val extended = Wire(SInt((inputBits + 1).W))
    extended := value
    val negative = extended < 0.S
    val magnitude = Mux(negative, (-extended).asUInt, extended.asUInt)
    val roundedMagnitude = MuxLookup(shift, 0.U((inputBits + 1).W))(
      (0 to maximumShift).map { amount =>
        val quotient = magnitude >> amount
        val rounded = if (amount == 0) {
          quotient
        } else {
          val remainderMask = (BigInt(1) << amount) - 1
          val halfway = BigInt(1) << (amount - 1)
          val remainder = magnitude & remainderMask.U
          val increment = remainder > halfway.U ||
            (remainder === halfway.U && quotient(0))
          quotient + increment
        }
        amount.U -> rounded
      }
    )
    val signedResult = Wire(SInt(roundedBits.W))
    signedResult := Mux(
      negative,
      -roundedMagnitude.asSInt,
      roundedMagnitude.asSInt
    )
    signedResult
  }

  val rangeWide = maximumReg -& minimumReg
  val rangeRaw = rangeWide.asUInt

  // These Q12 entry thresholds are generated from the exact FP32 path in
  // utils/compute.py: range * {0.03f, 0.10f}, log2f, round-to-even. Freezing
  // the values avoids both runtime FP arithmetic and an off-by-one mismatch at
  // K's k=5 rejection boundary (Q12 range codes 3089397/3089398).
  private val kEntryThresholds = IndexedSeq(
    1509, 3017, 6034, 12068, 24136, 48272,
    96544, 193088, 386175, 772350, 1544699, 3089398
  ).map(BigInt(_))
  private val vEntryThresholds = IndexedSeq(
    453, 906, 1811, 3621, 7241, 14482,
    28964, 57927, 115853, 231705, 463410, 926820
  ).map(BigInt(_))

  private def entryThresholdRaw(exponent: Int): BigInt = {
    val index = exponent - minimumExponent
    (if (isKey) kEntryThresholds else vEntryThresholds)(index)
  }

  val selectedExponentWide = Wire(SInt((metadataBits + 1).W))
  selectedExponentWide := (minimumExponent - 1).S
  for (exponent <- minimumExponent to maximumExponent + 1) {
    when(rangeRaw >= entryThresholdRaw(exponent).U) {
      selectedExponentWide := exponent.S
    }
  }
  val exponentValid = selectedExponentWide >= minimumExponent.S &&
    selectedExponentWide <= maximumExponent.S
  val selectedShift = (selectedExponentWide + inputFractionalBits.S).asUInt
  val selectedZeroWide = roundShiftEven(minimumReg, selectedShift)
  val selectedMaximumCode =
    roundShiftEven(maximumReg, selectedShift) - selectedZeroWide
  private val minimumZero = -(BigInt(1) << (zeroPointBits - 1))
  private val maximumZero = (BigInt(1) << (zeroPointBits - 1)) - 1
  private val maximumCode = (BigInt(1) << codeValueBits) - 1
  val zeroPointValid = selectedZeroWide >= minimumZero.S(roundedBits.W) &&
    selectedZeroWide <= maximumZero.S(roundedBits.W)
  val codeRangeValid = selectedMaximumCode >= 0.S &&
    selectedMaximumCode <= maximumCode.S
  val quantParametersValid = exponentValid && zeroPointValid && codeRangeValid

  val inputFire = io.in.valid && io.in.ready
  val metadataFire = io.metadataOut.valid && io.metadataOut.ready
  val qOutputFire = io.qOut.valid && io.qOut.ready

  val issueRead = state === sEmit && !readOutstanding && !qOutputValid
  val memoryReadValue = valueMemory.read(emitIndex, issueRead)
  val readResponseValid = RegNext(issueRead, false.B)

  when(readResponseValid) {
    val roundedValue = roundShiftEven(
      memoryReadValue,
      (exponentReg + inputFractionalBits.S).asUInt
    )
    val qWide = roundedValue - zeroPointReg
    qOutputReg.q := qWide.asUInt(codeValueBits - 1, 0)
    qOutputReg.tokenTag := tokenTagReg
    qOutputReg.featureIndex := emitIndex
    qOutputReg.last := emitIndex === featureDimReg - 1.U
    qOutputValid := true.B
    readOutstanding := false.B
  }
  when(issueRead) {
    readOutstanding := true.B
  }

  when(io.start && state === sIdle && !qOutputValid && !readOutstanding) {
    val commandValid = io.featureDim =/= 0.U &&
      io.featureDim <= maximumFeatureDim.U
    featureDimReg := io.featureDim
    tokenTagReg := io.tokenTag
    collectIndex := 0.U
    emitIndex := 0.U
    errorReg := !commandValid
    doneReg := !commandValid
    qOutputValid := false.B
    readOutstanding := false.B
    activeCycles := 0.U
    inputValues := 0.U
    outputValues := 0.U
    sourceStallCycles := 0.U
    metadataStallCycles := 0.U
    outputStallCycles := 0.U
    rejectedTokens := Mux(commandValid, 0.U, 1.U)
    state := Mux(commandValid, sCollect, sIdle)
  }.elsewhen(io.start) {
    errorReg := true.B
  }.otherwise {
    when(state =/= sIdle || qOutputValid || readOutstanding) {
      activeCycles := activeCycles + 1.U
    }
    when(state === sCollect && !io.in.valid) {
      sourceStallCycles := sourceStallCycles + 1.U
    }
    when(io.metadataOut.valid && !io.metadataOut.ready) {
      metadataStallCycles := metadataStallCycles + 1.U
    }
    when(io.qOut.valid && !io.qOut.ready) {
      outputStallCycles := outputStallCycles + 1.U
    }

    when(inputFire) {
      val expectedLast = collectIndex === featureDimReg - 1.U
      when(
        io.in.bits.featureIndex =/= collectIndex ||
          io.in.bits.last =/= expectedLast
      ) {
        errorReg := true.B
      }
      valueMemory.write(collectIndex, io.in.bits.fixedRaw)
      when(collectIndex === 0.U) {
        minimumReg := io.in.bits.fixedRaw
        maximumReg := io.in.bits.fixedRaw
      }.otherwise {
        when(io.in.bits.fixedRaw < minimumReg) {
          minimumReg := io.in.bits.fixedRaw
        }
        when(io.in.bits.fixedRaw > maximumReg) {
          maximumReg := io.in.bits.fixedRaw
        }
      }
      inputValues := inputValues + 1.U
      when(expectedLast) {
        state := sPrepare
      }.otherwise {
        collectIndex := collectIndex + 1.U
      }
    }

    when(state === sPrepare) {
      when(errorReg || !quantParametersValid) {
        errorReg := true.B
        rejectedTokens := rejectedTokens + 1.U
        doneReg := true.B
        state := sIdle
      }.otherwise {
        zeroPointReg := selectedZeroWide.asUInt(metadataBits - 1, 0).asSInt
        exponentReg :=
          selectedExponentWide.asUInt(metadataBits - 1, 0).asSInt
        state := sMetadata
      }
    }

    when(metadataFire) {
      emitIndex := 0.U
      state := sEmit
    }

    when(qOutputFire) {
      qOutputValid := false.B
      outputValues := outputValues + 1.U
      when(qOutputReg.last) {
        doneReg := true.B
        state := sIdle
      }.otherwise {
        emitIndex := emitIndex + 1.U
      }
    }
  }
}
