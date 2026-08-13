package briskkv

import chisel3._
import chisel3.util._

sealed trait QuantParameterArchitecture {
  def cliName: String
  def manifestName: String
  def extraParameterCyclesPerToken: Int
}

object QuantParameterArchitecture {
  case object V1SingleStage extends QuantParameterArchitecture {
    override val cliName = "v1"
    override val manifestName = "priority-exponent-combined-zero-range"
    override val extraParameterCyclesPerToken = 0
  }

  case object V2ThreeStage extends QuantParameterArchitecture {
    override val cliName = "v2"
    override val manifestName = "popcount-exponent-registered-zero-range"
    override val extraParameterCyclesPerToken = 2
  }

  case object V3LeadingOne extends QuantParameterArchitecture {
    override val cliName = "v3"
    override val manifestName =
      "leading-one-adjacent-threshold-combined-zero-range"
    override val extraParameterCyclesPerToken = 0
  }

  case object V4BalancedTree extends QuantParameterArchitecture {
    override val cliName = "v4"
    override val manifestName =
      "balanced-static-threshold-tree-combined-zero-range"
    override val extraParameterCyclesPerToken = 0
  }

  val supported: Seq[QuantParameterArchitecture] =
    Seq(V1SingleStage, V2ThreeStage, V3LeadingOne, V4BalancedTree)

  def fromCliName(name: String): QuantParameterArchitecture =
    supported.find(_.cliName == name).getOrElse(
      throw new IllegalArgumentException(
        s"Unsupported quant parameter architecture '$name'; " +
          s"expected one of ${supported.map(_.cliName).mkString(", ")}"
      )
    )
}

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
  enableStats: Boolean = true,
  parameterArchitecture: QuantParameterArchitecture =
    QuantParameterArchitecture.V1SingleStage
) extends Module {
  private val format = BriskKvFormatV0.params
  private val codeValueBits = if (isKey) format.kQuantBits else format.vQuantBits
  private val zeroPointBits = if (isKey) format.kZeroBits else format.vZeroBits
  private val maximumShift = inputFractionalBits + maximumExponent
  private val roundedBits = inputBits + 2
  private val featureCountBits = math.max(1, log2Ceil(maximumFeatureDim + 1))
  private val featureIndexBits = math.max(1, log2Ceil(maximumFeatureDim))

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

  // Keep the v1 state encoding identical to the 2026081205 DC baseline. v2
  // inserts two states, so its metadata and emit encodings are intentionally
  // different and remain isolated in its own generated RTL directory.
  private val singleStageParameters =
    parameterArchitecture != QuantParameterArchitecture.V2ThreeStage
  private val parameterStates = Enum(if (singleStageParameters) 5 else 7)
  private val sIdle = parameterStates(0)
  private val sCollect = parameterStates(1)
  private val sSelectExponent = parameterStates(2)
  private val sComputeZero =
    if (singleStageParameters) parameterStates(2)
    else parameterStates(3)
  private val sValidateRange =
    if (singleStageParameters) parameterStates(2)
    else parameterStates(4)
  private val sMetadata =
    if (singleStageParameters) parameterStates(3)
    else parameterStates(5)
  private val sEmit =
    if (singleStageParameters) parameterStates(4)
    else parameterStates(6)
  val state = RegInit(sIdle)
  val valueMemory = SyncReadMem(maximumFeatureDim, SInt(inputBits.W))
  // The public Format v0 interface remains countBits wide. Internally the
  // dimension must represent maximumFeatureDim, while indices only represent
  // 0 .. maximumFeatureDim - 1.
  val featureDimReg = RegInit(0.U(featureCountBits.W))
  val tokenTagReg = RegInit(0.U(tagBits.W))
  val collectIndex = RegInit(0.U(featureIndexBits.W))
  val emitIndex = RegInit(0.U(featureIndexBits.W))
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
  if (parameterArchitecture == QuantParameterArchitecture.V1SingleStage) {
    // DC reference architecture used by the 2026081205 baseline. The
    // monotonically increasing threshold bank is encoded as the original
    // priority selection and exponent/zero/range validation shares one state.
    selectedExponentWide := (minimumExponent - 1).S
    for (exponent <- minimumExponent to maximumExponent + 1) {
      when(rangeRaw >= entryThresholdRaw(exponent).U) {
        selectedExponentWide := exponent.S
      }
    }
  } else if (
    parameterArchitecture == QuantParameterArchitecture.V2ThreeStage
  ) {
    // v2 timing experiment: parallel threshold comparisons followed by a
    // population count. Later states register zero point and range checks.
    val thresholdMatches = VecInit(
      (minimumExponent to maximumExponent + 1).map { exponent =>
        rangeRaw >= entryThresholdRaw(exponent).U
      }
    )
    selectedExponentWide :=
      PopCount(thresholdMatches).zext + (minimumExponent - 1).S
  } else if (
    parameterArchitecture == QuantParameterArchitecture.V3LeadingOne
  ) {
    // v3 exploits the near-doubling of consecutive entry thresholds. The
    // leading-one position identifies the only exponent boundary that can lie
    // in the current power-of-two interval; one exact table lookup and one
    // comparison then corrects the adjacent candidate. Values outside the
    // frozen exponent table deliberately map to the same invalid sentinels as
    // v1, preserving reject behavior at both ends.
    val rangeWidth = rangeRaw.getWidth
    val leadingZeroCount = PriorityEncoder(Reverse(rangeRaw))
    val leadingOneIndex =
      (rangeWidth - 1).U - leadingZeroCount
    val exponentOffset = if (isKey) 16 else 14
    val baseExponent = Wire(SInt((metadataBits + 1).W))
    baseExponent := leadingOneIndex.zext - exponentOffset.S

    val minimumTableExponent = minimumExponent
    val maximumTableExponent = maximumExponent + 1
    val clampedBaseExponent = Mux(
      baseExponent < minimumTableExponent.S,
      minimumTableExponent.S,
      Mux(
        baseExponent > maximumTableExponent.S,
        maximumTableExponent.S,
        baseExponent
      )
    )
    val thresholdIndexWide =
      (clampedBaseExponent - minimumTableExponent.S).asUInt
    val thresholdIndexBits = log2Ceil(
      maximumTableExponent - minimumTableExponent + 1
    )
    val thresholdIndex = thresholdIndexWide(thresholdIndexBits - 1, 0)
    val thresholdTable = VecInit(
      (minimumTableExponent to maximumTableExponent).map { exponent =>
        entryThresholdRaw(exponent).U(rangeWidth.W)
      }
    )
    val correctedExponent = Mux(
      rangeRaw >= thresholdTable(thresholdIndex),
      baseExponent,
      baseExponent - 1.S
    )

    selectedExponentWide := Mux(
      !rangeRaw.orR || baseExponent < minimumTableExponent.S,
      (minimumExponent - 1).S,
      Mux(
        baseExponent > maximumTableExponent.S,
        maximumTableExponent.S,
        correctedExponent
      )
    )
  } else {
    // v4 keeps v1's one-cycle parameter schedule and exact frozen thresholds,
    // but expresses predecessor search as a static balanced binary tree. Each
    // root-to-leaf path contains at most four threshold comparisons for the
    // twelve-entry Format v0 table. Unlike v3, there is no leading-one network,
    // dynamic Vec lookup, candidate clamp, or adjacent correction datapath.
    val thresholds = if (isKey) kEntryThresholds else vEntryThresholds
    val exponentWidth = metadataBits + 1

    def balancedThresholdSelect(low: Int, high: Int): SInt = {
      if (low > high) {
        (minimumExponent + low - 1).S(exponentWidth.W)
      } else {
        val middle = (low + high) / 2
        Mux(
          rangeRaw >= thresholds(middle).U(rangeRaw.getWidth.W),
          balancedThresholdSelect(middle + 1, high),
          balancedThresholdSelect(low, middle - 1)
        )
      }
    }

    selectedExponentWide :=
      balancedThresholdSelect(0, thresholds.length - 1)
  }
  val selectedExponentValid = selectedExponentWide >= minimumExponent.S &&
    selectedExponentWide <= maximumExponent.S

  val selectedShift =
    (selectedExponentWide + inputFractionalBits.S).asUInt
  val selectedZeroWide = roundShiftEven(minimumReg, selectedShift)
  val selectedMaximumCode =
    roundShiftEven(maximumReg, selectedShift) - selectedZeroWide

  // Exponent, zero point, and maximum-code validation are intentionally based
  // on registers from the preceding stage. This prevents the threshold bank,
  // runtime power-of-two shift, and range checks from forming one path.
  val registeredShift = (exponentReg + inputFractionalBits.S).asUInt
  val computedZeroWide = roundShiftEven(minimumReg, registeredShift)
  val computedMaximumCode =
    roundShiftEven(maximumReg, registeredShift) - zeroPointReg
  private val minimumZero = -(BigInt(1) << (zeroPointBits - 1))
  private val maximumZero = (BigInt(1) << (zeroPointBits - 1)) - 1
  private val maximumCode = (BigInt(1) << codeValueBits) - 1
  val computedZeroValid = computedZeroWide >= minimumZero.S(roundedBits.W) &&
    computedZeroWide <= maximumZero.S(roundedBits.W)
  val computedCodeRangeValid = computedMaximumCode >= 0.S &&
    computedMaximumCode <= maximumCode.S
  val selectedZeroValid = selectedZeroWide >= minimumZero.S(roundedBits.W) &&
    selectedZeroWide <= maximumZero.S(roundedBits.W)
  val selectedCodeRangeValid = selectedMaximumCode >= 0.S &&
    selectedMaximumCode <= maximumCode.S
  val singleStageParametersValid = selectedExponentValid &&
    selectedZeroValid && selectedCodeRangeValid

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
    featureDimReg := io.featureDim(featureCountBits - 1, 0)
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
        state := sSelectExponent
      }.otherwise {
        collectIndex := collectIndex + 1.U
      }
    }

    if (singleStageParameters) {
      when(state === sSelectExponent) {
        when(errorReg || !singleStageParametersValid) {
          errorReg := true.B
          rejectedTokens := rejectedTokens + 1.U
          doneReg := true.B
          state := sIdle
        }.otherwise {
          zeroPointReg :=
            selectedZeroWide.asUInt(metadataBits - 1, 0).asSInt
          exponentReg :=
            selectedExponentWide.asUInt(metadataBits - 1, 0).asSInt
          state := sMetadata
        }
      }
    } else {
      when(state === sSelectExponent) {
        when(errorReg || !selectedExponentValid) {
          errorReg := true.B
          rejectedTokens := rejectedTokens + 1.U
          doneReg := true.B
          state := sIdle
        }.otherwise {
          exponentReg :=
            selectedExponentWide.asUInt(metadataBits - 1, 0).asSInt
          state := sComputeZero
        }
      }

      when(state === sComputeZero) {
        when(!computedZeroValid) {
          errorReg := true.B
          rejectedTokens := rejectedTokens + 1.U
          doneReg := true.B
          state := sIdle
        }.otherwise {
          zeroPointReg := computedZeroWide.asUInt(metadataBits - 1, 0).asSInt
          state := sValidateRange
        }
      }

      when(state === sValidateRange) {
        when(!computedCodeRangeValid) {
          errorReg := true.B
          rejectedTokens := rejectedTokens + 1.U
          doneReg := true.B
          state := sIdle
        }.otherwise {
          state := sMetadata
        }
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
