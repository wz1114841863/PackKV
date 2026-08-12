package briskkv

import chisel3._
import chisel3.util._

class AttentionOutputFeature(outputBits: Int, countBits: Int) extends Bundle {
  val value = SInt(outputBits.W)
  val featureIndex = UInt(countBits.W)
  val last = Bool()
}

class AvOutputQuantizerStats extends Bundle {
  val activeCycles = UInt(64.W)
  val inputFeatures = UInt(64.W)
  val outputFeatures = UInt(64.W)
  val positiveSaturations = UInt(64.W)
  val negativeSaturations = UInt(64.W)
  val downstreamStallCycles = UInt(64.W)
}

class AvOutputQuantizerIO(
  accumulatorBits: Int,
  outputBits: Int,
  countBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val in = Flipped(Decoupled(new AvFeatureResult(accumulatorBits, countBits)))
  val out = Decoupled(new AttentionOutputFeature(outputBits, countBits))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new AvOutputQuantizerStats)
}

/** Converts exact Q21 AV features to the signed Q6 compute-side format.
  *
  * Absolute-value rounding gives symmetric round-to-nearest with ties away
  * from zero. The restored signed value is saturated to the output width.
  */
class AvOutputQuantizer(
  accumulatorBits: Int = 50,
  inputFractionalBits: Int = 21,
  outputBits: Int = 18,
  outputFractionalBits: Int = 6,
  maximumFeatureDim: Int = 256,
  countBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  require(accumulatorBits > outputBits && outputBits >= 2)
  require(inputFractionalBits > outputFractionalBits)
  require(maximumFeatureDim > 0)

  private val fractionalShift = inputFractionalBits - outputFractionalBits
  private val wideBits = accumulatorBits + 1
  private val maximumOutput = (BigInt(1) << (outputBits - 1)) - 1
  private val minimumOutput = -(BigInt(1) << (outputBits - 1))

  val io = IO(
    new AvOutputQuantizerIO(accumulatorBits, outputBits, countBits)
  )

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val featureDimReg = RegInit(0.U(countBits.W))
  val expectedFeatureIndex = RegInit(0.U(countBits.W))
  val outputValid = RegInit(false.B)
  val outputReg = Reg(new AttentionOutputFeature(outputBits, countBits))

  val activeCycles = RegInit(0.U(64.W))
  val inputFeatures = RegInit(0.U(64.W))
  val outputFeatures = RegInit(0.U(64.W))
  val positiveSaturations = RegInit(0.U(64.W))
  val negativeSaturations = RegInit(0.U(64.W))
  val downstreamStallCycles = RegInit(0.U(64.W))

  doneReg := false.B
  io.busy := active || outputValid
  io.done := doneReg
  io.error := errorReg
  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.inputFeatures := Mux(enableStats.B, inputFeatures, 0.U)
  io.stats.outputFeatures := Mux(enableStats.B, outputFeatures, 0.U)
  io.stats.positiveSaturations := Mux(enableStats.B, positiveSaturations, 0.U)
  io.stats.negativeSaturations := Mux(enableStats.B, negativeSaturations, 0.U)
  io.stats.downstreamStallCycles := Mux(enableStats.B, downstreamStallCycles, 0.U)

  io.in.ready := active && (!outputValid || io.out.ready)
  io.out.valid := outputValid
  io.out.bits := outputReg
  val inputFire = io.in.valid && io.in.ready
  val outputFire = io.out.valid && io.out.ready

  when(io.start && !active && !outputValid) {
    val parametersValid = io.featureDim =/= 0.U &&
      io.featureDim <= maximumFeatureDim.U
    active := parametersValid
    doneReg := !parametersValid
    errorReg := !parametersValid
    featureDimReg := io.featureDim
    expectedFeatureIndex := 0.U
    activeCycles := 0.U
    inputFeatures := 0.U
    outputFeatures := 0.U
    positiveSaturations := 0.U
    negativeSaturations := 0.U
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
      outputFeatures := outputFeatures + 1.U
      when(outputReg.last) {
        doneReg := true.B
      }
    }

    when(inputFire) {
      val expectedLast = expectedFeatureIndex === featureDimReg - 1.U
      when(
        io.in.bits.featureIndex =/= expectedFeatureIndex ||
          io.in.bits.last =/= expectedLast
      ) {
        errorReg := true.B
      }

      val negative = io.in.bits.value < 0.S
      val magnitude = Mux(
        negative,
        (-io.in.bits.value).asUInt,
        io.in.bits.value.asUInt
      )
      val roundedMagnitude =
        (magnitude + (BigInt(1) << (fractionalShift - 1)).U) >>
          fractionalShift
      val roundedWide = Wire(SInt(wideBits.W))
      roundedWide := Mux(
        negative,
        -roundedMagnitude.asSInt,
        roundedMagnitude.asSInt
      )
      val positiveOverflow = roundedWide > maximumOutput.S(wideBits.W)
      val negativeOverflow = roundedWide < minimumOutput.S(wideBits.W)
      outputReg.value := Mux(
        positiveOverflow,
        maximumOutput.S(outputBits.W),
        Mux(
          negativeOverflow,
          minimumOutput.S(outputBits.W),
          roundedWide.asUInt(outputBits - 1, 0).asSInt
        )
      )
      outputReg.featureIndex := expectedFeatureIndex
      outputReg.last := expectedLast
      outputValid := true.B
      inputFeatures := inputFeatures + 1.U
      expectedFeatureIndex := expectedFeatureIndex + 1.U
      when(positiveOverflow) {
        positiveSaturations := positiveSaturations + 1.U
      }
      when(negativeOverflow) {
        negativeSaturations := negativeSaturations + 1.U
      }
      when(expectedLast) {
        active := false.B
      }
    }
  }
}
