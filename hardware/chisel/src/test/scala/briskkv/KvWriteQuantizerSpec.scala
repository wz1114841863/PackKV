package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class KvWriteQuantizerSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private val FractionalBits = 12

  private def raw(value: Double): Long =
    Math.round(value * (1L << FractionalBits))

  private def runGolden(
    isKey: Boolean,
    values: IndexedSeq[Double],
    expectedZero: Int,
    expectedExponent: Int,
    expectedQ: IndexedSeq[Int],
    randomBackpressure: Boolean,
    parameterArchitecture: QuantParameterArchitecture =
      QuantParameterArchitecture.V1SingleStage
  ): Unit = {
    val random = new Random(if (isKey) 0x4b51554e54L else 0x5651554e54L)
    simulate(
      new KvWriteQuantizer(
        isKey = isKey,
        inputFractionalBits = FractionalBits,
        maximumFeatureDim = 16,
        parameterArchitecture = parameterArchitecture
      )
    ) { dut =>
      dut.io.start.poke(false.B)
      dut.io.featureDim.poke(values.length.U)
      dut.io.tokenTag.poke(37.U)
      dut.io.in.valid.poke(false.B)
      dut.io.in.bits.fixedRaw.poke(0.S)
      dut.io.in.bits.featureIndex.poke(0.U)
      dut.io.in.bits.last.poke(false.B)
      dut.io.metadataOut.ready.poke(false.B)
      dut.io.qOut.ready.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var inputIndex = 0
      var metadataCount = 0
      var outputIndex = 0
      var cycles = 0
      var done = false
      while (!done && cycles < 1000) {
        val offerInput = inputIndex < values.length &&
          (!randomBackpressure || random.nextBoolean())
        val acceptMetadata = !randomBackpressure || random.nextBoolean()
        val acceptOutput = !randomBackpressure || random.nextBoolean()
        dut.io.in.valid.poke(offerInput.B)
        dut.io.in.bits.fixedRaw.poke(
          raw(values(Math.min(inputIndex, values.length - 1))).S
        )
        dut.io.in.bits.featureIndex.poke(inputIndex.U)
        dut.io.in.bits.last.poke((inputIndex == values.length - 1).B)
        dut.io.metadataOut.ready.poke(acceptMetadata.B)
        dut.io.qOut.ready.poke(acceptOutput.B)

        val inputFire = offerInput && dut.io.in.ready.peek().litToBoolean
        val metadataFire = acceptMetadata &&
          dut.io.metadataOut.valid.peek().litToBoolean
        val outputFire = acceptOutput && dut.io.qOut.valid.peek().litToBoolean
        if (metadataFire) {
          dut.io.metadataOut.bits.zeroPoint.expect(expectedZero.S)
          dut.io.metadataOut.bits.exponent.expect(expectedExponent.S)
          dut.io.metadataOut.bits.tokenTag.expect(37.U)
        }
        if (outputFire) {
          dut.io.qOut.bits.q.expect(expectedQ(outputIndex).U)
          dut.io.qOut.bits.tokenTag.expect(37.U)
          dut.io.qOut.bits.featureIndex.expect(outputIndex.U)
          dut.io.qOut.bits.last.expect((outputIndex == values.length - 1).B)
        }

        dut.clock.step()
        if (inputFire) inputIndex += 1
        if (metadataFire) metadataCount += 1
        if (outputFire) outputIndex += 1
        done = dut.io.done.peek().litToBoolean
        cycles += 1
      }

      done mustBe true
      inputIndex mustBe values.length
      metadataCount mustBe 1
      outputIndex mustBe values.length
      dut.io.error.expect(false.B)
      dut.io.stats.inputValues.expect(values.length.U)
      dut.io.stats.outputValues.expect(values.length.U)
      dut.io.stats.rejectedTokens.expect(0.U)
      dut.clock.step()
      dut.io.busy.expect(false.B)
    }
  }

  "Format v0 write-side token quantizer" - {
    "must register exponent, zero point, and range validation in separate cycles" in {
      simulate(
        new KvWriteQuantizer(
          isKey = true,
          inputFractionalBits = FractionalBits,
          maximumFeatureDim = 4,
          enableStats = false,
          parameterArchitecture = QuantParameterArchitecture.V2ThreeStage
        )
      ) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(2.U)
        dut.io.tokenTag.poke(3.U)
        dut.io.in.valid.poke(false.B)
        dut.io.metadataOut.ready.poke(false.B)
        dut.io.qOut.ready.poke(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        for ((value, index) <- Seq((0.0, 0), (1.0, 1))) {
          dut.io.in.valid.poke(true.B)
          dut.io.in.bits.fixedRaw.poke(raw(value).S)
          dut.io.in.bits.featureIndex.poke(index.U)
          dut.io.in.bits.last.poke((index == 1).B)
          dut.clock.step()
        }
        dut.io.in.valid.poke(false.B)

        // The cycle following the final input is exponent selection, followed
        // by zero-point computation and maximum-code validation.
        for (_ <- 0 until 3) {
          dut.io.metadataOut.valid.expect(false.B)
          dut.clock.step()
        }
        dut.io.metadataOut.valid.expect(true.B)
        dut.io.metadataOut.bits.zeroPoint.expect(0.S)
        dut.io.metadataOut.bits.exponent.expect((-5).S)

        dut.io.metadataOut.ready.poke(true.B)
        dut.clock.step()
        var cycles = 0
        while (!dut.io.done.peek().litToBoolean && cycles < 20) {
          dut.clock.step()
          cycles += 1
        }
        dut.io.done.expect(true.B)
        dut.io.error.expect(false.B)
      }
    }

    "must match software-derived K nearest-2^k values and ties-to-even" in {
      runGolden(
        isKey = true,
        values = IndexedSeq(-2.0, -0.375, -0.125, 0.125, 0.375, 2.0, 4.0),
        expectedZero = -8,
        expectedExponent = -2,
        expectedQ = IndexedSeq(0, 6, 8, 8, 10, 16, 24),
        randomBackpressure = true
      )
    }

    "must match software-derived V nearest-2^k values and ties-to-even" in {
      runGolden(
        isKey = false,
        values = IndexedSeq(-1.0, -0.375, -0.125, 0.125, 0.375, 1.0, 2.0),
        expectedZero = -4,
        expectedExponent = -2,
        expectedQ = IndexedSeq(0, 2, 4, 4, 6, 8, 12),
        randomBackpressure = true
      )
    }

    "v2 must preserve the v1 quantization result" in {
      runGolden(
        isKey = true,
        values = IndexedSeq(-2.0, -0.375, -0.125, 0.125, 0.375, 2.0, 4.0),
        expectedZero = -8,
        expectedExponent = -2,
        expectedQ = IndexedSeq(0, 6, 8, 8, 10, 16, 24),
        randomBackpressure = true,
        parameterArchitecture = QuantParameterArchitecture.V2ThreeStage
      )
    }

    "must match the K FP32 exponent boundary instead of the ideal-real threshold" in {
      val upperRaw = 3089397L
      runGolden(
        isKey = true,
        values = IndexedSeq(0.0, upperRaw.toDouble / (1L << FractionalBits)),
        expectedZero = 0,
        expectedExponent = 4,
        expectedQ = IndexedSeq(0, 47),
        randomBackpressure = false
      )
    }

    "must reject a token whose nearest exponent is outside Format v0" in {
      simulate(
        new KvWriteQuantizer(
          isKey = true,
          inputFractionalBits = FractionalBits,
          maximumFeatureDim = 4
        )
      ) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(2.U)
        dut.io.tokenTag.poke(9.U)
        dut.io.in.valid.poke(false.B)
        dut.io.metadataOut.ready.poke(true.B)
        dut.io.qOut.ready.poke(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        for (index <- 0 until 2) {
          dut.io.in.valid.poke(true.B)
          dut.io.in.bits.fixedRaw.poke(raw(1.0).S)
          dut.io.in.bits.featureIndex.poke(index.U)
          dut.io.in.bits.last.poke((index == 1).B)
          while (!dut.io.in.ready.peek().litToBoolean) dut.clock.step()
          dut.clock.step()
        }
        dut.io.in.valid.poke(false.B)
        var cycles = 0
        while (!dut.io.done.peek().litToBoolean && cycles < 20) {
          dut.io.metadataOut.valid.expect(false.B)
          dut.io.qOut.valid.expect(false.B)
          dut.clock.step()
          cycles += 1
        }
        dut.io.done.expect(true.B)
        dut.io.error.expect(true.B)
        dut.io.stats.rejectedTokens.expect(1.U)
      }
    }

    "must reject malformed feature ordering without emitting a partial token" in {
      simulate(
        new KvWriteQuantizer(
          isKey = false,
          inputFractionalBits = FractionalBits,
          maximumFeatureDim = 4
        )
      ) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(2.U)
        dut.io.tokenTag.poke(11.U)
        dut.io.in.valid.poke(false.B)
        dut.io.metadataOut.ready.poke(true.B)
        dut.io.qOut.ready.poke(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        for ((value, index) <- Seq((-1.0, 0), (2.0, 0))) {
          dut.io.in.valid.poke(true.B)
          dut.io.in.bits.fixedRaw.poke(raw(value).S)
          dut.io.in.bits.featureIndex.poke(index.U)
          dut.io.in.bits.last.poke((value == 2.0).B)
          dut.clock.step()
        }
        dut.io.in.valid.poke(false.B)
        var cycles = 0
        while (!dut.io.done.peek().litToBoolean && cycles < 20) {
          dut.io.metadataOut.valid.expect(false.B)
          dut.io.qOut.valid.expect(false.B)
          dut.clock.step()
          cycles += 1
        }
        dut.io.done.expect(true.B)
        dut.io.error.expect(true.B)
      }
    }
  }
}
