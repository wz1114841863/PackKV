package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class CompactMetadataDecoderSpec extends AnyFreeSpec with Matchers with ChiselSim {
  private def runVector(
    caseName: String,
    fieldBits: Int,
    inputFile: String,
    expectedFile: String,
    randomBackpressure: Boolean
  ): Unit = {
    val input = GoldenVectorLoader.unsigned(caseName, inputFile)
    val expected = GoldenVectorLoader.signed(caseName, expectedFile)
    val random = new Random(0x42524953L + fieldBits + expected.length)

    simulate(new CompactMetadataDecoder(fieldBits)) { dut =>
      dut.io.start.poke(false.B)
      dut.io.fieldCount.poke(expected.length.U)
      dut.io.in.valid.poke(false.B)
      dut.io.in.bits.poke(0.U)
      dut.io.out.ready.poke(false.B)

      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var inputIndex = 0
      var outputIndex = 0
      var cycles = 0
      var done = false
      while (!done && cycles < 10000) {
        val offerInput = inputIndex < input.length &&
          (!randomBackpressure || random.nextBoolean())
        val acceptOutput = !randomBackpressure || random.nextBoolean()
        dut.io.in.valid.poke(offerInput.B)
        dut.io.in.bits.poke((if (inputIndex < input.length) input(inputIndex) else 0).U)
        dut.io.out.ready.poke(acceptOutput.B)

        val inputFire = offerInput && dut.io.in.ready.peek().litToBoolean
        val outputFire = acceptOutput && dut.io.out.valid.peek().litToBoolean
        if (outputFire) {
          outputIndex must be < expected.length
          dut.io.out.bits.expect(expected(outputIndex).S(8.W))
        }

        dut.clock.step()
        if (inputFire) inputIndex += 1
        if (outputFire) outputIndex += 1
        done = dut.io.done.peek().litToBoolean
        cycles += 1
      }

      done mustBe true
      inputIndex mustBe input.length
      outputIndex mustBe expected.length
      dut.io.error.expect(false.B)
      dut.io.busy.expect(false.B)
    }
  }

  "Compact metadata decoder" - {
    "must decode all four non-identity metadata streams without stalls" in {
      runVector("directed_nonidentity", 7, "k_zero_points.bin", "expected_k_zero.bin", false)
      runVector("directed_nonidentity", 5, "v_zero_points.bin", "expected_v_zero.bin", false)
      runVector("directed_nonidentity", 4, "k_exponents.bin", "expected_k_exponent.bin", false)
      runVector("directed_nonidentity", 4, "v_exponents.bin", "expected_v_exponent.bin", false)
    }

    "must preserve values under independent input and output backpressure" in {
      runVector("random_seed_20260809", 7, "k_zero_points.bin", "expected_k_zero.bin", true)
      runVector("random_seed_20260809", 5, "v_zero_points.bin", "expected_v_zero.bin", true)
      runVector("random_seed_20260809", 4, "k_exponents.bin", "expected_k_exponent.bin", true)
      runVector("random_seed_20260809", 4, "v_exponents.bin", "expected_v_exponent.bin", true)
    }

    "must reject non-zero final padding bits" in {
      simulate(new CompactMetadataDecoder(fieldBits = 5)) { dut =>
        dut.io.start.poke(false.B)
        dut.io.fieldCount.poke(1.U)
        dut.io.in.valid.poke(false.B)
        dut.io.in.bits.poke(0.U)
        dut.io.out.ready.poke(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        // Raw value is 3; the upper three padding bits are deliberately one.
        dut.io.in.bits.poke(0xe3.U)
        dut.io.in.valid.poke(true.B)
        while (!dut.io.in.ready.peek().litToBoolean) dut.clock.step()
        dut.clock.step()
        dut.io.in.valid.poke(false.B)
        while (!dut.io.out.valid.peek().litToBoolean) dut.clock.step()
        dut.io.out.bits.expect(3.S)
        dut.clock.step()
        dut.io.done.expect(true.B)
        dut.io.error.expect(true.B)
      }
    }
  }
}
