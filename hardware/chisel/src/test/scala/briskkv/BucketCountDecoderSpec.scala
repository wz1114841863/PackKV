package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class BucketCountDecoderSpec extends AnyFreeSpec with Matchers with ChiselSim {
  private def runVector(caseName: String, randomBackpressure: Boolean): Unit = {
    val input = GoldenVectorLoader.unsigned(caseName, "bucket_counts.bin")
    val expectedFlat =
      GoldenVectorLoader.unsigned(caseName, "expected_bucket_counts.bin")
    expectedFlat.length % 4 mustBe 0
    val expected = expectedFlat.grouped(4).map(_.toIndexedSeq).toIndexedSeq
    val random = new Random(0x4255434bL + expected.length)

    simulate(new BucketCountDecoder()) { dut =>
      dut.io.start.poke(false.B)
      dut.io.blockCount.poke(expected.length.U)
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
          for (bucket <- 0 until 4) {
            dut.io.out.bits.counts(bucket).expect(expected(outputIndex)(bucket).U)
          }
          dut.io.out.bits.blockIndex.expect(outputIndex.U)
          dut.io.out.bits.last.expect((outputIndex == expected.length - 1).B)
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

  private def runInvalidHeader(bytes: Seq[Int]): Unit = {
    require(bytes.length == 3)
    simulate(new BucketCountDecoder()) { dut =>
      dut.io.start.poke(false.B)
      dut.io.blockCount.poke(1.U)
      dut.io.in.valid.poke(false.B)
      dut.io.in.bits.poke(0.U)
      dut.io.out.ready.poke(true.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      for (byte <- bytes) {
        dut.io.in.bits.poke(byte.U)
        dut.io.in.valid.poke(true.B)
        while (!dut.io.in.ready.peek().litToBoolean) dut.clock.step()
        dut.clock.step()
      }
      dut.io.in.valid.poke(false.B)
      while (!dut.io.out.valid.peek().litToBoolean) dut.clock.step()
      dut.clock.step()
      dut.io.done.expect(true.B)
      dut.io.error.expect(true.B)
    }
  }

  "Bucket count decoder" - {
    "must decode independently aligned single-block headers" in {
      runVector("directed_nonidentity", randomBackpressure = false)
      runVector("directed_width0", randomBackpressure = false)
    }

    "must decode multiple blocks under independent backpressure" in {
      runVector("random_seed_20260809", randomBackpressure = true)
    }

    "must reject non-zero per-block padding" in {
      runInvalidHeader(Seq(0x00, 0x00, 0x20))
    }

    "must reject counts whose sum exceeds 64" in {
      // LSB-first fields encode count0=64, count1=1, count2=0.
      runInvalidHeader(Seq(0xc0, 0x00, 0x00))
    }
  }
}
