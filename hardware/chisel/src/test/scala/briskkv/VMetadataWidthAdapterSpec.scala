package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class VMetadataWidthAdapterSpec extends AnyFreeSpec with Matchers with ChiselSim {
  private def pack(values: Seq[Int], bits: Int): Seq[Int] = {
    var reservoir = BigInt(0)
    var count = 0
    val out = collection.mutable.ArrayBuffer.empty[Int]
    values.foreach { value =>
      reservoir |= BigInt(value & ((1 << bits) - 1)) << count
      count += bits
      while (count >= 8) {
        out += (reservoir & 0xff).toInt
        reservoir >>= 8
        count -= 8
      }
    }
    if (count > 0) out += reservoir.toInt
    out.toSeq
  }

  "V compact metadata must widen without changing signed values" in {
    val minima = Seq(0, 1, 7, 15, 3, 11)
    val zeros = Seq(-16, -9, -1, 0, 7, 15)
    val minimumInput = pack(minima, 4)
    val zeroInput = pack(zeros, 5)
    val expectedMinimum = pack(minima, 6)
    val expectedZero = pack(zeros, 7)

    simulate(new VMetadataWidthAdapter()) { dut =>
      dut.io.start.poke(false.B)
      dut.io.descriptorCount.poke(minima.length.U)
      dut.io.tokenCount.poke(zeros.length.U)
      dut.io.minimumIn.valid.poke(false.B)
      dut.io.zeroPointIn.valid.poke(false.B)
      dut.io.minimumOut.ready.poke(true.B)
      dut.io.zeroPointOut.ready.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var minIn = 0
      var zeroIn = 0
      var minOut = Vector.empty[Int]
      var zeroOut = Vector.empty[Int]
      var cycles = 0
      while ((minOut.length < expectedMinimum.length ||
          zeroOut.length < expectedZero.length) && cycles < 1000) {
        dut.io.minimumIn.valid.poke((minIn < minimumInput.length).B)
        if (minIn < minimumInput.length)
          dut.io.minimumIn.bits.poke(minimumInput(minIn).U)
        dut.io.zeroPointIn.valid.poke((zeroIn < zeroInput.length).B)
        if (zeroIn < zeroInput.length)
          dut.io.zeroPointIn.bits.poke(zeroInput(zeroIn).U)
        if (dut.io.minimumIn.valid.peek().litToBoolean &&
            dut.io.minimumIn.ready.peek().litToBoolean) minIn += 1
        if (dut.io.zeroPointIn.valid.peek().litToBoolean &&
            dut.io.zeroPointIn.ready.peek().litToBoolean) zeroIn += 1
        if (dut.io.minimumOut.valid.peek().litToBoolean)
          minOut :+= dut.io.minimumOut.bits.peek().litValue.toInt
        if (dut.io.zeroPointOut.valid.peek().litToBoolean)
          zeroOut :+= dut.io.zeroPointOut.bits.peek().litValue.toInt
        dut.clock.step()
        cycles += 1
      }
      minOut mustBe expectedMinimum
      zeroOut mustBe expectedZero
      dut.io.error.expect(false.B)
    }
  }
}
