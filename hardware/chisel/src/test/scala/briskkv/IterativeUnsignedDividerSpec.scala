package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class IterativeUnsignedDividerSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  "Iterative unsigned divider" - {
    "must match exact integer division across Softmax reciprocal boundaries" in {
      val numeratorBits = 33
      val denominatorBits = 31
      val directedCases = IndexedSeq(
        (BigInt(0), BigInt(1)),
        (BigInt(1), BigInt(1)),
        (BigInt(1) << 32, BigInt(1) << 16),
        ((BigInt(1) << 32) + 123456, BigInt(654321)),
        ((BigInt(1) << 32) + (BigInt(1) << 29), BigInt(1)),
        ((BigInt(1) << 33) - 1, (BigInt(1) << 31) - 1)
      )
      val random = new Random(0x44495649444552L)
      val randomizedCases = IndexedSeq.fill(64) {
        val numerator = BigInt(random.nextLong()) & ((BigInt(1) << 33) - 1)
        val denominator =
          (BigInt(random.nextInt()) & ((BigInt(1) << 31) - 1)).max(1)
        (numerator, denominator)
      }
      val cases = directedCases ++ randomizedCases

      simulate(new IterativeUnsignedDivider(numeratorBits, denominatorBits)) {
        dut =>
          dut.io.start.poke(false.B)
          dut.io.numerator.poke(0.U)
          dut.io.denominator.poke(1.U)
          dut.clock.step()

          cases.foreach { case (numerator, denominator) =>
            dut.io.numerator.poke(numerator.U)
            dut.io.denominator.poke(denominator.U)
            dut.io.start.poke(true.B)
            dut.clock.step()
            dut.io.start.poke(false.B)
            dut.io.busy.expect(true.B)

            var cycles = 0
            while (!dut.io.done.peek().litToBoolean && cycles < 40) {
              dut.clock.step()
              cycles += 1
            }
            cycles mustBe numeratorBits
            dut.io.quotient.expect((numerator / denominator).U)
            dut.io.error.expect(false.B)
            dut.io.busy.expect(false.B)
            dut.clock.step()
            dut.io.done.expect(false.B)
          }
        }
    }

    "must terminate with an error on division by zero" in {
      simulate(new IterativeUnsignedDivider(33, 31)) { dut =>
        dut.io.start.poke(false.B)
        dut.io.numerator.poke(123.U)
        dut.io.denominator.poke(0.U)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        dut.io.done.expect(true.B)
        dut.io.busy.expect(false.B)
        dut.io.error.expect(true.B)
        dut.io.quotient.expect(0.U)
      }
    }
  }
}
