package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class ReplayByteStreamBufferSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  "Replay byte stream buffer" - {
    "must preserve byte order across seal, replay, and backpressure" in {
      val payload = IndexedSeq(0x12, 0x00, 0xff, 0x5a, 0x81)

      simulate(new ReplayByteStreamBuffer(capacityBytes = 8)) { dut =>
        dut.io.clear.poke(true.B)
        dut.io.seal.poke(false.B)
        dut.io.readStart.poke(false.B)
        dut.io.writeIn.valid.poke(false.B)
        dut.io.writeIn.bits.poke(0.U)
        dut.io.readOut.ready.poke(false.B)
        dut.clock.step()
        dut.io.clear.poke(false.B)

        payload.foreach { byte =>
          dut.io.writeIn.valid.poke(true.B)
          dut.io.writeIn.bits.poke(byte.U)
          dut.io.writeIn.ready.expect(true.B)
          dut.clock.step()
        }
        dut.io.writeIn.valid.poke(false.B)
        dut.io.length.expect(payload.length.U)

        dut.io.seal.poke(true.B)
        dut.clock.step()
        dut.io.seal.poke(false.B)
        dut.io.sealedStream.expect(true.B)
        dut.io.writeIn.ready.expect(false.B)

        dut.io.readStart.poke(true.B)
        dut.clock.step()
        dut.io.readStart.poke(false.B)

        var received = Vector.empty[Int]
        var cycles = 0
        var sawDone = false
        while ((!sawDone || received.length < payload.length) && cycles < 100) {
          val accept = cycles % 3 != 1
          dut.io.readOut.ready.poke(accept.B)
          if (accept && dut.io.readOut.valid.peek().litToBoolean) {
            received :+= dut.io.readOut.bits.peek().litValue.toInt
          }
          if (dut.io.readDone.peek().litToBoolean) sawDone = true
          dut.clock.step()
          cycles += 1
        }

        received mustBe payload
        sawDone mustBe true
        dut.io.overflow.expect(false.B)
      }
    }

    "must terminate the producer and report overflow explicitly" in {
      simulate(new ReplayByteStreamBuffer(capacityBytes = 2)) { dut =>
        dut.io.clear.poke(true.B)
        dut.io.seal.poke(false.B)
        dut.io.readStart.poke(false.B)
        dut.io.readOut.ready.poke(false.B)
        dut.io.writeIn.valid.poke(false.B)
        dut.io.writeIn.bits.poke(0.U)
        dut.clock.step()
        dut.io.clear.poke(false.B)

        (0 until 3).foreach { byte =>
          dut.io.writeIn.valid.poke(true.B)
          dut.io.writeIn.bits.poke(byte.U)
          dut.io.writeIn.ready.expect(true.B)
          dut.clock.step()
        }
        dut.io.writeIn.valid.poke(false.B)
        dut.io.length.expect(2.U)
        dut.io.overflow.expect(true.B)
      }
    }
  }
}
