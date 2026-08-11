package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class QueryReplayBufferSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  "Query replay buffer" - {
    "must load once and replay every feature for every pack under backpressure" in {
      val query = IndexedSeq(64, -32, 11, 127, -96)
      val packCount = 3
      val random = new Random(0x51524550L)

      simulate(new QueryReplayBuffer()) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(query.length.U)
        dut.io.packCount.poke(packCount.U)
        dut.io.loadIn.valid.poke(false.B)
        dut.io.loadIn.bits.poke(0.S)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var loadIndex = 0
        var outputIndex = 0
        var cycles = 0
        var done = false
        while (!done && cycles < 1000) {
          val offerLoad = loadIndex < query.length && random.nextBoolean()
          val acceptOutput = cycles > 10 && random.nextBoolean()
          dut.io.loadIn.valid.poke(offerLoad.B)
          dut.io.loadIn.bits.poke(
            (if (loadIndex < query.length) query(loadIndex) else 0).S
          )
          dut.io.out.ready.poke(acceptOutput.B)

          val loadFire = offerLoad && dut.io.loadIn.ready.peek().litToBoolean
          val outputFire = acceptOutput && dut.io.out.valid.peek().litToBoolean
          if (outputFire) {
            val packIndex = outputIndex / query.length
            val featureIndex = outputIndex % query.length
            dut.io.out.bits.value.expect(query(featureIndex).S)
            dut.io.out.bits.featureIndex.expect(featureIndex.U)
            dut.io.out.bits.packIndex.expect(packIndex.U)
            dut.io.out.bits.last.expect(
              (outputIndex == query.length * packCount - 1).B
            )
          }

          dut.clock.step()
          if (loadFire) loadIndex += 1
          if (outputFire) outputIndex += 1
          done = dut.io.done.peek().litToBoolean
          cycles += 1
        }

        done mustBe true
        loadIndex mustBe query.length
        outputIndex mustBe query.length * packCount
        dut.io.error.expect(false.B)
        dut.io.loaded.expect(true.B)
        dut.io.stats.loadedValues.expect(query.length.U)
        dut.io.stats.readRequests.expect((query.length * packCount).U)
        dut.io.stats.replayedValues.expect((query.length * packCount).U)
        dut.io.stats.downstreamStallCycles.peek().litValue must be > BigInt(0)
      }
    }

    "must reject invalid geometry without accepting query data" in {
      simulate(new QueryReplayBuffer(maximumFeatureDim = 8)) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(9.U)
        dut.io.packCount.poke(1.U)
        dut.io.loadIn.valid.poke(true.B)
        dut.io.loadIn.bits.poke(1.S)
        dut.io.out.ready.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        dut.io.done.expect(true.B)
        dut.io.error.expect(true.B)
        dut.io.busy.expect(false.B)
        dut.io.loadIn.ready.expect(false.B)
      }
    }

    "must sustain one replayed feature per cycle after SRAM priming" in {
      val query = IndexedSeq(7, -11, 19, 23)
      val packCount = 3
      simulate(new QueryReplayBuffer()) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(query.length.U)
        dut.io.packCount.poke(packCount.U)
        dut.io.loadIn.valid.poke(false.B)
        dut.io.out.ready.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        for (value <- query) {
          dut.io.loadIn.valid.poke(true.B)
          dut.io.loadIn.bits.poke(value.S)
          dut.io.loadIn.ready.expect(true.B)
          dut.clock.step()
        }
        dut.io.loadIn.valid.poke(false.B)

        val fireCycles = collection.mutable.ArrayBuffer.empty[Int]
        var cycle = 0
        var done = false
        while (!done && cycle < 100) {
          if (dut.io.out.valid.peek().litToBoolean) {
            fireCycles += cycle
          }
          dut.clock.step()
          done = dut.io.done.peek().litToBoolean
          cycle += 1
        }

        done mustBe true
        fireCycles.length mustBe query.length * packCount
        fireCycles.sliding(2).foreach { pair =>
          pair(1) - pair(0) mustBe 1
        }
        dut.io.error.expect(false.B)
      }
    }
  }
}
