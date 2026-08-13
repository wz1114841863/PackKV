package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class JitVAccumulatorSpec extends AnyFreeSpec with Matchers with ChiselSim {
  "JIT-V two-entry streaming accumulator" in {
    val packs = 4
    val features = 4
    val tokens = packs * 16
    val weights = IndexedSeq.tabulate(packs, 16) { (pack, lane) =>
      128 + pack * 7 + lane
    }
    val values = IndexedSeq.tabulate(packs, features, 16) {
      (pack, feature, lane) => pack * 13 + feature * 5 + lane - 20
    }
    val expected = IndexedSeq.tabulate(features) { feature =>
      (0 until packs).map { pack =>
        (0 until 16).map { lane =>
          BigInt(weights(pack)(lane)) * values(pack)(feature)(lane)
        }.sum
      }.sum
    }

    simulate(
      new JitVAccumulator(
        maximumFeatureDim = 8,
        maximumTokens = tokens
      )
    ) { dut =>
      dut.io.start.poke(false.B)
      dut.io.featureDim.poke(features.U)
      dut.io.tokenCount.poke(tokens.U)
      dut.io.weightIn.valid.poke(false.B)
      dut.io.vIn.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.clock.step()
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var weightIndex = 0
      var vIndex = 0
      var results = Vector.empty[BigInt]
      var cycles = 0
      var vStartedBeforeAllWeights = false
      while (results.length < features && cycles < 5000) {
        // Slow down the weight producer so that V must both overlap it and
        // exercise the queue's per-pack dependency backpressure.
        if (weightIndex < packs && cycles % 7 == 0) {
          dut.io.weightIn.valid.poke(true.B)
          dut.io.weightIn.bits.packIndex.poke(weightIndex.U)
          dut.io.weightIn.bits.blockIndex.poke((weightIndex / 4).U)
          dut.io.weightIn.bits.packWithinBlock.poke((weightIndex % 4).U)
          dut.io.weightIn.bits.validTokens.poke(16.U)
          dut.io.weightIn.bits.last.poke((weightIndex == packs - 1).B)
          weights(weightIndex).indices.foreach { lane =>
            dut.io.weightIn.bits.weights(lane).poke(weights(weightIndex)(lane).U)
          }
          if (dut.io.weightIn.ready.peek().litToBoolean) weightIndex += 1
        } else dut.io.weightIn.valid.poke(false.B)

        if (dut.io.vLaunchReady.peek().litToBoolean && vIndex < packs * features) {
          val pack = vIndex / features
          val feature = vIndex % features
          dut.io.vIn.valid.poke(true.B)
          dut.io.vIn.bits.validTokens.poke(16.U)
          dut.io.vIn.bits.descriptorIndex.poke(vIndex.U)
          dut.io.vIn.bits.packIndex.poke(pack.U)
          dut.io.vIn.bits.featureIndex.poke(feature.U)
          dut.io.vIn.bits.blockIndex.poke((pack / 4).U)
          dut.io.vIn.bits.packWithinBlock.poke((pack % 4).U)
          dut.io.vIn.bits.last.poke((vIndex == packs * features - 1).B)
          values(pack)(feature).indices.foreach { lane =>
            dut.io.vIn.bits.values(lane).poke(values(pack)(feature)(lane).S)
          }
          if (dut.io.vIn.ready.peek().litToBoolean) {
            if (weightIndex < packs) vStartedBeforeAllWeights = true
            vIndex += 1
          }
        } else dut.io.vIn.valid.poke(false.B)

        val accept = cycles % 4 != 1
        dut.io.out.ready.poke(accept.B)
        if (accept && dut.io.out.valid.peek().litToBoolean) {
          dut.io.out.bits.featureIndex.expect(results.length.U)
          results :+= dut.io.out.bits.value.peek().litValue
        }
        dut.clock.step()
        cycles += 1
      }

      weightIndex mustBe packs
      vIndex mustBe packs * features
      vStartedBeforeAllWeights mustBe true
      results mustBe expected
      dut.io.error.expect(false.B)
      dut.io.stats.acceptedVPackets.expect((packs * features).U)
    }
  }
}
