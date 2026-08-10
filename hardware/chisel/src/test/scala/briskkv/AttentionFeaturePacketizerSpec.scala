package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class AttentionFeaturePacketizerSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  "Attention feature packetizer" - {
    "must aggregate feature-major values and zero invalid lanes" in {
      val tokenCount = 35
      val featureDim = 3
      val packTokens = 16
      val packCount = 3
      val descriptorCount = packCount * featureDim
      val random = new Random(0x5041434bL)

      simulate(new AttentionFeaturePacketizer()) { dut =>
        dut.io.start.poke(false.B)
        dut.io.tokenCount.poke(tokenCount.U)
        dut.io.featureDim.poke(featureDim.U)
        dut.io.descriptorCount.poke(descriptorCount.U)
        dut.io.in.valid.poke(false.B)
        dut.io.in.bits.fixedRaw.poke(0.S)
        dut.io.in.bits.descriptorIndex.poke(0.U)
        dut.io.in.bits.tokenIndex.poke(0.U)
        dut.io.in.bits.last.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var descriptorIndex = 0
        var tokenIndex = 0
        var packetCount = 0
        var done = false
        var cycles = 0
        while (!done && cycles < 10000) {
          val packIndex = descriptorIndex / featureDim
          val featureIndex = descriptorIndex % featureDim
          val validTokens = math.min(packTokens, tokenCount - packIndex * packTokens)
          val globalToken = packIndex * packTokens + tokenIndex
          val value = globalToken * 10 + featureIndex
          val offerInput = descriptorIndex < descriptorCount && random.nextBoolean()
          val acceptOutput = random.nextBoolean()
          dut.io.in.valid.poke(offerInput.B)
          dut.io.in.bits.fixedRaw.poke(value.S)
          dut.io.in.bits.descriptorIndex.poke(descriptorIndex.U)
          dut.io.in.bits.tokenIndex.poke(tokenIndex.U)
          dut.io.in.bits.last.poke(
            (descriptorIndex == descriptorCount - 1 && tokenIndex == validTokens - 1).B
          )
          dut.io.out.ready.poke(acceptOutput.B)

          val inputFire = offerInput && dut.io.in.ready.peek().litToBoolean
          val outputFire = acceptOutput && dut.io.out.valid.peek().litToBoolean
          if (outputFire) {
            val outputPack = packetCount / featureDim
            val outputFeature = packetCount % featureDim
            val outputValid = math.min(
              packTokens,
              tokenCount - outputPack * packTokens
            )
            dut.io.out.bits.packIndex.expect(outputPack.U)
            dut.io.out.bits.featureIndex.expect(outputFeature.U)
            dut.io.out.bits.blockIndex.expect(0.U)
            dut.io.out.bits.packWithinBlock.expect(outputPack.U)
            dut.io.out.bits.validTokens.expect(outputValid.U)
            for (lane <- 0 until packTokens) {
              val expected = if (lane < outputValid) {
                (outputPack * packTokens + lane) * 10 + outputFeature
              } else 0
              dut.io.out.bits.values(lane).expect(expected.S)
            }
            dut.io.out.bits.last.expect((packetCount == descriptorCount - 1).B)
          }

          dut.clock.step()
          if (inputFire) {
            if (tokenIndex == validTokens - 1) {
              descriptorIndex += 1
              tokenIndex = 0
            } else tokenIndex += 1
          }
          if (outputFire) packetCount += 1
          done = dut.io.done.peek().litToBoolean
          cycles += 1
        }

        done mustBe true
        descriptorIndex mustBe descriptorCount
        packetCount mustBe descriptorCount
        dut.io.error.expect(false.B)
        dut.io.stats.inputValues.expect((tokenCount * featureDim).U)
        dut.io.stats.outputPackets.expect(descriptorCount.U)
      }
    }
  }
}
