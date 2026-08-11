package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class QkComputePipelineSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  "Query-replay QK compute pipeline" - {
    "must match Python logits while loading the query only once" in {
      val caseName = "directed_nonidentity"
      val featureDim = 4
      val tokenCount = 64
      val packTokens = 16
      val packCount = tokenCount / packTokens
      val descriptorCount = packCount * featureDim
      val query = GoldenVectorLoader.int32LittleEndian(
        caseName,
        "qk_query_q6_i32.bin"
      )
      val key = GoldenVectorLoader.float32LittleEndian(
        caseName,
        "expected_k_dequant_f32.bin"
      ).map(value => Math.round(value * 64.0f))
      val expected = GoldenVectorLoader.int64LittleEndian(
        caseName,
        "expected_qk_logits_q12_i64.bin"
      )
      val random = new Random(0x51504c4eL)

      simulate(new QkComputePipeline()) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(featureDim.U)
        dut.io.packCount.poke(packCount.U)
        dut.io.queryLoadIn.valid.poke(false.B)
        dut.io.queryLoadIn.bits.poke(0.S)
        dut.io.keyIn.valid.poke(false.B)
        dut.io.keyIn.bits.values.foreach(_.poke(0.S))
        dut.io.keyIn.bits.validTokens.poke(packTokens.U)
        dut.io.keyIn.bits.descriptorIndex.poke(0.U)
        dut.io.keyIn.bits.packIndex.poke(0.U)
        dut.io.keyIn.bits.featureIndex.poke(0.U)
        dut.io.keyIn.bits.blockIndex.poke(0.U)
        dut.io.keyIn.bits.packWithinBlock.poke(0.U)
        dut.io.keyIn.bits.last.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var queryIndex = 0
        var descriptor = 0
        var outputPack = 0
        var cycles = 0
        var done = false
        while (!done && cycles < 10000) {
          val queryRemains = queryIndex < query.length
          val keyRemains = descriptor < descriptorCount
          val packIndex = descriptor / featureDim
          val featureIndex = descriptor % featureDim
          val offerQuery = queryRemains && random.nextBoolean()
          val offerKey = keyRemains && random.nextBoolean()
          val acceptOutput = random.nextBoolean()

          dut.io.queryLoadIn.valid.poke(offerQuery.B)
          dut.io.queryLoadIn.bits.poke(
            (if (queryRemains) query(queryIndex) else 0).S
          )
          dut.io.keyIn.valid.poke(offerKey.B)
          dut.io.keyIn.bits.validTokens.poke(packTokens.U)
          dut.io.keyIn.bits.descriptorIndex.poke(descriptor.U)
          dut.io.keyIn.bits.packIndex.poke(
            (if (keyRemains) packIndex else 0).U
          )
          dut.io.keyIn.bits.featureIndex.poke(featureIndex.U)
          dut.io.keyIn.bits.blockIndex.poke(0.U)
          dut.io.keyIn.bits.packWithinBlock.poke(
            (if (keyRemains) packIndex else 0).U
          )
          dut.io.keyIn.bits.last.poke(
            (keyRemains && descriptor == descriptorCount - 1).B
          )
          for (lane <- 0 until packTokens) {
            val tokenIndex = packIndex * packTokens + lane
            val value =
              if (keyRemains) key(tokenIndex * featureDim + featureIndex)
              else 0
            dut.io.keyIn.bits.values(lane).poke(value.S)
          }
          dut.io.out.ready.poke(acceptOutput.B)

          val queryFire = offerQuery &&
            dut.io.queryLoadIn.ready.peek().litToBoolean
          val keyFire = offerKey && dut.io.keyIn.ready.peek().litToBoolean
          val outputFire = acceptOutput && dut.io.out.valid.peek().litToBoolean
          if (outputFire) {
            dut.io.out.bits.packIndex.expect(outputPack.U)
            dut.io.out.bits.validTokens.expect(packTokens.U)
            for (lane <- 0 until packTokens) {
              dut.io.out.bits.logits(lane).expect(
                expected(outputPack * packTokens + lane).S
              )
            }
            dut.io.out.bits.last.expect((outputPack == packCount - 1).B)
          }

          dut.clock.step()
          if (queryFire) queryIndex += 1
          if (keyFire) descriptor += 1
          if (outputFire) outputPack += 1
          done = dut.io.done.peek().litToBoolean
          cycles += 1
        }

        done mustBe true
        queryIndex mustBe featureDim
        descriptor mustBe descriptorCount
        outputPack mustBe packCount
        dut.io.error.expect(false.B)
        dut.io.queryLoaded.expect(true.B)
        dut.io.stats.queryReplay.loadedValues.expect(featureDim.U)
        dut.io.stats.queryReplay.replayedValues.expect(descriptorCount.U)
        dut.io.stats.accumulator.inputPackets.expect(descriptorCount.U)
        dut.io.stats.accumulator.outputPackets.expect(packCount.U)
      }
    }
  }
}
