package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class QkDotProductAccumulatorSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  "16-lane QK dot-product accumulator" - {
    "must match the Python Q12 golden logits" in {
      val caseName = "directed_nonidentity"
      val featureDim = 4
      val tokenCount = 64
      val packTokens = 16
      val descriptorCount = tokenCount / packTokens * featureDim
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
      val random = new Random(0x5059514bL)

      simulate(new QkDotProductAccumulator()) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(featureDim.U)
        dut.io.queryIn.valid.poke(false.B)
        dut.io.keyIn.valid.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var descriptor = 0
        var outputPack = 0
        var cycles = 0
        var done = false
        while (!done && cycles < 10000) {
          val inputsRemain = descriptor < descriptorCount
          val packIndex = descriptor / featureDim
          val featureIndex = descriptor % featureDim
          val offerQuery = inputsRemain && random.nextBoolean()
          val offerKey = inputsRemain && random.nextBoolean()
          val acceptOutput = random.nextBoolean()

          dut.io.queryIn.valid.poke(offerQuery.B)
          dut.io.queryIn.bits.value.poke(
            (if (inputsRemain) query(featureIndex) else 0).S
          )
          dut.io.queryIn.bits.featureIndex.poke(featureIndex.U)
          dut.io.queryIn.bits.packIndex.poke(
            (if (inputsRemain) packIndex else 0).U
          )
          dut.io.queryIn.bits.last.poke(
            (inputsRemain && descriptor == descriptorCount - 1).B
          )
          dut.io.keyIn.valid.poke(offerKey.B)
          dut.io.keyIn.bits.validTokens.poke(packTokens.U)
          dut.io.keyIn.bits.descriptorIndex.poke(descriptor.U)
          dut.io.keyIn.bits.packIndex.poke(
            (if (inputsRemain) packIndex else 0).U
          )
          dut.io.keyIn.bits.featureIndex.poke(featureIndex.U)
          dut.io.keyIn.bits.blockIndex.poke(0.U)
          dut.io.keyIn.bits.packWithinBlock.poke(
            (if (inputsRemain) packIndex else 0).U
          )
          dut.io.keyIn.bits.last.poke(
            (inputsRemain && descriptor == descriptorCount - 1).B
          )
          for (lane <- 0 until packTokens) {
            val tokenIndex = packIndex * packTokens + lane
            val value =
              if (inputsRemain) key(tokenIndex * featureDim + featureIndex)
              else 0
            dut.io.keyIn.bits.values(lane).poke(value.S)
          }
          dut.io.out.ready.poke(acceptOutput.B)

          val inputFire = offerQuery && offerKey &&
            dut.io.queryIn.ready.peek().litToBoolean &&
            dut.io.keyIn.ready.peek().litToBoolean
          val outputFire = acceptOutput && dut.io.out.valid.peek().litToBoolean
          if (outputFire) {
            dut.io.out.bits.packIndex.expect(outputPack.U)
            dut.io.out.bits.validTokens.expect(packTokens.U)
            for (lane <- 0 until packTokens) {
              dut.io.out.bits.logits(lane).expect(
                expected(outputPack * packTokens + lane).S
              )
            }
            dut.io.out.bits.last.expect((outputPack == 3).B)
          }

          dut.clock.step()
          if (inputFire) descriptor += 1
          if (outputFire) outputPack += 1
          done = dut.io.done.peek().litToBoolean
          cycles += 1
        }

        done mustBe true
        descriptor mustBe descriptorCount
        outputPack mustBe 4
        dut.io.error.expect(false.B)
        dut.io.stats.inputPackets.expect(descriptorCount.U)
        dut.io.stats.outputPackets.expect(4.U)
        dut.io.stats.macOperations.expect((tokenCount * featureDim).U)
      }
    }

    "must accumulate Q6 inputs into exact Q12 logits under backpressure" in {
      val featureDim = 3
      val packTokens = 16
      val validTokensByPack = IndexedSeq(16, 5)
      val query = IndexedSeq(64, -32, 128)
      val random = new Random(0x514b4d41L)

      simulate(new QkDotProductAccumulator()) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(featureDim.U)
        dut.io.queryIn.valid.poke(false.B)
        dut.io.queryIn.bits.value.poke(0.S)
        dut.io.queryIn.bits.featureIndex.poke(0.U)
        dut.io.queryIn.bits.packIndex.poke(0.U)
        dut.io.queryIn.bits.last.poke(false.B)
        dut.io.keyIn.valid.poke(false.B)
        dut.io.keyIn.bits.values.foreach(_.poke(0.S))
        dut.io.keyIn.bits.validTokens.poke(0.U)
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

        var descriptor = 0
        var outputPack = 0
        var cycles = 0
        var done = false
        while (!done && cycles < 1000) {
          val packIndex = descriptor / featureDim
          val featureIndex = descriptor % featureDim
          val inputsRemain = packIndex < validTokensByPack.length
          val offerQuery = inputsRemain && random.nextBoolean()
          val offerKey = inputsRemain && random.nextBoolean()
          val acceptOutput = cycles > 8 && random.nextBoolean()
          val validTokens =
            if (inputsRemain) validTokensByPack(packIndex) else 0

          dut.io.queryIn.valid.poke(offerQuery.B)
          dut.io.queryIn.bits.value.poke(
            (if (inputsRemain) query(featureIndex) else 0).S
          )
          dut.io.queryIn.bits.featureIndex.poke(featureIndex.U)
          dut.io.queryIn.bits.packIndex.poke(packIndex.U)
          dut.io.queryIn.bits.last.poke(
            (inputsRemain && packIndex == 1 && featureIndex == featureDim - 1).B
          )
          dut.io.keyIn.valid.poke(offerKey.B)
          dut.io.keyIn.bits.validTokens.poke(validTokens.U)
          dut.io.keyIn.bits.descriptorIndex.poke(descriptor.U)
          dut.io.keyIn.bits.packIndex.poke(packIndex.U)
          dut.io.keyIn.bits.featureIndex.poke(featureIndex.U)
          dut.io.keyIn.bits.blockIndex.poke(0.U)
          dut.io.keyIn.bits.packWithinBlock.poke(packIndex.U)
          dut.io.keyIn.bits.last.poke(
            (inputsRemain && packIndex == 1 && featureIndex == featureDim - 1).B
          )
          for (lane <- 0 until packTokens) {
            val key =
              if (inputsRemain && lane < validTokens)
                (packIndex + 1) * 100 + lane * 7 + featureIndex * 3 - 50
              else 0
            dut.io.keyIn.bits.values(lane).poke(key.S)
          }
          dut.io.out.ready.poke(acceptOutput.B)

          val inputFire = offerQuery && offerKey &&
            dut.io.queryIn.ready.peek().litToBoolean &&
            dut.io.keyIn.ready.peek().litToBoolean
          val outputFire = acceptOutput && dut.io.out.valid.peek().litToBoolean

          if (outputFire) {
            dut.io.out.bits.packIndex.expect(outputPack.U)
            dut.io.out.bits.blockIndex.expect(0.U)
            dut.io.out.bits.packWithinBlock.expect(outputPack.U)
            dut.io.out.bits.validTokens.expect(validTokensByPack(outputPack).U)
            for (lane <- 0 until packTokens) {
              val expected = if (lane < validTokensByPack(outputPack)) {
                query.indices.map { feature =>
                  val key =
                    (outputPack + 1) * 100 + lane * 7 + feature * 3 - 50
                  query(feature).toLong * key.toLong
                }.sum
              } else 0L
              dut.io.out.bits.logits(lane).expect(expected.S)
            }
            dut.io.out.bits.last.expect((outputPack == 1).B)
          }

          dut.clock.step()
          if (inputFire) descriptor += 1
          if (outputFire) outputPack += 1
          done = dut.io.done.peek().litToBoolean
          cycles += 1
        }

        done mustBe true
        descriptor mustBe featureDim * validTokensByPack.length
        outputPack mustBe validTokensByPack.length
        dut.io.error.expect(false.B)
        dut.io.stats.inputPackets.expect(6.U)
        dut.io.stats.outputPackets.expect(2.U)
        dut.io.stats.macOperations.expect(63.U)
        dut.io.stats.activeCycles.peek().litValue must be > BigInt(0)
        dut.io.stats.downstreamStallCycles.peek().litValue must be > BigInt(0)
      }
    }

    "must report malformed feature sequencing without consuming streams separately" in {
      simulate(new QkDotProductAccumulator()) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(2.U)
        dut.io.queryIn.valid.poke(false.B)
        dut.io.keyIn.valid.poke(false.B)
        dut.io.out.ready.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        dut.io.queryIn.valid.poke(true.B)
        dut.io.queryIn.bits.value.poke(64.S)
        dut.io.queryIn.bits.featureIndex.poke(1.U)
        dut.io.queryIn.bits.packIndex.poke(0.U)
        dut.io.queryIn.bits.last.poke(false.B)
        dut.io.keyIn.valid.poke(false.B)
        dut.io.queryIn.ready.expect(false.B)
        dut.clock.step()
        dut.io.stats.keyWaitCycles.expect(1.U)

        dut.io.keyIn.valid.poke(true.B)
        dut.io.keyIn.bits.values.foreach(_.poke(64.S))
        dut.io.keyIn.bits.validTokens.poke(16.U)
        dut.io.keyIn.bits.descriptorIndex.poke(0.U)
        dut.io.keyIn.bits.packIndex.poke(0.U)
        dut.io.keyIn.bits.featureIndex.poke(0.U)
        dut.io.keyIn.bits.blockIndex.poke(0.U)
        dut.io.keyIn.bits.packWithinBlock.poke(0.U)
        dut.io.keyIn.bits.last.poke(false.B)
        dut.io.queryIn.ready.expect(true.B)
        dut.io.keyIn.ready.expect(true.B)
        dut.clock.step()
        dut.io.error.expect(true.B)
        dut.io.stats.inputPackets.expect(1.U)
      }
    }

    "must reject an unsupported feature dimension before accepting data" in {
      simulate(new QkDotProductAccumulator(maximumFeatureDim = 256)) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(257.U)
        dut.io.queryIn.valid.poke(false.B)
        dut.io.keyIn.valid.poke(false.B)
        dut.io.out.ready.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        dut.io.done.expect(true.B)
        dut.io.error.expect(true.B)
        dut.io.busy.expect(false.B)
        dut.io.queryIn.ready.expect(false.B)
        dut.io.keyIn.ready.expect(false.B)
      }
    }
  }
}
