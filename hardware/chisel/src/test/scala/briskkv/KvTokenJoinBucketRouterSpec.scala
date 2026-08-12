package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class KvTokenJoinBucketRouterSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private val BlockTokens = 64

  private def driveIdle(dut: KvTokenJoinBucketRouter): Unit = {
    dut.io.start.poke(false.B)
    dut.io.kMetadataIn.valid.poke(false.B)
    dut.io.vMetadataIn.valid.poke(false.B)
    dut.io.kQIn.valid.poke(false.B)
    dut.io.vQIn.valid.poke(false.B)
    dut.io.bucketCountsOut.ready.poke(false.B)
    dut.io.metadataOut.ready.poke(false.B)
    dut.io.qOut.ready.poke(false.B)
  }

  "K/V token join and stable k_sum bucket router" - {
    "must preserve K/V/metadata association under independent backpressure" in {
      val featureDim = 3
      val scores = IndexedSeq.tabulate(BlockTokens)(token => (token * 17) % 64)
      val expectedOrder = (0 until 4).flatMap { bucket =>
        (0 until BlockTokens).filter { token => scores(token) / 16 == bucket }
      }
      val random = new Random(0x42524f555445L)

      simulate(
        new KvTokenJoinBucketRouter(maximumFeatureDim = 8)
      ) { dut =>
        driveIdle(dut)
        dut.io.featureDim.poke(featureDim.U)
        dut.io.blockIndex.poke(7.U)
        dut.io.blockLast.poke(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var inputToken = 0
        var inputFeature = -1
        var headerCount = 0
        var metadataIndex = 0
        var outputFeature = 0
        var cycles = 0
        var done = false
        while (!done && cycles < 50000) {
          val sendingMetadata = inputToken < BlockTokens && inputFeature < 0
          val sendingValue = inputToken < BlockTokens && inputFeature >= 0
          val offerKMetadata = sendingMetadata && random.nextBoolean()
          val offerVMetadata = sendingMetadata && random.nextBoolean()
          val offerKValue = sendingValue && random.nextBoolean()
          val offerVValue = sendingValue && random.nextBoolean()
          val drivenToken = math.min(inputToken, BlockTokens - 1)
          val tag = 1000 + drivenToken
          val feature = math.max(inputFeature, 0)

          dut.io.kMetadataIn.valid.poke(offerKMetadata.B)
          dut.io.vMetadataIn.valid.poke(offerVMetadata.B)
          dut.io.kMetadataIn.bits.zeroPoint.poke((-(drivenToken % 16)).S)
          dut.io.kMetadataIn.bits.exponent.poke(((drivenToken % 5) - 4).S)
          dut.io.kMetadataIn.bits.tokenTag.poke(tag.U)
          dut.io.vMetadataIn.bits.zeroPoint.poke((-(drivenToken % 8)).S)
          dut.io.vMetadataIn.bits.exponent.poke(((drivenToken % 4) - 3).S)
          dut.io.vMetadataIn.bits.tokenTag.poke(tag.U)

          dut.io.kQIn.valid.poke(offerKValue.B)
          dut.io.vQIn.valid.poke(offerVValue.B)
          dut.io.kQIn.bits.q.poke((if (feature == 0) scores(drivenToken) else 0).U)
          dut.io.vQIn.bits.q.poke(((drivenToken + feature) % 16).U)
          dut.io.kQIn.bits.tokenTag.poke(tag.U)
          dut.io.vQIn.bits.tokenTag.poke(tag.U)
          dut.io.kQIn.bits.featureIndex.poke(feature.U)
          dut.io.vQIn.bits.featureIndex.poke(feature.U)
          dut.io.kQIn.bits.last.poke((feature == featureDim - 1).B)
          dut.io.vQIn.bits.last.poke((feature == featureDim - 1).B)

          val acceptHeader = random.nextBoolean()
          val acceptMetadata = random.nextBoolean()
          val acceptQ = random.nextBoolean()
          dut.io.bucketCountsOut.ready.poke(acceptHeader.B)
          dut.io.metadataOut.ready.poke(acceptMetadata.B)
          dut.io.qOut.ready.poke(acceptQ.B)

          val kMetadataFire = offerKMetadata &&
            dut.io.kMetadataIn.ready.peek().litToBoolean
          val vMetadataFire = offerVMetadata &&
            dut.io.vMetadataIn.ready.peek().litToBoolean
          val kValueFire = offerKValue && dut.io.kQIn.ready.peek().litToBoolean
          val vValueFire = offerVValue && dut.io.vQIn.ready.peek().litToBoolean
          val headerFire = acceptHeader &&
            dut.io.bucketCountsOut.valid.peek().litToBoolean
          val metadataFire = acceptMetadata &&
            dut.io.metadataOut.valid.peek().litToBoolean
          val qFire = acceptQ && dut.io.qOut.valid.peek().litToBoolean

          kMetadataFire mustBe vMetadataFire
          kValueFire mustBe vValueFire
          if (headerFire) {
            for (bucket <- 0 until 4) {
              dut.io.bucketCountsOut.bits.counts(bucket).expect(16.U)
            }
            dut.io.bucketCountsOut.bits.blockIndex.expect(7.U)
            dut.io.bucketCountsOut.bits.last.expect(true.B)
          }
          if (metadataFire) {
            val original = expectedOrder(metadataIndex)
            val bucket = scores(original) / 16
            dut.io.metadataOut.bits.kZeroPoint.expect((-(original % 16)).S)
            dut.io.metadataOut.bits.kExponent.expect(((original % 5) - 4).S)
            dut.io.metadataOut.bits.vZeroPoint.expect((-(original % 8)).S)
            dut.io.metadataOut.bits.vExponent.expect(((original % 4) - 3).S)
            dut.io.metadataOut.bits.tokenTag.expect((1000 + original).U)
            dut.io.metadataOut.bits.originalTokenIndex.expect(original.U)
            dut.io.metadataOut.bits.routedTokenIndex.expect(metadataIndex.U)
            dut.io.metadataOut.bits.bucketId.expect(bucket.U)
            dut.io.metadataOut.bits.blockIndex.expect(7.U)
            dut.io.metadataOut.bits.last.expect((metadataIndex == 63).B)
          }
          if (qFire) {
            val original = expectedOrder(metadataIndex - 1)
            val bucket = scores(original) / 16
            dut.io.qOut.bits.kQ.expect(
              (if (outputFeature == 0) scores(original) else 0).U
            )
            dut.io.qOut.bits.vQ.expect(((original + outputFeature) % 16).U)
            dut.io.qOut.bits.tokenTag.expect((1000 + original).U)
            dut.io.qOut.bits.originalTokenIndex.expect(original.U)
            dut.io.qOut.bits.routedTokenIndex.expect((metadataIndex - 1).U)
            dut.io.qOut.bits.featureIndex.expect(outputFeature.U)
            dut.io.qOut.bits.bucketId.expect(bucket.U)
            dut.io.qOut.bits.lastFeature.expect((outputFeature == featureDim - 1).B)
            dut.io.qOut.bits.last.expect(
              (metadataIndex == 64 && outputFeature == featureDim - 1).B
            )
          }

          dut.clock.step()
          if (kMetadataFire) inputFeature = 0
          if (kValueFire) {
            if (inputFeature == featureDim - 1) {
              inputToken += 1
              inputFeature = -1
            } else inputFeature += 1
          }
          if (headerFire) headerCount += 1
          if (metadataFire) metadataIndex += 1
          if (qFire) {
            if (outputFeature == featureDim - 1) outputFeature = 0
            else outputFeature += 1
          }
          done = dut.io.done.peek().litToBoolean
          cycles += 1
        }

        done mustBe true
        inputToken mustBe BlockTokens
        headerCount mustBe 1
        metadataIndex mustBe BlockTokens
        outputFeature mustBe 0
        dut.io.error.expect(false.B)
        dut.io.stats.joinedTokens.expect(BlockTokens.U)
        dut.io.stats.joinedValues.expect((BlockTokens * featureDim).U)
        dut.io.stats.routedTokens.expect(BlockTokens.U)
        dut.io.stats.routedValues.expect((BlockTokens * featureDim).U)
        dut.io.stats.rejectedBlocks.expect(0.U)
        dut.clock.step()
        dut.io.busy.expect(false.B)
      }
    }

    "must reject a K/V token-tag mismatch before exposing a block" in {
      simulate(new KvTokenJoinBucketRouter(maximumFeatureDim = 2)) { dut =>
        driveIdle(dut)
        dut.io.featureDim.poke(1.U)
        dut.io.blockIndex.poke(0.U)
        dut.io.blockLast.poke(true.B)
        dut.io.bucketCountsOut.ready.poke(true.B)
        dut.io.metadataOut.ready.poke(true.B)
        dut.io.qOut.ready.poke(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        for (token <- 0 until BlockTokens) {
          dut.io.kMetadataIn.valid.poke(true.B)
          dut.io.vMetadataIn.valid.poke(true.B)
          dut.io.kMetadataIn.bits.zeroPoint.poke(0.S)
          dut.io.kMetadataIn.bits.exponent.poke(0.S)
          dut.io.kMetadataIn.bits.tokenTag.poke(token.U)
          dut.io.vMetadataIn.bits.zeroPoint.poke(0.S)
          dut.io.vMetadataIn.bits.exponent.poke(0.S)
          dut.io.vMetadataIn.bits.tokenTag.poke((token + (if (token == 0) 1 else 0)).U)
          while (!dut.io.kMetadataIn.ready.peek().litToBoolean) dut.clock.step()
          dut.clock.step()
          dut.io.kMetadataIn.valid.poke(false.B)
          dut.io.vMetadataIn.valid.poke(false.B)

          dut.io.kQIn.valid.poke(true.B)
          dut.io.vQIn.valid.poke(true.B)
          dut.io.kQIn.bits.q.poke((token % 64).U)
          dut.io.vQIn.bits.q.poke((token % 16).U)
          dut.io.kQIn.bits.tokenTag.poke(token.U)
          dut.io.vQIn.bits.tokenTag.poke(token.U)
          dut.io.kQIn.bits.featureIndex.poke(0.U)
          dut.io.vQIn.bits.featureIndex.poke(0.U)
          dut.io.kQIn.bits.last.poke(true.B)
          dut.io.vQIn.bits.last.poke(true.B)
          while (!dut.io.kQIn.ready.peek().litToBoolean) dut.clock.step()
          dut.clock.step()
          dut.io.kQIn.valid.poke(false.B)
          dut.io.vQIn.valid.poke(false.B)
        }

        var cycles = 0
        while (!dut.io.done.peek().litToBoolean && cycles < 200) {
          dut.io.bucketCountsOut.valid.expect(false.B)
          dut.io.metadataOut.valid.expect(false.B)
          dut.io.qOut.valid.expect(false.B)
          dut.clock.step()
          cycles += 1
        }
        dut.io.done.expect(true.B)
        dut.io.error.expect(true.B)
        dut.io.stats.rejectedBlocks.expect(1.U)
      }
    }
  }
}
