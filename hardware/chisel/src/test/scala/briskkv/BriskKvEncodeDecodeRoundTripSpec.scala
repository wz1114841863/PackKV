package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class BriskKvEncodeDecodeRoundTripSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private val BlockTokens = 64
  private val BlockCount = 2
  private val TokenCount = BlockTokens * BlockCount
  private val FeatureDim = 4
  private val PackTokens = 16
  private val FractionalBits = 12

  private def rawQuarter(quarters: Int): Int =
    quarters * (1 << FractionalBits) / 4

  private val kRaw = IndexedSeq.tabulate(TokenCount) { token =>
    IndexedSeq(
      rawQuarter(0),
      rawQuarter(token % 4),
      rawQuarter((token / 4) % 4),
      rawQuarter(4)
    )
  }
  private val vRaw = IndexedSeq.tabulate(TokenCount) { token =>
    IndexedSeq(
      rawQuarter(0),
      rawQuarter((token + 1) % 4),
      rawQuarter((token / 4 + 2) % 4),
      rawQuarter(4)
    )
  }

  // K range=1 selects 2^-5, V range=1 selects 2^-3. All quarter-point
  // inputs are exactly representable, so reconstructed Q6 values are exact.
  private val kQ = kRaw.map(_.map(value => value >> 7))
  private val vQ = vRaw.map(_.map(value => value >> 9))
  private val kScores = kQ.map(_.sum)
  private val scoreMinimum = kScores.min
  private val scoreMaximum = kScores.max
  private val scoreSpan = scoreMaximum - scoreMinimum + 1
  private val thresholds = (1 to 3).map { boundary =>
    scoreMinimum + (scoreSpan * boundary + 3) / 4
  }
  private val bucketIds = kScores.map(score => thresholds.count(score >= _))
  private val routedTokens = (0 until BlockCount).flatMap { block =>
    (0 until 4).flatMap { bucket =>
      val first = block * BlockTokens
      (first until first + BlockTokens).filter(token => bucketIds(token) == bucket)
    }
  }.toIndexedSeq
  private val expectedBucketCounts = (0 until BlockCount).map { block =>
    val first = block * BlockTokens
    (0 until 4).map { bucket =>
      bucketIds.slice(first, first + BlockTokens).count(_ == bucket)
    }.toIndexedSeq
  }.toIndexedSeq

  private def outputSignals(dut: BriskKvWriteEncoderTop): IndexedSeq[(Bool, UInt)] =
    IndexedSeq(
      dut.io.kMinimumOut.valid -> dut.io.kMinimumOut.bits,
      dut.io.kWidthOut.valid -> dut.io.kWidthOut.bits,
      dut.io.kPayloadOut.valid -> dut.io.kPayloadOut.bits,
      dut.io.vMinimumOut.valid -> dut.io.vMinimumOut.bits,
      dut.io.vWidthOut.valid -> dut.io.vWidthOut.bits,
      dut.io.vPayloadOut.valid -> dut.io.vPayloadOut.bits,
      dut.io.kZeroOut.valid -> dut.io.kZeroOut.bits,
      dut.io.kExponentOut.valid -> dut.io.kExponentOut.bits,
      dut.io.vZeroOut.valid -> dut.io.vZeroOut.bits,
      dut.io.vExponentOut.valid -> dut.io.vExponentOut.bits,
      dut.io.bucketCountOut.valid -> dut.io.bucketCountOut.bits
    )

  "Real write encoder to decompressor round trip" - {
    for (
      parameterArchitecture <- IndexedSeq(
        QuantParameterArchitecture.V1SingleStage,
        QuantParameterArchitecture.V3LeadingOne
      )
    ) {
      s"${parameterArchitecture.cliName} must preserve routed K/V values and bucket metadata" in {
      val encoded = IndexedSeq.fill(11)(ArrayBuffer.empty[Int])
      val encodeRandom = new Random(0x454e435254L)

      simulate(
        new BriskKvWriteEncoderTop(
          maximumFeatureDim = 8,
          enableStats = false,
          quantParameterArchitecture = parameterArchitecture
        )
      ) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(FeatureDim.U)
        dut.io.blockCount.poke(BlockCount.U)
        dut.io.firstBlockIndex.poke(0.U)
        dut.io.in.valid.poke(false.B)
        dut.io.kMinimumOut.ready.poke(false.B)
        dut.io.kWidthOut.ready.poke(false.B)
        dut.io.kPayloadOut.ready.poke(false.B)
        dut.io.vMinimumOut.ready.poke(false.B)
        dut.io.vWidthOut.ready.poke(false.B)
        dut.io.vPayloadOut.ready.poke(false.B)
        dut.io.kZeroOut.ready.poke(false.B)
        dut.io.kExponentOut.ready.poke(false.B)
        dut.io.vZeroOut.ready.poke(false.B)
        dut.io.vExponentOut.ready.poke(false.B)
        dut.io.bucketCountOut.ready.poke(false.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var token = 0
        var feature = 0
        var done = false
        var cycles = 0
        while (!done && cycles < 150000) {
          val offer = token < TokenCount && encodeRandom.nextBoolean()
          val drivenToken = math.min(token, TokenCount - 1)
          dut.io.in.valid.poke(offer.B)
          dut.io.in.bits.kFixedRaw.poke(kRaw(drivenToken)(feature).S)
          dut.io.in.bits.vFixedRaw.poke(vRaw(drivenToken)(feature).S)
          dut.io.in.bits.tokenTag.poke((2000 + drivenToken).U)
          dut.io.in.bits.blockIndex.poke((drivenToken / BlockTokens).U)
          dut.io.in.bits.tokenIndex.poke((drivenToken % BlockTokens).U)
          dut.io.in.bits.featureIndex.poke(feature.U)
          dut.io.in.bits.lastFeature.poke((feature == FeatureDim - 1).B)
          dut.io.in.bits.last.poke(
            (drivenToken == TokenCount - 1 && feature == FeatureDim - 1).B
          )

          val accepts = IndexedSeq.fill(11)(encodeRandom.nextBoolean())
          dut.io.kMinimumOut.ready.poke(accepts(0).B)
          dut.io.kWidthOut.ready.poke(accepts(1).B)
          dut.io.kPayloadOut.ready.poke(accepts(2).B)
          dut.io.vMinimumOut.ready.poke(accepts(3).B)
          dut.io.vWidthOut.ready.poke(accepts(4).B)
          dut.io.vPayloadOut.ready.poke(accepts(5).B)
          dut.io.kZeroOut.ready.poke(accepts(6).B)
          dut.io.kExponentOut.ready.poke(accepts(7).B)
          dut.io.vZeroOut.ready.poke(accepts(8).B)
          dut.io.vExponentOut.ready.poke(accepts(9).B)
          dut.io.bucketCountOut.ready.poke(accepts(10).B)

          val inputFire = offer && dut.io.in.ready.peek().litToBoolean
          val outputs = outputSignals(dut)
          val outputFires = outputs.indices.map { index =>
            accepts(index) && outputs(index)._1.peek().litToBoolean
          }
          val outputValues = outputs.map(_._2.peek().litValue.toInt)
          dut.clock.step()

          if (inputFire) {
            if (feature == FeatureDim - 1) {
              feature = 0
              token += 1
            } else feature += 1
          }
          outputFires.indices.foreach { index =>
            if (outputFires(index)) encoded(index) += outputValues(index)
          }
          done = dut.io.done.peek().litToBoolean
          cycles += 1
        }

        done mustBe true
        token mustBe TokenCount
        dut.io.error.expect(false.B)
      }

      encoded(2).nonEmpty mustBe true
      encoded(5).nonEmpty mustBe true
      encoded(10).length mustBe 3 * BlockCount

      val decodeRandom = new Random(0x4445435254L)
      simulate(
        new DualKvDecompressionController(enableStats = false)
      ) { dut =>
        dut.io.command.valid.poke(false.B)
        dut.io.command.bits.tag.poke(0.U)
        dut.io.command.bits.tokenCount.poke(0.U)
        dut.io.command.bits.featureDim.poke(0.U)
        dut.io.command.bits.descriptorCount.poke(0.U)
        dut.io.command.bits.kPayloadByteCount.poke(0.U)
        dut.io.command.bits.vPayloadByteCount.poke(0.U)
        val inputValids = IndexedSeq(
          dut.io.kMinimumIn.valid,
          dut.io.kWidthIn.valid,
          dut.io.kPayloadIn.valid,
          dut.io.vMinimumIn.valid,
          dut.io.vWidthIn.valid,
          dut.io.vPayloadIn.valid,
          dut.io.kZeroPointIn.valid,
          dut.io.kExponentIn.valid,
          dut.io.vZeroPointIn.valid,
          dut.io.vExponentIn.valid,
          dut.io.bucketCountIn.valid
        )
        val inputBits = IndexedSeq(
          dut.io.kMinimumIn.bits,
          dut.io.kWidthIn.bits,
          dut.io.kPayloadIn.bits,
          dut.io.vMinimumIn.bits,
          dut.io.vWidthIn.bits,
          dut.io.vPayloadIn.bits,
          dut.io.kZeroPointIn.bits,
          dut.io.kExponentIn.bits,
          dut.io.vZeroPointIn.bits,
          dut.io.vExponentIn.bits,
          dut.io.bucketCountIn.bits
        )
        val inputReadies = IndexedSeq(
          dut.io.kMinimumIn.ready,
          dut.io.kWidthIn.ready,
          dut.io.kPayloadIn.ready,
          dut.io.vMinimumIn.ready,
          dut.io.vWidthIn.ready,
          dut.io.vPayloadIn.ready,
          dut.io.kZeroPointIn.ready,
          dut.io.kExponentIn.ready,
          dut.io.vZeroPointIn.ready,
          dut.io.vExponentIn.ready,
          dut.io.bucketCountIn.ready
        )
        inputValids.foreach(_.poke(false.B))
        inputBits.foreach(_.poke(0.U))
        dut.io.kOut.ready.poke(false.B)
        dut.io.vOut.ready.poke(false.B)
        dut.io.bucketOut.ready.poke(false.B)
        dut.io.result.ready.poke(false.B)
        dut.clock.step()

        val descriptorCount = TokenCount / PackTokens * FeatureDim
        dut.io.command.bits.tag.poke(77.U)
        dut.io.command.bits.tokenCount.poke(TokenCount.U)
        dut.io.command.bits.featureDim.poke(FeatureDim.U)
        dut.io.command.bits.descriptorCount.poke(descriptorCount.U)
        dut.io.command.bits.kPayloadByteCount.poke(encoded(2).length.U)
        dut.io.command.bits.vPayloadByteCount.poke(encoded(5).length.U)
        dut.io.command.valid.poke(true.B)
        dut.io.command.ready.expect(true.B)
        dut.clock.step()
        dut.io.command.valid.poke(false.B)

        val indices = Array.fill(11)(0)
        var kOutputs = 0
        var vOutputs = 0
        var bucketOutputs = 0
        var resultSeen = false
        var cycles = 0
        while (!resultSeen && cycles < 200000) {
          val offers = encoded.indices.map { index =>
            indices(index) < encoded(index).length && decodeRandom.nextBoolean()
          }
          encoded.indices.foreach { index =>
            inputValids(index).poke(offers(index).B)
            inputBits(index).poke(
              (if (indices(index) < encoded(index).length)
                 encoded(index)(indices(index))
               else 0).U
            )
          }
          val acceptK = decodeRandom.nextBoolean()
          val acceptV = decodeRandom.nextBoolean()
          val acceptBucket = decodeRandom.nextBoolean()
          dut.io.kOut.ready.poke(acceptK.B)
          dut.io.vOut.ready.poke(acceptV.B)
          dut.io.bucketOut.ready.poke(acceptBucket.B)
          dut.io.result.ready.poke(false.B)

          val inputFires = encoded.indices.map { index =>
            offers(index) && inputReadies(index).peek().litToBoolean
          }
          val kFire = acceptK && dut.io.kOut.valid.peek().litToBoolean
          val vFire = acceptV && dut.io.vOut.valid.peek().litToBoolean
          val bucketFire = acceptBucket && dut.io.bucketOut.valid.peek().litToBoolean

          def checkValue(isKey: Boolean): Unit = {
            val bits = if (isKey) dut.io.kOut.bits else dut.io.vOut.bits
            val descriptor = bits.descriptorIndex.peek().litValue.toInt
            val lane = bits.tokenIndex.peek().litValue.toInt
            val pack = descriptor / FeatureDim
            val feature = descriptor % FeatureDim
            val routedToken = pack * PackTokens + lane
            val originalToken = routedTokens(routedToken)
            val source = if (isKey) kRaw else vRaw
            val expectedQ6 = source(originalToken)(feature) >> 6
            bits.fixedRaw.expect(expectedQ6.S(18.W))
            bits.last.expect(
              (descriptor == descriptorCount - 1 && lane == PackTokens - 1).B
            )
          }

          if (kFire) checkValue(isKey = true)
          if (vFire) checkValue(isKey = false)
          if (bucketFire) {
            bucketOutputs must be < BlockCount
            for (bucket <- 0 until 4) {
              dut.io.bucketOut.bits.counts(bucket).expect(
                expectedBucketCounts(bucketOutputs)(bucket).U
              )
            }
            dut.io.bucketOut.bits.blockIndex.expect(bucketOutputs.U)
            dut.io.bucketOut.bits.last.expect(
              (bucketOutputs == BlockCount - 1).B
            )
          }

          dut.clock.step()
          encoded.indices.foreach { index =>
            if (inputFires(index)) indices(index) += 1
          }
          if (kFire) kOutputs += 1
          if (vFire) vOutputs += 1
          if (bucketFire) bucketOutputs += 1

          if (dut.io.result.valid.peek().litToBoolean) {
            dut.io.result.bits.tag.expect(77.U)
            dut.io.result.bits.error.expect(false.B)
            dut.io.result.bits.tokenCount.expect(TokenCount.U)
            dut.io.result.bits.packCount.expect((TokenCount / PackTokens).U)
            dut.io.result.bits.blockCount.expect(BlockCount.U)
            dut.io.result.bits.descriptorCount.expect(descriptorCount.U)
            dut.io.result.bits.bucketRecords.expect(BlockCount.U)
            dut.io.result.ready.poke(true.B)
            dut.clock.step()
            resultSeen = true
          }
          cycles += 1
        }

        resultSeen mustBe true
        kOutputs mustBe TokenCount * FeatureDim
        vOutputs mustBe TokenCount * FeatureDim
        bucketOutputs mustBe BlockCount
        encoded.indices.foreach { index =>
          indices(index) mustBe encoded(index).length
        }
      }
    }
    }
  }
}
