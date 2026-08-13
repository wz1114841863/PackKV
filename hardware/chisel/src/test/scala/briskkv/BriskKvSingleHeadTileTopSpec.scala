package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class BriskKvSingleHeadTileTopSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private val FeatureDim = 4
  private val TokenCount = 64
  private val OutputMaximum = (1 << 17) - 1
  private val OutputMinimum = -(1 << 17)

  private def q12(quarterUnits: Int): Int = quarterUnits * 4096 / 4

  private val kRaw = IndexedSeq.tabulate(TokenCount) { token =>
    IndexedSeq(0, token % 4, (token / 4) % 4, 4).map(q12)
  }
  private val vRaw = IndexedSeq.tabulate(TokenCount) { token =>
    IndexedSeq(0, (token + 1) % 4, (token / 4 + 2) % 4, 4).map(q12)
  }

  private def quantizeQ21ToQ6(value: BigInt): Int = {
    val roundedMagnitude = (value.abs + (BigInt(1) << 14)) >> 15
    val rounded = if (value.signum < 0) -roundedMagnitude else roundedMagnitude
    rounded.max(OutputMinimum).min(OutputMaximum).toInt
  }

  private def fixedWeights(
    logits: IndexedSeq[Long],
    featureDim: Int
  ): IndexedSeq[Int] = {
    val scale = Math.round((1L << 18).toDouble / Math.sqrt(featureDim.toDouble))
    val scaled = logits.map { value =>
      val product = BigInt(value) * scale
      val rounded = (product.abs + (BigInt(1) << 17)) >> 18
      if (product.signum < 0) -rounded.toLong else rounded.toLong
    }
    val maximum = scaled.max
    val exponent = scaled.map { value =>
      val index = Math.min(128, ((maximum - value + 128) >> 8).toInt)
      Math.round(Math.exp(-index.toDouble / 16.0) * (1L << 16))
    }
    val sum = exponent.sum
    val reciprocal = ((BigInt(1) << 32) + sum / 2) / sum
    exponent.map { value =>
      (((BigInt(value) * reciprocal + (BigInt(1) << 16)) >> 17)
        .min(BigInt(1) << 15)).toInt
    }
  }

  "Single-head write-store-read-attention tile" - {
    "must match the fixed-point attention reference from raw K/V input" in {
      val query = IndexedSeq(8, -5, 3, 6)
      // Both directed tensors quantize and dequantize exactly at Q6 for their
      // selected power-of-two scales. A common token permutation therefore
      // cannot change the attention result.
      val kQ6 = kRaw.map(_.map(_ >> 6))
      val vQ6 = vRaw.map(_.map(_ >> 6))
      val logits = kQ6.map { token =>
        token.zip(query).map { case (k, q) => k.toLong * q }.sum
      }
      val weights = fixedWeights(logits, FeatureDim)
      val expected = IndexedSeq.tabulate(FeatureDim) { feature =>
        val q21 = (0 until TokenCount).foldLeft(BigInt(0)) { (sum, token) =>
          sum + BigInt(weights(token)) * vQ6(token)(feature)
        }
        quantizeQ21ToQ6(q21)
      }

      simulate(
        new BriskKvSingleHeadTileTop(
          maximumFeatureDim = 8,
          maximumTokens = TokenCount,
          enableStats = true
        )
      ) { dut =>
        dut.io.writeStart.poke(false.B)
        dut.io.featureDim.poke(FeatureDim.U)
        dut.io.blockCount.poke(1.U)
        dut.io.firstBlockIndex.poke(0.U)
        dut.io.writeIn.valid.poke(false.B)
        dut.io.attentionStart.poke(false.B)
        dut.io.attentionTag.poke(91.U)
        dut.io.queryIn.valid.poke(false.B)
        dut.io.queryIn.bits.poke(0.S)
        dut.io.bucketOut.ready.poke(true.B)
        dut.io.attentionOut.ready.poke(false.B)
        dut.io.result.ready.poke(false.B)
        dut.clock.step()

        dut.io.writeReady.expect(true.B)
        dut.io.writeStart.poke(true.B)
        dut.clock.step()
        dut.io.writeStart.poke(false.B)

        var pairIndex = 0
        var cycles = 0
        var writeDone = false
        while (!writeDone && cycles < 100000) {
          if (pairIndex < TokenCount * FeatureDim) {
            val token = pairIndex / FeatureDim
            val feature = pairIndex % FeatureDim
            dut.io.writeIn.valid.poke(true.B)
            dut.io.writeIn.bits.kFixedRaw.poke(kRaw(token)(feature).S)
            dut.io.writeIn.bits.vFixedRaw.poke(vRaw(token)(feature).S)
            dut.io.writeIn.bits.tokenTag.poke(token.U)
            dut.io.writeIn.bits.blockIndex.poke(0.U)
            dut.io.writeIn.bits.tokenIndex.poke(token.U)
            dut.io.writeIn.bits.featureIndex.poke(feature.U)
            dut.io.writeIn.bits.lastFeature.poke((feature == FeatureDim - 1).B)
            dut.io.writeIn.bits.last.poke(
              (token == TokenCount - 1 && feature == FeatureDim - 1).B
            )
            if (dut.io.writeIn.ready.peek().litToBoolean) pairIndex += 1
          } else {
            dut.io.writeIn.valid.poke(false.B)
          }
          if (dut.io.writeDone.peek().litToBoolean) writeDone = true
          dut.clock.step()
          cycles += 1
        }

        pairIndex mustBe TokenCount * FeatureDim
        writeDone mustBe true
        dut.io.error.expect(false.B)
        dut.io.encodedReady.expect(true.B)
        (0 until 11).foreach { stream =>
          dut.io.storedLengths(stream).peek().litValue must be > BigInt(0)
        }
        dut.io.writeStats.inputPairs.expect((TokenCount * FeatureDim).U)

        dut.io.attentionReady.expect(true.B)
        dut.io.attentionStart.poke(true.B)
        dut.clock.step()
        dut.io.attentionStart.poke(false.B)

        var queryIndex = 0
        var output = Vector.empty[Int]
        var bucketRecords = 0
        var resultSeen = false
        cycles = 0
        while (!resultSeen && cycles < 200000) {
          if (queryIndex < query.length) {
            dut.io.queryIn.valid.poke(true.B)
            dut.io.queryIn.bits.poke(query(queryIndex).S)
            if (dut.io.queryIn.ready.peek().litToBoolean) queryIndex += 1
          } else {
            dut.io.queryIn.valid.poke(false.B)
          }

          // Periodic compute-side backpressure also exercises SRAM replay.
          val acceptOutput = cycles % 5 != 2
          dut.io.attentionOut.ready.poke(acceptOutput.B)
          if (acceptOutput && dut.io.attentionOut.valid.peek().litToBoolean) {
            dut.io.attentionOut.bits.featureIndex.expect(output.length.U)
            output :+= dut.io.attentionOut.bits.value.peek().litValue.toInt
          }
          if (dut.io.bucketOut.valid.peek().litToBoolean) bucketRecords += 1

          if (dut.io.result.valid.peek().litToBoolean) {
            dut.io.result.bits.tag.expect(91.U)
            dut.io.result.bits.error.expect(false.B)
            dut.io.result.ready.poke(true.B)
            resultSeen = true
          }
          dut.clock.step()
          cycles += 1
        }

        queryIndex mustBe FeatureDim
        bucketRecords mustBe 1
        output mustBe expected
        resultSeen mustBe true
        dut.io.error.expect(false.B)
        // The encoded cache remains resident and can serve the next decode
        // query without rerunning quantization or bit-packing.
        dut.io.encodedReady.expect(true.B)
        dut.io.attentionReady.expect(true.B)
        dut.io.writeReady.expect(true.B)

        dut.io.attentionTag.poke(92.U)
        dut.io.attentionStart.poke(true.B)
        dut.io.result.ready.poke(false.B)
        dut.clock.step()
        dut.io.attentionStart.poke(false.B)

        queryIndex = 0
        output = Vector.empty[Int]
        resultSeen = false
        cycles = 0
        while (!resultSeen && cycles < 200000) {
          if (queryIndex < query.length) {
            dut.io.queryIn.valid.poke(true.B)
            dut.io.queryIn.bits.poke(query(queryIndex).S)
            if (dut.io.queryIn.ready.peek().litToBoolean) queryIndex += 1
          } else {
            dut.io.queryIn.valid.poke(false.B)
          }
          dut.io.attentionOut.ready.poke(true.B)
          if (dut.io.attentionOut.valid.peek().litToBoolean) {
            dut.io.attentionOut.bits.featureIndex.expect(output.length.U)
            output :+= dut.io.attentionOut.bits.value.peek().litValue.toInt
          }
          if (dut.io.result.valid.peek().litToBoolean) {
            dut.io.result.bits.tag.expect(92.U)
            dut.io.result.bits.error.expect(false.B)
            dut.io.result.ready.poke(true.B)
            resultSeen = true
          }
          dut.clock.step()
          cycles += 1
        }
        queryIndex mustBe FeatureDim
        output mustBe expected
        resultSeen mustBe true
        dut.io.encodedReady.expect(true.B)
      }
    }

    "must reject a non-zero first block index at the storage boundary" in {
      simulate(
        new BriskKvSingleHeadTileTop(
          maximumFeatureDim = 8,
          maximumTokens = TokenCount
        )
      ) { dut =>
        dut.io.writeStart.poke(false.B)
        dut.io.featureDim.poke(FeatureDim.U)
        dut.io.blockCount.poke(1.U)
        dut.io.firstBlockIndex.poke(1.U)
        dut.io.writeIn.valid.poke(false.B)
        dut.io.attentionStart.poke(false.B)
        dut.io.attentionTag.poke(0.U)
        dut.io.queryIn.valid.poke(false.B)
        dut.io.queryIn.bits.poke(0.S)
        dut.io.bucketOut.ready.poke(false.B)
        dut.io.attentionOut.ready.poke(false.B)
        dut.io.result.ready.poke(false.B)
        dut.clock.step()

        dut.io.writeStart.poke(true.B)
        dut.clock.step()
        dut.io.writeStart.poke(false.B)
        dut.io.writeDone.expect(true.B)
        dut.io.error.expect(true.B)
        dut.io.encodedReady.expect(false.B)
      }
    }
  }
}
