package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class SoftmaxVComputePipelineSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private val PackTokens = 16
  private val ScaleFractionalBits = 18
  private val LogitFractionalBits = 12
  private val ReciprocalFractionalBits = 32
  private val WeightFractionalBits = 15

  private def fixedWeights(
    logits: IndexedSeq[Long],
    featureDim: Int
  ): IndexedSeq[Int] = {
    val scale = Math.round(
      (1L << ScaleFractionalBits).toDouble / Math.sqrt(featureDim.toDouble)
    )
    val scaled = logits.map { value =>
      val product = BigInt(value) * scale
      val rounded =
        (product.abs + (BigInt(1) << (ScaleFractionalBits - 1))) >>
          ScaleFractionalBits
      if (product.signum < 0) -rounded.toLong else rounded.toLong
    }
    val maximum = scaled.max
    val expValues = scaled.map { value =>
      val delta = maximum - value
      val index = Math.min(
        128,
        ((delta + (1L << (LogitFractionalBits - 4 - 1))) >>
          (LogitFractionalBits - 4)).toInt
      )
      Math.round(Math.exp(-index.toDouble / 16.0) * (1L << 16))
    }
    val sum = expValues.sum
    val reciprocal = ((BigInt(1) << ReciprocalFractionalBits) + sum / 2) / sum
    expValues.map { value =>
      val normalized =
        (BigInt(value) * reciprocal +
          (BigInt(1) << (ReciprocalFractionalBits - WeightFractionalBits - 1))) >>
          (ReciprocalFractionalBits - WeightFractionalBits)
      normalized.min(BigInt(1) << WeightFractionalBits).toInt
    }
  }

  private def runCase(
    tokenCount: Int,
    featureDim: Int,
    vValues: IndexedSeq[Long],
    weights: IndexedSeq[Int],
    seed: Long
  ): Unit = {
    require(vValues.length == tokenCount * featureDim)
    require(weights.length == tokenCount)
    val packCount = (tokenCount + PackTokens - 1) / PackTokens
    val descriptorCount = packCount * featureDim
    val expected = IndexedSeq.tabulate(featureDim) { feature =>
      (0 until tokenCount).foldLeft(BigInt(0)) { (sum, token) =>
        sum + BigInt(weights(token)) * vValues(token * featureDim + feature)
      }
    }
    val random = new Random(seed)

    simulate(
      new SoftmaxVComputePipeline(
        maximumFeatureDim = 16,
        maximumTokens = 64
      )
    ) { dut =>
      dut.io.start.poke(false.B)
      dut.io.featureDim.poke(featureDim.U)
      dut.io.tokenCount.poke(tokenCount.U)
      dut.io.vIn.valid.poke(false.B)
      dut.io.vIn.bits.values.foreach(_.poke(0.S))
      dut.io.vIn.bits.validTokens.poke(0.U)
      dut.io.vIn.bits.descriptorIndex.poke(0.U)
      dut.io.vIn.bits.packIndex.poke(0.U)
      dut.io.vIn.bits.featureIndex.poke(0.U)
      dut.io.vIn.bits.blockIndex.poke(0.U)
      dut.io.vIn.bits.packWithinBlock.poke(0.U)
      dut.io.vIn.bits.last.poke(false.B)
      dut.io.weightIn.valid.poke(false.B)
      dut.io.weightIn.bits.weights.foreach(_.poke(0.U))
      dut.io.weightIn.bits.validTokens.poke(0.U)
      dut.io.weightIn.bits.packIndex.poke(0.U)
      dut.io.weightIn.bits.blockIndex.poke(0.U)
      dut.io.weightIn.bits.packWithinBlock.poke(0.U)
      dut.io.weightIn.bits.last.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.clock.step()
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var vDescriptor = 0
      var weightPack = 0
      var outputFeature = 0
      var cycles = 0
      while (outputFeature < featureDim && cycles < 10000) {
        val offerV = vDescriptor < descriptorCount && random.nextBoolean()
        val offerWeight = weightPack < packCount && random.nextBoolean()
        val acceptOutput = random.nextBoolean()
        val drivenDescriptor = Math.min(vDescriptor, descriptorCount - 1)
        val drivenVPack = drivenDescriptor / featureDim
        val drivenFeature = drivenDescriptor % featureDim
        val vValidTokens = Math.min(
          PackTokens,
          tokenCount - drivenVPack * PackTokens
        )
        dut.io.vIn.valid.poke(offerV.B)
        dut.io.vIn.bits.validTokens.poke(vValidTokens.U)
        dut.io.vIn.bits.descriptorIndex.poke(drivenDescriptor.U)
        dut.io.vIn.bits.packIndex.poke(drivenVPack.U)
        dut.io.vIn.bits.featureIndex.poke(drivenFeature.U)
        dut.io.vIn.bits.blockIndex.poke((drivenVPack / 4).U)
        dut.io.vIn.bits.packWithinBlock.poke((drivenVPack % 4).U)
        dut.io.vIn.bits.last.poke((drivenDescriptor == descriptorCount - 1).B)
        for (lane <- 0 until PackTokens) {
          val token = drivenVPack * PackTokens + lane
          val value =
            if (token < tokenCount) vValues(token * featureDim + drivenFeature)
            else 77777L
          dut.io.vIn.bits.values(lane).poke(value.S)
        }

        val drivenWeightPack = Math.min(weightPack, packCount - 1)
        val weightValidTokens = Math.min(
          PackTokens,
          tokenCount - drivenWeightPack * PackTokens
        )
        dut.io.weightIn.valid.poke(offerWeight.B)
        dut.io.weightIn.bits.validTokens.poke(weightValidTokens.U)
        dut.io.weightIn.bits.packIndex.poke(drivenWeightPack.U)
        dut.io.weightIn.bits.blockIndex.poke((drivenWeightPack / 4).U)
        dut.io.weightIn.bits.packWithinBlock.poke((drivenWeightPack % 4).U)
        dut.io.weightIn.bits.last.poke((drivenWeightPack == packCount - 1).B)
        for (lane <- 0 until PackTokens) {
          val token = drivenWeightPack * PackTokens + lane
          dut.io.weightIn.bits.weights(lane).poke(
            (if (token < tokenCount) weights(token) else 12345).U
          )
        }
        dut.io.out.ready.poke(acceptOutput.B)

        val vFire = offerV && dut.io.vIn.ready.peek().litToBoolean
        val weightFire = offerWeight && dut.io.weightIn.ready.peek().litToBoolean
        val outputFire = acceptOutput && dut.io.out.valid.peek().litToBoolean
        if (outputFire) {
          dut.io.out.bits.featureIndex.expect(outputFeature.U)
          dut.io.out.bits.value.expect(expected(outputFeature).S)
          dut.io.out.bits.last.expect((outputFeature == featureDim - 1).B)
        }

        dut.clock.step()
        if (vFire) vDescriptor += 1
        if (weightFire) weightPack += 1
        if (outputFire) outputFeature += 1
        cycles += 1
      }

      vDescriptor mustBe descriptorCount
      weightPack mustBe packCount
      outputFeature mustBe featureDim
      dut.io.error.expect(false.B)
      dut.io.stats.vBuffer.loadedPackets.expect(descriptorCount.U)
      dut.io.stats.accumulator.vReadRequests.expect(descriptorCount.U)
      dut.io.stats.accumulator.vReadResponses.expect(descriptorCount.U)
      dut.io.stats.accumulator.outputFeatures.expect(featureDim.U)
      dut.io.stats.accumulator.macOperations.expect((tokenCount * featureDim).U)
      dut.clock.step(2)
      dut.io.busy.expect(false.B)
    }
  }

  "Buffered Softmax-V compute pipeline" - {
    "must match Python-derived weights and V golden vectors under backpressure" in {
      val featureDim = 4
      val tokenCount = 64
      val logits = GoldenVectorLoader.int64LittleEndian(
        "directed_nonidentity",
        "expected_qk_logits_q12_i64.bin"
      )
      val weights = fixedWeights(logits, featureDim)
      val vValues = GoldenVectorLoader.float32LittleEndian(
        "directed_nonidentity",
        "expected_v_dequant_f32.bin"
      ).map(value => Math.round(value * 64.0f).toLong)

      runCase(
        tokenCount,
        featureDim,
        vValues,
        weights,
        0x4156474f4c44L
      )
    }

    "must ignore invalid lanes in a partial final pack" in {
      val tokenCount = 20
      val featureDim = 3
      val logits = IndexedSeq.tabulate(tokenCount) { index =>
        ((index % 9) - 4).toLong * 811L
      }
      val weights = fixedWeights(logits, featureDim)
      val vValues = IndexedSeq.tabulate(tokenCount * featureDim) { index =>
        ((index * 13) % 101 - 50).toLong
      }

      runCase(
        tokenCount,
        featureDim,
        vValues,
        weights,
        0x5041525449414cL
      )
    }

    "must reject unsupported geometry without consuming V or weights" in {
      simulate(
        new SoftmaxVComputePipeline(
          maximumFeatureDim = 8,
          maximumTokens = 64
        )
      ) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(9.U)
        dut.io.tokenCount.poke(16.U)
        dut.io.vIn.valid.poke(false.B)
        dut.io.vIn.bits.values.foreach(_.poke(0.S))
        dut.io.vIn.bits.validTokens.poke(0.U)
        dut.io.vIn.bits.descriptorIndex.poke(0.U)
        dut.io.vIn.bits.packIndex.poke(0.U)
        dut.io.vIn.bits.featureIndex.poke(0.U)
        dut.io.vIn.bits.blockIndex.poke(0.U)
        dut.io.vIn.bits.packWithinBlock.poke(0.U)
        dut.io.vIn.bits.last.poke(false.B)
        dut.io.weightIn.valid.poke(false.B)
        dut.io.weightIn.bits.weights.foreach(_.poke(0.U))
        dut.io.weightIn.bits.validTokens.poke(0.U)
        dut.io.weightIn.bits.packIndex.poke(0.U)
        dut.io.weightIn.bits.blockIndex.poke(0.U)
        dut.io.weightIn.bits.packWithinBlock.poke(0.U)
        dut.io.weightIn.bits.last.poke(false.B)
        dut.io.out.ready.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        dut.io.done.expect(true.B)
        dut.io.error.expect(true.B)
        dut.io.vIn.ready.expect(false.B)
        dut.io.weightIn.ready.expect(false.B)
      }
    }
  }
}
