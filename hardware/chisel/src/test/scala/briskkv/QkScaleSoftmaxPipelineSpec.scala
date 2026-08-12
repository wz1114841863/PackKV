package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class QkScaleSoftmaxPipelineSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private val PackTokens = 16
  private val LogitFractionalBits = 12
  private val ScaleFractionalBits = 18
  private val ExponentFractionalBits = 16
  private val ReciprocalFractionalBits = 32
  private val WeightFractionalBits = 15

  private def roundedScale(value: Long, featureDim: Int): Long = {
    val multiplier = Math.round(
      (1L << ScaleFractionalBits).toDouble / Math.sqrt(featureDim.toDouble)
    )
    val product = BigInt(value) * multiplier
    val rounded =
      (product.abs + (BigInt(1) << (ScaleFractionalBits - 1))) >>
        ScaleFractionalBits
    if (product.signum < 0) -rounded.toLong else rounded.toLong
  }

  private def fixedSoftmax(scaled: IndexedSeq[Long]): IndexedSeq[Int] = {
    val maximum = scaled.max
    val expValues = scaled.map { value =>
      val delta = maximum - value
      val index = Math.min(
        8 << 4,
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

  private def initializeScale(dut: AttentionScaleUnit): Unit = {
    dut.io.start.poke(false.B)
    dut.io.featureDim.poke(0.U)
    dut.io.tokenCount.poke(0.U)
    dut.io.in.valid.poke(false.B)
    dut.io.in.bits.logits.foreach(_.poke(0.S))
    dut.io.in.bits.validTokens.poke(0.U)
    dut.io.in.bits.packIndex.poke(0.U)
    dut.io.in.bits.blockIndex.poke(0.U)
    dut.io.in.bits.packWithinBlock.poke(0.U)
    dut.io.in.bits.last.poke(false.B)
    dut.io.out.ready.poke(false.B)
  }

  "Fixed-point attention scaler" - {
    "must apply Q18 reciprocal-square-root scaling with symmetric rounding" in {
      val logits = IndexedSeq.tabulate(20)(index => (index - 10).toLong * 513L)
      val expected = logits.map(roundedScale(_, 4))
      val random = new Random(0x5343414c45L)

      simulate(new AttentionScaleUnit(maximumTokens = 64)) { dut =>
        initializeScale(dut)
        dut.clock.step()
        dut.io.featureDim.poke(4.U)
        dut.io.tokenCount.poke(20.U)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        dut.io.scaleMultiplier.expect((1 << 17).U)

        var inputPack = 0
        var outputPack = 0
        var cycles = 0
        while (outputPack < 2 && cycles < 200) {
          val offer = inputPack < 2 && random.nextBoolean()
          val accept = random.nextBoolean()
          val validTokens = if (inputPack == 1) 4 else 16
          dut.io.in.valid.poke(offer.B)
          dut.io.in.bits.validTokens.poke(validTokens.U)
          dut.io.in.bits.packIndex.poke(inputPack.U)
          dut.io.in.bits.blockIndex.poke(0.U)
          dut.io.in.bits.packWithinBlock.poke(inputPack.U)
          dut.io.in.bits.last.poke((inputPack == 1).B)
          for (lane <- 0 until PackTokens) {
            val index = inputPack * PackTokens + lane
            dut.io.in.bits.logits(lane).poke(
              (if (index < logits.length) logits(index) else 999999L).S
            )
          }
          dut.io.out.ready.poke(accept.B)

          val inputFire = offer && dut.io.in.ready.peek().litToBoolean
          val outputFire = accept && dut.io.out.valid.peek().litToBoolean
          if (outputFire) {
            val outputValid = if (outputPack == 1) 4 else 16
            dut.io.out.bits.validTokens.expect(outputValid.U)
            dut.io.out.bits.packIndex.expect(outputPack.U)
            dut.io.out.bits.last.expect((outputPack == 1).B)
            for (lane <- 0 until PackTokens) {
              val index = outputPack * PackTokens + lane
              val value = if (index < expected.length) expected(index) else 0L
              dut.io.out.bits.logits(lane).expect(value.S)
            }
          }
          dut.clock.step()
          if (inputFire) inputPack += 1
          if (outputFire) outputPack += 1
          cycles += 1
        }

        outputPack mustBe 2
        dut.io.error.expect(false.B)
        dut.io.stats.inputPackets.expect(2.U)
        dut.io.stats.outputPackets.expect(2.U)
      }
    }

    "must reject unsupported geometry without accepting a packet" in {
      simulate(new AttentionScaleUnit(maximumFeatureDim = 8, maximumTokens = 64)) {
        dut =>
          initializeScale(dut)
          dut.clock.step()
          dut.io.featureDim.poke(9.U)
          dut.io.tokenCount.poke(16.U)
          dut.io.start.poke(true.B)
          dut.clock.step()
          dut.io.start.poke(false.B)
          dut.io.done.expect(true.B)
          dut.io.error.expect(true.B)
          dut.io.in.ready.expect(false.B)
      }
    }

    "must preserve signed Q12 logits at the d=1 scale boundary" in {
      val logits = IndexedSeq.tabulate(PackTokens) { lane =>
        if ((lane & 1) == 0) (lane + 1).toLong * 4097L
        else -(lane + 1).toLong * 4097L
      }
      simulate(new AttentionScaleUnit(maximumFeatureDim = 8, maximumTokens = 64)) {
        dut =>
          initializeScale(dut)
          dut.clock.step()
          dut.io.featureDim.poke(1.U)
          dut.io.tokenCount.poke(PackTokens.U)
          dut.io.start.poke(true.B)
          dut.clock.step()
          dut.io.start.poke(false.B)
          dut.io.scaleMultiplier.expect((1 << ScaleFractionalBits).U)

          dut.io.in.valid.poke(true.B)
          dut.io.in.bits.validTokens.poke(PackTokens.U)
          dut.io.in.bits.packIndex.poke(0.U)
          dut.io.in.bits.blockIndex.poke(0.U)
          dut.io.in.bits.packWithinBlock.poke(0.U)
          dut.io.in.bits.last.poke(true.B)
          logits.zip(dut.io.in.bits.logits).foreach { case (value, port) =>
            port.poke(value.S)
          }
          dut.io.out.ready.poke(false.B)
          dut.io.in.ready.expect(true.B)
          dut.clock.step()
          dut.io.in.valid.poke(false.B)

          for (_ <- 0 until 4) {
            dut.io.in.ready.expect(false.B)
            dut.io.out.valid.expect(false.B)
            dut.clock.step()
          }
          dut.io.out.valid.expect(true.B)
          logits.zip(dut.io.out.bits.logits).foreach { case (value, port) =>
            port.expect(value.S)
          }
          dut.io.error.expect(false.B)
          dut.io.out.ready.poke(true.B)
          dut.clock.step()
          dut.io.done.expect(true.B)
        }
    }

    "must keep function while tying counters to zero when stats are disabled" in {
      simulate(
        new AttentionScaleUnit(
          maximumFeatureDim = 8,
          maximumTokens = 64,
          enableStats = false
        )
      ) { dut =>
        initializeScale(dut)
        dut.clock.step()
        dut.io.featureDim.poke(4.U)
        dut.io.tokenCount.poke(PackTokens.U)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        dut.io.in.valid.poke(true.B)
        dut.io.in.bits.validTokens.poke(PackTokens.U)
        dut.io.in.bits.packIndex.poke(0.U)
        dut.io.in.bits.blockIndex.poke(0.U)
        dut.io.in.bits.packWithinBlock.poke(0.U)
        dut.io.in.bits.last.poke(true.B)
        dut.io.in.bits.logits.foreach(_.poke(8192.S))
        dut.clock.step()
        dut.io.in.valid.poke(false.B)
        dut.clock.step(4)

        dut.io.out.valid.expect(true.B)
        dut.io.out.bits.logits.foreach(_.expect(4096.S))
        dut.io.stats.activeCycles.expect(0.U)
        dut.io.stats.inputPackets.expect(0.U)
        dut.io.stats.outputPackets.expect(0.U)
        dut.io.stats.downstreamStallCycles.expect(0.U)
        dut.io.out.ready.poke(true.B)
        dut.clock.step()
        dut.io.done.expect(true.B)
      }
    }
  }

  "Streaming fixed-point softmax" - {
    "must normalize a partial final pack under independent backpressure" in {
      val logits = IndexedSeq.tabulate(20)(index => ((index % 7) - 3).toLong * 701L)
      val expected = fixedSoftmax(logits)
      val random = new Random(0x534f4654L)

      simulate(new StreamingSoftmax(maximumTokens = 64)) { dut =>
        dut.io.start.poke(false.B)
        dut.io.tokenCount.poke(20.U)
        dut.io.in.valid.poke(false.B)
        dut.io.in.bits.logits.foreach(_.poke(0.S))
        dut.io.in.bits.validTokens.poke(0.U)
        dut.io.in.bits.packIndex.poke(0.U)
        dut.io.in.bits.blockIndex.poke(0.U)
        dut.io.in.bits.packWithinBlock.poke(0.U)
        dut.io.in.bits.last.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var inputPack = 0
        var outputPack = 0
        var cycles = 0
        val observed = ArrayBuffer.empty[Int]
        while (outputPack < 2 && cycles < 500) {
          val offer = inputPack < 2 && random.nextBoolean()
          val accept = random.nextBoolean()
          val validTokens = if (inputPack == 1) 4 else 16
          dut.io.in.valid.poke(offer.B)
          dut.io.in.bits.validTokens.poke(validTokens.U)
          dut.io.in.bits.packIndex.poke(inputPack.U)
          dut.io.in.bits.blockIndex.poke(0.U)
          dut.io.in.bits.packWithinBlock.poke(inputPack.U)
          dut.io.in.bits.last.poke((inputPack == 1).B)
          for (lane <- 0 until PackTokens) {
            val index = inputPack * PackTokens + lane
            dut.io.in.bits.logits(lane).poke(
              (if (index < logits.length) logits(index) else 123456L).S
            )
          }
          dut.io.out.ready.poke(accept.B)

          val inputFire = offer && dut.io.in.ready.peek().litToBoolean
          val outputFire = accept && dut.io.out.valid.peek().litToBoolean
          if (outputFire) {
            val valid = if (outputPack == 1) 4 else 16
            dut.io.out.bits.validTokens.expect(valid.U)
            dut.io.out.bits.packIndex.expect(outputPack.U)
            dut.io.out.bits.last.expect((outputPack == 1).B)
            for (lane <- 0 until PackTokens) {
              val index = outputPack * PackTokens + lane
              val value = dut.io.out.bits.weights(lane).peek().litValue.toInt
              if (lane < valid) {
                value mustBe expected(index)
                observed += value
              } else value mustBe 0
            }
          }
          dut.clock.step()
          if (inputFire) inputPack += 1
          if (outputFire) outputPack += 1
          cycles += 1
        }

        outputPack mustBe 2
        observed.sum must be >= 32760
        observed.sum must be <= 32776
        dut.io.error.expect(false.B)
        dut.io.stats.inputPackets.expect(2.U)
        dut.io.stats.exponentPackets.expect(2.U)
        dut.io.stats.outputPackets.expect(2.U)
      }
    }

  }

  "QK scale-softmax pipeline" - {
    "must match the fixed-point model for Python golden QK logits" in {
      val logits = GoldenVectorLoader.int64LittleEndian(
        "directed_nonidentity",
        "expected_qk_logits_q12_i64.bin"
      )
      val scaled = logits.map(roundedScale(_, 4))
      val expected = fixedSoftmax(scaled)
      val random = new Random(0x51534d58L)

      simulate(new QkScaleSoftmaxPipeline(maximumTokens = 64)) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(4.U)
        dut.io.tokenCount.poke(64.U)
        dut.io.in.valid.poke(false.B)
        dut.io.in.bits.logits.foreach(_.poke(0.S))
        dut.io.in.bits.validTokens.poke(16.U)
        dut.io.in.bits.packIndex.poke(0.U)
        dut.io.in.bits.blockIndex.poke(0.U)
        dut.io.in.bits.packWithinBlock.poke(0.U)
        dut.io.in.bits.last.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var inputPack = 0
        var outputPack = 0
        var cycles = 0
        while (outputPack < 4 && cycles < 1000) {
          val offer = inputPack < 4 && random.nextBoolean()
          val accept = random.nextBoolean()
          val drivenPack = Math.min(inputPack, 3)
          dut.io.in.valid.poke(offer.B)
          dut.io.in.bits.validTokens.poke(16.U)
          dut.io.in.bits.packIndex.poke(drivenPack.U)
          dut.io.in.bits.blockIndex.poke(0.U)
          dut.io.in.bits.packWithinBlock.poke(drivenPack.U)
          dut.io.in.bits.last.poke((drivenPack == 3).B)
          for (lane <- 0 until PackTokens) {
            val index = drivenPack * PackTokens + lane
            dut.io.in.bits.logits(lane).poke(
              logits(index).S
            )
          }
          dut.io.out.ready.poke(accept.B)

          val inputFire = offer && dut.io.in.ready.peek().litToBoolean
          val outputFire = accept && dut.io.out.valid.peek().litToBoolean
          if (outputFire) {
            for (lane <- 0 until PackTokens) {
              dut.io.out.bits.weights(lane).expect(
                expected(outputPack * PackTokens + lane).U
              )
            }
          }
          dut.clock.step()
          if (inputFire) inputPack += 1
          if (outputFire) outputPack += 1
          cycles += 1
        }

        outputPack mustBe 4
        dut.io.error.expect(false.B)
        dut.io.scaleMultiplier.expect((1 << 17).U)
        dut.io.stats.scaling.inputPackets.expect(4.U)
        dut.io.stats.softmax.outputPackets.expect(4.U)
      }
    }
  }
}
