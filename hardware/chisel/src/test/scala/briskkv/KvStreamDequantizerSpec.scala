package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class KvStreamDequantizerSpec extends AnyFreeSpec with Matchers with ChiselSim {
  private case class VectorConfig(
    codeValueBits: Int,
    zeroPointBits: Int,
    encodeLengthBits: Int,
    featureDim: Int,
    packLen: Int,
    tokenCount: Int,
    paddedTokenCount: Int
  ) {
    val descriptorCount: Int =
      ((tokenCount + paddedTokenCount) / packLen) * featureDim
  }

  private def config(caseName: String, cache: String): VectorConfig =
    VectorConfig(
      codeValueBits = GoldenVectorLoader.bitpackInt(caseName, cache, "code_value_bits"),
      zeroPointBits = if (cache == "k") 7 else 5,
      encodeLengthBits = GoldenVectorLoader.bitpackInt(
        caseName,
        cache,
        "encode_length_field_bits"
      ),
      featureDim = GoldenVectorLoader.bitpackInt(caseName, cache, "feature_dim"),
      packLen = GoldenVectorLoader.bitpackInt(caseName, cache, "pack_len"),
      tokenCount = GoldenVectorLoader.bitpackInt(caseName, cache, "token_count"),
      paddedTokenCount = GoldenVectorLoader.bitpackInt(
        caseName,
        cache,
        "padded_token_count"
      )
    )

  private def runGolden(
    caseName: String,
    cache: String,
    randomBackpressure: Boolean
  ): Unit = {
    val cfg = config(caseName, cache)
    val minimumBytes = GoldenVectorLoader.unsigned(caseName, s"${cache}_pack_mins.bin")
    val widthBytes = GoldenVectorLoader.unsigned(caseName, s"${cache}_encode_lengths.bin")
    val payloadBytes = GoldenVectorLoader.unsigned(caseName, s"${cache}_payload.bin")
    val zeroBytes = GoldenVectorLoader.unsigned(caseName, s"${cache}_zero_points.bin")
    val exponentBytes = GoldenVectorLoader.unsigned(caseName, s"${cache}_exponents.bin")
    val expected = GoldenVectorLoader.float32LittleEndian(
      caseName,
      s"expected_${cache}_dequant_f32.bin"
    )
    expected.length mustBe cfg.tokenCount * cfg.featureDim
    val random = new Random(
      0x44455154L + caseName.length + cache.head.toInt + cfg.codeValueBits
    )

    simulate(
      new KvStreamDequantizer(
        codeValueBits = cfg.codeValueBits,
        zeroPointBits = cfg.zeroPointBits,
        encodeLengthBits = cfg.encodeLengthBits,
        packTokens = cfg.packLen
      )
    ) { dut =>
      dut.io.start.poke(false.B)
      dut.io.tokenCount.poke(cfg.tokenCount.U)
      dut.io.descriptorCount.poke(cfg.descriptorCount.U)
      dut.io.featureDim.poke(cfg.featureDim.U)
      dut.io.payloadByteCount.poke(payloadBytes.length.U)
      dut.io.minimumIn.valid.poke(false.B)
      dut.io.minimumIn.bits.poke(0.U)
      dut.io.widthIn.valid.poke(false.B)
      dut.io.widthIn.bits.poke(0.U)
      dut.io.payloadIn.valid.poke(false.B)
      dut.io.payloadIn.bits.poke(0.U)
      dut.io.zeroPointIn.valid.poke(false.B)
      dut.io.zeroPointIn.bits.poke(0.U)
      dut.io.exponentIn.valid.poke(false.B)
      dut.io.exponentIn.bits.poke(0.U)
      dut.io.out.ready.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      val streams = IndexedSeq(
        minimumBytes,
        widthBytes,
        payloadBytes,
        zeroBytes,
        exponentBytes
      )
      val indices = Array.fill(5)(0)
      var outputCount = 0
      var cycles = 0
      var done = false
      while (!done && cycles < 200000) {
        val offers = streams.indices.map { stream =>
          indices(stream) < streams(stream).length &&
            (!randomBackpressure || random.nextBoolean())
        }
        dut.io.minimumIn.valid.poke(offers(0).B)
        dut.io.minimumIn.bits.poke(
          (if (indices(0) < streams(0).length) streams(0)(indices(0)) else 0).U
        )
        dut.io.widthIn.valid.poke(offers(1).B)
        dut.io.widthIn.bits.poke(
          (if (indices(1) < streams(1).length) streams(1)(indices(1)) else 0).U
        )
        dut.io.payloadIn.valid.poke(offers(2).B)
        dut.io.payloadIn.bits.poke(
          (if (indices(2) < streams(2).length) streams(2)(indices(2)) else 0).U
        )
        dut.io.zeroPointIn.valid.poke(offers(3).B)
        dut.io.zeroPointIn.bits.poke(
          (if (indices(3) < streams(3).length) streams(3)(indices(3)) else 0).U
        )
        dut.io.exponentIn.valid.poke(offers(4).B)
        dut.io.exponentIn.bits.poke(
          (if (indices(4) < streams(4).length) streams(4)(indices(4)) else 0).U
        )
        val acceptOutput = !randomBackpressure || random.nextBoolean()
        dut.io.out.ready.poke(acceptOutput.B)

        val fires = IndexedSeq(
          offers(0) && dut.io.minimumIn.ready.peek().litToBoolean,
          offers(1) && dut.io.widthIn.ready.peek().litToBoolean,
          offers(2) && dut.io.payloadIn.ready.peek().litToBoolean,
          offers(3) && dut.io.zeroPointIn.ready.peek().litToBoolean,
          offers(4) && dut.io.exponentIn.ready.peek().litToBoolean
        )
        val outputFire = acceptOutput && dut.io.out.valid.peek().litToBoolean
        if (outputFire) {
          val descriptorIndex = dut.io.out.bits.descriptorIndex.peek().litValue.toInt
          val tokenIndex = dut.io.out.bits.tokenIndex.peek().litValue.toInt
          val packIndex = descriptorIndex / cfg.featureDim
          val featureIndex = descriptorIndex % cfg.featureDim
          val globalToken = packIndex * cfg.packLen + tokenIndex
          val expectedIndex = globalToken * cfg.featureDim + featureIndex
          val expectedRaw = Math.round(expected(expectedIndex) * 64.0f)
          dut.io.out.bits.fixedRaw.expect(expectedRaw.S(18.W))
          dut.io.out.bits.last.expect((expectedIndex == expected.length - 1).B)
        }

        dut.clock.step()
        streams.indices.foreach { stream => if (fires(stream)) indices(stream) += 1 }
        if (outputFire) outputCount += 1
        done = dut.io.done.peek().litToBoolean
        cycles += 1
      }

      done mustBe true
      streams.indices.foreach { stream =>
        withClue(s"stream $stream: ") { indices(stream) mustBe streams(stream).length }
      }
      outputCount mustBe expected.length
      dut.io.error.expect(false.B)
      dut.io.busy.expect(false.B)
    }
  }

  "Power-of-two fixed-point dequantizer" - {
    "must preserve exact values at both exponent boundaries" in {
      simulate(
        new Po2FixedPointDequantizer(codeValueBits = 6, zeroPointBits = 7)
      ) { dut =>
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(true.B)
        val cases = Seq(
          (48, -34, -6, 14),
          (20, -6, 0, 896),
          (48, -6, 4, 43008)
        )
        for ((q, zero, exponent, expectedRaw) <- cases) {
          dut.io.in.bits.q.poke(q.U)
          dut.io.in.bits.zeroPoint.poke(zero.S)
          dut.io.in.bits.exponent.poke(exponent.S)
          dut.io.in.valid.poke(true.B)
          dut.io.out.valid.expect(true.B)
          dut.io.out.bits.fixedRaw.expect(expectedRaw.S)
          dut.io.out.bits.error.expect(false.B)
          dut.clock.step()
        }
        dut.io.in.bits.exponent.poke((-7).S)
        dut.io.out.bits.error.expect(true.B)
        dut.io.in.bits.exponent.poke(0.S)
        dut.io.in.bits.zeroPoint.poke(64.S)
        dut.io.out.bits.error.expect(true.B)
      }
    }
  }

  "End-to-end KV stream dequantizer" - {
    "must exactly reconstruct directed and zero-width K/V vectors" in {
      for {
        caseName <- Seq("directed_nonidentity", "directed_width0")
        cache <- Seq("k", "v")
      } runGolden(caseName, cache, randomBackpressure = false)
    }

    "must exactly reconstruct random K/V vectors under independent backpressure" in {
      runGolden("random_seed_20260809", "k", randomBackpressure = true)
      runGolden("random_seed_20260809", "v", randomBackpressure = true)
    }

    "must reconstruct zero-width K/V vectors under independent backpressure" in {
      runGolden("directed_width0", "k", randomBackpressure = true)
      runGolden("directed_width0", "v", randomBackpressure = true)
    }

    "must discard repeated padding tokens from a partial final pack" in {
      simulate(
        new PackMetadataDequantizer(codeValueBits = 6, zeroPointBits = 7)
      ) { dut =>
        dut.io.start.poke(false.B)
        dut.io.tokenCount.poke(3.U)
        dut.io.descriptorCount.poke(2.U)
        dut.io.featureDim.poke(2.U)
        dut.io.qIn.valid.poke(false.B)
        dut.io.zeroIn.valid.poke(false.B)
        dut.io.exponentIn.valid.poke(false.B)
        dut.io.out.ready.poke(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        val zeros = IndexedSeq(-1, -2, -3)
        val exponents = IndexedSeq(-6, -5, -4)
        var metadataIndex = 0
        var descriptorIndex = 0
        var tokenIndex = 0
        var outputCount = 0
        var done = false
        var cycles = 0
        while (!done && cycles < 1000) {
          val offerMetadata = metadataIndex < zeros.length
          dut.io.zeroIn.valid.poke(offerMetadata.B)
          dut.io.zeroIn.bits.poke(
            (if (offerMetadata) zeros(metadataIndex) else 0).S
          )
          dut.io.exponentIn.valid.poke(offerMetadata.B)
          dut.io.exponentIn.bits.poke(
            (if (offerMetadata) exponents(metadataIndex) else 0).S
          )
          val offerQ = descriptorIndex < 2
          dut.io.qIn.valid.poke(offerQ.B)
          dut.io.qIn.bits.value.poke((10 + math.min(tokenIndex, 2)).S)
          dut.io.qIn.bits.descriptorIndex.poke(descriptorIndex.U)
          dut.io.qIn.bits.tokenIndex.poke(tokenIndex.U)
          dut.io.qIn.bits.last.poke((descriptorIndex == 1 && tokenIndex == 15).B)

          val metadataFire = offerMetadata &&
            dut.io.zeroIn.ready.peek().litToBoolean &&
            dut.io.exponentIn.ready.peek().litToBoolean
          val qFire = offerQ && dut.io.qIn.ready.peek().litToBoolean
          val outputFire = dut.io.out.valid.peek().litToBoolean
          if (outputFire) {
            val expectedToken = outputCount % 3
            val mantissa = 10 + expectedToken + zeros(expectedToken)
            val expectedRaw = mantissa << (exponents(expectedToken) + 6)
            dut.io.out.bits.fixedRaw.expect(expectedRaw.S)
            dut.io.out.bits.last.expect((outputCount == 5).B)
          }

          dut.clock.step()
          if (metadataFire) metadataIndex += 1
          if (qFire) {
            if (tokenIndex == 15) {
              tokenIndex = 0
              descriptorIndex += 1
            } else tokenIndex += 1
          }
          if (outputFire) outputCount += 1
          done = dut.io.done.peek().litToBoolean
          cycles += 1
        }

        done mustBe true
        metadataIndex mustBe 3
        outputCount mustBe 6
        dut.io.error.expect(false.B)
      }
    }
  }
}
