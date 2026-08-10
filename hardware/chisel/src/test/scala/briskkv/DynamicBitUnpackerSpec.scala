package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class DynamicBitUnpackerSpec extends AnyFreeSpec with Matchers with ChiselSim {
  private case class VectorConfig(
    codeValueBits: Int,
    encodeLengthBits: Int,
    featureDim: Int,
    packLen: Int,
    tokenCount: Int,
    paddedTokenCount: Int,
    signedValues: Boolean
  ) {
    val packCount: Int = (tokenCount + paddedTokenCount) / packLen
    val descriptorCount: Int = packCount * featureDim
  }

  private def config(caseName: String, cache: String): VectorConfig =
    VectorConfig(
      codeValueBits = GoldenVectorLoader.bitpackInt(caseName, cache, "code_value_bits"),
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
      ),
      signedValues = GoldenVectorLoader.bitpackBoolean(
        caseName,
        cache,
        "signed_values"
      )
    )

  private def runDescriptorVector(
    caseName: String,
    cache: String,
    randomBackpressure: Boolean
  ): Unit = {
    val cfg = config(caseName, cache)
    val minimumBytes = GoldenVectorLoader.unsigned(caseName, s"${cache}_pack_mins.bin")
    val widthBytes = GoldenVectorLoader.unsigned(caseName, s"${cache}_encode_lengths.bin")
    val expectedMinimum = GoldenVectorLoader.unpackFixed(
      minimumBytes,
      cfg.codeValueBits,
      cfg.descriptorCount,
      cfg.signedValues
    )
    val expectedWidth = GoldenVectorLoader.unpackFixed(
      widthBytes,
      cfg.encodeLengthBits,
      cfg.descriptorCount,
      signed = false
    )
    val random = new Random(0x44455343L + cfg.codeValueBits + cfg.descriptorCount)

    simulate(
      new PackDescriptorDecoder(
        cfg.codeValueBits,
        cfg.encodeLengthBits,
        cfg.signedValues
      )
    ) { dut =>
      dut.io.start.poke(false.B)
      dut.io.descriptorCount.poke(cfg.descriptorCount.U)
      dut.io.minimumIn.valid.poke(false.B)
      dut.io.minimumIn.bits.poke(0.U)
      dut.io.widthIn.valid.poke(false.B)
      dut.io.widthIn.bits.poke(0.U)
      dut.io.out.ready.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var minimumIndex = 0
      var widthIndex = 0
      var outputIndex = 0
      var cycles = 0
      var done = false
      while (!done && cycles < 20000) {
        val offerMinimum = minimumIndex < minimumBytes.length &&
          (!randomBackpressure || random.nextBoolean())
        val offerWidth = widthIndex < widthBytes.length &&
          (!randomBackpressure || random.nextBoolean())
        val acceptOutput = !randomBackpressure || random.nextBoolean()
        dut.io.minimumIn.valid.poke(offerMinimum.B)
        dut.io.minimumIn.bits.poke(
          (if (minimumIndex < minimumBytes.length) minimumBytes(minimumIndex) else 0).U
        )
        dut.io.widthIn.valid.poke(offerWidth.B)
        dut.io.widthIn.bits.poke(
          (if (widthIndex < widthBytes.length) widthBytes(widthIndex) else 0).U
        )
        dut.io.out.ready.poke(acceptOutput.B)

        val minimumFire = offerMinimum && dut.io.minimumIn.ready.peek().litToBoolean
        val widthFire = offerWidth && dut.io.widthIn.ready.peek().litToBoolean
        val outputFire = acceptOutput && dut.io.out.valid.peek().litToBoolean
        if (outputFire) {
          outputIndex must be < cfg.descriptorCount
          dut.io.out.bits.minimum.expect(expectedMinimum(outputIndex).S(8.W))
          dut.io.out.bits.bitWidth.expect(expectedWidth(outputIndex).U)
          dut.io.out.bits.descriptorIndex.expect(outputIndex.U)
          dut.io.out.bits.last.expect((outputIndex == cfg.descriptorCount - 1).B)
        }

        dut.clock.step()
        if (minimumFire) minimumIndex += 1
        if (widthFire) widthIndex += 1
        if (outputFire) outputIndex += 1
        done = dut.io.done.peek().litToBoolean
        cycles += 1
      }

      done mustBe true
      minimumIndex mustBe minimumBytes.length
      widthIndex mustBe widthBytes.length
      outputIndex mustBe cfg.descriptorCount
      dut.io.error.expect(false.B)
      dut.io.busy.expect(false.B)
    }
  }

  private def runFullVector(
    caseName: String,
    cache: String,
    randomBackpressure: Boolean
  ): Unit = {
    val cfg = config(caseName, cache)
    cfg.packLen mustBe 16
    cfg.paddedTokenCount mustBe 0
    val minimumBytes = GoldenVectorLoader.unsigned(caseName, s"${cache}_pack_mins.bin")
    val widthBytes = GoldenVectorLoader.unsigned(caseName, s"${cache}_encode_lengths.bin")
    val payloadBytes = GoldenVectorLoader.unsigned(caseName, s"${cache}_payload.bin")
    val expectedQ = GoldenVectorLoader.unsigned(caseName, s"expected_${cache}_q.bin")
    expectedQ.length mustBe cfg.tokenCount * cfg.featureDim
    val random = new Random(
      0x5041594cL + cfg.codeValueBits + cfg.descriptorCount + caseName.length
    )

    simulate(
      new DynamicBitUnpacker(
        cfg.codeValueBits,
        cfg.encodeLengthBits,
        cfg.signedValues,
        cfg.packLen
      )
    ) { dut =>
      dut.io.start.poke(false.B)
      dut.io.descriptorCount.poke(cfg.descriptorCount.U)
      dut.io.payloadByteCount.poke(payloadBytes.length.U)
      dut.io.minimumIn.valid.poke(false.B)
      dut.io.minimumIn.bits.poke(0.U)
      dut.io.widthIn.valid.poke(false.B)
      dut.io.widthIn.bits.poke(0.U)
      dut.io.payloadIn.valid.poke(false.B)
      dut.io.payloadIn.bits.poke(0.U)
      dut.io.out.ready.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var minimumIndex = 0
      var widthIndex = 0
      var payloadIndex = 0
      var outputCount = 0
      var cycles = 0
      var done = false
      while (!done && cycles < 100000) {
        val offerMinimum = minimumIndex < minimumBytes.length &&
          (!randomBackpressure || random.nextBoolean())
        val offerWidth = widthIndex < widthBytes.length &&
          (!randomBackpressure || random.nextBoolean())
        val offerPayload = payloadIndex < payloadBytes.length &&
          (!randomBackpressure || random.nextBoolean())
        val acceptOutput = !randomBackpressure || random.nextBoolean()

        dut.io.minimumIn.valid.poke(offerMinimum.B)
        dut.io.minimumIn.bits.poke(
          (if (minimumIndex < minimumBytes.length) minimumBytes(minimumIndex) else 0).U
        )
        dut.io.widthIn.valid.poke(offerWidth.B)
        dut.io.widthIn.bits.poke(
          (if (widthIndex < widthBytes.length) widthBytes(widthIndex) else 0).U
        )
        dut.io.payloadIn.valid.poke(offerPayload.B)
        dut.io.payloadIn.bits.poke(
          (if (payloadIndex < payloadBytes.length) payloadBytes(payloadIndex) else 0).U
        )
        dut.io.out.ready.poke(acceptOutput.B)

        val minimumFire = offerMinimum && dut.io.minimumIn.ready.peek().litToBoolean
        val widthFire = offerWidth && dut.io.widthIn.ready.peek().litToBoolean
        val payloadFire = offerPayload && dut.io.payloadIn.ready.peek().litToBoolean
        val outputFire = acceptOutput && dut.io.out.valid.peek().litToBoolean
        if (outputFire) {
          val descriptorIndex = dut.io.out.bits.descriptorIndex.peek().litValue.toInt
          val tokenIndex = dut.io.out.bits.tokenIndex.peek().litValue.toInt
          descriptorIndex must be < cfg.descriptorCount
          tokenIndex must be < cfg.packLen
          val packIndex = descriptorIndex / cfg.featureDim
          val featureIndex = descriptorIndex % cfg.featureDim
          val globalToken = packIndex * cfg.packLen + tokenIndex
          val expectedIndex = globalToken * cfg.featureDim + featureIndex
          dut.io.out.bits.value.expect(expectedQ(expectedIndex).S(8.W))
          dut.io.out.bits.last.expect(
            (descriptorIndex == cfg.descriptorCount - 1 && tokenIndex == cfg.packLen - 1).B
          )
        }

        dut.clock.step()
        if (minimumFire) minimumIndex += 1
        if (widthFire) widthIndex += 1
        if (payloadFire) payloadIndex += 1
        if (outputFire) outputCount += 1
        done = dut.io.done.peek().litToBoolean
        cycles += 1
      }

      done mustBe true
      minimumIndex mustBe minimumBytes.length
      widthIndex mustBe widthBytes.length
      payloadIndex mustBe payloadBytes.length
      outputCount mustBe cfg.descriptorCount * cfg.packLen
      dut.io.error.expect(false.B)
      dut.io.busy.expect(false.B)
    }
  }

  private def runMalformedPayload(
    descriptorWidth: Int,
    declaredPayloadBytes: Int,
    suppliedPayloadBytes: Int
  ): Unit = {
    simulate(
      new DynamicPayloadUnpacker(
        codeValueBits = 6,
        widthBits = 3,
        signedValues = false
      )
    ) { dut =>
      dut.io.start.poke(false.B)
      dut.io.descriptorCount.poke(1.U)
      dut.io.payloadByteCount.poke(declaredPayloadBytes.U)
      dut.io.descriptorIn.valid.poke(false.B)
      dut.io.descriptorIn.bits.minimum.poke(0.S)
      dut.io.descriptorIn.bits.bitWidth.poke(descriptorWidth.U)
      dut.io.descriptorIn.bits.descriptorIndex.poke(0.U)
      dut.io.descriptorIn.bits.last.poke(true.B)
      dut.io.payloadIn.valid.poke(false.B)
      dut.io.payloadIn.bits.poke(0.U)
      dut.io.out.ready.poke(true.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var descriptorSent = false
      var payloadIndex = 0
      var cycles = 0
      var done = false
      while (!done && cycles < 1000) {
        dut.io.descriptorIn.valid.poke((!descriptorSent).B)
        val offerPayload = payloadIndex < suppliedPayloadBytes
        dut.io.payloadIn.valid.poke(offerPayload.B)
        dut.io.payloadIn.bits.poke(0.U)
        val descriptorFire = !descriptorSent &&
          dut.io.descriptorIn.ready.peek().litToBoolean
        val payloadFire = offerPayload && dut.io.payloadIn.ready.peek().litToBoolean
        dut.clock.step()
        if (descriptorFire) descriptorSent = true
        if (payloadFire) payloadIndex += 1
        done = dut.io.done.peek().litToBoolean
        cycles += 1
      }

      done mustBe true
      descriptorSent mustBe true
      payloadIndex mustBe suppliedPayloadBytes
      dut.io.error.expect(true.B)
      dut.io.busy.expect(false.B)
    }
  }

  "Pack descriptor decoder" - {
    "must pair independently stalled minimum and width streams" in {
      runDescriptorVector("directed_nonidentity", "k", randomBackpressure = true)
    }
  }

  "Dynamic bit unpacker" - {
    "must reconstruct directed K and V q values" in {
      runFullVector("directed_nonidentity", "k", randomBackpressure = false)
      runFullVector("directed_nonidentity", "v", randomBackpressure = false)
    }

    "must emit constant packs without reading payload bytes" in {
      runFullVector("directed_width0", "k", randomBackpressure = true)
      runFullVector("directed_width0", "v", randomBackpressure = true)
    }

    "must preserve random K and V values under independent backpressure" in {
      runFullVector("random_seed_20260809", "k", randomBackpressure = true)
      runFullVector("random_seed_20260809", "v", randomBackpressure = true)
    }

    "must reject a truncated payload instead of deadlocking" in {
      // A width-six, 16-value descriptor requires exactly 12 payload bytes.
      runMalformedPayload(
        descriptorWidth = 6,
        declaredPayloadBytes = 11,
        suppliedPayloadBytes = 11
      )
    }

    "must reject extra zero payload bytes" in {
      runMalformedPayload(
        descriptorWidth = 0,
        declaredPayloadBytes = 1,
        suppliedPayloadBytes = 1
      )
    }
  }
}
