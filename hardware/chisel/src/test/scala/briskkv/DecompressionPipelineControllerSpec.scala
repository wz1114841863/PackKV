package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class DecompressionPipelineControllerSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private def initialize(dut: DecompressionPipelineController): Unit = {
    dut.io.command.valid.poke(false.B)
    dut.io.command.bits.tag.poke(0.U)
    dut.io.command.bits.tokenCount.poke(0.U)
    dut.io.command.bits.featureDim.poke(0.U)
    dut.io.command.bits.descriptorCount.poke(0.U)
    dut.io.command.bits.payloadByteCount.poke(0.U)
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
    dut.io.result.ready.poke(false.B)
  }

  "Decompression pipeline controller" - {
    "must execute one tagged command and report progress and statistics" in {
      val caseName = "directed_nonidentity"
      val cache = "k"
      val tokenCount = GoldenVectorLoader.bitpackInt(caseName, cache, "token_count")
      val featureDim = GoldenVectorLoader.bitpackInt(caseName, cache, "feature_dim")
      val packLen = GoldenVectorLoader.bitpackInt(caseName, cache, "pack_len")
      val descriptorCount = (tokenCount / packLen) * featureDim
      val minimumBytes = GoldenVectorLoader.unsigned(caseName, "k_pack_mins.bin")
      val widthBytes = GoldenVectorLoader.unsigned(caseName, "k_encode_lengths.bin")
      val payloadBytes = GoldenVectorLoader.unsigned(caseName, "k_payload.bin")
      val zeroBytes = GoldenVectorLoader.unsigned(caseName, "k_zero_points.bin")
      val exponentBytes = GoldenVectorLoader.unsigned(caseName, "k_exponents.bin")
      val expected = GoldenVectorLoader.float32LittleEndian(
        caseName,
        "expected_k_dequant_f32.bin"
      )
      val random = new Random(0x4354524cL)

      simulate(new DecompressionPipelineController(6, 7)) { dut =>
        initialize(dut)
        dut.clock.step()
        dut.io.command.bits.tag.poke(42.U)
        dut.io.command.bits.tokenCount.poke(tokenCount.U)
        dut.io.command.bits.featureDim.poke(featureDim.U)
        dut.io.command.bits.descriptorCount.poke(descriptorCount.U)
        dut.io.command.bits.payloadByteCount.poke(payloadBytes.length.U)
        dut.io.command.valid.poke(true.B)
        dut.io.command.ready.expect(true.B)
        dut.clock.step()
        dut.io.command.valid.poke(false.B)

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
        var resultSeen = false
        while (!resultSeen && cycles < 100000) {
          val offers = streams.indices.map { index =>
            indices(index) < streams(index).length && random.nextBoolean()
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
          val acceptOutput = random.nextBoolean()
          dut.io.out.ready.poke(acceptOutput.B)
          dut.io.result.ready.poke(false.B)

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
            val packIndex = descriptorIndex / featureDim
            val featureIndex = descriptorIndex % featureDim
            val globalToken = packIndex * packLen + tokenIndex
            val expectedIndex = globalToken * featureDim + featureIndex
            dut.io.out.bits.fixedRaw.expect(
              Math.round(expected(expectedIndex) * 64.0f).S(18.W)
            )
            dut.io.out.bits.last.expect((expectedIndex == expected.length - 1).B)
          }

          dut.clock.step()
          streams.indices.foreach { index => if (fires(index)) indices(index) += 1 }
          if (outputFire) outputCount += 1
          if (dut.io.result.valid.peek().litToBoolean) {
            dut.io.command.ready.expect(false.B)
            dut.io.busy.expect(false.B)
            dut.io.result.bits.tag.expect(42.U)
            dut.io.result.bits.error.expect(false.B)
            dut.io.result.bits.tokenCount.expect(64.U)
            dut.io.result.bits.packCount.expect(4.U)
            dut.io.result.bits.blockCount.expect(1.U)
            dut.io.result.bits.descriptorCount.expect(16.U)
            dut.io.result.bits.stats.outputValues.expect(256.U)
            dut.io.progress.completedValues.expect(256.U)
            dut.io.progress.completedDescriptors.expect(16.U)
            dut.io.progress.completedPacks.expect(4.U)
            dut.io.progress.completedBlocks.expect(1.U)
            dut.io.result.ready.poke(true.B)
            dut.clock.step()
            resultSeen = true
          }
          cycles += 1
        }

        resultSeen mustBe true
        outputCount mustBe expected.length
        streams.indices.foreach { index => indices(index) mustBe streams(index).length }
        dut.io.command.ready.expect(true.B)
      }
    }

    "must reject an inconsistent descriptor count without consuming streams" in {
      simulate(new DecompressionPipelineController(6, 7)) { dut =>
        initialize(dut)
        dut.clock.step()
        dut.io.command.bits.tag.poke(7.U)
        dut.io.command.bits.tokenCount.poke(64.U)
        dut.io.command.bits.featureDim.poke(4.U)
        dut.io.command.bits.descriptorCount.poke(15.U)
        dut.io.command.bits.payloadByteCount.poke(100.U)
        dut.io.command.valid.poke(true.B)
        dut.clock.step()
        dut.io.command.valid.poke(false.B)

        dut.io.result.valid.expect(true.B)
        dut.io.result.bits.tag.expect(7.U)
        dut.io.result.bits.error.expect(true.B)
        dut.io.result.bits.packCount.expect(4.U)
        dut.io.result.bits.blockCount.expect(1.U)
        dut.io.result.bits.stats.activeCycles.expect(0.U)
        dut.io.minimumIn.ready.expect(false.B)
        dut.io.widthIn.ready.expect(false.B)
        dut.io.payloadIn.ready.expect(false.B)
        dut.io.zeroPointIn.ready.expect(false.B)
        dut.io.exponentIn.ready.expect(false.B)

        dut.io.result.ready.poke(true.B)
        dut.clock.step()
        dut.io.command.ready.expect(true.B)
      }
    }
  }
}
