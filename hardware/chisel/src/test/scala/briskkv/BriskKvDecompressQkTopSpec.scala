package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class BriskKvDecompressQkTopSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  "Full decompression-to-QK top" - {
    "must decode compressed K/V and match Python QK logits under backpressure" in {
      val caseName = "directed_nonidentity"
      val tokenCount = 64
      val featureDim = 4
      val packTokens = 16
      val packCount = tokenCount / packTokens
      val descriptorCount = packCount * featureDim
      val query = GoldenVectorLoader.int32LittleEndian(
        caseName,
        "qk_query_q6_i32.bin"
      )
      val expectedLogits = GoldenVectorLoader.int64LittleEndian(
        caseName,
        "expected_qk_logits_q12_i64.bin"
      )
      val expectedV = GoldenVectorLoader.float32LittleEndian(
        caseName,
        "expected_v_dequant_f32.bin"
      )
      val streams = IndexedSeq(
        GoldenVectorLoader.unsigned(caseName, "k_pack_mins.bin"),
        GoldenVectorLoader.unsigned(caseName, "k_encode_lengths.bin"),
        GoldenVectorLoader.unsigned(caseName, "k_payload.bin"),
        GoldenVectorLoader.unsigned(caseName, "k_zero_points.bin"),
        GoldenVectorLoader.unsigned(caseName, "k_exponents.bin"),
        GoldenVectorLoader.unsigned(caseName, "v_pack_mins.bin"),
        GoldenVectorLoader.unsigned(caseName, "v_encode_lengths.bin"),
        GoldenVectorLoader.unsigned(caseName, "v_payload.bin"),
        GoldenVectorLoader.unsigned(caseName, "v_zero_points.bin"),
        GoldenVectorLoader.unsigned(caseName, "v_exponents.bin"),
        GoldenVectorLoader.unsigned(caseName, "bucket_counts.bin")
      )
      val random = new Random(0x46554c4cL)

      simulate(new BriskKvDecompressQkTop()) { dut =>
        dut.io.command.valid.poke(false.B)
        dut.io.queryLoadIn.valid.poke(false.B)
        dut.io.queryLoadIn.bits.poke(0.S)
        val inputValids = IndexedSeq(
          dut.io.kMinimumIn.valid,
          dut.io.kWidthIn.valid,
          dut.io.kPayloadIn.valid,
          dut.io.kZeroPointIn.valid,
          dut.io.kExponentIn.valid,
          dut.io.vMinimumIn.valid,
          dut.io.vWidthIn.valid,
          dut.io.vPayloadIn.valid,
          dut.io.vZeroPointIn.valid,
          dut.io.vExponentIn.valid,
          dut.io.bucketCountIn.valid
        )
        val inputBits = IndexedSeq(
          dut.io.kMinimumIn.bits,
          dut.io.kWidthIn.bits,
          dut.io.kPayloadIn.bits,
          dut.io.kZeroPointIn.bits,
          dut.io.kExponentIn.bits,
          dut.io.vMinimumIn.bits,
          dut.io.vWidthIn.bits,
          dut.io.vPayloadIn.bits,
          dut.io.vZeroPointIn.bits,
          dut.io.vExponentIn.bits,
          dut.io.bucketCountIn.bits
        )
        val inputReadies = IndexedSeq(
          dut.io.kMinimumIn.ready,
          dut.io.kWidthIn.ready,
          dut.io.kPayloadIn.ready,
          dut.io.kZeroPointIn.ready,
          dut.io.kExponentIn.ready,
          dut.io.vMinimumIn.ready,
          dut.io.vWidthIn.ready,
          dut.io.vPayloadIn.ready,
          dut.io.vZeroPointIn.ready,
          dut.io.vExponentIn.ready,
          dut.io.bucketCountIn.ready
        )
        inputValids.foreach(_.poke(false.B))
        inputBits.foreach(_.poke(0.U))
        dut.io.qkLogitsOut.ready.poke(false.B)
        dut.io.vFeatureOut.ready.poke(false.B)
        dut.io.bucketOut.ready.poke(false.B)
        dut.io.result.ready.poke(false.B)
        dut.clock.step()

        dut.io.command.bits.tag.poke(23.U)
        dut.io.command.bits.tokenCount.poke(tokenCount.U)
        dut.io.command.bits.featureDim.poke(featureDim.U)
        dut.io.command.bits.descriptorCount.poke(descriptorCount.U)
        dut.io.command.bits.kPayloadByteCount.poke(streams(2).length.U)
        dut.io.command.bits.vPayloadByteCount.poke(streams(7).length.U)
        dut.io.command.valid.poke(true.B)
        dut.io.command.ready.expect(true.B)
        dut.clock.step()
        dut.io.command.valid.poke(false.B)

        val indices = Array.fill(streams.length)(0)
        var queryIndex = 0
        var qkPacks = 0
        var vPackets = 0
        var bucketRecords = 0
        var cycles = 0
        var resultSeen = false
        while (!resultSeen && cycles < 200000) {
          val offerQuery = queryIndex < query.length && random.nextBoolean()
          dut.io.queryLoadIn.valid.poke(offerQuery.B)
          dut.io.queryLoadIn.bits.poke(
            (if (queryIndex < query.length) query(queryIndex) else 0).S
          )
          val offers = streams.indices.map { index =>
            indices(index) < streams(index).length && random.nextBoolean()
          }
          streams.indices.foreach { index =>
            inputValids(index).poke(offers(index).B)
            inputBits(index).poke(
              (if (indices(index) < streams(index).length)
                 streams(index)(indices(index))
               else 0).U
            )
          }

          val acceptQk = cycles > 100 && random.nextBoolean()
          val acceptV = random.nextBoolean()
          val acceptBucket = random.nextBoolean()
          dut.io.qkLogitsOut.ready.poke(acceptQk.B)
          dut.io.vFeatureOut.ready.poke(acceptV.B)
          dut.io.bucketOut.ready.poke(acceptBucket.B)
          dut.io.result.ready.poke(false.B)

          val queryFire = offerQuery &&
            dut.io.queryLoadIn.ready.peek().litToBoolean
          val inputFires = streams.indices.map { index =>
            offers(index) && inputReadies(index).peek().litToBoolean
          }
          val qkFire = acceptQk && dut.io.qkLogitsOut.valid.peek().litToBoolean
          val vFire = acceptV && dut.io.vFeatureOut.valid.peek().litToBoolean
          val bucketFire = acceptBucket && dut.io.bucketOut.valid.peek().litToBoolean

          if (qkFire) {
            dut.io.qkLogitsOut.bits.packIndex.expect(qkPacks.U)
            dut.io.qkLogitsOut.bits.validTokens.expect(packTokens.U)
            for (lane <- 0 until packTokens) {
              dut.io.qkLogitsOut.bits.logits(lane).expect(
                expectedLogits(qkPacks * packTokens + lane).S
              )
            }
            dut.io.qkLogitsOut.bits.last.expect((qkPacks == packCount - 1).B)
          }
          if (vFire) {
            val packIndex = vPackets / featureDim
            val featureIndex = vPackets % featureDim
            dut.io.vFeatureOut.bits.packIndex.expect(packIndex.U)
            dut.io.vFeatureOut.bits.featureIndex.expect(featureIndex.U)
            for (lane <- 0 until packTokens) {
              val expectedIndex =
                (packIndex * packTokens + lane) * featureDim + featureIndex
              dut.io.vFeatureOut.bits.values(lane).expect(
                Math.round(expectedV(expectedIndex) * 64.0f).S(18.W)
              )
            }
          }

          dut.clock.step()
          if (queryFire) queryIndex += 1
          streams.indices.foreach { index =>
            if (inputFires(index)) indices(index) += 1
          }
          if (qkFire) qkPacks += 1
          if (vFire) vPackets += 1
          if (bucketFire) bucketRecords += 1

          if (dut.io.result.valid.peek().litToBoolean) {
            qkPacks mustBe packCount
            vPackets mustBe descriptorCount
            bucketRecords mustBe 1
            dut.io.result.bits.tag.expect(23.U)
            dut.io.result.bits.error.expect(false.B)
            dut.io.result.bits.packCount.expect(packCount.U)
            dut.io.progress.qk.queryReplay.loadedValues.expect(featureDim.U)
            dut.io.progress.qk.accumulator.inputPackets.expect(descriptorCount.U)
            dut.io.progress.qk.accumulator.outputPackets.expect(packCount.U)
            dut.io.result.ready.poke(true.B)
            dut.clock.step()
            resultSeen = true
          }
          cycles += 1
        }

        resultSeen mustBe true
        queryIndex mustBe featureDim
        streams.indices.foreach { index =>
          indices(index) mustBe streams(index).length
        }
      }
    }

    "must reject a feature dimension beyond the query memory capacity" in {
      simulate(new BriskKvDecompressQkTop(maximumFeatureDim = 8)) { dut =>
        dut.io.command.valid.poke(false.B)
        dut.io.command.bits.tag.poke(31.U)
        dut.io.command.bits.tokenCount.poke(64.U)
        dut.io.command.bits.featureDim.poke(9.U)
        dut.io.command.bits.descriptorCount.poke(36.U)
        dut.io.command.bits.kPayloadByteCount.poke(0.U)
        dut.io.command.bits.vPayloadByteCount.poke(0.U)
        dut.io.queryLoadIn.valid.poke(false.B)
        dut.io.qkLogitsOut.ready.poke(true.B)
        dut.io.vFeatureOut.ready.poke(true.B)
        dut.io.bucketOut.ready.poke(true.B)
        dut.io.result.ready.poke(false.B)
        dut.io.kMinimumIn.valid.poke(false.B)
        dut.io.kWidthIn.valid.poke(false.B)
        dut.io.kPayloadIn.valid.poke(false.B)
        dut.io.kZeroPointIn.valid.poke(false.B)
        dut.io.kExponentIn.valid.poke(false.B)
        dut.io.vMinimumIn.valid.poke(false.B)
        dut.io.vWidthIn.valid.poke(false.B)
        dut.io.vPayloadIn.valid.poke(false.B)
        dut.io.vZeroPointIn.valid.poke(false.B)
        dut.io.vExponentIn.valid.poke(false.B)
        dut.io.bucketCountIn.valid.poke(false.B)
        dut.clock.step()

        dut.io.command.valid.poke(true.B)
        dut.io.command.ready.expect(true.B)
        dut.clock.step()
        dut.io.command.valid.poke(false.B)
        dut.io.result.valid.expect(true.B)
        dut.io.result.bits.tag.expect(31.U)
        dut.io.result.bits.error.expect(true.B)
        dut.io.queryLoadIn.ready.expect(false.B)
        dut.io.result.ready.poke(true.B)
        dut.clock.step()
        dut.io.busy.expect(false.B)
      }
    }

    "must return an error for descriptor geometry without starting QK" in {
      simulate(new BriskKvDecompressQkTop()) { dut =>
        dut.io.command.valid.poke(false.B)
        dut.io.command.bits.tag.poke(41.U)
        dut.io.command.bits.tokenCount.poke(64.U)
        dut.io.command.bits.featureDim.poke(4.U)
        dut.io.command.bits.descriptorCount.poke(15.U)
        dut.io.command.bits.kPayloadByteCount.poke(0.U)
        dut.io.command.bits.vPayloadByteCount.poke(0.U)
        dut.io.queryLoadIn.valid.poke(false.B)
        dut.io.qkLogitsOut.ready.poke(true.B)
        dut.io.vFeatureOut.ready.poke(true.B)
        dut.io.bucketOut.ready.poke(true.B)
        dut.io.result.ready.poke(false.B)
        dut.io.kMinimumIn.valid.poke(false.B)
        dut.io.kWidthIn.valid.poke(false.B)
        dut.io.kPayloadIn.valid.poke(false.B)
        dut.io.kZeroPointIn.valid.poke(false.B)
        dut.io.kExponentIn.valid.poke(false.B)
        dut.io.vMinimumIn.valid.poke(false.B)
        dut.io.vWidthIn.valid.poke(false.B)
        dut.io.vPayloadIn.valid.poke(false.B)
        dut.io.vZeroPointIn.valid.poke(false.B)
        dut.io.vExponentIn.valid.poke(false.B)
        dut.io.bucketCountIn.valid.poke(false.B)
        dut.clock.step()

        dut.io.command.valid.poke(true.B)
        dut.io.command.ready.expect(true.B)
        dut.clock.step()
        dut.io.command.valid.poke(false.B)

        var cycles = 0
        while (!dut.io.result.valid.peek().litToBoolean && cycles < 20) {
          dut.io.queryLoadIn.ready.expect(false.B)
          dut.clock.step()
          cycles += 1
        }
        dut.io.result.valid.expect(true.B)
        dut.io.result.bits.tag.expect(41.U)
        dut.io.result.bits.error.expect(true.B)
        dut.io.progress.qk.queryReplay.loadedValues.expect(0.U)
        dut.io.progress.qk.accumulator.inputPackets.expect(0.U)
        dut.io.result.ready.poke(true.B)
        dut.clock.step()
        dut.io.busy.expect(false.B)
      }
    }
  }
}
