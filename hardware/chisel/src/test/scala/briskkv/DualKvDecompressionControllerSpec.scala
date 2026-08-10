package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class DualKvDecompressionControllerSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  "Dual K/V decompression controller" - {
    "must coordinate independent K, V, and bucket streams under backpressure" in {
      val caseName = "directed_nonidentity"
      val tokenCount = GoldenVectorLoader.bitpackInt(caseName, "k", "token_count")
      val featureDim = GoldenVectorLoader.bitpackInt(caseName, "k", "feature_dim")
      val packLen = GoldenVectorLoader.bitpackInt(caseName, "k", "pack_len")
      val descriptorCount = (tokenCount / packLen) * featureDim
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
      val expectedK = GoldenVectorLoader.float32LittleEndian(
        caseName,
        "expected_k_dequant_f32.bin"
      )
      val expectedV = GoldenVectorLoader.float32LittleEndian(
        caseName,
        "expected_v_dequant_f32.bin"
      )
      val expectedBuckets = GoldenVectorLoader
        .unsigned(caseName, "expected_bucket_counts.bin")
        .grouped(4)
        .map(_.toIndexedSeq)
        .toIndexedSeq
      val random = new Random(0x4455414cL)

      simulate(new DualKvDecompressionController()) { dut =>
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
        dut.io.kOut.ready.poke(false.B)
        dut.io.vOut.ready.poke(false.B)
        dut.io.bucketOut.ready.poke(false.B)
        dut.io.result.ready.poke(false.B)
        dut.clock.step()

        dut.io.command.bits.tag.poke(99.U)
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
        var kOutputCount = 0
        var vOutputCount = 0
        var bucketOutputCount = 0
        var cycles = 0
        var resultSeen = false
        while (!resultSeen && cycles < 200000) {
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
          // Hold K briefly while V and bucket streams continue independently.
          val acceptK = cycles > 40 && random.nextBoolean()
          val acceptV = random.nextBoolean()
          val acceptBucket = random.nextBoolean()
          dut.io.kOut.ready.poke(acceptK.B)
          dut.io.vOut.ready.poke(acceptV.B)
          dut.io.bucketOut.ready.poke(acceptBucket.B)
          dut.io.result.ready.poke(false.B)

          val inputFires = streams.indices.map { index =>
            offers(index) && inputReadies(index).peek().litToBoolean
          }
          val kFire = acceptK && dut.io.kOut.valid.peek().litToBoolean
          val vFire = acceptV && dut.io.vOut.valid.peek().litToBoolean
          val bucketFire = acceptBucket && dut.io.bucketOut.valid.peek().litToBoolean

          def checkValue(
            descriptorIndex: Int,
            tokenIndex: Int,
            expected: IndexedSeq[Float],
            signal: SInt,
            last: Bool
          ): Unit = {
            val packIndex = descriptorIndex / featureDim
            val featureIndex = descriptorIndex % featureDim
            val globalToken = packIndex * packLen + tokenIndex
            val expectedIndex = globalToken * featureDim + featureIndex
            signal.expect(Math.round(expected(expectedIndex) * 64.0f).S(18.W))
            last.expect((expectedIndex == expected.length - 1).B)
          }

          if (kFire) {
            checkValue(
              dut.io.kOut.bits.descriptorIndex.peek().litValue.toInt,
              dut.io.kOut.bits.tokenIndex.peek().litValue.toInt,
              expectedK,
              dut.io.kOut.bits.fixedRaw,
              dut.io.kOut.bits.last
            )
          }
          if (vFire) {
            checkValue(
              dut.io.vOut.bits.descriptorIndex.peek().litValue.toInt,
              dut.io.vOut.bits.tokenIndex.peek().litValue.toInt,
              expectedV,
              dut.io.vOut.bits.fixedRaw,
              dut.io.vOut.bits.last
            )
          }
          if (bucketFire) {
            bucketOutputCount must be < expectedBuckets.length
            for (bucket <- 0 until 4) {
              dut.io.bucketOut.bits.counts(bucket).expect(
                expectedBuckets(bucketOutputCount)(bucket).U
              )
            }
            dut.io.bucketOut.bits.blockIndex.expect(bucketOutputCount.U)
            dut.io.bucketOut.bits.last.expect(
              (bucketOutputCount == expectedBuckets.length - 1).B
            )
          }

          dut.clock.step()
          streams.indices.foreach { index => if (inputFires(index)) indices(index) += 1 }
          if (kFire) kOutputCount += 1
          if (vFire) vOutputCount += 1
          if (bucketFire) bucketOutputCount += 1

          if (dut.io.result.valid.peek().litToBoolean) {
            dut.io.busy.expect(false.B)
            dut.io.command.ready.expect(false.B)
            dut.io.result.bits.tag.expect(99.U)
            dut.io.result.bits.error.expect(false.B)
            dut.io.result.bits.tokenCount.expect(64.U)
            dut.io.result.bits.packCount.expect(4.U)
            dut.io.result.bits.blockCount.expect(1.U)
            dut.io.result.bits.descriptorCount.expect(16.U)
            dut.io.result.bits.bucketRecords.expect(1.U)
            dut.io.result.bits.kStats.outputValues.expect(256.U)
            dut.io.result.bits.vStats.outputValues.expect(256.U)
            dut.io.progress.k.completedValues.expect(256.U)
            dut.io.progress.v.completedValues.expect(256.U)
            dut.io.progress.bucketRecords.expect(1.U)
            dut.io.result.ready.poke(true.B)
            dut.clock.step()
            resultSeen = true
          }
          cycles += 1
        }

        resultSeen mustBe true
        kOutputCount mustBe expectedK.length
        vOutputCount mustBe expectedV.length
        bucketOutputCount mustBe expectedBuckets.length
        streams.indices.foreach { index => indices(index) mustBe streams(index).length }
        dut.io.command.ready.expect(true.B)
      }
    }

    "must reject invalid shared geometry before launching any child stream" in {
      simulate(new DualKvDecompressionController()) { dut =>
        dut.io.command.valid.poke(false.B)
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
        dut.io.kOut.ready.poke(true.B)
        dut.io.vOut.ready.poke(true.B)
        dut.io.bucketOut.ready.poke(true.B)
        dut.io.result.ready.poke(false.B)
        dut.clock.step()

        dut.io.command.bits.tag.poke(3.U)
        dut.io.command.bits.tokenCount.poke(64.U)
        dut.io.command.bits.featureDim.poke(4.U)
        dut.io.command.bits.descriptorCount.poke(15.U)
        dut.io.command.bits.kPayloadByteCount.poke(1.U)
        dut.io.command.bits.vPayloadByteCount.poke(1.U)
        dut.io.command.valid.poke(true.B)
        dut.clock.step()
        dut.io.command.valid.poke(false.B)

        dut.io.result.valid.expect(true.B)
        dut.io.result.bits.error.expect(true.B)
        dut.io.result.bits.kStats.activeCycles.expect(0.U)
        dut.io.result.bits.vStats.activeCycles.expect(0.U)
        dut.io.progress.bucketRecords.expect(0.U)
        dut.io.kMinimumIn.ready.expect(false.B)
        dut.io.vMinimumIn.ready.expect(false.B)
        dut.io.bucketCountIn.ready.expect(false.B)
      }
    }
  }
}
