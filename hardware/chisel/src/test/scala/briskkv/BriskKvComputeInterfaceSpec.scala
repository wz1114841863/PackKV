package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class BriskKvComputeInterfaceSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  "BRISK-KV compute interface" - {
    "must expose aligned K/V feature packets and delay completion until drained" in {
      val caseName = "directed_nonidentity"
      val tokenCount = 64
      val featureDim = 4
      val packTokens = 16
      val descriptorCount = 16
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
      val random = new Random(0x434f4d50L)

      simulate(new BriskKvComputeInterface()) { dut =>
        dut.io.command.valid.poke(false.B)
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
        dut.io.kFeatureOut.ready.poke(false.B)
        dut.io.vFeatureOut.ready.poke(false.B)
        dut.io.bucketOut.ready.poke(true.B)
        dut.io.result.ready.poke(false.B)
        dut.clock.step()

        dut.io.command.bits.tag.poke(12.U)
        dut.io.command.bits.tokenCount.poke(tokenCount.U)
        dut.io.command.bits.featureDim.poke(featureDim.U)
        dut.io.command.bits.descriptorCount.poke(descriptorCount.U)
        dut.io.command.bits.kPayloadByteCount.poke(streams(2).length.U)
        dut.io.command.bits.vPayloadByteCount.poke(streams(7).length.U)
        dut.io.command.valid.poke(true.B)
        dut.clock.step()
        dut.io.command.valid.poke(false.B)

        val indices = Array.fill(streams.length)(0)
        var kPackets = 0
        var vPackets = 0
        var bucketRecords = 0
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
          val acceptK = cycles > 60 && random.nextBoolean()
          val acceptV = random.nextBoolean()
          dut.io.kFeatureOut.ready.poke(acceptK.B)
          dut.io.vFeatureOut.ready.poke(acceptV.B)
          dut.io.bucketOut.ready.poke(true.B)
          dut.io.result.ready.poke(false.B)

          val inputFires = streams.indices.map { index =>
            offers(index) && inputReadies(index).peek().litToBoolean
          }
          val kFire = acceptK && dut.io.kFeatureOut.valid.peek().litToBoolean
          val vFire = acceptV && dut.io.vFeatureOut.valid.peek().litToBoolean
          val bucketFire = dut.io.bucketOut.valid.peek().litToBoolean

          def checkPacket(
            packetNumber: Int,
            expected: IndexedSeq[Float],
            packet: AttentionFeaturePacket
          ): Unit = {
            val packIndex = packetNumber / featureDim
            val featureIndex = packetNumber % featureDim
            packet.packIndex.expect(packIndex.U)
            packet.featureIndex.expect(featureIndex.U)
            packet.blockIndex.expect(0.U)
            packet.packWithinBlock.expect(packIndex.U)
            packet.validTokens.expect(packTokens.U)
            for (lane <- 0 until packTokens) {
              val expectedIndex =
                (packIndex * packTokens + lane) * featureDim + featureIndex
              packet.values(lane).expect(
                Math.round(expected(expectedIndex) * 64.0f).S(18.W)
              )
            }
            packet.last.expect((packetNumber == descriptorCount - 1).B)
          }

          if (kFire) checkPacket(kPackets, expectedK, dut.io.kFeatureOut.bits)
          if (vFire) checkPacket(vPackets, expectedV, dut.io.vFeatureOut.bits)

          dut.clock.step()
          streams.indices.foreach { index => if (inputFires(index)) indices(index) += 1 }
          if (kFire) kPackets += 1
          if (vFire) vPackets += 1
          if (bucketFire) bucketRecords += 1

          if (dut.io.result.valid.peek().litToBoolean) {
            kPackets mustBe descriptorCount
            vPackets mustBe descriptorCount
            dut.io.result.bits.error.expect(false.B)
            dut.io.result.bits.tag.expect(12.U)
            dut.io.progress.kPacketizer.inputValues.expect(256.U)
            dut.io.progress.vPacketizer.inputValues.expect(256.U)
            dut.io.progress.kPacketizer.outputPackets.expect(16.U)
            dut.io.progress.vPacketizer.outputPackets.expect(16.U)
            dut.io.result.ready.poke(true.B)
            dut.clock.step()
            resultSeen = true
          }
          cycles += 1
        }

        resultSeen mustBe true
        kPackets mustBe descriptorCount
        vPackets mustBe descriptorCount
        bucketRecords mustBe 1
        streams.indices.foreach { index => indices(index) mustBe streams(index).length }
      }
    }
  }
}
