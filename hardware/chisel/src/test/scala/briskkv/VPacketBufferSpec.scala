package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.collection.mutable
import scala.util.Random

class VPacketBufferSpec extends AnyFreeSpec with Matchers with ChiselSim {
  "Values-only V packet buffer" - {
    "must reconstruct every packet field for partial and cross-block reads" in {
      val packTokens = 16
      val blockTokens = 64
      val tokenCount = 68
      val featureDim = 3
      val packCount = 5
      val descriptorCount = packCount * featureDim
      val random = new Random(0x565041434b4554L)
      val requests = random.shuffle(
        (for {
          pack <- 0 until packCount
          feature <- 0 until featureDim
        } yield (pack, feature)).toList
      )

      simulate(
        new VPacketBuffer(
          maximumFeatureDim = 8,
          maximumTokens = 80
        )
      ) { dut =>
        dut.io.start.poke(false.B)
        dut.io.finish.poke(false.B)
        dut.io.featureDim.poke(featureDim.U)
        dut.io.tokenCount.poke(tokenCount.U)
        dut.io.loadIn.valid.poke(false.B)
        dut.io.readRequest.valid.poke(false.B)
        dut.io.readResponse.ready.poke(false.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var descriptor = 0
        var cycles = 0
        while (!dut.io.loaded.peek().litToBoolean && cycles < 2000) {
          val offer = descriptor < descriptorCount && random.nextBoolean()
          val driven = math.min(descriptor, descriptorCount - 1)
          val pack = driven / featureDim
          val feature = driven % featureDim
          val validTokens = math.min(packTokens, tokenCount - pack * packTokens)
          dut.io.loadIn.valid.poke(offer.B)
          dut.io.loadIn.bits.descriptorIndex.poke(driven.U)
          dut.io.loadIn.bits.packIndex.poke(pack.U)
          dut.io.loadIn.bits.featureIndex.poke(feature.U)
          dut.io.loadIn.bits.blockIndex.poke((pack / (blockTokens / packTokens)).U)
          dut.io.loadIn.bits.packWithinBlock.poke((pack % (blockTokens / packTokens)).U)
          dut.io.loadIn.bits.validTokens.poke(validTokens.U)
          dut.io.loadIn.bits.last.poke((driven == descriptorCount - 1).B)
          for (lane <- 0 until packTokens) {
            dut.io.loadIn.bits.values(lane).poke((driven * 32 + lane - 200).S)
          }
          val fire = offer && dut.io.loadIn.ready.peek().litToBoolean
          dut.clock.step()
          if (fire) descriptor += 1
          cycles += 1
        }
        descriptor mustBe descriptorCount
        dut.io.loaded.expect(true.B)
        dut.io.loadIn.valid.poke(false.B)

        val expectedResponses = mutable.Queue.empty[(Int, Int)]
        var requestIndex = 0
        var responseCount = 0
        while (responseCount < requests.length && cycles < 10000) {
          val offer = requestIndex < requests.length && random.nextBoolean()
          val (pack, feature) =
            if (requestIndex < requests.length) requests(requestIndex) else (0, 0)
          val accept = random.nextBoolean()
          dut.io.readRequest.valid.poke(offer.B)
          dut.io.readRequest.bits.packIndex.poke(pack.U)
          dut.io.readRequest.bits.featureIndex.poke(feature.U)
          dut.io.readResponse.ready.poke(accept.B)

          val requestFire = offer && dut.io.readRequest.ready.peek().litToBoolean
          val responseFire = accept && dut.io.readResponse.valid.peek().litToBoolean
          if (responseFire) {
            expectedResponses.nonEmpty mustBe true
            val (expectedPack, expectedFeature) = expectedResponses.front
            val expectedDescriptor = expectedPack * featureDim + expectedFeature
            val expectedValidTokens = math.min(
              packTokens,
              tokenCount - expectedPack * packTokens
            )
            dut.io.readResponse.bits.descriptorIndex.expect(expectedDescriptor.U)
            dut.io.readResponse.bits.packIndex.expect(expectedPack.U)
            dut.io.readResponse.bits.featureIndex.expect(expectedFeature.U)
            dut.io.readResponse.bits.blockIndex.expect((expectedPack / 4).U)
            dut.io.readResponse.bits.packWithinBlock.expect((expectedPack % 4).U)
            dut.io.readResponse.bits.validTokens.expect(expectedValidTokens.U)
            dut.io.readResponse.bits.last.expect(
              (expectedPack == packCount - 1 && expectedFeature == featureDim - 1).B
            )
            for (lane <- 0 until packTokens) {
              dut.io.readResponse.bits.values(lane).expect(
                (expectedDescriptor * 32 + lane - 200).S
              )
            }
          }

          dut.clock.step()
          if (requestFire) {
            expectedResponses.enqueue((pack, feature))
            requestIndex += 1
          }
          if (responseFire) {
            expectedResponses.dequeue()
            responseCount += 1
          }
          cycles += 1
        }

        requestIndex mustBe requests.length
        responseCount mustBe requests.length
        expectedResponses mustBe empty
        dut.io.error.expect(false.B)
        dut.io.stats.loadedPackets.expect(descriptorCount.U)
        dut.io.stats.readRequests.expect(requests.length.U)
        dut.io.stats.readResponses.expect(requests.length.U)

        dut.io.readRequest.valid.poke(false.B)
        dut.io.readResponse.ready.poke(false.B)
        dut.io.finish.poke(true.B)
        dut.clock.step()
        dut.io.finish.poke(false.B)
        dut.io.busy.expect(false.B)
      }
    }
  }
}
