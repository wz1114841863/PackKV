package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class BriskKvWriteEncoderTopSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private val BlockTokens = 64
  private val BlockCount = 2
  private val FeatureDim = 2
  private val FractionalBits = 12

  private def packFields(values: Seq[Int], widths: Seq[Int]): IndexedSeq[Int] = {
    var reservoir = BigInt(0)
    var count = 0
    val output = ArrayBuffer.empty[Int]
    values.zip(widths).foreach { case (value, width) =>
      reservoir |= BigInt(value) << count
      count += width
      while (count >= 8) {
        output += (reservoir & 0xff).toInt
        reservoir >>= 8
        count -= 8
      }
    }
    if (count > 0) output += (reservoir & 0xff).toInt
    output.toIndexedSeq
  }

  "Unified Format v0 write encoder" - {
    "must concatenate all eleven streams across blocks under backpressure" in {
      // Every token is K=[0,1], V=[0,1]. The qualified quantizers produce
      // K q=[0,32], exponent=-5 and V q=[0,8], exponent=-3, all with zero=0.
      // Equal K scores preserve token order and place all tokens in bucket 0.
      val descriptorKMinimums = IndexedSeq.fill(4)(IndexedSeq(0, 32)).flatten
      val descriptorVMinimums = IndexedSeq.fill(4)(IndexedSeq(0, 8)).flatten
      val descriptorWidths = IndexedSeq.fill(8)(0)
      val expectedPerBlock = IndexedSeq(
        packFields(descriptorKMinimums, Seq.fill(8)(6)),
        packFields(descriptorWidths, Seq.fill(8)(3)),
        IndexedSeq.empty[Int],
        packFields(descriptorVMinimums, Seq.fill(8)(4)),
        packFields(descriptorWidths, Seq.fill(8)(3)),
        IndexedSeq.empty[Int],
        packFields(Seq.fill(BlockTokens)(0), Seq.fill(BlockTokens)(7)),
        packFields(Seq.fill(BlockTokens)(0xb), Seq.fill(BlockTokens)(4)),
        packFields(Seq.fill(BlockTokens)(0), Seq.fill(BlockTokens)(5)),
        packFields(Seq.fill(BlockTokens)(0xd), Seq.fill(BlockTokens)(4)),
        packFields(Seq(64, 0, 0), Seq(7, 7, 7))
      )
      val expected = expectedPerBlock.map { stream =>
        IndexedSeq.fill(BlockCount)(stream).flatten
      }
      val random = new Random(0x5752495445544f50L)

      simulate(
        new BriskKvWriteEncoderTop(maximumFeatureDim = 8)
      ) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(FeatureDim.U)
        dut.io.blockCount.poke(BlockCount.U)
        dut.io.firstBlockIndex.poke(3.U)
        dut.io.in.valid.poke(false.B)
        dut.io.kMinimumOut.ready.poke(false.B)
        dut.io.kWidthOut.ready.poke(false.B)
        dut.io.kPayloadOut.ready.poke(false.B)
        dut.io.vMinimumOut.ready.poke(false.B)
        dut.io.vWidthOut.ready.poke(false.B)
        dut.io.vPayloadOut.ready.poke(false.B)
        dut.io.kZeroOut.ready.poke(false.B)
        dut.io.kExponentOut.ready.poke(false.B)
        dut.io.vZeroOut.ready.poke(false.B)
        dut.io.vExponentOut.ready.poke(false.B)
        dut.io.bucketCountOut.ready.poke(false.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var inputToken = 0
        var inputFeature = 0
        val outputIndices = Array.fill(11)(0)
        var cycles = 0
        var done = false
        while (!done && cycles < 100000) {
          val totalTokens = BlockTokens * BlockCount
          val offerInput = inputToken < totalTokens && random.nextBoolean()
          val drivenToken = math.min(inputToken, totalTokens - 1)
          val tokenInBlock = drivenToken % BlockTokens
          val blockOffset = drivenToken / BlockTokens
          dut.io.in.valid.poke(offerInput.B)
          dut.io.in.bits.kFixedRaw.poke(
            (if (inputFeature == 0) 0 else 1 << FractionalBits).S
          )
          dut.io.in.bits.vFixedRaw.poke(
            (if (inputFeature == 0) 0 else 1 << FractionalBits).S
          )
          dut.io.in.bits.tokenTag.poke((1000 + drivenToken).U)
          dut.io.in.bits.blockIndex.poke((3 + blockOffset).U)
          dut.io.in.bits.tokenIndex.poke(tokenInBlock.U)
          dut.io.in.bits.featureIndex.poke(inputFeature.U)
          dut.io.in.bits.lastFeature.poke((inputFeature == FeatureDim - 1).B)
          dut.io.in.bits.last.poke(
            (drivenToken == totalTokens - 1 && inputFeature == FeatureDim - 1).B
          )

          val accepts = IndexedSeq.fill(11)(random.nextBoolean())
          dut.io.kMinimumOut.ready.poke(accepts(0).B)
          dut.io.kWidthOut.ready.poke(accepts(1).B)
          dut.io.kPayloadOut.ready.poke(accepts(2).B)
          dut.io.vMinimumOut.ready.poke(accepts(3).B)
          dut.io.vWidthOut.ready.poke(accepts(4).B)
          dut.io.vPayloadOut.ready.poke(accepts(5).B)
          dut.io.kZeroOut.ready.poke(accepts(6).B)
          dut.io.kExponentOut.ready.poke(accepts(7).B)
          dut.io.vZeroOut.ready.poke(accepts(8).B)
          dut.io.vExponentOut.ready.poke(accepts(9).B)
          dut.io.bucketCountOut.ready.poke(accepts(10).B)

          val valids = IndexedSeq(
            dut.io.kMinimumOut.valid.peek().litToBoolean,
            dut.io.kWidthOut.valid.peek().litToBoolean,
            dut.io.kPayloadOut.valid.peek().litToBoolean,
            dut.io.vMinimumOut.valid.peek().litToBoolean,
            dut.io.vWidthOut.valid.peek().litToBoolean,
            dut.io.vPayloadOut.valid.peek().litToBoolean,
            dut.io.kZeroOut.valid.peek().litToBoolean,
            dut.io.kExponentOut.valid.peek().litToBoolean,
            dut.io.vZeroOut.valid.peek().litToBoolean,
            dut.io.vExponentOut.valid.peek().litToBoolean,
            dut.io.bucketCountOut.valid.peek().litToBoolean
          )
          val bits = IndexedSeq(
            dut.io.kMinimumOut.bits.peek().litValue.toInt,
            dut.io.kWidthOut.bits.peek().litValue.toInt,
            dut.io.kPayloadOut.bits.peek().litValue.toInt,
            dut.io.vMinimumOut.bits.peek().litValue.toInt,
            dut.io.vWidthOut.bits.peek().litValue.toInt,
            dut.io.vPayloadOut.bits.peek().litValue.toInt,
            dut.io.kZeroOut.bits.peek().litValue.toInt,
            dut.io.kExponentOut.bits.peek().litValue.toInt,
            dut.io.vZeroOut.bits.peek().litValue.toInt,
            dut.io.vExponentOut.bits.peek().litValue.toInt,
            dut.io.bucketCountOut.bits.peek().litValue.toInt
          )
          val inputFire = offerInput && dut.io.in.ready.peek().litToBoolean
          for (stream <- 0 until 11 if accepts(stream) && valids(stream)) {
            outputIndices(stream) must be < expected(stream).length
            bits(stream) mustBe expected(stream)(outputIndices(stream))
          }

          dut.clock.step()
          if (inputFire) {
            if (inputFeature == FeatureDim - 1) {
              inputFeature = 0
              inputToken += 1
            } else inputFeature += 1
          }
          for (stream <- 0 until 11 if accepts(stream) && valids(stream)) {
            outputIndices(stream) += 1
          }
          done = dut.io.done.peek().litToBoolean
          cycles += 1
        }

        done mustBe true
        inputToken mustBe BlockTokens * BlockCount
        for (stream <- 0 until 11) {
          withClue(s"stream $stream: ") {
            outputIndices(stream) mustBe expected(stream).length
          }
        }
        dut.io.error.expect(false.B)
        dut.io.stats.inputPairs.expect((BlockTokens * BlockCount * FeatureDim).U)
        dut.io.stats.completedTokens.expect((BlockTokens * BlockCount).U)
        dut.io.stats.completedBlocks.expect(BlockCount.U)
      }
    }

    "must reject an odd feature dimension before accepting data" in {
      simulate(
        new BriskKvWriteEncoderTop(maximumFeatureDim = 8)
      ) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(3.U)
        dut.io.blockCount.poke(1.U)
        dut.io.firstBlockIndex.poke(0.U)
        dut.io.in.valid.poke(true.B)
        dut.io.in.bits.kFixedRaw.poke(0.S)
        dut.io.in.bits.vFixedRaw.poke(0.S)
        dut.io.in.bits.tokenTag.poke(0.U)
        dut.io.in.bits.blockIndex.poke(0.U)
        dut.io.in.bits.tokenIndex.poke(0.U)
        dut.io.in.bits.featureIndex.poke(0.U)
        dut.io.in.bits.lastFeature.poke(false.B)
        dut.io.in.bits.last.poke(false.B)
        dut.io.kMinimumOut.ready.poke(true.B)
        dut.io.kWidthOut.ready.poke(true.B)
        dut.io.kPayloadOut.ready.poke(true.B)
        dut.io.vMinimumOut.ready.poke(true.B)
        dut.io.vWidthOut.ready.poke(true.B)
        dut.io.vPayloadOut.ready.poke(true.B)
        dut.io.kZeroOut.ready.poke(true.B)
        dut.io.kExponentOut.ready.poke(true.B)
        dut.io.vZeroOut.ready.poke(true.B)
        dut.io.vExponentOut.ready.poke(true.B)
        dut.io.bucketCountOut.ready.poke(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        dut.io.done.expect(true.B)
        dut.io.error.expect(true.B)
        dut.io.busy.expect(false.B)
        dut.io.in.ready.expect(false.B)
        dut.io.stats.rejectedTransactions.expect(1.U)
      }
    }

    "must reject a transaction larger than the configured token bound" in {
      simulate(
        new BriskKvWriteEncoderTop(
          maximumFeatureDim = 8,
          maximumTokens = 128,
          enableStats = false
        )
      ) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(2.U)
        dut.io.blockCount.poke(3.U)
        dut.io.firstBlockIndex.poke(0.U)
        dut.io.in.valid.poke(false.B)
        dut.io.kMinimumOut.ready.poke(false.B)
        dut.io.kWidthOut.ready.poke(false.B)
        dut.io.kPayloadOut.ready.poke(false.B)
        dut.io.vMinimumOut.ready.poke(false.B)
        dut.io.vWidthOut.ready.poke(false.B)
        dut.io.vPayloadOut.ready.poke(false.B)
        dut.io.kZeroOut.ready.poke(false.B)
        dut.io.kExponentOut.ready.poke(false.B)
        dut.io.vZeroOut.ready.poke(false.B)
        dut.io.vExponentOut.ready.poke(false.B)
        dut.io.bucketCountOut.ready.poke(false.B)

        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        dut.io.done.expect(true.B)
        dut.io.error.expect(true.B)
        dut.io.busy.expect(false.B)
        dut.io.in.ready.expect(false.B)
      }
    }
  }
}
