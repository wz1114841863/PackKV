package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class KvPackTransposeBitPackEncoderSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private val TokenCount = 64
  private val PackTokens = 16

  private case class Encoded(
    minimums: IndexedSeq[Int],
    widths: IndexedSeq[Int],
    payload: IndexedSeq[Int]
  )

  private def packFields(
    values: IndexedSeq[Int],
    widths: IndexedSeq[Int]
  ): IndexedSeq[Int] = {
    var reservoir = BigInt(0)
    var count = 0
    val bytes = ArrayBuffer.empty[Int]
    values.zip(widths).foreach { case (value, width) =>
      require(width >= 0)
      require(value >= 0 && (width == 0 && value == 0 || value < (1 << width)))
      reservoir |= BigInt(value) << count
      count += width
      while (count >= 8) {
        bytes += (reservoir & 0xff).toInt
        reservoir >>= 8
        count -= 8
      }
    }
    if (count > 0) bytes += (reservoir & 0xff).toInt
    bytes.toIndexedSeq
  }

  private def encodeReference(
    values: IndexedSeq[IndexedSeq[Int]],
    codeBits: Int
  ): Encoded = {
    val featureDim = values.head.length
    val minimumFields = ArrayBuffer.empty[Int]
    val widthFields = ArrayBuffer.empty[Int]
    val payloadValues = ArrayBuffer.empty[Int]
    val payloadWidths = ArrayBuffer.empty[Int]
    for {
      pack <- 0 until TokenCount / PackTokens
      feature <- 0 until featureDim
    } {
      val lanes = (0 until PackTokens).map { lane =>
        values(pack * PackTokens + lane)(feature)
      }
      val minimum = lanes.min
      val difference = lanes.max - minimum
      val width = if (difference == 0) 0 else 32 - Integer.numberOfLeadingZeros(difference)
      minimumFields += minimum
      widthFields += width
      lanes.foreach { value =>
        payloadValues += value - minimum
        payloadWidths += width
      }
    }
    val widthBits = math.max(1, 32 - Integer.numberOfLeadingZeros(codeBits))
    Encoded(
      packFields(
        minimumFields.toIndexedSeq,
        IndexedSeq.fill(minimumFields.length)(codeBits)
      ),
      packFields(
        widthFields.toIndexedSeq,
        IndexedSeq.fill(widthFields.length)(widthBits)
      ),
      packFields(payloadValues.toIndexedSeq, payloadWidths.toIndexedSeq)
    )
  }

  private def runCase(
    featureDim: Int,
    kValues: IndexedSeq[IndexedSeq[Int]],
    vValues: IndexedSeq[IndexedSeq[Int]],
    randomBackpressure: Boolean
  ): Unit = {
    val expectedK = encodeReference(kValues, 6)
    val expectedV = encodeReference(vValues, 4)
    val expected = IndexedSeq(
      expectedK.minimums,
      expectedK.widths,
      expectedK.payload,
      expectedV.minimums,
      expectedV.widths,
      expectedV.payload
    )
    val random = new Random(0x5041434b454e43L + featureDim)

    simulate(
      new KvPackTransposeBitPackEncoder(maximumFeatureDim = 8)
    ) { dut =>
      dut.io.start.poke(false.B)
      dut.io.featureDim.poke(featureDim.U)
      dut.io.blockIndex.poke(3.U)
      dut.io.in.valid.poke(false.B)
      dut.io.kMinimumOut.ready.poke(false.B)
      dut.io.kWidthOut.ready.poke(false.B)
      dut.io.kPayloadOut.ready.poke(false.B)
      dut.io.vMinimumOut.ready.poke(false.B)
      dut.io.vWidthOut.ready.poke(false.B)
      dut.io.vPayloadOut.ready.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var inputToken = 0
      var inputFeature = 0
      val outputIndices = Array.fill(6)(0)
      var cycles = 0
      var done = false
      while (!done && cycles < 100000) {
        val offerInput = inputToken < TokenCount &&
          (!randomBackpressure || random.nextBoolean())
        dut.io.in.valid.poke(offerInput.B)
        val drivenToken = math.min(inputToken, TokenCount - 1)
        dut.io.in.bits.kQ.poke(kValues(drivenToken)(inputFeature).U)
        dut.io.in.bits.vQ.poke(vValues(drivenToken)(inputFeature).U)
        dut.io.in.bits.tokenTag.poke((500 + drivenToken).U)
        dut.io.in.bits.originalTokenIndex.poke(drivenToken.U)
        dut.io.in.bits.routedTokenIndex.poke(drivenToken.U)
        dut.io.in.bits.featureIndex.poke(inputFeature.U)
        dut.io.in.bits.bucketId.poke((drivenToken / 16).U)
        dut.io.in.bits.blockIndex.poke(3.U)
        dut.io.in.bits.lastFeature.poke((inputFeature == featureDim - 1).B)
        dut.io.in.bits.last.poke(
          (drivenToken == TokenCount - 1 && inputFeature == featureDim - 1).B
        )

        val accepts = IndexedSeq.fill(6)(
          !randomBackpressure || random.nextBoolean()
        )
        dut.io.kMinimumOut.ready.poke(accepts(0).B)
        dut.io.kWidthOut.ready.poke(accepts(1).B)
        dut.io.kPayloadOut.ready.poke(accepts(2).B)
        dut.io.vMinimumOut.ready.poke(accepts(3).B)
        dut.io.vWidthOut.ready.poke(accepts(4).B)
        dut.io.vPayloadOut.ready.poke(accepts(5).B)

        val inputFire = offerInput && dut.io.in.ready.peek().litToBoolean
        val valids = IndexedSeq(
          dut.io.kMinimumOut.valid.peek().litToBoolean,
          dut.io.kWidthOut.valid.peek().litToBoolean,
          dut.io.kPayloadOut.valid.peek().litToBoolean,
          dut.io.vMinimumOut.valid.peek().litToBoolean,
          dut.io.vWidthOut.valid.peek().litToBoolean,
          dut.io.vPayloadOut.valid.peek().litToBoolean
        )
        val bits = IndexedSeq(
          dut.io.kMinimumOut.bits.peek().litValue.toInt,
          dut.io.kWidthOut.bits.peek().litValue.toInt,
          dut.io.kPayloadOut.bits.peek().litValue.toInt,
          dut.io.vMinimumOut.bits.peek().litValue.toInt,
          dut.io.vWidthOut.bits.peek().litValue.toInt,
          dut.io.vPayloadOut.bits.peek().litValue.toInt
        )
        for (stream <- 0 until 6 if accepts(stream) && valids(stream)) {
          outputIndices(stream) must be < expected(stream).length
          bits(stream) mustBe expected(stream)(outputIndices(stream))
        }

        dut.clock.step()
        if (inputFire) {
          if (inputFeature == featureDim - 1) {
            inputFeature = 0
            inputToken += 1
          } else inputFeature += 1
        }
        for (stream <- 0 until 6 if accepts(stream) && valids(stream)) {
          outputIndices(stream) += 1
        }
        done = dut.io.done.peek().litToBoolean
        cycles += 1
      }

      done mustBe true
      inputToken mustBe TokenCount
      for (stream <- 0 until 6) {
        withClue(s"stream $stream: ") {
          outputIndices(stream) mustBe expected(stream).length
        }
      }
      dut.io.error.expect(false.B)
      dut.io.transposeStats.inputValues.expect((TokenCount * featureDim).U)
      dut.io.transposeStats.outputDescriptors.expect((4 * featureDim).U)
      dut.io.kEncoderStats.inputDescriptors.expect((4 * featureDim).U)
      dut.io.vEncoderStats.inputDescriptors.expect((4 * featureDim).U)
      dut.io.kEncoderStats.minimumBytes.expect(expectedK.minimums.length.U)
      dut.io.kEncoderStats.widthBytes.expect(expectedK.widths.length.U)
      dut.io.kEncoderStats.payloadBytes.expect(expectedK.payload.length.U)
      dut.io.vEncoderStats.minimumBytes.expect(expectedV.minimums.length.U)
      dut.io.vEncoderStats.widthBytes.expect(expectedV.widths.length.U)
      dut.io.vEncoderStats.payloadBytes.expect(expectedV.payload.length.U)
    }
  }

  "16-token transpose and dynamic bit-pack encoder" - {
    "must exactly match software-format K/V bytes under backpressure" in {
      val featureDim = 3
      val kValues = IndexedSeq.tabulate(TokenCount) { token =>
        IndexedSeq.tabulate(featureDim) { feature =>
          (token * 11 + feature * 19 + token / 16) % 64
        }
      }
      val vValues = IndexedSeq.tabulate(TokenCount) { token =>
        IndexedSeq.tabulate(featureDim) { feature =>
          (token * 5 + feature * 7 + token / 16) % 16
        }
      }
      runCase(featureDim, kValues, vValues, randomBackpressure = true)
    }

    "must emit no payload bytes for all zero-width descriptors" in {
      val featureDim = 2
      val kValues = IndexedSeq.tabulate(TokenCount) { token =>
        IndexedSeq.tabulate(featureDim)(feature => (token / 16 + feature) % 64)
      }
      val vValues = IndexedSeq.tabulate(TokenCount) { token =>
        IndexedSeq.tabulate(featureDim)(feature => (token / 16 + feature * 2) % 16)
      }
      runCase(featureDim, kValues, vValues, randomBackpressure = true)
    }

    "must reject unsupported geometry without starting either encoder" in {
      simulate(
        new KvPackTransposeBitPackEncoder(maximumFeatureDim = 8)
      ) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(9.U)
        dut.io.blockIndex.poke(0.U)
        dut.io.in.valid.poke(false.B)
        dut.io.kMinimumOut.ready.poke(true.B)
        dut.io.kWidthOut.ready.poke(true.B)
        dut.io.kPayloadOut.ready.poke(true.B)
        dut.io.vMinimumOut.ready.poke(true.B)
        dut.io.vWidthOut.ready.poke(true.B)
        dut.io.vPayloadOut.ready.poke(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        dut.io.done.expect(true.B)
        dut.io.error.expect(true.B)
        dut.io.busy.expect(false.B)
        dut.io.in.ready.expect(false.B)
        dut.io.kEncoderStats.inputDescriptors.expect(0.U)
        dut.io.vEncoderStats.inputDescriptors.expect(0.U)
      }
    }
  }
}
