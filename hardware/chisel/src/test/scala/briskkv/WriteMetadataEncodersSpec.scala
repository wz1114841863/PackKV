package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class WriteMetadataEncodersSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
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

  private def encodedSigned(value: Int, bits: Int): Int =
    value & ((1 << bits) - 1)

  "Fixed-width field packer" - {
    "must pad only the high bits of the final byte" in {
      val fields = IndexedSeq(3, 17, 7)
      val expected = packFields(fields, Seq.fill(fields.length)(5))
      simulate(new FixedWidthFieldPacker(fieldBits = 5)) { dut =>
        dut.io.start.poke(false.B)
        dut.io.fieldCount.poke(fields.length.U)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        var input = 0
        var output = 0
        while (!dut.io.done.peek().litToBoolean) {
          dut.io.in.valid.poke((input < fields.length).B)
          dut.io.in.bits.poke((if (input < fields.length) fields(input) else 0).U)
          val inFire = input < fields.length && dut.io.in.ready.peek().litToBoolean
          val outFire = dut.io.out.valid.peek().litToBoolean
          if (outFire) dut.io.out.bits.expect(expected(output).U)
          dut.clock.step()
          if (inFire) input += 1
          if (outFire) output += 1
        }
        input mustBe fields.length
        output mustBe expected.length
        dut.io.error.expect(false.B)
      }
    }
  }

  "Compact routed K/V metadata encoder" - {
    "must match all four software-format streams across two blocks" in {
      val count = 128
      val firstBlock = 5
      val kZero = IndexedSeq.tabulate(count)(index => -(index % 32))
      val kExponent = IndexedSeq.tabulate(count)(index => (index % 11) - 6)
      val vZero = IndexedSeq.tabulate(count)(index => -(index % 16))
      val vExponent = IndexedSeq.tabulate(count)(index => (index % 11) - 6)
      val expected = IndexedSeq(
        packFields(kZero.map(encodedSigned(_, 7)), Seq.fill(count)(7)),
        packFields(kExponent.map(encodedSigned(_, 4)), Seq.fill(count)(4)),
        packFields(vZero.map(encodedSigned(_, 5)), Seq.fill(count)(5)),
        packFields(vExponent.map(encodedSigned(_, 4)), Seq.fill(count)(4))
      )
      val random = new Random(0x4d455441454e43L)

      simulate(new CompactKvMetadataEncoder()) { dut =>
        dut.io.start.poke(false.B)
        dut.io.parameterCount.poke(count.U)
        dut.io.firstBlockIndex.poke(firstBlock.U)
        dut.io.in.valid.poke(false.B)
        dut.io.kZeroOut.ready.poke(false.B)
        dut.io.kExponentOut.ready.poke(false.B)
        dut.io.vZeroOut.ready.poke(false.B)
        dut.io.vExponentOut.ready.poke(false.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var inputIndex = 0
        val outputIndices = Array.fill(4)(0)
        var cycles = 0
        var done = false
        while (!done && cycles < 50000) {
          val offer = inputIndex < count && random.nextBoolean()
          val driven = math.min(inputIndex, count - 1)
          dut.io.in.valid.poke(offer.B)
          dut.io.in.bits.kZeroPoint.poke(kZero(driven).S)
          dut.io.in.bits.kExponent.poke(kExponent(driven).S)
          dut.io.in.bits.vZeroPoint.poke(vZero(driven).S)
          dut.io.in.bits.vExponent.poke(vExponent(driven).S)
          dut.io.in.bits.tokenTag.poke((1000 + driven).U)
          dut.io.in.bits.originalTokenIndex.poke((driven % 64).U)
          dut.io.in.bits.routedTokenIndex.poke((driven % 64).U)
          dut.io.in.bits.bucketId.poke(((driven % 64) / 16).U)
          dut.io.in.bits.blockIndex.poke((firstBlock + driven / 64).U)
          dut.io.in.bits.last.poke((driven % 64 == 63).B)

          val accepts = IndexedSeq.fill(4)(random.nextBoolean())
          dut.io.kZeroOut.ready.poke(accepts(0).B)
          dut.io.kExponentOut.ready.poke(accepts(1).B)
          dut.io.vZeroOut.ready.poke(accepts(2).B)
          dut.io.vExponentOut.ready.poke(accepts(3).B)
          val valids = IndexedSeq(
            dut.io.kZeroOut.valid.peek().litToBoolean,
            dut.io.kExponentOut.valid.peek().litToBoolean,
            dut.io.vZeroOut.valid.peek().litToBoolean,
            dut.io.vExponentOut.valid.peek().litToBoolean
          )
          val bytes = IndexedSeq(
            dut.io.kZeroOut.bits.peek().litValue.toInt,
            dut.io.kExponentOut.bits.peek().litValue.toInt,
            dut.io.vZeroOut.bits.peek().litValue.toInt,
            dut.io.vExponentOut.bits.peek().litValue.toInt
          )
          val inputFire = offer && dut.io.in.ready.peek().litToBoolean
          for (stream <- 0 until 4 if accepts(stream) && valids(stream)) {
            bytes(stream) mustBe expected(stream)(outputIndices(stream))
          }
          dut.clock.step()
          if (inputFire) inputIndex += 1
          for (stream <- 0 until 4 if accepts(stream) && valids(stream)) {
            outputIndices(stream) += 1
          }
          done = dut.io.done.peek().litToBoolean
          cycles += 1
        }

        done mustBe true
        inputIndex mustBe count
        for (stream <- 0 until 4) outputIndices(stream) mustBe expected(stream).length
        dut.io.error.expect(false.B)
        dut.io.kZeroStats.outputBytes.expect(expected(0).length.U)
        dut.io.kExponentStats.outputBytes.expect(expected(1).length.U)
        dut.io.vZeroStats.outputBytes.expect(expected(2).length.U)
        dut.io.vExponentStats.outputBytes.expect(expected(3).length.U)
      }
    }
  }

  "Bucket count header encoder" - {
    "must independently align every 21-bit block header" in {
      val records = IndexedSeq(
        IndexedSeq(16, 16, 16, 16),
        IndexedSeq(0, 1, 30, 33),
        IndexedSeq(64, 0, 0, 0)
      )
      val expected = records.flatMap { counts =>
        packFields(counts.take(3), Seq(7, 7, 7))
      }
      val random = new Random(0x4255434b454e43L)
      simulate(new BucketCountEncoder()) { dut =>
        dut.io.start.poke(false.B)
        dut.io.blockCount.poke(records.length.U)
        dut.io.firstBlockIndex.poke(7.U)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        var input = 0
        var output = 0
        var done = false
        while (!done) {
          val offer = input < records.length && random.nextBoolean()
          val driven = math.min(input, records.length - 1)
          dut.io.in.valid.poke(offer.B)
          for (bucket <- 0 until 4) {
            dut.io.in.bits.counts(bucket).poke(records(driven)(bucket).U)
          }
          dut.io.in.bits.blockIndex.poke((7 + driven).U)
          dut.io.in.bits.last.poke((driven == records.length - 1).B)
          val accept = random.nextBoolean()
          dut.io.out.ready.poke(accept.B)
          val inputFire = offer && dut.io.in.ready.peek().litToBoolean
          val outputFire = accept && dut.io.out.valid.peek().litToBoolean
          if (outputFire) dut.io.out.bits.expect(expected(output).U)
          dut.clock.step()
          if (inputFire) input += 1
          if (outputFire) output += 1
          done = dut.io.done.peek().litToBoolean
        }
        input mustBe records.length
        output mustBe expected.length
        dut.io.error.expect(false.B)
        dut.io.stats.outputBytes.expect(expected.length.U)
      }
    }

    "must reject an invalid occupancy sum without emitting a header" in {
      simulate(new BucketCountEncoder()) { dut =>
        dut.io.start.poke(false.B)
        dut.io.blockCount.poke(1.U)
        dut.io.firstBlockIndex.poke(0.U)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        dut.io.in.valid.poke(true.B)
        Seq(20, 20, 20, 20).zipWithIndex.foreach { case (count, bucket) =>
          dut.io.in.bits.counts(bucket).poke(count.U)
        }
        dut.io.in.bits.blockIndex.poke(0.U)
        dut.io.in.bits.last.poke(true.B)
        dut.clock.step()
        dut.io.in.valid.poke(false.B)
        dut.io.done.expect(true.B)
        dut.io.error.expect(true.B)
        dut.io.out.valid.expect(false.B)
        dut.io.stats.outputBytes.expect(0.U)
      }
    }
  }
}
