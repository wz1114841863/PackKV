package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class BriskKvAttentionTopSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private val PackTokens = 16
  private val OutputMaximum = (1 << 17) - 1
  private val OutputMinimum = -(1 << 17)

  private def quantizeQ21ToQ6(value: BigInt): Int = {
    val roundedMagnitude = (value.abs + (BigInt(1) << 14)) >> 15
    val rounded = if (value.signum < 0) -roundedMagnitude else roundedMagnitude
    rounded.max(OutputMinimum).min(OutputMaximum).toInt
  }

  private def fixedWeights(
    logits: IndexedSeq[Long],
    featureDim: Int
  ): IndexedSeq[Int] = {
    val scale = Math.round((1L << 18).toDouble / Math.sqrt(featureDim.toDouble))
    val scaled = logits.map { value =>
      val product = BigInt(value) * scale
      val rounded = (product.abs + (BigInt(1) << 17)) >> 18
      if (product.signum < 0) -rounded.toLong else rounded.toLong
    }
    val maximum = scaled.max
    val exponent = scaled.map { value =>
      val index = Math.min(128, ((maximum - value + 128) >> 8).toInt)
      Math.round(Math.exp(-index.toDouble / 16.0) * (1L << 16))
    }
    val sum = exponent.sum
    val reciprocal = ((BigInt(1) << 32) + sum / 2) / sum
    exponent.map { value =>
      (((BigInt(value) * reciprocal + (BigInt(1) << 16)) >> 17)
        .min(BigInt(1) << 15)).toInt
    }
  }

  "Q21 to Q6 output quantizer" - {
    "must round symmetrically and saturate both signed boundaries" in {
      val inputs = IndexedSeq[BigInt](
        0,
        BigInt(1) << 14,
        -(BigInt(1) << 14),
        BigInt(OutputMaximum) << 15,
        BigInt(OutputMaximum + 1) << 15,
        BigInt(OutputMinimum) << 15,
        BigInt(OutputMinimum - 1) << 15
      )
      val expected = inputs.map(quantizeQ21ToQ6)
      val random = new Random(0x5155414e54L)

      simulate(new AvOutputQuantizer(maximumFeatureDim = 8)) { dut =>
        dut.io.start.poke(false.B)
        dut.io.featureDim.poke(inputs.length.U)
        dut.io.in.valid.poke(false.B)
        dut.io.in.bits.value.poke(0.S)
        dut.io.in.bits.featureIndex.poke(0.U)
        dut.io.in.bits.last.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        var inputIndex = 0
        var outputIndex = 0
        var cycles = 0
        while (outputIndex < inputs.length && cycles < 500) {
          val offer = inputIndex < inputs.length && random.nextBoolean()
          val accept = random.nextBoolean()
          val driven = Math.min(inputIndex, inputs.length - 1)
          dut.io.in.valid.poke(offer.B)
          dut.io.in.bits.value.poke(inputs(driven).S)
          dut.io.in.bits.featureIndex.poke(driven.U)
          dut.io.in.bits.last.poke((driven == inputs.length - 1).B)
          dut.io.out.ready.poke(accept.B)
          val inputFire = offer && dut.io.in.ready.peek().litToBoolean
          val outputFire = accept && dut.io.out.valid.peek().litToBoolean
          if (outputFire) {
            dut.io.out.bits.featureIndex.expect(outputIndex.U)
            dut.io.out.bits.value.expect(expected(outputIndex).S)
            dut.io.out.bits.last.expect((outputIndex == inputs.length - 1).B)
          }
          dut.clock.step()
          if (inputFire) inputIndex += 1
          if (outputFire) outputIndex += 1
          cycles += 1
        }

        outputIndex mustBe inputs.length
        dut.io.error.expect(false.B)
        dut.io.stats.positiveSaturations.expect(1.U)
        dut.io.stats.negativeSaturations.expect(1.U)
      }
    }
  }

  "Unified BRISK-KV attention top" - {
    "must match every compressed Golden Vector through Q6 output under backpressure" in {
      val normalCases = IndexedSeq(
        "directed_nonidentity",
        "directed_width0",
        "random_seed_20260809"
      )
      normalCases.foreach { caseName =>
      val tokenCount = GoldenVectorLoader.bitpackInt(
        caseName,
        "k",
        "token_count"
      )
      val featureDim = 4
      val packCount = tokenCount / PackTokens
      val blockCount = (tokenCount + 63) / 64
      val descriptorCount = packCount * featureDim
      val query = GoldenVectorLoader.int32LittleEndian(
        caseName,
        "qk_query_q6_i32.bin"
      )
      val logits = GoldenVectorLoader.int64LittleEndian(
        caseName,
        "expected_qk_logits_q12_i64.bin"
      )
      val weights = fixedWeights(logits, featureDim)
      val vValues = GoldenVectorLoader.float32LittleEndian(
        caseName,
        "expected_v_dequant_f32.bin"
      ).map(value => Math.round(value * 64.0f).toLong)
      val expectedOutput = IndexedSeq.tabulate(featureDim) { feature =>
        val q21 = (0 until tokenCount).foldLeft(BigInt(0)) { (sum, token) =>
          sum + BigInt(weights(token)) * vValues(token * featureDim + feature)
        }
        quantizeQ21ToQ6(q21)
      }
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
      val random = new Random(0x4154544e544f50L ^ caseName.hashCode.toLong)

      simulate(
        new BriskKvAttentionTop(
          maximumFeatureDim = 8,
          maximumTokens = tokenCount
        )
      ) { dut =>
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
        dut.io.bucketOut.ready.poke(false.B)
        dut.io.attentionOut.ready.poke(false.B)
        dut.io.result.ready.poke(false.B)
        dut.clock.step()

        dut.io.command.bits.tag.poke(73.U)
        dut.io.command.bits.tokenCount.poke(tokenCount.U)
        dut.io.command.bits.featureDim.poke(featureDim.U)
        dut.io.command.bits.descriptorCount.poke(descriptorCount.U)
        dut.io.command.bits.kPayloadByteCount.poke(streams(2).length.U)
        dut.io.command.bits.vPayloadByteCount.poke(streams(7).length.U)
        dut.io.command.valid.poke(true.B)
        dut.io.command.ready.expect(true.B)
        dut.clock.step()
        dut.io.command.valid.poke(false.B)

        val streamIndices = Array.fill(streams.length)(0)
        var queryIndex = 0
        var bucketRecords = 0
        var outputFeature = 0
        var resultSeen = false
        var cycles = 0
        val cycleLimit = 20000
        while (!resultSeen && cycles < cycleLimit) {
          val offerQuery = queryIndex < query.length && random.nextBoolean()
          dut.io.queryLoadIn.valid.poke(offerQuery.B)
          dut.io.queryLoadIn.bits.poke(
            (if (queryIndex < query.length) query(queryIndex) else 0).S
          )
          val offers = streams.indices.map { index =>
            streamIndices(index) < streams(index).length && random.nextBoolean()
          }
          streams.indices.foreach { index =>
            inputValids(index).poke(offers(index).B)
            inputBits(index).poke(
              (if (streamIndices(index) < streams(index).length)
                 streams(index)(streamIndices(index))
               else 0).U
            )
          }
          val acceptBucket = random.nextBoolean()
          val acceptOutput = cycles > 150 && random.nextBoolean()
          dut.io.bucketOut.ready.poke(acceptBucket.B)
          dut.io.attentionOut.ready.poke(acceptOutput.B)
          dut.io.result.ready.poke(false.B)

          val queryFire = offerQuery &&
            dut.io.queryLoadIn.ready.peek().litToBoolean
          val inputFires = streams.indices.map { index =>
            offers(index) && inputReadies(index).peek().litToBoolean
          }
          val bucketFire = acceptBucket &&
            dut.io.bucketOut.valid.peek().litToBoolean
          val outputFire = acceptOutput &&
            dut.io.attentionOut.valid.peek().litToBoolean
          if (outputFire) {
            dut.io.attentionOut.bits.featureIndex.expect(outputFeature.U)
            dut.io.attentionOut.bits.value.expect(expectedOutput(outputFeature).S)
            dut.io.attentionOut.bits.last.expect((outputFeature == featureDim - 1).B)
          }
          if (outputFeature < featureDim) {
            dut.io.result.valid.expect(false.B)
          }

          dut.clock.step()
          if (queryFire) queryIndex += 1
          streams.indices.foreach { index =>
            if (inputFires(index)) streamIndices(index) += 1
          }
          if (bucketFire) bucketRecords += 1
          if (outputFire) outputFeature += 1

          if (dut.io.result.valid.peek().litToBoolean) {
            outputFeature mustBe featureDim
            bucketRecords mustBe blockCount
            dut.io.result.bits.tag.expect(73.U)
            dut.io.result.bits.error.expect(false.B)
            dut.io.progress.scaleSoftmax.softmax.outputPackets.expect(packCount.U)
            dut.io.progress.decompressionQk.qk.queryReplay.loadedValues.expect(
              featureDim.U
            )
            dut.io.progress.decompressionQk.qk.queryReplay.replayedValues.expect(
              descriptorCount.U
            )
            dut.io.progress.decompressionQk.qk.accumulator.inputPackets.expect(
              descriptorCount.U
            )
            dut.io.progress.decompressionQk.qk.accumulator.outputPackets.expect(
              packCount.U
            )
            dut.io.progress.decompressionQk.qk.accumulator.macOperations.expect(
              (tokenCount * featureDim).U
            )
            dut.io.progress.scaleSoftmax.scaling.inputPackets.expect(packCount.U)
            dut.io.progress.scaleSoftmax.scaling.outputPackets.expect(packCount.U)
            dut.io.progress.scaleSoftmax.softmax.inputPackets.expect(packCount.U)
            dut.io.progress.scaleSoftmax.softmax.exponentPackets.expect(packCount.U)
            dut.io.progress.softmaxV.vBuffer.loadedPackets.expect(descriptorCount.U)
            dut.io.progress.softmaxV.vBuffer.readRequests.expect(descriptorCount.U)
            dut.io.progress.softmaxV.vBuffer.readResponses.expect(descriptorCount.U)
            dut.io.progress.softmaxV.accumulator.loadedWeightPackets.expect(packCount.U)
            dut.io.progress.softmaxV.accumulator.vReadRequests.expect(descriptorCount.U)
            dut.io.progress.softmaxV.accumulator.vReadResponses.expect(descriptorCount.U)
            dut.io.progress.softmaxV.accumulator.outputFeatures.expect(featureDim.U)
            dut.io.progress.outputQuantizer.inputFeatures.expect(featureDim.U)
            dut.io.progress.outputQuantizer.outputFeatures.expect(featureDim.U)
            dut.io.result.ready.poke(true.B)
            dut.clock.step()
            resultSeen = true
          }
          cycles += 1
        }

        val completionState =
          s"cycles=$cycles busy=${dut.io.busy.peek().litToBoolean} " +
            s"queryLoaded=${dut.io.queryLoaded.peek().litToBoolean} " +
            s"vLoaded=${dut.io.vLoaded.peek().litToBoolean} " +
            s"buckets=$bucketRecords outputs=$outputFeature " +
            s"streams=${streamIndices.mkString("[", ",", "]")} " +
            s"kDeqValues=${dut.io.progress.decompressionQk.compute.decompression.k.completedValues.peek().litValue} " +
            s"vDeqValues=${dut.io.progress.decompressionQk.compute.decompression.v.completedValues.peek().litValue} " +
            s"kDeqDesc=${dut.io.progress.decompressionQk.compute.decompression.k.completedDescriptors.peek().litValue} " +
            s"vDeqDesc=${dut.io.progress.decompressionQk.compute.decompression.v.completedDescriptors.peek().litValue} " +
            s"kPktIn=${dut.io.progress.decompressionQk.compute.kPacketizer.inputValues.peek().litValue} " +
            s"vPktIn=${dut.io.progress.decompressionQk.compute.vPacketizer.inputValues.peek().litValue} " +
            s"kPktOut=${dut.io.progress.decompressionQk.compute.kPacketizer.outputPackets.peek().litValue} " +
            s"vPktOut=${dut.io.progress.decompressionQk.compute.vPacketizer.outputPackets.peek().litValue} " +
            s"qkOut=${dut.io.progress.decompressionQk.qk.accumulator.outputPackets.peek().litValue} " +
            s"softmaxIn=${dut.io.progress.scaleSoftmax.softmax.inputPackets.peek().litValue} " +
            s"softmaxOut=${dut.io.progress.scaleSoftmax.softmax.outputPackets.peek().litValue} " +
            s"vPackets=${dut.io.progress.softmaxV.vBuffer.loadedPackets.peek().litValue} " +
            s"avOut=${dut.io.progress.softmaxV.accumulator.outputFeatures.peek().litValue}"
        withClue(s"$caseName completion ($completionState): ") {
          resultSeen mustBe true
        }
        withClue(s"$caseName query input: ") {
          queryIndex mustBe featureDim
        }
        withClue(s"$caseName output features: ") {
          outputFeature mustBe featureDim
        }
        streams.indices.foreach { index =>
          withClue(s"$caseName stream[$index]: ") {
            streamIndices(index) mustBe streams(index).length
          }
        }
        dut.io.progress.softmaxV.accumulator.macOperations.expect(
          (tokenCount * featureDim).U
        )
      }
      }
    }

    "must reject a command beyond the unified token capacity" in {
      simulate(
        new BriskKvAttentionTop(
          maximumFeatureDim = 8,
          maximumTokens = 64
        )
      ) { dut =>
        dut.io.command.valid.poke(false.B)
        dut.io.command.bits.tag.poke(91.U)
        dut.io.command.bits.tokenCount.poke(65.U)
        dut.io.command.bits.featureDim.poke(4.U)
        dut.io.command.bits.descriptorCount.poke(20.U)
        dut.io.command.bits.kPayloadByteCount.poke(0.U)
        dut.io.command.bits.vPayloadByteCount.poke(0.U)
        dut.io.queryLoadIn.valid.poke(false.B)
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
        dut.io.bucketOut.ready.poke(true.B)
        dut.io.attentionOut.ready.poke(true.B)
        dut.io.result.ready.poke(false.B)
        dut.clock.step()
        dut.io.command.valid.poke(true.B)
        dut.io.command.ready.expect(true.B)
        dut.clock.step()
        dut.io.command.valid.poke(false.B)
        dut.io.result.valid.expect(true.B)
        dut.io.result.bits.tag.expect(91.U)
        dut.io.result.bits.error.expect(true.B)
        dut.io.queryLoadIn.ready.expect(false.B)
        dut.io.result.ready.poke(true.B)
        dut.clock.step()
        dut.io.busy.expect(false.B)
      }
    }
  }
}
