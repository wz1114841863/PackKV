package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import scala.util.Random

class MetadataDequantizerHarness(buffered: Boolean) extends Module {
  private val packTokens = 16
  private val tokenIndexBits = 4
  val io = IO(
    new PackMetadataDequantizerIO(
      outputBits = 18,
      metadataBits = 8,
      descriptorIndexBits = 32,
      tokenIndexBits = tokenIndexBits,
      countBits = 32
    )
  )

  private val implementation = if (buffered) {
    Module(new BufferedPackMetadataDequantizer(6, 7, packTokens))
  } else {
    Module(new PackMetadataDequantizer(6, 7, packTokens))
  }

  implementation.io.start := io.start
  implementation.io.tokenCount := io.tokenCount
  implementation.io.descriptorCount := io.descriptorCount
  implementation.io.featureDim := io.featureDim
  implementation.io.qIn <> io.qIn
  implementation.io.zeroIn <> io.zeroIn
  implementation.io.exponentIn <> io.exponentIn
  io.out <> implementation.io.out
  io.busy := implementation.io.busy
  io.done := implementation.io.done
  io.error := implementation.io.error
  io.stats := implementation.io.stats
}

class BufferedPackMetadataDequantizerSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private case class RunResult(
    outputs: IndexedSeq[Int],
    activeCycles: Long,
    outputValues: Long,
    metadataStallCycles: Long,
    downstreamStallCycles: Long
  )

  private def run(
    buffered: Boolean,
    tokenCount: Int,
    featureDim: Int,
    randomBackpressure: Boolean
  ): RunResult = {
    val packTokens = 16
    val packCount = (tokenCount + packTokens - 1) / packTokens
    val descriptorCount = packCount * featureDim
    val zeros = IndexedSeq.tabulate(tokenCount)(index => (index % 7) - 3)
    val exponents = IndexedSeq.tabulate(tokenCount)(index => (index % 11) - 6)
    val random = new Random(0x50494e47L + tokenCount + featureDim)
    var result: Option[RunResult] = None

    simulate(new MetadataDequantizerHarness(buffered)) { dut =>
      dut.io.start.poke(false.B)
      dut.io.tokenCount.poke(tokenCount.U)
      dut.io.descriptorCount.poke(descriptorCount.U)
      dut.io.featureDim.poke(featureDim.U)
      dut.io.qIn.valid.poke(false.B)
      dut.io.qIn.bits.value.poke(0.S)
      dut.io.qIn.bits.descriptorIndex.poke(0.U)
      dut.io.qIn.bits.tokenIndex.poke(0.U)
      dut.io.qIn.bits.last.poke(false.B)
      dut.io.zeroIn.valid.poke(false.B)
      dut.io.zeroIn.bits.poke(0.S)
      dut.io.exponentIn.valid.poke(false.B)
      dut.io.exponentIn.bits.poke(0.S)
      dut.io.out.ready.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var metadataIndex = 0
      var descriptorIndex = 0
      var tokenIndex = 0
      var cycles = 0
      var done = false
      val outputs = IndexedSeq.newBuilder[Int]
      while (!done && cycles < 100000) {
        val offerMetadata = metadataIndex < tokenCount &&
          (!randomBackpressure || random.nextBoolean())
        val offerQ = descriptorIndex < descriptorCount
        val acceptOutput = !randomBackpressure || random.nextBoolean()
        val packIndex = descriptorIndex / featureDim
        val featureIndex = descriptorIndex % featureDim
        val globalToken = packIndex * packTokens + tokenIndex
        val validGlobalToken = math.min(globalToken, tokenCount - 1)
        val q = (validGlobalToken + featureIndex) % 49

        dut.io.zeroIn.valid.poke(offerMetadata.B)
        dut.io.zeroIn.bits.poke(
          (if (metadataIndex < tokenCount) zeros(metadataIndex) else 0).S
        )
        dut.io.exponentIn.valid.poke(offerMetadata.B)
        dut.io.exponentIn.bits.poke(
          (if (metadataIndex < tokenCount) exponents(metadataIndex) else 0).S
        )
        dut.io.qIn.valid.poke(offerQ.B)
        dut.io.qIn.bits.value.poke(q.S)
        dut.io.qIn.bits.descriptorIndex.poke(descriptorIndex.U)
        dut.io.qIn.bits.tokenIndex.poke(tokenIndex.U)
        dut.io.qIn.bits.last.poke(
          (descriptorIndex == descriptorCount - 1 && tokenIndex == packTokens - 1).B
        )
        dut.io.out.ready.poke(acceptOutput.B)

        val metadataFire = offerMetadata &&
          dut.io.zeroIn.ready.peek().litToBoolean &&
          dut.io.exponentIn.ready.peek().litToBoolean
        val qFire = offerQ && dut.io.qIn.ready.peek().litToBoolean
        val outputFire = acceptOutput && dut.io.out.valid.peek().litToBoolean
        if (outputFire) {
          val outputDescriptor = dut.io.out.bits.descriptorIndex.peek().litValue.toInt
          val outputToken = dut.io.out.bits.tokenIndex.peek().litValue.toInt
          val outputPack = outputDescriptor / featureDim
          val outputFeature = outputDescriptor % featureDim
          val outputGlobalToken = outputPack * packTokens + outputToken
          outputGlobalToken must be < tokenCount
          val expectedQ = (outputGlobalToken + outputFeature) % 49
          val mantissa = expectedQ + zeros(outputGlobalToken)
          val expectedRaw = mantissa << (exponents(outputGlobalToken) + 6)
          dut.io.out.bits.fixedRaw.expect(expectedRaw.S)
          dut.io.out.bits.last.expect(
            (outputGlobalToken == tokenCount - 1 && outputFeature == featureDim - 1).B
          )
          outputs += expectedRaw
        }

        dut.clock.step()
        if (metadataFire) metadataIndex += 1
        if (qFire) {
          if (tokenIndex == packTokens - 1) {
            tokenIndex = 0
            descriptorIndex += 1
          } else tokenIndex += 1
        }
        done = dut.io.done.peek().litToBoolean
        cycles += 1
      }

      done mustBe true
      metadataIndex mustBe tokenCount
      descriptorIndex mustBe descriptorCount
      dut.io.error.expect(false.B)
      result = Some(
        RunResult(
          outputs = outputs.result(),
          activeCycles = dut.io.stats.activeCycles.peek().litValue.toLong,
          outputValues = dut.io.stats.outputValues.peek().litValue.toLong,
          metadataStallCycles =
            dut.io.stats.metadataStallCycles.peek().litValue.toLong,
          downstreamStallCycles =
            dut.io.stats.downstreamStallCycles.peek().litValue.toLong
        )
      )
    }
    result.get
  }

  "Ping-pong metadata dequantizer" - {
    "must match the single-buffer output with partial-pack padding and stalls" in {
      val single = run(
        buffered = false,
        tokenCount = 35,
        featureDim = 3,
        randomBackpressure = true
      )
      val buffered = run(
        buffered = true,
        tokenCount = 35,
        featureDim = 3,
        randomBackpressure = true
      )
      buffered.outputs mustBe single.outputs
      buffered.outputValues mustBe 35L * 3L
    }

    "must remove inter-pack metadata bubbles in the cycle-level comparison" in {
      val single = run(
        buffered = false,
        tokenCount = 64,
        featureDim = 4,
        randomBackpressure = false
      )
      val buffered = run(
        buffered = true,
        tokenCount = 64,
        featureDim = 4,
        randomBackpressure = false
      )

      buffered.outputs mustBe single.outputs
      single.outputValues mustBe 256L
      buffered.outputValues mustBe 256L
      single.downstreamStallCycles mustBe 0L
      buffered.downstreamStallCycles mustBe 0L
      buffered.activeCycles must be < single.activeCycles
      buffered.metadataStallCycles must be < single.metadataStallCycles
      info(
        s"cycle comparison: single=${single.activeCycles}, " +
          s"buffered=${buffered.activeCycles}, " +
          s"single-metadata-stall=${single.metadataStallCycles}, " +
          s"buffered-metadata-stall=${buffered.metadataStallCycles}"
      )
    }
  }
}
