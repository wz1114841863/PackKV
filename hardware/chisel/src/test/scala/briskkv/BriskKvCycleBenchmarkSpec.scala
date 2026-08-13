package briskkv

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.util._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

class BriskKvCycleBenchmarkProgress extends Bundle {
  val kCompletedValues = UInt(64.W)
  val vCompletedValues = UInt(64.W)
  val softmaxOutputPackets = UInt(64.W)
  val vStreamStart = Bool()
  val vInputStallCycles = UInt(64.W)
  val outputStallCycles = UInt(64.W)
}

class BriskKvCycleBenchmarkHarnessIO(
  inputBits: Int,
  valueBits: Int,
  outputBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int,
  bucketCountBits: Int
) extends Bundle {
  val writeStart = Input(Bool())
  val writeReady = Output(Bool())
  val featureDim = Input(UInt(countBits.W))
  val blockCount = Input(UInt(countBits.W))
  val firstBlockIndex = Input(UInt(countBits.W))
  val writeIn = Flipped(
    Decoupled(new BriskKvWriteInput(inputBits, countBits, tagBits, tokenIndexBits))
  )
  val attentionStart = Input(Bool())
  val attentionReady = Output(Bool())
  val attentionTag = Input(UInt(tagBits.W))
  val queryIn = Flipped(Decoupled(SInt(valueBits.W)))
  val bucketOut = Decoupled(new BucketCountRecord(bucketCountBits, countBits))
  val attentionOut = Decoupled(new AttentionOutputFeature(outputBits, countBits))
  val result = Decoupled(new DualKvDecompressionResult(countBits, tagBits))
  val writeDone = Output(Bool())
  val encodedReady = Output(Bool())
  val busy = Output(Bool())
  val error = Output(Bool())
  val progress = Output(new BriskKvCycleBenchmarkProgress)
}

/** Test-only common interface for cycle-comparable Full-V/JIT-V runs. */
class BriskKvCycleBenchmarkHarness(
  architecture: String,
  maximumFeatureDim: Int = 128,
  maximumTokens: Int = 1024,
  inputBits: Int = 24,
  valueBits: Int = 18,
  outputBits: Int = 18,
  countBits: Int = 32,
  tagBits: Int = 16
) extends Module {
  require(Set("full_v", "jit_v_dual", "jit_v_shared").contains(architecture))
  private val params = BriskKvFormatV0.params
  private val tokenIndexBits = log2Ceil(params.blockTokens)
  private val bucketCountBits = log2Ceil(params.blockTokens + 1)
  val io = IO(
    new BriskKvCycleBenchmarkHarnessIO(
      inputBits,
      valueBits,
      outputBits,
      countBits,
      tagBits,
      tokenIndexBits,
      bucketCountBits
    )
  )

  if (architecture == "full_v") {
    val dut = Module(
      new BriskKvSingleHeadTileTop(
        inputBits = inputBits,
        valueBits = valueBits,
        outputBits = outputBits,
        countBits = countBits,
        tagBits = tagBits,
        maximumFeatureDim = maximumFeatureDim,
        maximumTokens = maximumTokens,
        enableStats = true
      )
    )
    dut.io.writeStart := io.writeStart
    io.writeReady := dut.io.writeReady
    dut.io.featureDim := io.featureDim
    dut.io.blockCount := io.blockCount
    dut.io.firstBlockIndex := io.firstBlockIndex
    dut.io.writeIn <> io.writeIn
    dut.io.attentionStart := io.attentionStart
    io.attentionReady := dut.io.attentionReady
    dut.io.attentionTag := io.attentionTag
    dut.io.queryIn <> io.queryIn
    io.bucketOut <> dut.io.bucketOut
    io.attentionOut <> dut.io.attentionOut
    io.result <> dut.io.result
    io.writeDone := dut.io.writeDone
    io.encodedReady := dut.io.encodedReady
    io.busy := dut.io.busy
    io.error := dut.io.error
    io.progress.kCompletedValues :=
      dut.io.attentionProgress.decompressionQk.compute.decompression.k.completedValues
    io.progress.vCompletedValues :=
      dut.io.attentionProgress.decompressionQk.compute.decompression.v.completedValues
    io.progress.softmaxOutputPackets :=
      dut.io.attentionProgress.scaleSoftmax.softmax.outputPackets
    // Full-V launches both component streams with the attention transaction.
    io.progress.vStreamStart := io.attentionStart && io.attentionReady
    io.progress.vInputStallCycles :=
      dut.io.attentionProgress.softmaxV.vBuffer.loadStallCycles
    io.progress.outputStallCycles :=
      dut.io.attentionProgress.softmaxV.accumulator.downstreamStallCycles
  } else {
    val dut = Module(
      new BriskKvJitVSingleHeadTileTop(
        inputBits = inputBits,
        valueBits = valueBits,
        outputBits = outputBits,
        countBits = countBits,
        tagBits = tagBits,
        maximumFeatureDim = maximumFeatureDim,
        maximumTokens = maximumTokens,
        enableStats = true,
        sharedDecompressor = architecture == "jit_v_shared"
      )
    )
    dut.io.writeStart := io.writeStart
    io.writeReady := dut.io.writeReady
    dut.io.featureDim := io.featureDim
    dut.io.blockCount := io.blockCount
    dut.io.firstBlockIndex := io.firstBlockIndex
    dut.io.writeIn <> io.writeIn
    dut.io.attentionStart := io.attentionStart
    io.attentionReady := dut.io.attentionReady
    dut.io.attentionTag := io.attentionTag
    dut.io.queryIn <> io.queryIn
    io.bucketOut <> dut.io.bucketOut
    io.attentionOut <> dut.io.attentionOut
    io.result <> dut.io.result
    io.writeDone := dut.io.writeDone
    io.encodedReady := dut.io.encodedReady
    io.busy := dut.io.busy
    io.error := dut.io.error
    io.progress.kCompletedValues :=
      dut.io.attentionProgress.kDecompression.completedValues
    io.progress.vCompletedValues :=
      dut.io.attentionProgress.vDecompression.completedValues
    io.progress.softmaxOutputPackets :=
      dut.io.attentionProgress.scaleSoftmax.softmax.outputPackets
    io.progress.vStreamStart := dut.io.attentionProgress.vLaunched
    io.progress.vInputStallCycles :=
      dut.io.attentionProgress.jitV.inputStallCycles
    io.progress.outputStallCycles :=
      dut.io.attentionProgress.jitV.downstreamStallCycles
  }
}

class BriskKvCycleBenchmarkSpec
    extends AnyFreeSpec
    with Matchers
    with ChiselSim {
  private val FeatureDim = sys.props.getOrElse("briskkv.benchmarkFeatureDim", "8").toInt
  private val TokenCounts = sys.props
    .getOrElse("briskkv.benchmarkTokens", "64,256,1024")
    .split(",")
    .map(_.trim.toInt)
    .toSeq
  private val MaximumTokens = TokenCounts.max
  private val Architecture =
    sys.props.getOrElse("briskkv.benchmarkArchitecture", "full_v")
  private val Cases = for {
    tokens <- TokenCounts
    backpressure <- Seq("none", "periodic")
  } yield (tokens, backpressure)

  private case class Row(
    architecture: String,
    tokens: Int,
    featureDim: Int,
    backpressure: String,
    writeCycles: Long,
    attentionCycles: Long,
    kDoneCycle: Long,
    firstWeightCycle: Long,
    weightsLoadedCycle: Long,
    vFirstValueCycle: Long,
    vDoneCycle: Long,
    firstOutputCycle: Long,
    lastOutputCycle: Long,
    resultCycle: Long,
    vInputStallCycles: Long,
    outputStallCycles: Long,
    outputCount: Int,
    outputChecksum: BigInt
  ) {
    def csv: String = productIterator.mkString(",")
  }

  private def q12(quarterUnits: Int): Int = quarterUnits * 4096 / 4

  "cycle benchmark must complete and export a comparable CSV" in {
    val rows = collection.mutable.ArrayBuffer.empty[Row]
    simulate(
      new BriskKvCycleBenchmarkHarness(
        Architecture,
        // Match the exercised geometry to keep cycle simulation lightweight.
        // All architectural SRAMs remain one-cycle SyncReadMem, so reducing
        // unused capacity does not change the scheduled cycle count.
        maximumFeatureDim = FeatureDim,
        maximumTokens = MaximumTokens
      )
    ) { dut =>
      dut.io.writeStart.poke(false.B)
      dut.io.writeIn.valid.poke(false.B)
      dut.io.attentionStart.poke(false.B)
      dut.io.queryIn.valid.poke(false.B)
      dut.io.bucketOut.ready.poke(true.B)
      dut.io.attentionOut.ready.poke(false.B)
      dut.io.result.ready.poke(false.B)
      dut.io.firstBlockIndex.poke(0.U)
      dut.clock.step()

      Cases.zipWithIndex.foreach { case ((tokenCount, backpressure), caseIndex) =>
        dut.io.featureDim.poke(FeatureDim.U)
        dut.io.blockCount.poke((tokenCount / 64).U)
        dut.io.attentionTag.poke((200 + caseIndex).U)
        dut.io.writeStart.poke(true.B)
        dut.clock.step()
        dut.io.writeStart.poke(false.B)

        var pair = 0
        var writeCycles = 1L
        var writeDone = false
        while (!writeDone && writeCycles < 2000000L) {
          if (pair < tokenCount * FeatureDim) {
            val token = pair / FeatureDim
            val feature = pair % FeatureDim
            val kQuarter = (token + feature * 3) % 5
            val vQuarter = (token * 3 + feature * 2 + 1) % 5
            dut.io.writeIn.valid.poke(true.B)
            dut.io.writeIn.bits.kFixedRaw.poke(q12(kQuarter).S)
            dut.io.writeIn.bits.vFixedRaw.poke(q12(vQuarter).S)
            dut.io.writeIn.bits.tokenTag.poke(token.U)
            dut.io.writeIn.bits.blockIndex.poke((token / 64).U)
            dut.io.writeIn.bits.tokenIndex.poke((token % 64).U)
            dut.io.writeIn.bits.featureIndex.poke(feature.U)
            dut.io.writeIn.bits.lastFeature.poke((feature == FeatureDim - 1).B)
            dut.io.writeIn.bits.last.poke(
              (token == tokenCount - 1 && feature == FeatureDim - 1).B
            )
            if (dut.io.writeIn.ready.peek().litToBoolean) pair += 1
          } else {
            dut.io.writeIn.valid.poke(false.B)
          }
          if (dut.io.writeDone.peek().litToBoolean) writeDone = true
          dut.clock.step()
          writeCycles += 1
        }
        pair mustBe tokenCount * FeatureDim
        writeDone mustBe true
        dut.io.error.expect(false.B)
        dut.io.encodedReady.expect(true.B)

        dut.io.attentionStart.poke(true.B)
        dut.clock.step()
        dut.io.attentionStart.poke(false.B)
        var cycle = 1L
        var queryIndex = 0
        var kDoneCycle = -1L
        var firstWeightCycle = -1L
        var weightsLoadedCycle = -1L
        var vFirstValueCycle = -1L
        var vDoneCycle = -1L
        var firstOutputCycle = -1L
        var lastOutputCycle = -1L
        var resultCycle = -1L
        var outputCount = 0
        var outputChecksum = BigInt(0)
        var externalOutputStallCycles = 0L
        var kCounterArmed = false
        var softmaxCounterArmed = false
        var vCounterArmed = false
        val totalValues = tokenCount.toLong * FeatureDim
        val packCount = tokenCount / 16

        while (resultCycle < 0 && cycle < 2000000L) {
          if (queryIndex < FeatureDim) {
            dut.io.queryIn.valid.poke(true.B)
            dut.io.queryIn.bits.poke(((queryIndex * 5 + 3) % 11 - 5).S)
            if (dut.io.queryIn.ready.peek().litToBoolean) queryIndex += 1
          } else {
            dut.io.queryIn.valid.poke(false.B)
          }

          // In the periodic case accept at most one output every four cycles;
          // a held valid response is therefore guaranteed to exercise stalls.
          val outputReady = backpressure == "none" || cycle % 4 == 0
          dut.io.attentionOut.ready.poke(outputReady.B)
          dut.io.result.ready.poke(true.B)
          if (!outputReady && dut.io.attentionOut.valid.peek().litToBoolean)
            externalOutputStallCycles += 1
          val kValues = dut.io.progress.kCompletedValues.peek().litValue
          val softmaxPackets = dut.io.progress.softmaxOutputPackets.peek().litValue
          val vValues = dut.io.progress.vCompletedValues.peek().litValue
          if (kValues == 0) kCounterArmed = true
          if (softmaxPackets == 0) softmaxCounterArmed = true
          if (vValues == 0) vCounterArmed = true
          if (kDoneCycle < 0 && kCounterArmed && kValues >= totalValues)
            kDoneCycle = cycle
          if (firstWeightCycle < 0 && softmaxCounterArmed && softmaxPackets > 0)
            firstWeightCycle = cycle
          if (weightsLoadedCycle < 0 && softmaxCounterArmed &&
              softmaxPackets >= packCount)
            weightsLoadedCycle = cycle
          if (vFirstValueCycle < 0 && vCounterArmed && vValues > 0)
            vFirstValueCycle = cycle
          if (vDoneCycle < 0 && vCounterArmed && vValues >= totalValues)
            vDoneCycle = cycle
          if (outputReady && dut.io.attentionOut.valid.peek().litToBoolean) {
            if (firstOutputCycle < 0) firstOutputCycle = cycle
            lastOutputCycle = cycle
            val value = dut.io.attentionOut.bits.value.peek().litValue
            outputChecksum = outputChecksum * 131 + value
            outputCount += 1
          }
          if (dut.io.result.valid.peek().litToBoolean) {
            dut.io.result.bits.error.expect(false.B)
            resultCycle = cycle
          }
          dut.clock.step()
          cycle += 1
        }

        queryIndex mustBe FeatureDim
        outputCount mustBe FeatureDim
        kDoneCycle must be >= 0L
        firstWeightCycle must be >= 0L
        weightsLoadedCycle must be >= 0L
        vFirstValueCycle must be >= 0L
        vDoneCycle must be >= 0L
        resultCycle must be >= 0L
        rows += Row(
          Architecture,
          tokenCount,
          FeatureDim,
          backpressure,
          writeCycles,
          resultCycle,
          kDoneCycle,
          firstWeightCycle,
          weightsLoadedCycle,
          vFirstValueCycle,
          vDoneCycle,
          firstOutputCycle,
          lastOutputCycle,
          resultCycle,
          dut.io.progress.vInputStallCycles.peek().litValue.toLong,
          externalOutputStallCycles,
          outputCount,
          outputChecksum
        )
        dut.clock.step(2)
      }
    }

    val output = Paths.get(
      sys.props.getOrElse(
        "briskkv.benchmarkOutput",
        s"target/cycle_benchmark/$Architecture.csv"
      )
    ).toAbsolutePath.normalize
    Files.createDirectories(output.getParent)
    val header =
      "architecture,tokens,feature_dim,backpressure,write_cycles," +
        "attention_cycles,k_done_cycle,first_weight_cycle,weights_loaded_cycle," +
        "v_first_value_cycle," +
        "v_done_cycle,first_output_cycle,last_output_cycle,result_cycle," +
        "v_input_stall_cycles,output_stall_cycles,output_count,output_checksum\n"
    Files.writeString(
      output,
      header + rows.map(_.csv).mkString("\n") + "\n",
      StandardCharsets.UTF_8
    )
    println(s"BRISK-KV cycle benchmark: $output")
  }
}
