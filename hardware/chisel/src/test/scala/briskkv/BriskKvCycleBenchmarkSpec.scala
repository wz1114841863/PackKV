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
  val qkActiveCycles = UInt(64.W)
  val softmaxActiveCycles = UInt(64.W)
  val avActiveCycles = UInt(64.W)
  val softmaxOutputPackets = UInt(64.W)
  val vStreamStart = Bool()
  val vPacketizerInputValues = UInt(64.W)
  val vPacketizerOutputPackets = UInt(64.W)
  val vPacketizerDownstreamStallCycles = UInt(64.W)
  val jitVActiveCycles = UInt(64.W)
  val jitVLoadedWeightPackets = UInt(64.W)
  val jitVAcceptedVPackets = UInt(64.W)
  val jitVOutputFeatures = UInt(64.W)
  val jitVMacOperations = UInt(64.W)
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
  require(
    Set("full_v", "jit_v_dual", "jit_v_shared", "jit_v_shared_writer_cg")
      .contains(architecture)
  )
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
    io.progress.qkActiveCycles :=
      dut.io.attentionProgress.decompressionQk.qk.accumulator.activeCycles
    io.progress.softmaxActiveCycles :=
      dut.io.attentionProgress.scaleSoftmax.softmax.activeCycles
    io.progress.avActiveCycles :=
      dut.io.attentionProgress.softmaxV.accumulator.activeCycles
    io.progress.softmaxOutputPackets :=
      dut.io.attentionProgress.scaleSoftmax.softmax.outputPackets
    // Full-V launches both component streams with the attention transaction.
    io.progress.vStreamStart := io.attentionStart && io.attentionReady
    io.progress.vPacketizerInputValues := 0.U
    io.progress.vPacketizerOutputPackets := 0.U
    io.progress.vPacketizerDownstreamStallCycles := 0.U
    io.progress.jitVActiveCycles := 0.U
    io.progress.jitVLoadedWeightPackets := 0.U
    io.progress.jitVAcceptedVPackets := 0.U
    io.progress.jitVOutputFeatures := 0.U
    io.progress.jitVMacOperations := 0.U
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
        sharedDecompressor = architecture.startsWith("jit_v_shared"),
        gateWriterClock = architecture == "jit_v_shared_writer_cg"
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
    io.progress.qkActiveCycles :=
      dut.io.attentionProgress.qk.accumulator.activeCycles
    io.progress.softmaxActiveCycles :=
      dut.io.attentionProgress.scaleSoftmax.softmax.activeCycles
    io.progress.avActiveCycles := dut.io.attentionProgress.jitV.activeCycles
    io.progress.softmaxOutputPackets :=
      dut.io.attentionProgress.scaleSoftmax.softmax.outputPackets
    io.progress.vStreamStart := dut.io.attentionProgress.vLaunched
    io.progress.vPacketizerInputValues :=
      dut.io.attentionProgress.vPacketizer.inputValues
    io.progress.vPacketizerOutputPackets :=
      dut.io.attentionProgress.vPacketizer.outputPackets
    io.progress.vPacketizerDownstreamStallCycles :=
      dut.io.attentionProgress.vPacketizer.downstreamStallCycles
    io.progress.jitVActiveCycles :=
      dut.io.attentionProgress.jitV.activeCycles
    io.progress.jitVLoadedWeightPackets :=
      dut.io.attentionProgress.jitV.loadedWeightPackets
    io.progress.jitVAcceptedVPackets :=
      dut.io.attentionProgress.jitV.acceptedVPackets
    io.progress.jitVOutputFeatures :=
      dut.io.attentionProgress.jitV.outputFeatures
    io.progress.jitVMacOperations :=
      dut.io.attentionProgress.jitV.macOperations
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
  private val BackpressureModes = sys.props
    .getOrElse("briskkv.benchmarkBackpressure", "none,periodic")
    .split(",")
    .map(_.trim)
    .toSeq
  require(BackpressureModes.nonEmpty)
  require(BackpressureModes.forall(Set("none", "periodic").contains))
  private val Cases = for {
    tokens <- TokenCounts
    backpressure <- BackpressureModes
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
    kActiveCycles: Long,
    qkActiveCycles: Long,
    softmaxActiveCycles: Long,
    avActiveCycles: Long,
    kOutputValues: Long,
    kMetadataStallCycles: Long,
    kDownstreamStallCycles: Long,
    vActiveCycles: Long,
    vOutputValues: Long,
    vMetadataStallCycles: Long,
    vDownstreamStallCycles: Long,
    vPacketizerInputValues: Long,
    vPacketizerOutputPackets: Long,
    vPacketizerDownstreamStallCycles: Long,
    jitVActiveCycles: Long,
    jitVLoadedWeightPackets: Long,
    jitVAcceptedVPackets: Long,
    jitVOutputFeatures: Long,
    jitVMacOperations: Long,
    vInputStallCycles: Long,
    jitVDownstreamStallCycles: Long,
    externalOutputStallCycles: Long,
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
        var kActiveCycles = -1L
        var qkActiveCycles = -1L
        var softmaxActiveCycles = -1L
        var avActiveCycles = -1L
        var kOutputValues = -1L
        var kMetadataStallCycles = -1L
        var kDownstreamStallCycles = -1L
        var vActiveCycles = -1L
        var vOutputValues = -1L
        var vMetadataStallCycles = -1L
        var vDownstreamStallCycles = -1L
        var vPacketizerInputValues = -1L
        var vPacketizerOutputPackets = -1L
        var vPacketizerDownstreamStallCycles = -1L
        var jitVActiveCycles = -1L
        var jitVLoadedWeightPackets = -1L
        var jitVAcceptedVPackets = -1L
        var jitVOutputFeatures = -1L
        var jitVMacOperations = -1L
        var jitVInputStallCycles = -1L
        var jitVDownstreamStallCycles = -1L
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
          // `cycle` starts at one here, while the VCS power testbench's
          // attention counter starts at zero. Subtract one so the periodic
          // ready pattern is phase-aligned with the archived SAIF workload.
          val outputReady = backpressure == "none" || (cycle - 1) % 4 == 0
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
            kActiveCycles =
              dut.io.result.bits.kStats.activeCycles.peek().litValue.toLong
            qkActiveCycles =
              dut.io.progress.qkActiveCycles.peek().litValue.toLong
            softmaxActiveCycles =
              dut.io.progress.softmaxActiveCycles.peek().litValue.toLong
            avActiveCycles =
              dut.io.progress.avActiveCycles.peek().litValue.toLong
            kOutputValues =
              dut.io.result.bits.kStats.outputValues.peek().litValue.toLong
            kMetadataStallCycles =
              dut.io.result.bits.kStats.metadataStallCycles.peek().litValue.toLong
            kDownstreamStallCycles =
              dut.io.result.bits.kStats.downstreamStallCycles.peek().litValue.toLong
            vActiveCycles =
              dut.io.result.bits.vStats.activeCycles.peek().litValue.toLong
            vOutputValues =
              dut.io.result.bits.vStats.outputValues.peek().litValue.toLong
            vMetadataStallCycles =
              dut.io.result.bits.vStats.metadataStallCycles.peek().litValue.toLong
            vDownstreamStallCycles =
              dut.io.result.bits.vStats.downstreamStallCycles.peek().litValue.toLong
            vPacketizerInputValues =
              dut.io.progress.vPacketizerInputValues.peek().litValue.toLong
            vPacketizerOutputPackets =
              dut.io.progress.vPacketizerOutputPackets.peek().litValue.toLong
            vPacketizerDownstreamStallCycles =
              dut.io.progress.vPacketizerDownstreamStallCycles.peek().litValue.toLong
            jitVActiveCycles =
              dut.io.progress.jitVActiveCycles.peek().litValue.toLong
            jitVLoadedWeightPackets =
              dut.io.progress.jitVLoadedWeightPackets.peek().litValue.toLong
            jitVAcceptedVPackets =
              dut.io.progress.jitVAcceptedVPackets.peek().litValue.toLong
            jitVOutputFeatures =
              dut.io.progress.jitVOutputFeatures.peek().litValue.toLong
            jitVMacOperations =
              dut.io.progress.jitVMacOperations.peek().litValue.toLong
            jitVInputStallCycles =
              dut.io.progress.vInputStallCycles.peek().litValue.toLong
            jitVDownstreamStallCycles =
              dut.io.progress.outputStallCycles.peek().litValue.toLong
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
        kActiveCycles must be >= 0L
        vActiveCycles must be >= 0L
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
          kActiveCycles,
          qkActiveCycles,
          softmaxActiveCycles,
          avActiveCycles,
          kOutputValues,
          kMetadataStallCycles,
          kDownstreamStallCycles,
          vActiveCycles,
          vOutputValues,
          vMetadataStallCycles,
          vDownstreamStallCycles,
          vPacketizerInputValues,
          vPacketizerOutputPackets,
          vPacketizerDownstreamStallCycles,
          jitVActiveCycles,
          jitVLoadedWeightPackets,
          jitVAcceptedVPackets,
          jitVOutputFeatures,
          jitVMacOperations,
          jitVInputStallCycles,
          jitVDownstreamStallCycles,
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
        "k_active_cycles,qk_active_cycles,softmax_active_cycles,av_active_cycles," +
        "k_output_values,k_metadata_stall_cycles," +
        "k_downstream_stall_cycles,v_active_cycles,v_output_values," +
        "v_metadata_stall_cycles,v_downstream_stall_cycles," +
        "v_packetizer_input_values,v_packetizer_output_packets," +
        "v_packetizer_downstream_stall_cycles,jit_v_active_cycles," +
        "jit_v_loaded_weight_packets,jit_v_accepted_v_packets," +
        "jit_v_output_features,jit_v_mac_operations," +
        "v_input_stall_cycles,jit_v_downstream_stall_cycles," +
        "external_output_stall_cycles,output_count,output_checksum\n"
    Files.writeString(
      output,
      header + rows.map(_.csv).mkString("\n") + "\n",
      StandardCharsets.UTF_8
    )
    println(s"BRISK-KV cycle benchmark: $output")
  }
}
