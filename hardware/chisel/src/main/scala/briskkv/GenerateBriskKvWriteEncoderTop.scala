package briskkv

import circt.stage.ChiselStage

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters._
import scala.util.matching.Regex

object GenerateBriskKvWriteEncoderTop {
  private final case class Config(
    targetDir: Path = Paths.get(
      "generated/briskkv_write_v1_t1024_f128_i24/full"
    ),
    maximumFeatureDim: Int = 128,
    maximumTokens: Int = 1024,
    inputBits: Int = 24,
    enableStats: Boolean = true,
    quantArchitecture: String = "v1",
    mode: String = "full"
  )

  private final case class MemoryRow(
    module: String,
    depth: Int,
    width: Int,
    instances: Int
  )

  private def usage(): String =
    """Usage: GenerateBriskKvWriteEncoderTop [options]
      |  --target-dir <path>          output directory
      |  --maximum-feature-dim <int>  maximum head dimension (default: 128)
      |  --maximum-tokens <int>       maximum full-block transaction tokens (default: 1024)
      |  --input-bits <int>           signed Q12 input width (default: 24)
      |  --enable-stats <true|false>  elaborate performance counters (default: true)
      |  --quant-architecture <v1|v2|v3> quantizer parameter microarchitecture (default: v1)
      |  --mode <full|dc_logic>       retain inferred memories or externalize them
      |""".stripMargin

  private def parse(args: List[String], config: Config = Config()): Config =
    args match {
      case Nil => config
      case "--target-dir" :: value :: tail =>
        parse(tail, config.copy(targetDir = Paths.get(value)))
      case "--maximum-feature-dim" :: value :: tail =>
        parse(tail, config.copy(maximumFeatureDim = value.toInt))
      case "--maximum-tokens" :: value :: tail =>
        parse(tail, config.copy(maximumTokens = value.toInt))
      case "--input-bits" :: value :: tail =>
        parse(tail, config.copy(inputBits = value.toInt))
      case "--enable-stats" :: value :: tail =>
        require(Set("true", "false").contains(value))
        parse(tail, config.copy(enableStats = value.toBoolean))
      case "--quant-architecture" :: value :: tail =>
        parse(tail, config.copy(quantArchitecture = value))
      case "--mode" :: value :: tail =>
        parse(tail, config.copy(mode = value))
      case "--help" :: _ =>
        println(usage())
        sys.exit(0)
      case option :: _ =>
        throw new IllegalArgumentException(
          s"Unknown or incomplete option: $option\n${usage()}"
        )
    }

  private def addressBits(depth: Int): Int =
    math.max(1, 32 - Integer.numberOfLeadingZeros(depth - 1))

  private def memoryBlackBoxStub(module: String, depth: Int, width: Int): String = {
    val addrWidth = addressBits(depth)
    s"""// BRISK-KV write-side DC logic-only SRAM stub.
       |// The behavioral SyncReadMem implementation is intentionally absent.
       |(* black_box = 1, syn_black_box = 1 *)
       |module $module(
       |  input  [${addrWidth - 1}:0] R0_addr,
       |  input                  R0_en,
       |                         R0_clk,
       |  output [${width - 1}:0] R0_data,
       |  input  [${addrWidth - 1}:0] W0_addr,
       |  input                  W0_en,
       |                         W0_clk,
       |  input  [${width - 1}:0] W0_data
       |);
       |endmodule
       |""".stripMargin
  }

  private val modulePattern: Regex =
    """(?m)^module\s+([A-Za-z_][A-Za-z0-9_$]*)\(""".r
  private val memoryPattern: Regex =
    """(?m)^\s*reg\s+\[(\d+):0\]\s+Memory\[0:(\d+)\];""".r

  private def systemVerilogFiles(targetDir: Path): IndexedSeq[Path] = {
    val stream = Files.list(targetDir)
    try stream.iterator().asScala
      .filter(path => path.getFileName.toString.endsWith(".sv"))
      .toIndexedSeq
      .sortBy(_.getFileName.toString)
    finally stream.close()
  }

  /** Discover CIRCT split-memory modules from their behavioral declarations.
    *
    * This deliberately derives the inventory from emitted RTL instead of
    * guessing FIRRTL/CIRCT module names. A changed memory geometry therefore
    * changes both the stubs and CACTI manifest in the same generation run.
    */
  private def discoverMemories(targetDir: Path): IndexedSeq[MemoryRow] = {
    val files = systemVerilogFiles(targetDir)
    val texts = files.map { path =>
      path -> Files.readString(path, StandardCharsets.UTF_8)
    }
    val allText = texts.map(_._2).mkString("\n")
    texts.flatMap { case (_, source) =>
      for {
        moduleMatch <- modulePattern.findFirstMatchIn(source)
        memoryMatch <- memoryPattern.findFirstMatchIn(source)
      } yield {
        val module = moduleMatch.group(1)
        val width = memoryMatch.group(1).toInt + 1
        val depth = memoryMatch.group(2).toInt + 1
        val instancePattern =
          ("(?m)^\\s*" + Regex.quote(module) + "\\s+[A-Za-z_][A-Za-z0-9_$]*\\s*\\(").r
        val instances = instancePattern.findAllMatchIn(allText).length
        require(instances > 0, s"No instance found for generated memory $module")
        MemoryRow(module, depth, width, instances)
      }
    }.sortBy(_.module)
  }

  def main(args: Array[String]): Unit = {
    val config = parse(args.toList)
    require(
      config.maximumFeatureDim >= 2 &&
        config.maximumFeatureDim <= 256 &&
        (config.maximumFeatureDim & (config.maximumFeatureDim - 1)) == 0
    )
    require(config.inputBits >= 3)
    require(config.maximumTokens >= BriskKvFormatV0.params.blockTokens)
    require(config.maximumTokens % BriskKvFormatV0.params.blockTokens == 0)
    require(Set("full", "dc_logic").contains(config.mode))
    val quantArchitecture =
      QuantParameterArchitecture.fromCliName(config.quantArchitecture)

    val targetDir = config.targetDir.toAbsolutePath.normalize
    Files.createDirectories(targetDir)
    ChiselStage.emitSystemVerilogFile(
      new BriskKvWriteEncoderTop(
        inputBits = config.inputBits,
        maximumFeatureDim = config.maximumFeatureDim,
        maximumTokens = config.maximumTokens,
        enableStats = config.enableStats,
        quantParameterArchitecture = quantArchitecture
      ),
      args = Array("--target-dir", targetDir.toString),
      firtoolOpts = Array("--disable-all-randomization", "--strip-debug-info")
    )

    val memoryRows = discoverMemories(targetDir)
    require(memoryRows.nonEmpty, "No write-side architectural memories discovered")
    if (config.mode == "dc_logic") {
      memoryRows.foreach { memory =>
        Files.writeString(
          targetDir.resolve(s"${memory.module}.sv"),
          memoryBlackBoxStub(memory.module, memory.depth, memory.width),
          StandardCharsets.UTF_8
        )
      }
    }

    val memoryImplementation =
      if (config.mode == "dc_logic") "bodyless_blackbox_stubs"
      else "behavioral_sync_read_mem"
    val maximumBlocks =
      config.maximumTokens / BriskKvFormatV0.params.blockTokens
    val maximumDescriptors =
      config.maximumFeatureDim *
        (BriskKvFormatV0.params.blockTokens / BriskKvFormatV0.params.packTokens)
    val microarchitectureRevision =
      if (quantArchitecture == QuantParameterArchitecture.V1SingleStage)
        "write-v1-narrow-counters-router-pipeline"
      else if (quantArchitecture == QuantParameterArchitecture.V2ThreeStage)
        "write-v2-quant-parameter-pipeline"
      else "write-v3-leading-one-exponent-selector"
    val manifest =
      s"""{
         |  "top": "BriskKvWriteEncoderTop",
         |  "mode": "${config.mode}",
         |  "microarchitecture_revision": "$microarchitectureRevision",
         |  "router_classification_pipeline": "registered-thresholds-split-count",
         |  "quant_parameter_architecture": "${quantArchitecture.cliName}",
         |  "quant_parameter_pipeline": "${quantArchitecture.manifestName}",
         |  "quant_parameter_extra_cycles_per_token": ${quantArchitecture.extraParameterCyclesPerToken},
         |  "maximum_feature_dim": ${config.maximumFeatureDim},
         |  "maximum_tokens": ${config.maximumTokens},
         |  "maximum_blocks_per_transaction": $maximumBlocks,
         |  "internal_feature_dim_bits": ${math.max(1, 32 - Integer.numberOfLeadingZeros(config.maximumFeatureDim))},
         |  "internal_feature_index_bits": ${addressBits(config.maximumFeatureDim)},
         |  "internal_token_index_bits": ${addressBits(BriskKvFormatV0.params.blockTokens)},
         |  "internal_blocks_remaining_bits": ${math.max(1, 32 - Integer.numberOfLeadingZeros(maximumBlocks))},
         |  "internal_descriptor_index_bits": ${addressBits(maximumDescriptors)},
         |  "internal_descriptors_remaining_bits": ${math.max(1, 32 - Integer.numberOfLeadingZeros(maximumDescriptors))},
         |  "internal_metadata_parameter_bits": ${math.max(1, 32 - Integer.numberOfLeadingZeros(config.maximumTokens))},
         |  "input_format": "signed Q12 / ${config.inputBits} bits",
         |  "performance_stats_enabled": ${config.enableStats},
         |  "pack_tokens": ${BriskKvFormatV0.params.packTokens},
         |  "block_tokens": ${BriskKvFormatV0.params.blockTokens},
         |  "bucket_count": ${BriskKvFormatV0.params.bucketCount},
         |  "output_component_streams": 11,
         |  "memory_implementation": "$memoryImplementation"
         |}
         |""".stripMargin
    Files.writeString(
      targetDir.resolve("manifest.json"),
      manifest,
      StandardCharsets.UTF_8
    )

    val memoryCsv =
      "module,purpose,depth,width_bits,instances,total_bits,read_ports,write_ports,readwrite_ports,access_mode\n" +
        memoryRows.map { memory =>
          val totalBits = memory.depth.toLong * memory.width * memory.instances
          s"${memory.module},write_pipeline_sram,${memory.depth},${memory.width}," +
            s"${memory.instances},$totalBits,1,1,0,parallel_width"
        }.mkString("\n") + "\n"
    Files.writeString(
      targetDir.resolve("memories.csv"),
      memoryCsv,
      StandardCharsets.UTF_8
    )
    if (config.mode == "dc_logic") {
      val memoryTcl =
        "# Auto-generated write-side SRAM modules excluded from DC logic PPA.\n" +
          "set BRISKKV_MEMORY_MODULES {\n" +
          memoryRows.map(memory => s"  ${memory.module}").mkString("\n") +
          "\n}\n"
      Files.writeString(
        targetDir.resolve("memory_modules.tcl"),
        memoryTcl,
        StandardCharsets.UTF_8
      )
    }

    println(s"Generated write-side ${config.mode} SystemVerilog in $targetDir")
    println(
      s"Discovered ${memoryRows.length} memory modules / " +
        s"${memoryRows.map(_.instances).sum} instances"
    )
    if (config.mode == "dc_logic")
      println(s"DC black-box list: ${targetDir.resolve("memory_modules.tcl")}")
  }
}
