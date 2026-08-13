package briskkv

import circt.stage.ChiselStage

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters._
import scala.util.matching.Regex

/** SystemVerilog export for the complete single-head encode/store/attention
  * tile. Memory modules are discovered from emitted CIRCT RTL so that the DC
  * black-box list and CACTI inventory cannot silently drift from elaboration.
  */
object GenerateBriskKvSingleHeadTileTop {
  private final case class Config(
    targetDir: Path = Paths.get("generated/briskkv_single_head_tile/full"),
    maximumFeatureDim: Int = 128,
    maximumTokens: Int = 1024,
    inputBits: Int = 24,
    scaleLanes: Int = 4,
    enableStats: Boolean = false,
    quantArchitecture: String = "v1",
    mode: String = "full"
  )

  private final case class MemoryRow(
    module: String,
    depth: Int,
    width: Int,
    instances: Int
  )

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
      case "--scale-lanes" :: value :: tail =>
        parse(tail, config.copy(scaleLanes = value.toInt))
      case "--enable-stats" :: value :: tail =>
        require(Set("true", "false").contains(value))
        parse(tail, config.copy(enableStats = value.toBoolean))
      case "--quant-architecture" :: value :: tail =>
        parse(tail, config.copy(quantArchitecture = value))
      case "--mode" :: value :: tail =>
        parse(tail, config.copy(mode = value))
      case option :: _ =>
        throw new IllegalArgumentException(
          s"Unknown or incomplete option: $option"
        )
    }

  private def addressBits(depth: Int): Int =
    math.max(1, 32 - Integer.numberOfLeadingZeros(depth - 1))

  private val modulePattern: Regex =
    """(?m)^module\s+([A-Za-z_][A-Za-z0-9_$]*)\(""".r
  private val memoryPattern: Regex =
    """(?m)^\s*reg\s+\[(\d+):0\]\s+Memory\[0:(\d+)\];""".r

  private def systemVerilogFiles(targetDir: Path): IndexedSeq[Path] = {
    val stream = Files.list(targetDir)
    try stream.iterator().asScala
      .filter(_.getFileName.toString.endsWith(".sv"))
      .toIndexedSeq
      .sortBy(_.getFileName.toString)
    finally stream.close()
  }

  private def discoverMemories(targetDir: Path): IndexedSeq[MemoryRow] = {
    val texts = systemVerilogFiles(targetDir).map { path =>
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
          ("(?m)^\\s*" + Regex.quote(module) +
            "\\s+[A-Za-z_][A-Za-z0-9_$]*\\s*\\(").r
        val instances = instancePattern.findAllMatchIn(allText).length
        require(instances > 0, s"No instance found for generated memory $module")
        MemoryRow(module, depth, width, instances)
      }
    }.sortBy(_.module)
  }

  private def memoryBlackBoxStub(memory: MemoryRow): String = {
    val addrWidth = addressBits(memory.depth)
    s"""// BRISK-KV unified-tile DC logic-only SRAM stub.
       |(* black_box = 1, syn_black_box = 1 *)
       |module ${memory.module}(
       |  input  [${addrWidth - 1}:0] R0_addr,
       |  input                  R0_en,
       |                         R0_clk,
       |  output [${memory.width - 1}:0] R0_data,
       |  input  [${addrWidth - 1}:0] W0_addr,
       |  input                  W0_en,
       |                         W0_clk,
       |  input  [${memory.width - 1}:0] W0_data
       |);
       |endmodule
       |""".stripMargin
  }

  def main(args: Array[String]): Unit = {
    val config = parse(args.toList)
    require(config.maximumFeatureDim >= 2)
    require((config.maximumFeatureDim & (config.maximumFeatureDim - 1)) == 0)
    require(config.maximumTokens >= BriskKvFormatV0.params.blockTokens)
    require(config.maximumTokens % BriskKvFormatV0.params.blockTokens == 0)
    require(config.inputBits >= 3)
    require(config.scaleLanes > 0)
    require(Set("full", "dc_logic").contains(config.mode))
    val quantArchitecture =
      QuantParameterArchitecture.fromCliName(config.quantArchitecture)
    val targetDir = config.targetDir.toAbsolutePath.normalize
    Files.createDirectories(targetDir)

    ChiselStage.emitSystemVerilogFile(
      new BriskKvSingleHeadTileTop(
        inputBits = config.inputBits,
        maximumFeatureDim = config.maximumFeatureDim,
        maximumTokens = config.maximumTokens,
        scaleLanes = config.scaleLanes,
        enableStats = config.enableStats,
        quantParameterArchitecture = quantArchitecture
      ),
      args = Array("--target-dir", targetDir.toString),
      firtoolOpts = Array("--disable-all-randomization", "--strip-debug-info")
    )

    val memories = discoverMemories(targetDir)
    require(memories.nonEmpty, "No unified-tile architectural memories discovered")
    if (config.mode == "dc_logic") {
      memories.foreach { memory =>
        Files.writeString(
          targetDir.resolve(s"${memory.module}.sv"),
          memoryBlackBoxStub(memory),
          StandardCharsets.UTF_8
        )
      }
    }

    val memoryBits = memories.map { memory =>
      memory.depth.toLong * memory.width * memory.instances
    }.sum
    val manifest =
      s"""{
         |  "top": "BriskKvSingleHeadTileTop",
         |  "mode": "${config.mode}",
         |  "architecture": "phase-separated-write-store-read-attention",
         |  "stored_transactions": 1,
         |  "repeat_attention_on_resident_transaction": true,
         |  "stored_component_streams": 11,
         |  "first_block_index_supported": 0,
         |  "maximum_feature_dim": ${config.maximumFeatureDim},
         |  "maximum_tokens": ${config.maximumTokens},
         |  "input_bits": ${config.inputBits},
         |  "scale_lanes": ${config.scaleLanes},
         |  "quant_parameter_architecture": "${quantArchitecture.cliName}",
         |  "performance_stats_enabled": ${config.enableStats},
         |  "memory_module_types": ${memories.length},
         |  "memory_instances": ${memories.map(_.instances).sum},
         |  "total_inferred_memory_bits": $memoryBits,
         |  "memory_implementation": "${if (config.mode == "dc_logic") "bodyless_blackbox_stubs" else "behavioral_sync_read_mem"}"
         |}
         |""".stripMargin
    Files.writeString(
      targetDir.resolve("manifest.json"),
      manifest,
      StandardCharsets.UTF_8
    )

    val csv =
      "module,purpose,depth,width_bits,instances,total_bits,read_ports,write_ports,readwrite_ports,access_mode\n" +
        memories.map { memory =>
          val purpose =
            if (memory.width == 8) "format_v0_stream_store_or_byte_buffer"
            else "attention_or_write_pipeline_sram"
          val totalBits = memory.depth.toLong * memory.width * memory.instances
          s"${memory.module},$purpose,${memory.depth},${memory.width}," +
            s"${memory.instances},$totalBits,1,1,0,phase_separated"
        }.mkString("\n") + "\n"
    Files.writeString(targetDir.resolve("memories.csv"), csv, StandardCharsets.UTF_8)

    if (config.mode == "dc_logic") {
      val tcl =
        "# Auto-generated unified-tile SRAM modules excluded from DC logic PPA.\n" +
          "set BRISKKV_MEMORY_MODULES {\n" +
          memories.map(memory => s"  ${memory.module}").mkString("\n") +
          "\n}\n"
      Files.writeString(
        targetDir.resolve("memory_modules.tcl"),
        tcl,
        StandardCharsets.UTF_8
      )
    }

    println(s"Generated unified-tile ${config.mode} SystemVerilog in $targetDir")
    println(
      s"Discovered ${memories.length} memory modules / " +
        s"${memories.map(_.instances).sum} instances / $memoryBits bits"
    )
  }
}
