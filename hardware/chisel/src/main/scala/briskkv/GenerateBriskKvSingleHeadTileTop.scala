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
  // Arrays shallower than this are intentionally synthesized as registers.
  // CACTI cannot form a meaningful SRAM organization for the two-entry query
  // queue, and black-boxing it would omit its implementation cost from both
  // the DC logic report and the CACTI memory report.
  private val MinimumExternalSramDepth = 16

  private final case class Config(
    targetDir: Path = Paths.get("generated/briskkv_single_head_tile/full"),
    maximumFeatureDim: Int = 128,
    maximumTokens: Int = 1024,
    inputBits: Int = 24,
    scaleLanes: Int = 4,
    enableStats: Boolean = false,
    quantArchitecture: String = "v1",
    attentionArchitecture: String = "full_v",
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
      case "--attention-architecture" :: value :: tail =>
        parse(tail, config.copy(attentionArchitecture = value))
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

  private def discoverMemories(
    targetDir: Path,
    topModule: String
  ): IndexedSeq[MemoryRow] = {
    val texts = systemVerilogFiles(targetDir).map { path =>
      path -> Files.readString(path, StandardCharsets.UTF_8)
    }
    val sourcesByModule = texts.flatMap { case (_, source) =>
      modulePattern.findFirstMatchIn(source).map(_.group(1) -> source)
    }.toMap
    require(sourcesByModule.contains(topModule), s"Missing generated top $topModule")

    // CIRCT reuses one wrapper module when two streams have equal geometry.
    // Count module instances through the complete hierarchy instead of merely
    // counting textual memory instantiations. Otherwise, for example, the K/V
    // width streams would be reported as one SRAM although the top contains
    // two instances of their shared ReplayByteStreamBuffer wrapper.
    val moduleNames = sourcesByModule.keySet
    val childCounts = sourcesByModule.map { case (parent, source) =>
      val children = moduleNames.toIndexedSeq.flatMap { child =>
        val instancePattern =
          ("(?m)^\\s*" + Regex.quote(child) +
            "\\s+[A-Za-z_][A-Za-z0-9_$]*\\s*\\(").r
        val count = instancePattern.findAllMatchIn(source).length
        Option.when(count > 0)(child -> count)
      }.toMap
      parent -> children
    }
    val hierarchyInstances = scala.collection.mutable.Map(topModule -> 1L)
    val pending = scala.collection.mutable.Queue(topModule -> 1L)
    while (pending.nonEmpty) {
      val (parent, parentDelta) = pending.dequeue()
      childCounts.getOrElse(parent, Map.empty).foreach { case (child, count) =>
        val childDelta = parentDelta * count
        hierarchyInstances.update(
          child,
          hierarchyInstances.getOrElse(child, 0L) + childDelta
        )
        pending.enqueue(child -> childDelta)
      }
    }

    texts.flatMap { case (_, source) =>
      for {
        moduleMatch <- modulePattern.findFirstMatchIn(source)
        memoryMatch <- memoryPattern.findFirstMatchIn(source)
      } yield {
        val module = moduleMatch.group(1)
        val width = memoryMatch.group(1).toInt + 1
        val depth = memoryMatch.group(2).toInt + 1
        val instances = hierarchyInstances.getOrElse(module, 0L)
        require(instances > 0, s"No reachable instance found for generated memory $module")
        require(instances <= Int.MaxValue, s"Memory instance count overflow: $module")
        MemoryRow(module, depth, width, instances.toInt)
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
    require(
      Set("full_v", "jit_v_dual", "jit_v_shared")
        .contains(config.attentionArchitecture),
      s"Unsupported attention architecture: ${config.attentionArchitecture}"
    )
    val quantArchitecture =
      QuantParameterArchitecture.fromCliName(config.quantArchitecture)
    val targetDir = config.targetDir.toAbsolutePath.normalize
    Files.createDirectories(targetDir)

    val topName = config.attentionArchitecture match {
      case "full_v"     => "BriskKvSingleHeadTileTop"
      case "jit_v_dual" => "BriskKvJitVSingleHeadTileTop"
      case "jit_v_shared" => "BriskKvSharedJitVSingleHeadTileTop"
    }
    // Keep this as a method: ChiselStage evaluates its generator by name inside
    // Builder context. Constructing a Module in a strict local val is illegal.
    def top = config.attentionArchitecture match {
      case "full_v" =>
        new BriskKvSingleHeadTileTop(
          inputBits = config.inputBits,
          maximumFeatureDim = config.maximumFeatureDim,
          maximumTokens = config.maximumTokens,
          scaleLanes = config.scaleLanes,
          enableStats = config.enableStats,
          quantParameterArchitecture = quantArchitecture
        )
      case "jit_v_dual" =>
        new BriskKvJitVSingleHeadTileTop(
          inputBits = config.inputBits,
          maximumFeatureDim = config.maximumFeatureDim,
          maximumTokens = config.maximumTokens,
          scaleLanes = config.scaleLanes,
          enableStats = config.enableStats,
          quantParameterArchitecture = quantArchitecture
        )
      case "jit_v_shared" =>
        new BriskKvSharedJitVSingleHeadTileTop(
          inputBits = config.inputBits,
          maximumFeatureDim = config.maximumFeatureDim,
          maximumTokens = config.maximumTokens,
          scaleLanes = config.scaleLanes,
          enableStats = config.enableStats,
          quantParameterArchitecture = quantArchitecture
        )
    }
    ChiselStage.emitSystemVerilogFile(
      top,
      args = Array("--target-dir", targetDir.toString),
      firtoolOpts = Array("--disable-all-randomization", "--strip-debug-info")
    )

    val discoveredMemories = discoverMemories(
      targetDir,
      topName
    )
    val (memories, logicMemories) = discoveredMemories.partition(
      _.depth >= MinimumExternalSramDepth
    )
    require(memories.nonEmpty, "No unified-tile architectural SRAMs discovered")
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
    val logicMemoryBits = logicMemories.map { memory =>
      memory.depth.toLong * memory.width * memory.instances
    }.sum
    val manifest =
      s"""{
         |  "top": "$topName",
         |  "mode": "${config.mode}",
         |  "architecture": "${config.attentionArchitecture}",
         |  "v_materialization": "${if (config.attentionArchitecture == "full_v") "full_token_feature_buffer" else "two_packet_jit_buffer_plus_feature_partial_sums"}",
         |  "decompression_datapaths": ${if (config.attentionArchitecture == "jit_v_shared") 1 else 2},
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
         |  "logic_memory_module_types": ${logicMemories.length},
         |  "logic_memory_instances": ${logicMemories.map(_.instances).sum},
         |  "logic_memory_bits": $logicMemoryBits,
         |  "minimum_external_sram_depth": $MinimumExternalSramDepth,
         |  "memory_implementation": "${if (config.mode == "dc_logic") "external_sram_blackboxes_plus_small_logic_arrays" else "behavioral_sync_read_mem"}"
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
            // `access_mode` describes how width slices form one logical SRAM
            // access, not whether tile reads and writes overlap in time. Every
            // slice of a wide word is active in parallel; the phase-separated
            // tile schedule is recorded independently in manifest.json.
            s"${memory.instances},$totalBits,1,1,0,parallel_width"
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
    println(
      s"Retained ${logicMemories.length} shallow memory modules / " +
        s"${logicMemories.map(_.instances).sum} instances / $logicMemoryBits bits as logic"
    )
  }
}
