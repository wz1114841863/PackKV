package briskkv

import circt.stage.ChiselStage

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}

object GenerateBriskKvAttentionTop {
  private final case class Config(
    targetDir: Path = Paths.get("generated/briskkv_attention_t1024_f128/full"),
    maximumTokens: Int = 1024,
    maximumFeatureDim: Int = 128,
    mode: String = "full"
  )

  private def usage(): String =
    """Usage: GenerateBriskKvAttentionTop [options]
      |  --target-dir <path>          output directory
      |  --maximum-tokens <int>       maximum context tokens (default: 1024)
      |  --maximum-feature-dim <int>  maximum head dimension (default: 128)
      |  --mode <full|dc_logic>       retain inferred memories or externalize them
      |""".stripMargin

  private def addressBits(depth: Int): Int =
    math.max(1, 32 - Integer.numberOfLeadingZeros(depth - 1))

  private def memoryBlackBoxStub(module: String, depth: Int, width: Int): String = {
    val addrWidth = addressBits(depth)
    s"""// BRISK-KV DC logic-only SRAM stub.
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

  private def parse(args: List[String], config: Config = Config()): Config =
    args match {
      case Nil => config
      case "--target-dir" :: value :: tail =>
        parse(tail, config.copy(targetDir = Paths.get(value)))
      case "--maximum-tokens" :: value :: tail =>
        parse(tail, config.copy(maximumTokens = value.toInt))
      case "--maximum-feature-dim" :: value :: tail =>
        parse(tail, config.copy(maximumFeatureDim = value.toInt))
      case "--mode" :: value :: tail =>
        parse(tail, config.copy(mode = value))
      case "--help" :: _ =>
        println(usage())
        sys.exit(0)
      case option :: _ =>
        throw new IllegalArgumentException(s"Unknown or incomplete option: $option\n${usage()}")
    }

  def main(args: Array[String]): Unit = {
    val config = parse(args.toList)
    require(config.maximumTokens >= BriskKvFormatV0.params.packTokens)
    require(config.maximumTokens % BriskKvFormatV0.params.packTokens == 0)
    require(config.maximumFeatureDim > 0 && config.maximumFeatureDim <= 256)
    require(Set("full", "dc_logic").contains(config.mode))

    val targetDir = config.targetDir.toAbsolutePath.normalize
    Files.createDirectories(targetDir)
    val firtoolOptions =
      Array("--disable-all-randomization", "--strip-debug-info")

    ChiselStage.emitSystemVerilogFile(
      new BriskKvAttentionTop(
        maximumTokens = config.maximumTokens,
        maximumFeatureDim = config.maximumFeatureDim
      ),
      args = Array("--target-dir", targetDir.toString),
      firtoolOpts = firtoolOptions
    )

    val memoryImplementation =
      if (config.mode == "dc_logic") "bodyless_blackbox_stubs" else "behavioral_sync_read_mem"
    val manifest =
      s"""{
         |  "top": "BriskKvAttentionTop",
         |  "mode": "${config.mode}",
         |  "maximum_tokens": ${config.maximumTokens},
         |  "maximum_feature_dim": ${config.maximumFeatureDim},
         |  "pack_tokens": ${BriskKvFormatV0.params.packTokens},
         |  "block_tokens": ${BriskKvFormatV0.params.blockTokens},
         |  "value_format": "signed Q6 / 18 bits",
         |  "qk_format": "signed Q12 / 44 bits",
         |  "softmax_format": "unsigned Q0.15 / 16 bits",
         |  "av_format": "signed Q21 / 50 bits",
         |  "output_format": "signed Q6 / 18 bits",
         |  "memory_implementation": "$memoryImplementation"
         |}
         |""".stripMargin
    Files.writeString(
      targetDir.resolve("manifest.json"),
      manifest,
      StandardCharsets.UTF_8
    )
    if (config.mode == "dc_logic") {
      val packCount = config.maximumTokens / BriskKvFormatV0.params.packTokens
      val memoryRows = Seq(
        (s"queryMemory_${config.maximumFeatureDim}x18", "query_replay", config.maximumFeatureDim, 18),
        (s"logitMemory_${packCount}x709", "softmax_logits", packCount, 709),
        (s"exponentMemory_${packCount}x277", "softmax_exponents", packCount, 277),
        (
          s"memory_${packCount * config.maximumFeatureDim}x288",
          "dequantized_v_packets",
          packCount * config.maximumFeatureDim,
          288
        ),
        (s"weightMemory_${packCount}x256", "softmax_weights", packCount, 256)
      )
      memoryRows.foreach { case (module, _, depth, width) =>
        val modulePath = targetDir.resolve(s"$module.sv")
        require(
          Files.isRegularFile(modulePath),
          s"Expected split memory module was not generated: $modulePath"
        )
        Files.writeString(
          modulePath,
          memoryBlackBoxStub(module, depth, width),
          StandardCharsets.UTF_8
        )
      }
      val memoryCsv =
        "module,purpose,depth,width_bits,instances,total_bits,read_ports,write_ports,readwrite_ports,access_mode\n" +
          memoryRows.map { case (module, purpose, depth, width) =>
            s"$module,$purpose,$depth,$width,1,${depth.toLong * width},1,1,0,parallel_width"
          }.mkString("\n") + "\n"
      Files.writeString(
        targetDir.resolve("memories.csv"),
        memoryCsv,
        StandardCharsets.UTF_8
      )
      val memoryTcl =
        "# Auto-generated architectural SRAM modules excluded from DC logic PPA.\n" +
          "set BRISKKV_MEMORY_MODULES {\n" +
          memoryRows.map { case (module, _, _, _) => s"  $module" }.mkString("\n") +
          "\n}\n"
      Files.writeString(
        targetDir.resolve("memory_modules.tcl"),
        memoryTcl,
        StandardCharsets.UTF_8
      )
    }
    println(s"Generated ${config.mode} SystemVerilog in $targetDir")
    if (config.mode == "dc_logic")
      println(s"DC black-box list: ${targetDir.resolve("memory_modules.tcl")}")
  }
}
