package briskkv

import chisel3._
import chisel3.util._

class JitVDecompressionPairIO(
  outputBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int
) extends Bundle {
  val kStart = Input(Bool())
  val vStart = Input(Bool())
  val tag = Input(UInt(tagBits.W))
  val tokenCount = Input(UInt(countBits.W))
  val featureDim = Input(UInt(countBits.W))
  val descriptorCount = Input(UInt(countBits.W))
  val kPayloadByteCount = Input(UInt(countBits.W))
  val vPayloadByteCount = Input(UInt(countBits.W))
  val kReady = Output(Bool())
  val vReady = Output(Bool())
  val kMinimumIn = Flipped(Decoupled(UInt(8.W)))
  val kWidthIn = Flipped(Decoupled(UInt(8.W)))
  val kPayloadIn = Flipped(Decoupled(UInt(8.W)))
  val kZeroPointIn = Flipped(Decoupled(UInt(8.W)))
  val kExponentIn = Flipped(Decoupled(UInt(8.W)))
  val vMinimumIn = Flipped(Decoupled(UInt(8.W)))
  val vWidthIn = Flipped(Decoupled(UInt(8.W)))
  val vPayloadIn = Flipped(Decoupled(UInt(8.W)))
  val vZeroPointIn = Flipped(Decoupled(UInt(8.W)))
  val vExponentIn = Flipped(Decoupled(UInt(8.W)))
  val kOut = Decoupled(
    new FixedPointDequantizedValue(outputBits, countBits, tokenIndexBits)
  )
  val vOut = Decoupled(
    new FixedPointDequantizedValue(outputBits, countBits, tokenIndexBits)
  )
  val kResult = Decoupled(new DecompressionResult(countBits, tagBits))
  val vResult = Decoupled(new DecompressionResult(countBits, tagBits))
  val busy = Output(Bool())
  val error = Output(Bool())
  val kProgress = Output(new DecompressionProgress(countBits))
  val vProgress = Output(new DecompressionProgress(countBits))
}

abstract class JitVDecompressionPairBase(
  outputBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int
) extends Module {
  val io = IO(
    new JitVDecompressionPairIO(outputBits, countBits, tagBits, tokenIndexBits)
  )
}

/** Storage-first JIT-V baseline: retain one native decoder per K/V format. */
class DualJitVDecompressionPair(
  outputBits: Int,
  countBits: Int,
  tagBits: Int,
  useBufferedMetadata: Boolean,
  enableStats: Boolean
) extends JitVDecompressionPairBase(
      outputBits,
      countBits,
      tagBits,
      log2Ceil(BriskKvFormatV0.params.packTokens)
    ) {
  private val params = BriskKvFormatV0.params
  val k = Module(
    new DecompressionPipelineController(
      params.kQuantBits,
      params.kZeroBits,
      outputBits = outputBits,
      countBits = countBits,
      tagBits = tagBits,
      useBufferedMetadata = useBufferedMetadata,
      enableStats = enableStats
    )
  )
  val v = Module(
    new DecompressionPipelineController(
      params.vQuantBits,
      params.vZeroBits,
      outputBits = outputBits,
      countBits = countBits,
      tagBits = tagBits,
      useBufferedMetadata = useBufferedMetadata,
      enableStats = enableStats
    )
  )

  io.kReady := k.io.command.ready
  io.vReady := v.io.command.ready
  k.io.command.valid := io.kStart
  v.io.command.valid := io.vStart
  for ((controller, payloadBytes) <- Seq(
      k -> io.kPayloadByteCount,
      v -> io.vPayloadByteCount
    )) {
    controller.io.command.bits.tag := io.tag
    controller.io.command.bits.tokenCount := io.tokenCount
    controller.io.command.bits.featureDim := io.featureDim
    controller.io.command.bits.descriptorCount := io.descriptorCount
    controller.io.command.bits.payloadByteCount := payloadBytes
  }
  k.io.minimumIn <> io.kMinimumIn
  k.io.widthIn <> io.kWidthIn
  k.io.payloadIn <> io.kPayloadIn
  k.io.zeroPointIn <> io.kZeroPointIn
  k.io.exponentIn <> io.kExponentIn
  v.io.minimumIn <> io.vMinimumIn
  v.io.widthIn <> io.vWidthIn
  v.io.payloadIn <> io.vPayloadIn
  v.io.zeroPointIn <> io.vZeroPointIn
  v.io.exponentIn <> io.vExponentIn
  io.kOut <> k.io.out
  io.vOut <> v.io.out
  io.kResult <> k.io.result
  io.vResult <> v.io.result
  io.busy := k.io.busy || v.io.busy
  io.error := false.B
  io.kProgress := k.io.progress
  io.vProgress := v.io.progress
}

/** Area-first ablation: K and V time-share one K-capable decoder.
  *
  * V's narrower metadata is widened losslessly by VMetadataWidthAdapter. The
  * payload, width and exponent streams are already format-compatible. This
  * serializes V decoding behind K/Softmax weight generation, which is already
  * required by the JIT-V schedule.
  */
class SharedJitVDecompressionPair(
  outputBits: Int,
  countBits: Int,
  tagBits: Int,
  useBufferedMetadata: Boolean,
  enableStats: Boolean
) extends JitVDecompressionPairBase(
      outputBits,
      countBits,
      tagBits,
      log2Ceil(BriskKvFormatV0.params.packTokens)
    ) {
  private val params = BriskKvFormatV0.params
  private object Phase extends ChiselEnum {
    val idle, k, v = Value
  }
  private val phase = RegInit(Phase.idle)
  val controller = Module(
    new DecompressionPipelineController(
      params.kQuantBits,
      params.kZeroBits,
      outputBits = outputBits,
      countBits = countBits,
      tagBits = tagBits,
      useBufferedMetadata = useBufferedMetadata,
      enableStats = enableStats
    )
  )
  val adapter = Module(new VMetadataWidthAdapter(countBits, enableStats = false))

  val startAny = io.kStart || io.vStart
  val selectingV = phase === Phase.v || (phase === Phase.idle && io.vStart)
  io.kReady := phase === Phase.idle && controller.io.command.ready
  io.vReady := phase === Phase.idle && controller.io.command.ready
  controller.io.command.valid := startAny
  controller.io.command.bits.tag := io.tag
  controller.io.command.bits.tokenCount := io.tokenCount
  controller.io.command.bits.featureDim := io.featureDim
  controller.io.command.bits.descriptorCount := io.descriptorCount
  controller.io.command.bits.payloadByteCount := Mux(
    selectingV,
    io.vPayloadByteCount,
    io.kPayloadByteCount
  )

  when(phase === Phase.idle && io.kStart && !io.vStart) { phase := Phase.k }
  when(phase === Phase.idle && io.vStart && !io.kStart) { phase := Phase.v }

  adapter.io.start := io.vStart
  adapter.io.descriptorCount := io.descriptorCount
  adapter.io.tokenCount := io.tokenCount
  adapter.io.minimumIn <> io.vMinimumIn
  adapter.io.zeroPointIn <> io.vZeroPointIn

  controller.io.minimumIn.valid := Mux(
    selectingV,
    adapter.io.minimumOut.valid,
    io.kMinimumIn.valid
  )
  controller.io.minimumIn.bits := Mux(
    selectingV,
    adapter.io.minimumOut.bits,
    io.kMinimumIn.bits
  )
  adapter.io.minimumOut.ready := selectingV && controller.io.minimumIn.ready
  io.kMinimumIn.ready := !selectingV && controller.io.minimumIn.ready

  controller.io.zeroPointIn.valid := Mux(
    selectingV,
    adapter.io.zeroPointOut.valid,
    io.kZeroPointIn.valid
  )
  controller.io.zeroPointIn.bits := Mux(
    selectingV,
    adapter.io.zeroPointOut.bits,
    io.kZeroPointIn.bits
  )
  adapter.io.zeroPointOut.ready := selectingV && controller.io.zeroPointIn.ready
  io.kZeroPointIn.ready := !selectingV && controller.io.zeroPointIn.ready

  private def selectByteStream(
    sink: DecoupledIO[UInt],
    kSource: DecoupledIO[UInt],
    vSource: DecoupledIO[UInt]
  ): Unit = {
    sink.valid := Mux(selectingV, vSource.valid, kSource.valid)
    sink.bits := Mux(selectingV, vSource.bits, kSource.bits)
    kSource.ready := !selectingV && sink.ready
    vSource.ready := selectingV && sink.ready
  }
  selectByteStream(controller.io.widthIn, io.kWidthIn, io.vWidthIn)
  selectByteStream(controller.io.payloadIn, io.kPayloadIn, io.vPayloadIn)
  selectByteStream(controller.io.exponentIn, io.kExponentIn, io.vExponentIn)

  io.kOut.valid := phase === Phase.k && controller.io.out.valid
  io.kOut.bits := controller.io.out.bits
  io.vOut.valid := phase === Phase.v && controller.io.out.valid
  io.vOut.bits := controller.io.out.bits
  controller.io.out.ready := Mux(
    phase === Phase.v,
    io.vOut.ready,
    io.kOut.ready
  )

  io.kResult.valid := phase === Phase.k && controller.io.result.valid
  io.kResult.bits := controller.io.result.bits
  io.vResult.valid := phase === Phase.v && controller.io.result.valid
  io.vResult.bits := controller.io.result.bits
  controller.io.result.ready := Mux(
    phase === Phase.v,
    io.vResult.ready,
    io.kResult.ready
  )
  when(controller.io.result.valid && controller.io.result.ready) {
    phase := Phase.idle
  }

  io.busy := phase =/= Phase.idle || controller.io.busy || adapter.io.busy
  io.error := adapter.io.error || (io.kStart && io.vStart)
  io.kProgress := Mux(
    phase === Phase.k,
    controller.io.progress,
    0.U.asTypeOf(new DecompressionProgress(countBits))
  )
  io.vProgress := Mux(
    phase === Phase.v,
    controller.io.progress,
    0.U.asTypeOf(new DecompressionProgress(countBits))
  )
}
