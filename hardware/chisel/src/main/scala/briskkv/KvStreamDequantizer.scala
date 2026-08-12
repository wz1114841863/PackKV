package briskkv

import chisel3._
import chisel3.util._

class KvStreamDequantizerIO(
  outputBits: Int,
  countBits: Int,
  tokenIndexBits: Int
) extends Bundle {
  val start = Input(Bool())
  val tokenCount = Input(UInt(countBits.W))
  val descriptorCount = Input(UInt(countBits.W))
  val featureDim = Input(UInt(countBits.W))
  val payloadByteCount = Input(UInt(countBits.W))
  val minimumIn = Flipped(Decoupled(UInt(8.W)))
  val widthIn = Flipped(Decoupled(UInt(8.W)))
  val payloadIn = Flipped(Decoupled(UInt(8.W)))
  val zeroPointIn = Flipped(Decoupled(UInt(8.W)))
  val exponentIn = Flipped(Decoupled(UInt(8.W)))
  val out = Decoupled(
    new FixedPointDequantizedValue(outputBits, countBits, tokenIndexBits)
  )
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new DequantizerPerformanceStats)
}

/** End-to-end Format v0 decoder for one K or V component set. */
class KvStreamDequantizer(
  codeValueBits: Int,
  zeroPointBits: Int,
  exponentBits: Int = BriskKvFormatV0.params.exponentBits,
  encodeLengthBits: Int = BriskKvFormatV0.params.packWidthBits,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  outputBits: Int = 18,
  countBits: Int = 32,
  useBufferedMetadata: Boolean = true,
  enableStats: Boolean = true
) extends Module {
  private val tokenIndexBits = log2Ceil(packTokens)

  val io = IO(new KvStreamDequantizerIO(outputBits, countBits, tokenIndexBits))

  val qDecoder = Module(
    new DynamicBitUnpacker(
      codeValueBits = codeValueBits,
      encodeLengthBits = encodeLengthBits,
      signedValues = false,
      packTokens = packTokens
    )
  )
  val zeroDecoder = Module(new CompactMetadataDecoder(zeroPointBits))
  val exponentDecoder = Module(new CompactMetadataDecoder(exponentBits))
  val joiner: PackMetadataDequantizerBase = if (useBufferedMetadata) {
    Module(
      new BufferedPackMetadataDequantizer(
        codeValueBits = codeValueBits,
        zeroPointBits = zeroPointBits,
        packTokens = packTokens,
        outputBits = outputBits,
        enableStats = enableStats
      )
    )
  } else {
    Module(
      new PackMetadataDequantizer(
        codeValueBits = codeValueBits,
        zeroPointBits = zeroPointBits,
        packTokens = packTokens,
        outputBits = outputBits,
        enableStats = enableStats
      )
    )
  }

  qDecoder.io.start := io.start
  qDecoder.io.descriptorCount := io.descriptorCount
  qDecoder.io.payloadByteCount := io.payloadByteCount
  qDecoder.io.minimumIn <> io.minimumIn
  qDecoder.io.widthIn <> io.widthIn
  qDecoder.io.payloadIn <> io.payloadIn

  zeroDecoder.io.start := io.start
  zeroDecoder.io.fieldCount := io.tokenCount
  zeroDecoder.io.in <> io.zeroPointIn
  exponentDecoder.io.start := io.start
  exponentDecoder.io.fieldCount := io.tokenCount
  exponentDecoder.io.in <> io.exponentIn

  joiner.io.start := io.start
  joiner.io.tokenCount := io.tokenCount
  joiner.io.descriptorCount := io.descriptorCount
  joiner.io.featureDim := io.featureDim
  joiner.io.qIn <> qDecoder.io.out
  joiner.io.zeroIn <> zeroDecoder.io.out
  joiner.io.exponentIn <> exponentDecoder.io.out
  io.out <> joiner.io.out

  io.busy := qDecoder.io.busy || zeroDecoder.io.busy ||
    exponentDecoder.io.busy || joiner.io.busy
  io.done := joiner.io.done
  io.error := qDecoder.io.error || zeroDecoder.io.error ||
    exponentDecoder.io.error || joiner.io.error
  io.stats := joiner.io.stats
}
