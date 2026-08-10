package briskkv

import chisel3._
import chisel3.util._

class DynamicBitUnpackerIO(
  outputBits: Int,
  descriptorIndexBits: Int,
  tokenIndexBits: Int,
  byteCountBits: Int
) extends Bundle {
  val start = Input(Bool())
  val descriptorCount = Input(UInt(descriptorIndexBits.W))
  val payloadByteCount = Input(UInt(byteCountBits.W))
  val minimumIn = Flipped(Decoupled(UInt(8.W)))
  val widthIn = Flipped(Decoupled(UInt(8.W)))
  val payloadIn = Flipped(Decoupled(UInt(8.W)))
  val out = Decoupled(
    new DynamicUnpackedValue(outputBits, descriptorIndexBits, tokenIndexBits)
  )
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
}

/** Top-level three-stream bit-packed K or V integer decoder. */
class DynamicBitUnpacker(
  codeValueBits: Int,
  encodeLengthBits: Int,
  signedValues: Boolean,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  outputBits: Int = 8,
  descriptorIndexBits: Int = 32,
  byteCountBits: Int = 32
) extends Module {
  private val tokenIndexBits = log2Ceil(packTokens)
  val io = IO(
    new DynamicBitUnpackerIO(
      outputBits,
      descriptorIndexBits,
      tokenIndexBits,
      byteCountBits
    )
  )

  private val descriptorDecoder = Module(
    new PackDescriptorDecoder(
      codeValueBits,
      encodeLengthBits,
      signedValues,
      outputBits,
      descriptorIndexBits
    )
  )
  private val payloadDecoder = Module(
    new DynamicPayloadUnpacker(
      codeValueBits,
      encodeLengthBits,
      signedValues,
      packTokens,
      outputBits,
      descriptorIndexBits,
      byteCountBits
    )
  )

  descriptorDecoder.io.start := io.start
  descriptorDecoder.io.descriptorCount := io.descriptorCount
  descriptorDecoder.io.minimumIn.valid := io.minimumIn.valid
  descriptorDecoder.io.minimumIn.bits := io.minimumIn.bits
  io.minimumIn.ready := descriptorDecoder.io.minimumIn.ready
  descriptorDecoder.io.widthIn.valid := io.widthIn.valid
  descriptorDecoder.io.widthIn.bits := io.widthIn.bits
  io.widthIn.ready := descriptorDecoder.io.widthIn.ready

  payloadDecoder.io.start := io.start
  payloadDecoder.io.descriptorCount := io.descriptorCount
  payloadDecoder.io.payloadByteCount := io.payloadByteCount
  payloadDecoder.io.descriptorIn <> descriptorDecoder.io.out
  payloadDecoder.io.payloadIn.valid := io.payloadIn.valid
  payloadDecoder.io.payloadIn.bits := io.payloadIn.bits
  io.payloadIn.ready := payloadDecoder.io.payloadIn.ready

  io.out <> payloadDecoder.io.out
  io.busy := descriptorDecoder.io.busy || payloadDecoder.io.busy
  io.done := payloadDecoder.io.done
  io.error := descriptorDecoder.io.error || payloadDecoder.io.error
}
