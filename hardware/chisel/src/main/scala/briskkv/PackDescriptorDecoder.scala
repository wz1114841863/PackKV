package briskkv

import chisel3._
import chisel3.util._

class PackDescriptor(
  outputBits: Int,
  encodeLengthBits: Int,
  descriptorIndexBits: Int
) extends Bundle {
  val minimum = SInt(outputBits.W)
  val bitWidth = UInt(encodeLengthBits.W)
  val descriptorIndex = UInt(descriptorIndexBits.W)
  val last = Bool()
}

class PackDescriptorDecoderIO(
  outputBits: Int,
  encodeLengthBits: Int,
  descriptorIndexBits: Int
) extends Bundle {
  val start = Input(Bool())
  val descriptorCount = Input(UInt(descriptorIndexBits.W))
  val minimumIn = Flipped(Decoupled(UInt(8.W)))
  val widthIn = Flipped(Decoupled(UInt(8.W)))
  val out = Decoupled(
    new PackDescriptor(outputBits, encodeLengthBits, descriptorIndexBits)
  )
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
}

/** Synchronizes fixed-width minimum and encode-length component streams. */
class PackDescriptorDecoder(
  codeValueBits: Int,
  encodeLengthBits: Int,
  signedValues: Boolean,
  outputBits: Int = 8,
  descriptorIndexBits: Int = 32
) extends Module {
  require(codeValueBits >= 1 && codeValueBits <= 8)
  require(encodeLengthBits >= 1 && encodeLengthBits <= 8)
  require(outputBits >= codeValueBits)

  val io = IO(
    new PackDescriptorDecoderIO(
      outputBits,
      encodeLengthBits,
      descriptorIndexBits
    )
  )

  private val minimumDecoder =
    Module(new FixedWidthFieldUnpacker(codeValueBits, descriptorIndexBits))
  private val widthDecoder =
    Module(new FixedWidthFieldUnpacker(encodeLengthBits, descriptorIndexBits))

  minimumDecoder.io.start := io.start
  minimumDecoder.io.fieldCount := io.descriptorCount
  minimumDecoder.io.in.valid := io.minimumIn.valid
  minimumDecoder.io.in.bits := io.minimumIn.bits
  io.minimumIn.ready := minimumDecoder.io.in.ready

  widthDecoder.io.start := io.start
  widthDecoder.io.fieldCount := io.descriptorCount
  widthDecoder.io.in.valid := io.widthIn.valid
  widthDecoder.io.in.bits := io.widthIn.bits
  io.widthIn.ready := widthDecoder.io.in.ready

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val descriptorsRemaining = RegInit(0.U(descriptorIndexBits.W))
  val descriptorIndex = RegInit(0.U(descriptorIndexBits.W))

  val pairValid = minimumDecoder.io.out.valid && widthDecoder.io.out.valid
  io.out.valid := active && pairValid
  minimumDecoder.io.out.ready := active && io.out.ready && widthDecoder.io.out.valid
  widthDecoder.io.out.ready := active && io.out.ready && minimumDecoder.io.out.valid

  private val signFill = if (signedValues) {
    minimumDecoder.io.out.bits(codeValueBits - 1)
  } else {
    false.B
  }
  io.out.bits.minimum := Cat(
    Fill(outputBits - codeValueBits, signFill),
    minimumDecoder.io.out.bits
  ).asSInt
  io.out.bits.bitWidth := widthDecoder.io.out.bits
  io.out.bits.descriptorIndex := descriptorIndex
  io.out.bits.last := descriptorsRemaining === 1.U

  io.busy := active || minimumDecoder.io.busy || widthDecoder.io.busy
  io.done := doneReg
  io.error := errorReg || minimumDecoder.io.error || widthDecoder.io.error
  doneReg := false.B

  val outputFire = io.out.valid && io.out.ready
  when(io.start && !active) {
    descriptorsRemaining := io.descriptorCount
    descriptorIndex := 0.U
    errorReg := false.B
    when(io.descriptorCount === 0.U) {
      active := false.B
      doneReg := true.B
      errorReg := true.B
    }.otherwise {
      active := true.B
    }
  }.elsewhen(io.start && active) {
    errorReg := true.B
  }.elsewhen(active && outputFire) {
    when(widthDecoder.io.out.bits > codeValueBits.U) {
      errorReg := true.B
    }
    when(descriptorsRemaining === 1.U) {
      descriptorsRemaining := 0.U
      active := false.B
      doneReg := true.B
    }.otherwise {
      descriptorsRemaining := descriptorsRemaining - 1.U
      descriptorIndex := descriptorIndex + 1.U
    }
  }
}
