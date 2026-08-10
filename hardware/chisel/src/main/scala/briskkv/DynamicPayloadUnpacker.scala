package briskkv

import chisel3._
import chisel3.util._

class DynamicUnpackedValue(
  outputBits: Int,
  descriptorIndexBits: Int,
  tokenIndexBits: Int
) extends Bundle {
  val value = SInt(outputBits.W)
  val descriptorIndex = UInt(descriptorIndexBits.W)
  val tokenIndex = UInt(tokenIndexBits.W)
  val last = Bool()
}

class DynamicPayloadUnpackerIO(
  outputBits: Int,
  widthBits: Int,
  descriptorIndexBits: Int,
  tokenIndexBits: Int,
  byteCountBits: Int
) extends Bundle {
  val start = Input(Bool())
  val descriptorCount = Input(UInt(descriptorIndexBits.W))
  val payloadByteCount = Input(UInt(byteCountBits.W))
  val descriptorIn = Flipped(
    Decoupled(
      new PackDescriptor(outputBits, widthBits, descriptorIndexBits)
    )
  )
  val payloadIn = Flipped(Decoupled(UInt(8.W)))
  val out = Decoupled(
    new DynamicUnpackedValue(outputBits, descriptorIndexBits, tokenIndexBits)
  )
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
}

/** Runtime-width LSB-first payload decoder for one K or V cache stream. */
class DynamicPayloadUnpacker(
  codeValueBits: Int,
  widthBits: Int,
  signedValues: Boolean,
  valuesPerDescriptor: Int = BriskKvFormatV0.params.packTokens,
  outputBits: Int = 8,
  descriptorIndexBits: Int = 32,
  byteCountBits: Int = 32
) extends Module {
  require(codeValueBits >= 1 && codeValueBits <= 8)
  require(widthBits >= 1 && widthBits <= 8)
  require(valuesPerDescriptor > 0 && isPow2(valuesPerDescriptor))
  require(valuesPerDescriptor % 8 == 0)
  require(outputBits >= codeValueBits)

  private val tokenIndexBits = log2Ceil(valuesPerDescriptor)
  private val reservoirBits = 24
  private val reservoirCountBits = log2Ceil(reservoirBits + 1)

  val io = IO(
    new DynamicPayloadUnpackerIO(
      outputBits,
      widthBits,
      descriptorIndexBits,
      tokenIndexBits,
      byteCountBits
    )
  )

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val payloadBytesRemaining = RegInit(0.U(byteCountBits.W))
  val declaredPayloadBytes = RegInit(0.U(byteCountBits.W))
  val expectedPayloadBytes = RegInit(0.U(byteCountBits.W))
  val descriptorsRemaining = RegInit(0.U(descriptorIndexBits.W))
  val expectedDescriptorIndex = RegInit(0.U(descriptorIndexBits.W))
  val descriptorValid = RegInit(false.B)
  val descriptorMinimum = RegInit(0.S(outputBits.W))
  val descriptorWidth = RegInit(0.U(widthBits.W))
  val descriptorIndex = RegInit(0.U(descriptorIndexBits.W))
  val descriptorLast = RegInit(false.B)
  val tokenIndex = RegInit(0.U(tokenIndexBits.W))
  val reservoir = RegInit(0.U(reservoirBits.W))
  val reservoirCount = RegInit(0.U(reservoirCountBits.W))

  io.busy := active
  io.done := doneReg
  io.error := errorReg
  doneReg := false.B

  io.descriptorIn.ready := active && !descriptorValid
  io.payloadIn.ready := active && payloadBytesRemaining =/= 0.U &&
    reservoirCount <= (reservoirBits - 8).U

  val widthAvailable = descriptorWidth === 0.U ||
    reservoirCount >= descriptorWidth
  io.out.valid := active && descriptorValid && widthAvailable

  val deltaMaskWide = (1.U((reservoirBits + 1).W) << descriptorWidth) - 1.U
  val delta = reservoir & deltaMaskWide(reservoirBits - 1, 0)
  val deltaExtended = Cat(0.U(1.W), delta(outputBits - 1, 0)).asSInt
  val restoredValue = descriptorMinimum +& deltaExtended
  io.out.bits.value := restoredValue(outputBits - 1, 0).asSInt
  io.out.bits.descriptorIndex := descriptorIndex
  io.out.bits.tokenIndex := tokenIndex
  io.out.bits.last := descriptorLast && tokenIndex === (valuesPerDescriptor - 1).U

  val descriptorFire = io.descriptorIn.valid && io.descriptorIn.ready
  val payloadFire = io.payloadIn.valid && io.payloadIn.ready
  val outputFire = io.out.valid && io.out.ready
  val countAfterOutput = reservoirCount - descriptorWidth
  val reservoirAfterOutput = reservoir >> descriptorWidth
  val shiftedPayload =
    (io.payloadIn.bits << reservoirCount)(reservoirBits - 1, 0)
  val shiftedPayloadAfterOutput =
    (io.payloadIn.bits << countAfterOutput)(reservoirBits - 1, 0)
  val reservoirAfterBoth = reservoirAfterOutput | shiftedPayloadAfterOutput

  val signedLower = (-(1 << (codeValueBits - 1))).S((outputBits + 1).W)
  val signedUpper = ((1 << (codeValueBits - 1)) - 1).S((outputBits + 1).W)
  val unsignedLower = 0.S((outputBits + 1).W)
  val unsignedUpper = ((1 << codeValueBits) - 1).S((outputBits + 1).W)
  val valueInRange = if (signedValues) {
    restoredValue >= signedLower && restoredValue <= signedUpper
  } else {
    restoredValue >= unsignedLower && restoredValue <= unsignedUpper
  }

  when(io.start && !active) {
    payloadBytesRemaining := io.payloadByteCount
    declaredPayloadBytes := io.payloadByteCount
    expectedPayloadBytes := 0.U
    descriptorsRemaining := io.descriptorCount
    expectedDescriptorIndex := 0.U
    descriptorValid := false.B
    tokenIndex := 0.U
    reservoir := 0.U
    reservoirCount := 0.U
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
  }.elsewhen(active) {
    when(descriptorFire) {
      descriptorValid := true.B
      descriptorMinimum := io.descriptorIn.bits.minimum
      descriptorWidth := io.descriptorIn.bits.bitWidth
      descriptorIndex := io.descriptorIn.bits.descriptorIndex
      descriptorLast := io.descriptorIn.bits.last
      tokenIndex := 0.U
      expectedPayloadBytes := expectedPayloadBytes +
        io.descriptorIn.bits.bitWidth * (valuesPerDescriptor / 8).U
      when(
        io.descriptorIn.bits.bitWidth > codeValueBits.U ||
          io.descriptorIn.bits.descriptorIndex =/= expectedDescriptorIndex ||
          io.descriptorIn.bits.last =/= (descriptorsRemaining === 1.U)
      ) {
        errorReg := true.B
      }
    }

    when(payloadFire && outputFire) {
      reservoir := reservoirAfterBoth
      reservoirCount := reservoirCount + 8.U - descriptorWidth
    }.elsewhen(payloadFire) {
      reservoir := reservoir | shiftedPayload
      reservoirCount := reservoirCount + 8.U
    }.elsewhen(outputFire) {
      reservoir := reservoirAfterOutput
      reservoirCount := countAfterOutput
    }
    when(payloadFire) {
      payloadBytesRemaining := payloadBytesRemaining - 1.U
    }

    when(outputFire) {
      when(!valueInRange) {
        errorReg := true.B
      }
      when(tokenIndex === (valuesPerDescriptor - 1).U) {
        descriptorValid := false.B
        tokenIndex := 0.U
        when(descriptorsRemaining === 1.U) {
          val residual = Mux(payloadFire, reservoirAfterBoth, reservoirAfterOutput)
          val remainingBytes = payloadBytesRemaining - payloadFire.asUInt
          descriptorsRemaining := 0.U
          active := false.B
          doneReg := true.B
          when(
            residual.orR || remainingBytes =/= 0.U ||
              declaredPayloadBytes =/= expectedPayloadBytes
          ) {
            errorReg := true.B
          }
        }.otherwise {
          descriptorsRemaining := descriptorsRemaining - 1.U
          expectedDescriptorIndex := expectedDescriptorIndex + 1.U
        }
      }.otherwise {
        tokenIndex := tokenIndex + 1.U
      }
    }

    when(
      descriptorValid && descriptorWidth =/= 0.U &&
        reservoirCount < descriptorWidth && payloadBytesRemaining === 0.U
    ) {
      active := false.B
      doneReg := true.B
      errorReg := true.B
    }
  }
}
