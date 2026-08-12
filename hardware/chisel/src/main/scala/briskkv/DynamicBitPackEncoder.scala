package briskkv

import chisel3._
import chisel3.util._

class DynamicPackVector(
  codeValueBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val values = Vec(packTokens, UInt(codeValueBits.W))
  val descriptorIndex = UInt(countBits.W)
  val last = Bool()
}

class DynamicBitPackEncoderStats extends Bundle {
  val activeCycles = UInt(64.W)
  val inputDescriptors = UInt(64.W)
  val minimumBytes = UInt(64.W)
  val widthBytes = UInt(64.W)
  val payloadBytes = UInt(64.W)
  val outputStallCycles = UInt(64.W)
  val zeroWidthDescriptors = UInt(64.W)
}

class DynamicBitPackEncoderIO(
  codeValueBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val start = Input(Bool())
  val descriptorCount = Input(UInt(countBits.W))
  val in = Flipped(
    Decoupled(new DynamicPackVector(codeValueBits, packTokens, countBits))
  )
  val minimumOut = Decoupled(UInt(8.W))
  val widthOut = Decoupled(UInt(8.W))
  val payloadOut = Decoupled(UInt(8.W))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new DynamicBitPackEncoderStats)
}

/** LSB-first dynamic bit-pack encoder for one K or V component.
  *
  * For each 16-value descriptor it emits the fixed-width minimum, a three-bit
  * dynamic width, and sixteen unsigned deltas. The three component streams are
  * independently continuous and are padded only at transaction end. A
  * zero-width descriptor emits no payload bits.
  */
class DynamicBitPackEncoder(
  codeValueBits: Int,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  countBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  private val widthBits = log2Ceil(codeValueBits + 1)
  private val laneBits = log2Ceil(packTokens)
  private val reservoirBits = 16
  private val reservoirCountBits = log2Ceil(reservoirBits + 1)

  require(codeValueBits >= 1 && codeValueBits <= 8)
  require(packTokens == 16)
  require(widthBits <= 8)

  val io = IO(
    new DynamicBitPackEncoderIO(codeValueBits, packTokens, countBits)
  )

  private val Seq(
    sIdle,
    sWaitDescriptor,
    sAppendMinimum,
    sEmitMinimum,
    sAppendWidth,
    sEmitWidth,
    sAppendPayload,
    sEmitPayload,
    sFlushMinimum,
    sFlushWidth,
    sFlushPayload
  ) = Enum(11)
  val state = RegInit(sIdle)
  val valuesReg = Reg(Vec(packTokens, UInt(codeValueBits.W)))
  val minimumReg = RegInit(0.U(codeValueBits.W))
  val widthReg = RegInit(0.U(widthBits.W))
  val descriptorLastReg = RegInit(false.B)
  val expectedDescriptorIndex = RegInit(0.U(countBits.W))
  val descriptorsRemaining = RegInit(0.U(countBits.W))
  val payloadLane = RegInit(0.U(laneBits.W))

  val minimumReservoir = RegInit(0.U(reservoirBits.W))
  val widthReservoir = RegInit(0.U(reservoirBits.W))
  val payloadReservoir = RegInit(0.U(reservoirBits.W))
  val minimumCount = RegInit(0.U(reservoirCountBits.W))
  val widthCount = RegInit(0.U(reservoirCountBits.W))
  val payloadCount = RegInit(0.U(reservoirCountBits.W))
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)

  val activeCycles = RegInit(0.U(64.W))
  val inputDescriptors = RegInit(0.U(64.W))
  val minimumBytes = RegInit(0.U(64.W))
  val widthBytes = RegInit(0.U(64.W))
  val payloadBytes = RegInit(0.U(64.W))
  val outputStallCycles = RegInit(0.U(64.W))
  val zeroWidthDescriptors = RegInit(0.U(64.W))

  io.busy := state =/= sIdle
  io.done := doneReg
  io.error := errorReg
  doneReg := false.B
  io.in.ready := state === sWaitDescriptor
  io.minimumOut.valid := state === sEmitMinimum ||
    (state === sFlushMinimum && minimumCount =/= 0.U)
  io.minimumOut.bits := minimumReservoir(7, 0)
  io.widthOut.valid := state === sEmitWidth ||
    (state === sFlushWidth && widthCount =/= 0.U)
  io.widthOut.bits := widthReservoir(7, 0)
  io.payloadOut.valid := state === sEmitPayload ||
    (state === sFlushPayload && payloadCount =/= 0.U)
  io.payloadOut.bits := payloadReservoir(7, 0)

  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.inputDescriptors := Mux(enableStats.B, inputDescriptors, 0.U)
  io.stats.minimumBytes := Mux(enableStats.B, minimumBytes, 0.U)
  io.stats.widthBytes := Mux(enableStats.B, widthBytes, 0.U)
  io.stats.payloadBytes := Mux(enableStats.B, payloadBytes, 0.U)
  io.stats.outputStallCycles := Mux(enableStats.B, outputStallCycles, 0.U)
  io.stats.zeroWidthDescriptors := Mux(enableStats.B, zeroWidthDescriptors, 0.U)

  private def balancedMinimum(values: Seq[UInt]): UInt = {
    if (values.length == 1) values.head
    else balancedMinimum(values.grouped(2).map {
      case Seq(a, b) => Mux(a < b, a, b)
      case Seq(a) => a
    }.toSeq)
  }

  private def balancedMaximum(values: Seq[UInt]): UInt = {
    if (values.length == 1) values.head
    else balancedMaximum(values.grouped(2).map {
      case Seq(a, b) => Mux(a > b, a, b)
      case Seq(a) => a
    }.toSeq)
  }

  val inputMinimum = balancedMinimum(io.in.bits.values.toSeq)
  val inputMaximum = balancedMaximum(io.in.bits.values.toSeq)
  val inputDifference = inputMaximum - inputMinimum
  val inputWidth = WireDefault(0.U(widthBits.W))
  for (bit <- 0 until codeValueBits) {
    when(inputDifference(bit)) {
      inputWidth := (bit + 1).U
    }
  }

  val inputFire = io.in.valid && io.in.ready
  val minimumFire = io.minimumOut.valid && io.minimumOut.ready
  val widthFire = io.widthOut.valid && io.widthOut.ready
  val payloadFire = io.payloadOut.valid && io.payloadOut.ready

  val minimumAppended = minimumReservoir |
    (minimumReg << minimumCount)(reservoirBits - 1, 0)
  val minimumCountAppended = minimumCount + codeValueBits.U
  val widthAppended = widthReservoir |
    (widthReg << widthCount)(reservoirBits - 1, 0)
  val widthCountAppended = widthCount + widthBits.U
  val payloadDelta = valuesReg(payloadLane) - minimumReg
  val payloadMask = (1.U((codeValueBits + 1).W) << widthReg) - 1.U
  val maskedPayloadDelta = payloadDelta & payloadMask(codeValueBits - 1, 0)
  val payloadAppended = payloadReservoir |
    (maskedPayloadDelta << payloadCount)(reservoirBits - 1, 0)
  val payloadCountAppended = payloadCount + widthReg

  def finishDescriptor(): Unit = {
    when(descriptorLastReg) {
      state := sFlushMinimum
    }.otherwise {
      descriptorsRemaining := descriptorsRemaining - 1.U
      expectedDescriptorIndex := expectedDescriptorIndex + 1.U
      state := sWaitDescriptor
    }
  }

  when(io.start && state === sIdle) {
    val commandValid = io.descriptorCount =/= 0.U
    descriptorsRemaining := io.descriptorCount
    expectedDescriptorIndex := 0.U
    minimumReservoir := 0.U
    widthReservoir := 0.U
    payloadReservoir := 0.U
    minimumCount := 0.U
    widthCount := 0.U
    payloadCount := 0.U
    errorReg := !commandValid
    doneReg := !commandValid
    activeCycles := 0.U
    inputDescriptors := 0.U
    minimumBytes := 0.U
    widthBytes := 0.U
    payloadBytes := 0.U
    outputStallCycles := 0.U
    zeroWidthDescriptors := 0.U
    state := Mux(commandValid, sWaitDescriptor, sIdle)
  }.elsewhen(io.start) {
    errorReg := true.B
  }.otherwise {
    when(state =/= sIdle) {
      activeCycles := activeCycles + 1.U
    }
    when(
      (io.minimumOut.valid && !io.minimumOut.ready) ||
        (io.widthOut.valid && !io.widthOut.ready) ||
        (io.payloadOut.valid && !io.payloadOut.ready)
    ) {
      outputStallCycles := outputStallCycles + 1.U
    }

    when(inputFire) {
      valuesReg := io.in.bits.values
      minimumReg := inputMinimum
      widthReg := inputWidth
      descriptorLastReg := io.in.bits.last
      payloadLane := 0.U
      inputDescriptors := inputDescriptors + 1.U
      when(
        io.in.bits.descriptorIndex =/= expectedDescriptorIndex ||
          io.in.bits.last =/= (descriptorsRemaining === 1.U)
      ) {
        errorReg := true.B
      }
      when(inputWidth === 0.U) {
        zeroWidthDescriptors := zeroWidthDescriptors + 1.U
      }
      state := sAppendMinimum
    }

    when(state === sAppendMinimum) {
      minimumReservoir := minimumAppended
      minimumCount := minimumCountAppended
      state := Mux(minimumCountAppended >= 8.U, sEmitMinimum, sAppendWidth)
    }
    when(state === sEmitMinimum && minimumFire) {
      minimumReservoir := minimumReservoir >> 8
      minimumCount := minimumCount - 8.U
      minimumBytes := minimumBytes + 1.U
      state := sAppendWidth
    }

    when(state === sAppendWidth) {
      widthReservoir := widthAppended
      widthCount := widthCountAppended
      state := Mux(widthCountAppended >= 8.U, sEmitWidth, sAppendPayload)
    }
    when(state === sEmitWidth && widthFire) {
      widthReservoir := widthReservoir >> 8
      widthCount := widthCount - 8.U
      widthBytes := widthBytes + 1.U
      state := sAppendPayload
    }

    when(state === sAppendPayload) {
      when(widthReg === 0.U) {
        finishDescriptor()
      }.otherwise {
        payloadReservoir := payloadAppended
        payloadCount := payloadCountAppended
        when(payloadCountAppended >= 8.U) {
          state := sEmitPayload
        }.elsewhen(payloadLane === (packTokens - 1).U) {
          finishDescriptor()
        }.otherwise {
          payloadLane := payloadLane + 1.U
        }
      }
    }
    when(state === sEmitPayload && payloadFire) {
      payloadReservoir := payloadReservoir >> 8
      payloadCount := payloadCount - 8.U
      payloadBytes := payloadBytes + 1.U
      when(payloadLane === (packTokens - 1).U) {
        finishDescriptor()
      }.otherwise {
        payloadLane := payloadLane + 1.U
        state := sAppendPayload
      }
    }

    when(state === sFlushMinimum) {
      when(minimumCount === 0.U) {
        state := sFlushWidth
      }.elsewhen(minimumFire) {
        minimumReservoir := 0.U
        minimumCount := 0.U
        minimumBytes := minimumBytes + 1.U
        state := sFlushWidth
      }
    }
    when(state === sFlushWidth) {
      when(widthCount === 0.U) {
        state := sFlushPayload
      }.elsewhen(widthFire) {
        widthReservoir := 0.U
        widthCount := 0.U
        widthBytes := widthBytes + 1.U
        state := sFlushPayload
      }
    }
    when(state === sFlushPayload) {
      when(payloadCount === 0.U) {
        descriptorsRemaining := 0.U
        state := sIdle
        doneReg := true.B
      }.elsewhen(payloadFire) {
        payloadReservoir := 0.U
        payloadCount := 0.U
        payloadBytes := payloadBytes + 1.U
        descriptorsRemaining := 0.U
        state := sIdle
        doneReg := true.B
      }
    }
  }
}

