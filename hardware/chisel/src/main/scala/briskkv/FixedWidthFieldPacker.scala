package briskkv

import chisel3._
import chisel3.util._

class FixedWidthFieldPackerStats extends Bundle {
  val activeCycles = UInt(64.W)
  val inputFields = UInt(64.W)
  val outputBytes = UInt(64.W)
  val sourceWaitCycles = UInt(64.W)
  val sinkStallCycles = UInt(64.W)
}

class FixedWidthFieldPackerIO(fieldBits: Int, countBits: Int) extends Bundle {
  val start = Input(Bool())
  val fieldCount = Input(UInt(countBits.W))
  val in = Flipped(Decoupled(UInt(fieldBits.W)))
  val out = Decoupled(UInt(8.W))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new FixedWidthFieldPackerStats)
}

/** Fixed-width fields to one continuous LSB-first byte stream. */
class FixedWidthFieldPacker(
  fieldBits: Int,
  countBits: Int = 32,
  enableStats: Boolean = true,
  maximumFieldCount: Option[Int] = None
) extends Module {
  require(fieldBits >= 1 && fieldBits <= 8)

  private val reservoirBits = 16
  private val reservoirCountBits = log2Ceil(reservoirBits + 1)
  private val fieldCounterBits = maximumFieldCount match {
    case Some(maximum) => math.max(1, log2Ceil(maximum + 1))
    case None => countBits
  }
  private val sIdle :: sAccept :: sEmit :: sFlush :: Nil = Enum(4)

  maximumFieldCount.foreach(maximum => require(maximum > 0))

  val io = IO(new FixedWidthFieldPackerIO(fieldBits, countBits))
  val state = RegInit(sIdle)
  val reservoir = RegInit(0.U(reservoirBits.W))
  val reservoirCount = RegInit(0.U(reservoirCountBits.W))
  val fieldsRemaining = RegInit(0.U(fieldCounterBits.W))
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val activeCycles = RegInit(0.U(64.W))
  val inputFields = RegInit(0.U(64.W))
  val outputBytes = RegInit(0.U(64.W))
  val sourceWaitCycles = RegInit(0.U(64.W))
  val sinkStallCycles = RegInit(0.U(64.W))

  io.busy := state =/= sIdle
  io.done := doneReg
  io.error := errorReg
  doneReg := false.B
  io.in.ready := state === sAccept
  io.out.valid := state === sEmit || state === sFlush
  io.out.bits := reservoir(7, 0)
  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.inputFields := Mux(enableStats.B, inputFields, 0.U)
  io.stats.outputBytes := Mux(enableStats.B, outputBytes, 0.U)
  io.stats.sourceWaitCycles := Mux(enableStats.B, sourceWaitCycles, 0.U)
  io.stats.sinkStallCycles := Mux(enableStats.B, sinkStallCycles, 0.U)

  val inputFire = io.in.valid && io.in.ready
  val outputFire = io.out.valid && io.out.ready
  val appended = reservoir |
    (io.in.bits << reservoirCount)(reservoirBits - 1, 0)
  val countAppended = reservoirCount + fieldBits.U

  when(io.start && state === sIdle) {
    val withinConfiguredMaximum = maximumFieldCount match {
      case Some(maximum) => io.fieldCount <= maximum.U
      case None => true.B
    }
    val commandValid = io.fieldCount =/= 0.U && withinConfiguredMaximum
    reservoir := 0.U
    reservoirCount := 0.U
    fieldsRemaining := io.fieldCount(fieldCounterBits - 1, 0)
    doneReg := !commandValid
    errorReg := !commandValid
    activeCycles := 0.U
    inputFields := 0.U
    outputBytes := 0.U
    sourceWaitCycles := 0.U
    sinkStallCycles := 0.U
    state := Mux(commandValid, sAccept, sIdle)
  }.elsewhen(io.start) {
    errorReg := true.B
  }.otherwise {
    when(state =/= sIdle) {
      activeCycles := activeCycles + 1.U
    }
    when(state === sAccept && !io.in.valid) {
      sourceWaitCycles := sourceWaitCycles + 1.U
    }
    when(io.out.valid && !io.out.ready) {
      sinkStallCycles := sinkStallCycles + 1.U
    }

    when(inputFire) {
      reservoir := appended
      reservoirCount := countAppended
      fieldsRemaining := fieldsRemaining - 1.U
      inputFields := inputFields + 1.U
      when(countAppended >= 8.U) {
        state := sEmit
      }.elsewhen(fieldsRemaining === 1.U) {
        state := sFlush
      }
    }

    when(state === sEmit && outputFire) {
      val countAfter = reservoirCount - 8.U
      reservoir := reservoir >> 8
      reservoirCount := countAfter
      outputBytes := outputBytes + 1.U
      when(fieldsRemaining === 0.U) {
        when(countAfter === 0.U) {
          state := sIdle
          doneReg := true.B
        }.otherwise {
          state := sFlush
        }
      }.otherwise {
        state := sAccept
      }
    }

    when(state === sFlush && outputFire) {
      reservoir := 0.U
      reservoirCount := 0.U
      outputBytes := outputBytes + 1.U
      state := sIdle
      doneReg := true.B
    }
  }
}
