package briskkv

import chisel3._
import chisel3.util._

class BucketCountEncoderStats extends Bundle {
  val activeCycles = UInt(64.W)
  val inputRecords = UInt(64.W)
  val outputBytes = UInt(64.W)
  val sourceWaitCycles = UInt(64.W)
  val sinkStallCycles = UInt(64.W)
}

class BucketCountEncoderIO(countBits: Int, blockIndexBits: Int) extends Bundle {
  val start = Input(Bool())
  val blockCount = Input(UInt(blockIndexBits.W))
  val firstBlockIndex = Input(UInt(blockIndexBits.W))
  val in = Flipped(
    Decoupled(new BucketCountRecord(countBits, blockIndexBits))
  )
  val out = Decoupled(UInt(8.W))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new BucketCountEncoderStats)
}

/** Encodes three 7-bit occupancies into one independently aligned header. */
class BucketCountEncoder(
  blockIndexBits: Int = 32,
  enableStats: Boolean = true,
  maximumBlockCount: Option[Int] = None
) extends Module {
  private val format = BriskKvFormatV0.params
  private val countBits = format.bucketCountBits
  private val headerBits = format.bucketHeaderBytesPerBlock * 8
  private val byteIndexBits = log2Ceil(format.bucketHeaderBytesPerBlock)
  private val blockCounterBits = maximumBlockCount match {
    case Some(maximum) => math.max(1, log2Ceil(maximum + 1))
    case None => blockIndexBits
  }

  require(format.blockTokens == 64)
  require(format.bucketCount == 4)
  require(format.bucketHeaderBytesPerBlock == 3)
  maximumBlockCount.foreach(maximum => require(maximum > 0))

  val io = IO(new BucketCountEncoderIO(countBits, blockIndexBits))
  private val sIdle :: sAccept :: sEmit :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val blocksRemaining = RegInit(0.U(blockCounterBits.W))
  val expectedBlockIndex = RegInit(0.U(blockIndexBits.W))
  val header = RegInit(0.U(headerBits.W))
  val byteIndex = RegInit(0.U(byteIndexBits.W))
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val activeCycles = RegInit(0.U(64.W))
  val inputRecords = RegInit(0.U(64.W))
  val outputBytes = RegInit(0.U(64.W))
  val sourceWaitCycles = RegInit(0.U(64.W))
  val sinkStallCycles = RegInit(0.U(64.W))

  io.busy := state =/= sIdle
  io.done := doneReg
  io.error := errorReg
  doneReg := false.B
  io.in.ready := state === sAccept
  io.out.valid := state === sEmit
  io.out.bits := MuxLookup(byteIndex, header(7, 0))(
    Seq(
      0.U -> header(7, 0),
      1.U -> header(15, 8),
      2.U -> header(23, 16)
    )
  )
  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.inputRecords := Mux(enableStats.B, inputRecords, 0.U)
  io.stats.outputBytes := Mux(enableStats.B, outputBytes, 0.U)
  io.stats.sourceWaitCycles := Mux(enableStats.B, sourceWaitCycles, 0.U)
  io.stats.sinkStallCycles := Mux(enableStats.B, sinkStallCycles, 0.U)

  val inputFire = io.in.valid && io.in.ready
  val outputFire = io.out.valid && io.out.ready
  val countSum = io.in.bits.counts.reduce(_ +& _)
  val storedSum = io.in.bits.counts(0) +& io.in.bits.counts(1) +&
    io.in.bits.counts(2)
  val recordValid = countSum === format.blockTokens.U &&
    storedSum <= format.blockTokens.U &&
    io.in.bits.counts(3) === format.blockTokens.U - storedSum &&
    io.in.bits.blockIndex === expectedBlockIndex &&
    io.in.bits.last === (blocksRemaining === 1.U)
  val assembledHeader = Cat(
    0.U(3.W),
    io.in.bits.counts(2),
    io.in.bits.counts(1),
    io.in.bits.counts(0)
  )

  when(io.start && state === sIdle) {
    val withinConfiguredMaximum = maximumBlockCount match {
      case Some(maximum) => io.blockCount <= maximum.U
      case None => true.B
    }
    val commandValid = io.blockCount =/= 0.U && withinConfiguredMaximum
    blocksRemaining := io.blockCount(blockCounterBits - 1, 0)
    expectedBlockIndex := io.firstBlockIndex
    header := 0.U
    byteIndex := 0.U
    doneReg := !commandValid
    errorReg := !commandValid
    activeCycles := 0.U
    inputRecords := 0.U
    outputBytes := 0.U
    sourceWaitCycles := 0.U
    sinkStallCycles := 0.U
    state := Mux(commandValid, sAccept, sIdle)
  }.elsewhen(io.start) {
    errorReg := true.B
  }.otherwise {
    when(state =/= sIdle) { activeCycles := activeCycles + 1.U }
    when(state === sAccept && !io.in.valid) {
      sourceWaitCycles := sourceWaitCycles + 1.U
    }
    when(io.out.valid && !io.out.ready) {
      sinkStallCycles := sinkStallCycles + 1.U
    }
    when(inputFire) {
      when(recordValid) {
        header := assembledHeader
        byteIndex := 0.U
        inputRecords := inputRecords + 1.U
        state := sEmit
      }.otherwise {
        errorReg := true.B
        doneReg := true.B
        state := sIdle
      }
    }
    when(state === sEmit && outputFire) {
      outputBytes := outputBytes + 1.U
      when(byteIndex === (format.bucketHeaderBytesPerBlock - 1).U) {
        when(blocksRemaining === 1.U) {
          blocksRemaining := 0.U
          state := sIdle
          doneReg := true.B
        }.otherwise {
          blocksRemaining := blocksRemaining - 1.U
          expectedBlockIndex := expectedBlockIndex + 1.U
          state := sAccept
        }
      }.otherwise {
        byteIndex := byteIndex + 1.U
      }
    }
  }
}
