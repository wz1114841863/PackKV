package briskkv

import chisel3._
import chisel3.util._

class ReplayByteStreamBufferIO(countBits: Int) extends Bundle {
  val clear = Input(Bool())
  val seal = Input(Bool())
  val readStart = Input(Bool())
  val writeIn = Flipped(Decoupled(UInt(8.W)))
  val readOut = Decoupled(UInt(8.W))
  val length = Output(UInt(countBits.W))
  val sealedStream = Output(Bool())
  val reading = Output(Bool())
  val readDone = Output(Bool())
  val overflow = Output(Bool())
}

/** Single-port byte SRAM with phase-separated write and replay operation.
  *
  * A stream is cleared, filled through `writeIn`, and sealed exactly once.
  * `readStart` then replays the stored bytes in original order under arbitrary
  * downstream backpressure. Writes and reads never overlap, matching the
  * current single-head tile's encode-then-attend schedule.
  */
class ReplayByteStreamBuffer(
  capacityBytes: Int,
  countBits: Int = 32
) extends Module {
  require(capacityBytes > 0)
  require(BigInt(capacityBytes) < (BigInt(1) << countBits))

  private val addressBits = math.max(1, log2Ceil(capacityBytes))
  private val lengthBits = math.max(1, log2Ceil(capacityBytes + 1))
  private val responseQueueDepth = 2

  val io = IO(new ReplayByteStreamBufferIO(countBits))
  val memory = SyncReadMem(capacityBytes, UInt(8.W))
  val writePointer = RegInit(0.U(lengthBits.W))
  val readIssueIndex = RegInit(0.U(lengthBits.W))
  val readDelivered = RegInit(0.U(lengthBits.W))
  val sealedReg = RegInit(false.B)
  val readingReg = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val overflowReg = RegInit(false.B)
  val responseQueue = withReset(reset.asBool || io.clear) {
    Module(
      new Queue(
        UInt(8.W),
        entries = responseQueueDepth,
        pipe = true,
        flow = false
      )
    )
  }

  io.writeIn.ready := !sealedReg
  io.readOut <> responseQueue.io.deq
  io.length := writePointer
  io.sealedStream := sealedReg
  io.reading := readingReg
  io.readDone := doneReg
  io.overflow := overflowReg
  doneReg := false.B

  val writeFire = io.writeIn.valid && io.writeIn.ready
  val outputFire = io.readOut.valid && io.readOut.ready
  val outstandingReads = readIssueIndex - readDelivered
  val outstandingAfterDelivery = outstandingReads - outputFire.asUInt
  val issueRead = readingReg && readIssueIndex < writePointer &&
    outstandingAfterDelivery < responseQueueDepth.U
  val memoryReadByte = memory.read(
    readIssueIndex(addressBits - 1, 0),
    issueRead
  )
  val readResponseValid = RegNext(issueRead, false.B)
  responseQueue.io.enq.valid := readResponseValid
  responseQueue.io.enq.bits := memoryReadByte

  when(io.clear) {
    writePointer := 0.U
    readIssueIndex := 0.U
    readDelivered := 0.U
    sealedReg := false.B
    readingReg := false.B
    overflowReg := false.B
  }.otherwise {
    when(writeFire) {
      when(writePointer < capacityBytes.U) {
        memory.write(
          writePointer(addressBits - 1, 0),
          io.writeIn.bits
        )
        writePointer := writePointer + 1.U
      }.otherwise {
        // Consume the byte to let the producer terminate, but make the
        // transaction failure explicit instead of deadlocking at full.
        overflowReg := true.B
      }
    }

    when(io.seal) {
      sealedReg := true.B
    }

    when(io.readStart && sealedReg && !readingReg && outstandingReads === 0.U) {
      readIssueIndex := 0.U
      readDelivered := 0.U
      when(writePointer === 0.U) {
        doneReg := true.B
      }.otherwise {
        readingReg := true.B
      }
    }

    when(issueRead) {
      readIssueIndex := readIssueIndex + 1.U
    }
    when(outputFire) {
      readDelivered := readDelivered + 1.U
      when(readDelivered + 1.U === writePointer) {
        readingReg := false.B
        doneReg := true.B
      }
    }
  }
}
