package briskkv

import chisel3._
import chisel3.util._

class BucketCountRecord(countBits: Int, blockIndexBits: Int) extends Bundle {
  val counts = Vec(4, UInt(countBits.W))
  val blockIndex = UInt(blockIndexBits.W)
  val last = Bool()
}

class BucketCountDecoderIO(countBits: Int, blockIndexBits: Int) extends Bundle {
  val start = Input(Bool())
  val blockCount = Input(UInt(blockIndexBits.W))
  val in = Flipped(Decoupled(UInt(8.W)))
  val out = Decoupled(new BucketCountRecord(countBits, blockIndexBits))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
}

/** BRISK-KV Format v0 four-bucket count-header decoder.
  *
  * Every 64-token block is encoded independently as three 7-bit unsigned
  * counts in a three-byte, LSB-first component. Bits 23:21 are byte-alignment
  * padding and must be zero. The fourth count is reconstructed as
  * `64 - count0 - count1 - count2`.
  */
class BucketCountDecoder(
  blockTokens: Int = BriskKvFormatV0.params.blockTokens,
  bucketCount: Int = BriskKvFormatV0.params.bucketCount,
  blockIndexBits: Int = 32
) extends Module {
  require(blockTokens == 64, "Format v0 fixes blockTokens at 64")
  require(bucketCount == 4, "Format v0 fixes bucketCount at 4")
  require(blockIndexBits >= 2)

  private val countBits = log2Ceil(blockTokens + 1)
  private val storedCounts = bucketCount - 1
  private val headerBits = storedCounts * countBits
  private val headerBytes = (headerBits + 7) / 8
  require(countBits == 7)
  require(headerBits == 21)
  require(headerBytes == 3)

  val io = IO(new BucketCountDecoderIO(countBits, blockIndexBits))

  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val header = RegInit(0.U((headerBytes * 8).W))
  val byteIndex = RegInit(0.U(log2Ceil(headerBytes).W))
  val blocksRemaining = RegInit(0.U(blockIndexBits.W))
  val currentBlockIndex = RegInit(0.U(blockIndexBits.W))
  val recordValid = RegInit(false.B)
  val recordCounts = Reg(Vec(bucketCount, UInt(countBits.W)))

  io.busy := active
  io.done := doneReg
  io.error := errorReg
  doneReg := false.B

  io.in.ready := active && !recordValid
  io.out.valid := recordValid
  io.out.bits.counts := recordCounts
  io.out.bits.blockIndex := currentBlockIndex
  io.out.bits.last := blocksRemaining === 1.U

  val inputFire = io.in.valid && io.in.ready
  val outputFire = io.out.valid && io.out.ready
  val completedHeader = WireDefault(header)
  switch(byteIndex) {
    is(0.U) {
      completedHeader := Cat(0.U(16.W), io.in.bits)
    }
    is(1.U) {
      completedHeader := Cat(0.U(8.W), io.in.bits, header(7, 0))
    }
    is(2.U) {
      completedHeader := Cat(io.in.bits, header(15, 0))
    }
  }

  when(io.start && !active) {
    header := 0.U
    byteIndex := 0.U
    blocksRemaining := io.blockCount
    currentBlockIndex := 0.U
    recordValid := false.B
    errorReg := false.B
    when(io.blockCount === 0.U) {
      active := false.B
      doneReg := true.B
      errorReg := true.B
    }.otherwise {
      active := true.B
    }
  }.elsewhen(io.start && active) {
    errorReg := true.B
  }.elsewhen(active) {
    when(inputFire) {
      when(byteIndex === (headerBytes - 1).U) {
        val count0 = completedHeader(6, 0)
        val count1 = completedHeader(13, 7)
        val count2 = completedHeader(20, 14)
        val countSum = count0 +& count1 +& count2
        val validSum = countSum <= blockTokens.U
        recordCounts(0) := count0
        recordCounts(1) := count1
        recordCounts(2) := count2
        recordCounts(3) := Mux(
          validSum,
          (blockTokens.U - countSum)(countBits - 1, 0),
          0.U
        )
        recordValid := true.B
        header := 0.U
        byteIndex := 0.U
        when(completedHeader(23, headerBits).orR || !validSum) {
          errorReg := true.B
        }
      }.otherwise {
        header := completedHeader
        byteIndex := byteIndex + 1.U
      }
    }

    when(outputFire) {
      recordValid := false.B
      when(blocksRemaining === 1.U) {
        blocksRemaining := 0.U
        active := false.B
        doneReg := true.B
      }.otherwise {
        blocksRemaining := blocksRemaining - 1.U
        currentBlockIndex := currentBlockIndex + 1.U
      }
    }
  }
}
