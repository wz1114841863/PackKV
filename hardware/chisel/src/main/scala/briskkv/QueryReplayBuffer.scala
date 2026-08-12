package briskkv

import chisel3._
import chisel3.util._

class QueryReplayStats extends Bundle {
  val activeCycles = UInt(64.W)
  val loadedValues = UInt(64.W)
  val readRequests = UInt(64.W)
  val replayedValues = UInt(64.W)
  val loadWaitCycles = UInt(64.W)
  val downstreamStallCycles = UInt(64.W)
}

class QueryReplayBufferIO(
  valueBits: Int,
  countBits: Int
) extends Bundle {
  val start = Input(Bool())
  val featureDim = Input(UInt(countBits.W))
  val packCount = Input(UInt(countBits.W))
  val loadIn = Flipped(Decoupled(SInt(valueBits.W)))
  val out = Decoupled(new QueryFeature(valueBits, countBits))
  val busy = Output(Bool())
  val loaded = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val stats = Output(new QueryReplayStats)
}

/** Loads one query vector and replays it once for every K token pack.
  *
  * The query memory uses a synchronous read port. A two-entry response queue
  * absorbs the one-cycle memory latency and preserves output stability under
  * downstream backpressure.
  */
class QueryReplayBuffer(
  valueBits: Int = 18,
  maximumFeatureDim: Int = 256,
  countBits: Int = 32,
  responseQueueDepth: Int = 2,
  enableStats: Boolean = true
) extends Module {
  require(valueBits >= 2)
  require(maximumFeatureDim > 0)
  require(responseQueueDepth >= 2)

  val io = IO(new QueryReplayBufferIO(valueBits, countBits))

  val queryMemory = SyncReadMem(maximumFeatureDim, SInt(valueBits.W))
  val responseQueue = Module(
    new Queue(
      new QueryFeature(valueBits, countBits),
      responseQueueDepth,
      pipe = true,
      flow = false
    )
  )

  val loading = RegInit(false.B)
  val replaying = RegInit(false.B)
  val loadedReg = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val featureDimReg = RegInit(0.U(countBits.W))
  val packCountReg = RegInit(0.U(countBits.W))
  val loadIndex = RegInit(0.U(countBits.W))
  val issueFeatureIndex = RegInit(0.U(countBits.W))
  val issuePackIndex = RegInit(0.U(countBits.W))
  val issuesComplete = RegInit(false.B)

  val activeCycles = RegInit(0.U(64.W))
  val loadedValues = RegInit(0.U(64.W))
  val readRequests = RegInit(0.U(64.W))
  val replayedValues = RegInit(0.U(64.W))
  val loadWaitCycles = RegInit(0.U(64.W))
  val downstreamStallCycles = RegInit(0.U(64.W))

  io.done := doneReg
  io.error := errorReg
  io.loaded := loadedReg
  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.loadedValues := Mux(enableStats.B, loadedValues, 0.U)
  io.stats.readRequests := Mux(enableStats.B, readRequests, 0.U)
  io.stats.replayedValues := Mux(enableStats.B, replayedValues, 0.U)
  io.stats.loadWaitCycles := Mux(enableStats.B, loadWaitCycles, 0.U)
  io.stats.downstreamStallCycles := Mux(enableStats.B, downstreamStallCycles, 0.U)
  doneReg := false.B

  io.loadIn.ready := loading
  val loadFire = io.loadIn.valid && io.loadIn.ready
  when(loadFire) {
    queryMemory.write(loadIndex, io.loadIn.bits)
  }

  val responseValid = RegInit(false.B)
  val responseFeatureIndex = RegInit(0.U(countBits.W))
  val responsePackIndex = RegInit(0.U(countBits.W))
  val responseLast = RegInit(false.B)
  val dequeueWillFire = responseQueue.io.deq.valid && io.out.ready
  val occupiedWithResponse = responseQueue.io.count +& responseValid.asUInt -
    dequeueWillFire.asUInt
  val canIssue = replaying && !issuesComplete &&
    occupiedWithResponse < responseQueueDepth.U
  val readData = queryMemory.read(issueFeatureIndex, canIssue)

  responseValid := canIssue
  when(canIssue) {
    responseFeatureIndex := issueFeatureIndex
    responsePackIndex := issuePackIndex
    responseLast := issueFeatureIndex === featureDimReg - 1.U &&
      issuePackIndex === packCountReg - 1.U
  }

  responseQueue.io.enq.valid := responseValid
  responseQueue.io.enq.bits.value := readData
  responseQueue.io.enq.bits.featureIndex := responseFeatureIndex
  responseQueue.io.enq.bits.packIndex := responsePackIndex
  responseQueue.io.enq.bits.last := responseLast
  io.out <> responseQueue.io.deq

  val outputFire = io.out.valid && io.out.ready
  val busyInternal = loading || replaying || responseValid ||
    responseQueue.io.deq.valid
  io.busy := busyInternal

  when(io.start && !busyInternal) {
    val parametersValid = io.featureDim =/= 0.U &&
      io.featureDim <= maximumFeatureDim.U && io.packCount =/= 0.U
    loading := parametersValid
    replaying := false.B
    loadedReg := false.B
    doneReg := !parametersValid
    errorReg := !parametersValid
    featureDimReg := io.featureDim
    packCountReg := io.packCount
    loadIndex := 0.U
    issueFeatureIndex := 0.U
    issuePackIndex := 0.U
    issuesComplete := false.B
    activeCycles := 0.U
    loadedValues := 0.U
    readRequests := 0.U
    replayedValues := 0.U
    loadWaitCycles := 0.U
    downstreamStallCycles := 0.U
  }.elsewhen(io.start && busyInternal) {
    errorReg := true.B
  }.otherwise {
    when(busyInternal) {
      activeCycles := activeCycles + 1.U
    }
    when(loading && !io.loadIn.valid) {
      loadWaitCycles := loadWaitCycles + 1.U
    }
    when(io.out.valid && !io.out.ready) {
      downstreamStallCycles := downstreamStallCycles + 1.U
    }
    when(responseValid && !responseQueue.io.enq.ready) {
      errorReg := true.B
    }

    when(loadFire) {
      loadedValues := loadedValues + 1.U
      when(loadIndex === featureDimReg - 1.U) {
        loading := false.B
        replaying := true.B
        loadedReg := true.B
        issueFeatureIndex := 0.U
        issuePackIndex := 0.U
      }.otherwise {
        loadIndex := loadIndex + 1.U
      }
    }

    when(canIssue) {
      readRequests := readRequests + 1.U
      when(
        issueFeatureIndex === featureDimReg - 1.U &&
          issuePackIndex === packCountReg - 1.U
      ) {
        issuesComplete := true.B
      }.elsewhen(issueFeatureIndex === featureDimReg - 1.U) {
        issueFeatureIndex := 0.U
        issuePackIndex := issuePackIndex + 1.U
      }.otherwise {
        issueFeatureIndex := issueFeatureIndex + 1.U
      }
    }

    when(outputFire) {
      replayedValues := replayedValues + 1.U
      when(io.out.bits.last) {
        replaying := false.B
        doneReg := true.B
      }
    }
  }
}
