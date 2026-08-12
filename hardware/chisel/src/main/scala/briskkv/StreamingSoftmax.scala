package briskkv

import chisel3._
import chisel3.util._

class AttentionWeightPacket(
  weightBits: Int,
  packTokens: Int,
  countBits: Int
) extends Bundle {
  val weights = Vec(packTokens, UInt(weightBits.W))
  val validTokens = UInt(log2Ceil(packTokens + 1).W)
  val packIndex = UInt(countBits.W)
  val blockIndex = UInt(countBits.W)
  val packWithinBlock = UInt(
    math.max(1, log2Ceil(BriskKvFormatV0.params.blockTokens / packTokens)).W
  )
  val last = Bool()
}

class StreamingSoftmaxStats extends Bundle {
  val activeCycles = UInt(64.W)
  val inputPackets = UInt(64.W)
  val exponentPackets = UInt(64.W)
  val outputPackets = UInt(64.W)
  val downstreamStallCycles = UInt(64.W)
  val maximumUpdates = UInt(64.W)
}

class StreamingSoftmaxIO(
  logitBits: Int,
  weightBits: Int,
  packTokens: Int,
  countBits: Int,
  sumBits: Int
) extends Bundle {
  val start = Input(Bool())
  val tokenCount = Input(UInt(countBits.W))
  val in = Flipped(Decoupled(new QkLogitPacket(logitBits, packTokens, countBits)))
  val out = Decoupled(new AttentionWeightPacket(weightBits, packTokens, countBits))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val maximum = Output(SInt(logitBits.W))
  val exponentSum = Output(UInt(sumBits.W))
  val stats = Output(new StreamingSoftmaxStats)
}

class StoredLogitPacket(logitBits: Int, packTokens: Int) extends Bundle {
  val logits = Vec(packTokens, SInt(logitBits.W))
  val validTokens = UInt(log2Ceil(packTokens + 1).W)
}

class StoredExponentPacket(expBits: Int, packTokens: Int) extends Bundle {
  val values = Vec(packTokens, UInt(expBits.W))
  val validTokens = UInt(log2Ceil(packTokens + 1).W)
}

/** Numerically stable, sequence-wide fixed-point softmax.
  *
  * Packets stream into an SRAM while the global maximum is collected. A
  * second SRAM pass evaluates an exp(-x) LUT and accumulates the denominator.
  * A single reciprocal is then shared by all 16 output lanes. The final pass
  * streams normalized Q0.15 weights under normal Decoupled backpressure.
  */
class StreamingSoftmax(
  logitBits: Int = 44,
  logitFractionalBits: Int = 12,
  weightBits: Int = 16,
  weightFractionalBits: Int = 15,
  exponentFractionalBits: Int = 16,
  reciprocalFractionalBits: Int = 32,
  expClampInteger: Int = 8,
  expStepFractionalBits: Int = 4,
  packTokens: Int = BriskKvFormatV0.params.packTokens,
  blockTokens: Int = BriskKvFormatV0.params.blockTokens,
  maximumTokens: Int = 16384,
  countBits: Int = 32,
  enableStats: Boolean = true
) extends Module {
  require(logitBits >= 2 && logitFractionalBits > expStepFractionalBits)
  require(weightBits > weightFractionalBits)
  require(exponentFractionalBits > 0)
  require(reciprocalFractionalBits > weightFractionalBits)
  require(expClampInteger > 0)
  require(packTokens > 0 && isPow2(packTokens))
  require(blockTokens >= packTokens && blockTokens % packTokens == 0)
  require(maximumTokens >= packTokens)

  private val packShift = log2Ceil(packTokens)
  private val maximumPacks = (maximumTokens + packTokens - 1) / packTokens
  private val packAddressBits = math.max(1, log2Ceil(maximumPacks))
  private val packsPerBlock = blockTokens / packTokens
  private val expIndexCount = expClampInteger << expStepFractionalBits
  private val expIndexBits = math.max(1, log2Ceil(expIndexCount + 1))
  private val expBits = exponentFractionalBits + 1
  private val sumBits = exponentFractionalBits + log2Ceil(maximumTokens + 1) + 1
  private val normalizationShift = reciprocalFractionalBits - weightFractionalBits
  private val reciprocalBits = reciprocalFractionalBits + 1

  private def balancedMaximum(values: Seq[SInt]): SInt = {
    require(values.nonEmpty)
    if (values.length == 1) values.head
    else {
      val nextLevel = values.grouped(2).map {
        case Seq(left, right) => Mux(left > right, left, right)
        case Seq(single) => single
      }.toSeq
      balancedMaximum(nextLevel)
    }
  }

  private def balancedSum(values: Seq[UInt]): UInt = {
    require(values.nonEmpty)
    if (values.length == 1) values.head
    else {
      val nextLevel = values.grouped(2).map {
        case Seq(left, right) => left +& right
        case Seq(single) => single
      }.toSeq
      balancedSum(nextLevel)
    }
  }

  val io = IO(
    new StreamingSoftmaxIO(
      logitBits,
      weightBits,
      packTokens,
      countBits,
      sumBits
    )
  )

  val expRom = VecInit((0 to expIndexCount).map { index =>
    val value = Math.round(
      Math.exp(-index.toDouble / (1 << expStepFractionalBits)) *
        (1L << exponentFractionalBits)
    )
    value.U(expBits.W)
  })

  val logitMemory = SyncReadMem(
    maximumPacks,
    new StoredLogitPacket(logitBits, packTokens)
  )
  val exponentMemory = SyncReadMem(
    maximumPacks,
    new StoredExponentPacket(expBits, packTokens)
  )
  val reciprocalDivider = Module(
    new IterativeUnsignedDivider(reciprocalBits, sumBits)
  )

  val Seq(
    sIdle,
    sCollect,
    sExpRead,
    sExpCompute,
    sReciprocalStart,
    sReciprocalWait,
    sOutputRead,
    sOutputCompute,
    sOutputHold
  ) = Enum(9)
  val state = RegInit(sIdle)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val packCountReg = RegInit(0.U(countBits.W))
  val finalValidTokensReg = RegInit(0.U(log2Ceil(packTokens + 1).W))
  val collectIndex = RegInit(0.U(countBits.W))
  val passIndex = RegInit(0.U(countBits.W))
  val globalMaximum = RegInit((-(BigInt(1) << (logitBits - 1))).S(logitBits.W))
  val exponentSum = RegInit(0.U(sumBits.W))
  val reciprocal = RegInit(0.U(reciprocalBits.W))
  val outputPacket = Reg(
    new AttentionWeightPacket(weightBits, packTokens, countBits)
  )

  val activeCycles = RegInit(0.U(64.W))
  val inputPackets = RegInit(0.U(64.W))
  val exponentPackets = RegInit(0.U(64.W))
  val outputPackets = RegInit(0.U(64.W))
  val downstreamStallCycles = RegInit(0.U(64.W))
  val maximumUpdates = RegInit(0.U(64.W))

  doneReg := false.B
  io.busy := state =/= sIdle
  io.done := doneReg
  io.error := errorReg
  io.maximum := globalMaximum
  io.exponentSum := exponentSum
  io.stats.activeCycles := Mux(enableStats.B, activeCycles, 0.U)
  io.stats.inputPackets := Mux(enableStats.B, inputPackets, 0.U)
  io.stats.exponentPackets := Mux(enableStats.B, exponentPackets, 0.U)
  io.stats.outputPackets := Mux(enableStats.B, outputPackets, 0.U)
  io.stats.downstreamStallCycles := Mux(enableStats.B, downstreamStallCycles, 0.U)
  io.stats.maximumUpdates := Mux(enableStats.B, maximumUpdates, 0.U)

  io.in.ready := state === sCollect
  io.out.valid := state === sOutputHold
  io.out.bits := outputPacket

  val logitReadEnable = state === sExpRead
  val logitReadData = logitMemory.read(
    passIndex(packAddressBits - 1, 0),
    logitReadEnable
  )
  val exponentReadEnable = state === sOutputRead
  val exponentReadData = exponentMemory.read(
    passIndex(packAddressBits - 1, 0),
    exponentReadEnable
  )

  val inputFire = io.in.valid && io.in.ready
  val outputFire = io.out.valid && io.out.ready
  val reciprocalNumerator = Wire(UInt(reciprocalBits.W))
  reciprocalNumerator :=
    (BigInt(1) << reciprocalFractionalBits).U + (exponentSum >> 1)
  reciprocalDivider.io.start := state === sReciprocalStart
  reciprocalDivider.io.numerator := reciprocalNumerator
  reciprocalDivider.io.denominator := exponentSum

  when(io.start && state === sIdle) {
    val parametersValid = io.tokenCount =/= 0.U &&
      io.tokenCount <= maximumTokens.U
    state := Mux(parametersValid, sCollect, sIdle)
    doneReg := !parametersValid
    errorReg := !parametersValid
    packCountReg := (io.tokenCount + (packTokens - 1).U) >> packShift
    val remainder = io.tokenCount & (packTokens - 1).U
    finalValidTokensReg := Mux(remainder === 0.U, packTokens.U, remainder)
    collectIndex := 0.U
    passIndex := 0.U
    globalMaximum := (-(BigInt(1) << (logitBits - 1))).S
    exponentSum := 0.U
    reciprocal := 0.U
    activeCycles := 0.U
    inputPackets := 0.U
    exponentPackets := 0.U
    outputPackets := 0.U
    downstreamStallCycles := 0.U
    maximumUpdates := 0.U
  }.elsewhen(io.start && state =/= sIdle) {
    errorReg := true.B
  }.otherwise {
    when(state =/= sIdle) {
      activeCycles := activeCycles + 1.U
    }
    when(io.out.valid && !io.out.ready) {
      downstreamStallCycles := downstreamStallCycles + 1.U
    }

    switch(state) {
      is(sIdle) {}

      is(sCollect) {
        when(inputFire) {
          val expectedLast = collectIndex === packCountReg - 1.U
          val expectedValidTokens = Mux(
            expectedLast,
            finalValidTokensReg,
            packTokens.U
          )
          val expectedBlock = collectIndex / packsPerBlock.U
          val expectedWithinBlock = collectIndex % packsPerBlock.U
          val packetValid = io.in.bits.packIndex === collectIndex &&
            io.in.bits.blockIndex === expectedBlock &&
            io.in.bits.packWithinBlock === expectedWithinBlock &&
            io.in.bits.validTokens === expectedValidTokens &&
            io.in.bits.last === expectedLast
          when(!packetValid) {
            errorReg := true.B
          }

          val stored = Wire(new StoredLogitPacket(logitBits, packTokens))
          stored.logits := io.in.bits.logits
          stored.validTokens := expectedValidTokens
          logitMemory.write(collectIndex(packAddressBits - 1, 0), stored)

          val candidates = Wire(Vec(packTokens, SInt(logitBits.W)))
          for (lane <- 0 until packTokens) {
            candidates(lane) := Mux(
              lane.U < expectedValidTokens,
              io.in.bits.logits(lane),
              (-(BigInt(1) << (logitBits - 1))).S(logitBits.W)
            )
          }
          val packetMaximum = balancedMaximum(candidates.toSeq)
          when(packetMaximum > globalMaximum) {
            globalMaximum := packetMaximum
            maximumUpdates := maximumUpdates + 1.U
          }
          inputPackets := inputPackets + 1.U
          collectIndex := collectIndex + 1.U
          when(expectedLast) {
            passIndex := 0.U
            state := sExpRead
          }
        }
      }

      is(sExpRead) {
        state := sExpCompute
      }

      is(sExpCompute) {
        val expPacket = Wire(new StoredExponentPacket(expBits, packTokens))
        val laneValues = Wire(Vec(packTokens, UInt(expBits.W)))
        for (lane <- 0 until packTokens) {
          val delta = (globalMaximum - logitReadData.logits(lane)).asUInt
          val roundedIndex =
            (delta + (BigInt(1) <<
              (logitFractionalBits - expStepFractionalBits - 1)).U) >>
              (logitFractionalBits - expStepFractionalBits)
          val clippedIndex = Mux(
            roundedIndex > expIndexCount.U,
            expIndexCount.U,
            roundedIndex(expIndexBits - 1, 0)
          )
          laneValues(lane) := Mux(
            lane.U < logitReadData.validTokens,
            expRom(clippedIndex),
            0.U
          )
        }
        expPacket.values := laneValues
        expPacket.validTokens := logitReadData.validTokens
        exponentMemory.write(passIndex(packAddressBits - 1, 0), expPacket)
        val packetSum = balancedSum(laneValues.toSeq)
        exponentSum := exponentSum + packetSum
        exponentPackets := exponentPackets + 1.U
        when(passIndex === packCountReg - 1.U) {
          state := sReciprocalStart
        }.otherwise {
          passIndex := passIndex + 1.U
          state := sExpRead
        }
      }

      is(sReciprocalStart) {
        state := sReciprocalWait
      }

      is(sReciprocalWait) {
        when(reciprocalDivider.io.done) {
          reciprocal := reciprocalDivider.io.quotient
          when(reciprocalDivider.io.error) {
            errorReg := true.B
          }
          passIndex := 0.U
          state := sOutputRead
        }
      }

      is(sOutputRead) {
        state := sOutputCompute
      }

      is(sOutputCompute) {
        val expectedLast = passIndex === packCountReg - 1.U
        outputPacket.validTokens := exponentReadData.validTokens
        outputPacket.packIndex := passIndex
        outputPacket.blockIndex := passIndex / packsPerBlock.U
        outputPacket.packWithinBlock := passIndex % packsPerBlock.U
        outputPacket.last := expectedLast
        for (lane <- 0 until packTokens) {
          val product = exponentReadData.values(lane) * reciprocal
          val normalized =
            (product + (BigInt(1) << (normalizationShift - 1)).U) >>
              normalizationShift
          val maximumWeight = BigInt(1) << weightFractionalBits
          outputPacket.weights(lane) := Mux(
            normalized > maximumWeight.U,
            maximumWeight.U,
            normalized(weightBits - 1, 0)
          )
        }
        state := sOutputHold
      }

      is(sOutputHold) {
        when(outputFire) {
          outputPackets := outputPackets + 1.U
          when(outputPacket.last) {
            state := sIdle
            doneReg := true.B
          }.otherwise {
            passIndex := passIndex + 1.U
            state := sOutputRead
          }
        }
      }
    }
  }
}
