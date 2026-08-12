package briskkv

import chisel3._
import chisel3.util._

class CompactKvMetadataEncoderIO(
  metadataBits: Int,
  countBits: Int,
  tagBits: Int,
  tokenIndexBits: Int,
  bucketIdBits: Int
) extends Bundle {
  val start = Input(Bool())
  val parameterCount = Input(UInt(countBits.W))
  val firstBlockIndex = Input(UInt(countBits.W))
  val in = Flipped(
    Decoupled(
      new RoutedKvTokenMetadata(
        metadataBits,
        countBits,
        tagBits,
        tokenIndexBits,
        bucketIdBits
      )
    )
  )
  val kZeroOut = Decoupled(UInt(8.W))
  val kExponentOut = Decoupled(UInt(8.W))
  val vZeroOut = Decoupled(UInt(8.W))
  val vExponentOut = Decoupled(UInt(8.W))
  val busy = Output(Bool())
  val done = Output(Bool())
  val error = Output(Bool())
  val kZeroStats = Output(new FixedWidthFieldPackerStats)
  val kExponentStats = Output(new FixedWidthFieldPackerStats)
  val vZeroStats = Output(new FixedWidthFieldPackerStats)
  val vExponentStats = Output(new FixedWidthFieldPackerStats)
}

/** Encodes routed K/V zero points and exponents as four compact streams. */
class CompactKvMetadataEncoder(
  metadataBits: Int = 8,
  countBits: Int = 32,
  tagBits: Int = 32,
  enableStats: Boolean = true,
  maximumParameterCount: Option[Int] = None
) extends Module {
  private val format = BriskKvFormatV0.params
  private val tokenIndexBits = log2Ceil(format.blockTokens)
  private val parameterCounterBits = maximumParameterCount match {
    case Some(maximum) => math.max(1, log2Ceil(maximum + 1))
    case None => countBits
  }

  maximumParameterCount.foreach { maximum =>
    require(maximum > 0 && maximum % format.blockTokens == 0)
  }

  val io = IO(
    new CompactKvMetadataEncoderIO(
      metadataBits,
      countBits,
      tagBits,
      tokenIndexBits,
      format.bucketIdBits
    )
  )

  val kZero = Module(
    new FixedWidthFieldPacker(
      format.kZeroBits,
      countBits,
      enableStats,
      maximumParameterCount
    )
  )
  val kExponent = Module(
    new FixedWidthFieldPacker(
      format.exponentBits,
      countBits,
      enableStats,
      maximumParameterCount
    )
  )
  val vZero = Module(
    new FixedWidthFieldPacker(
      format.vZeroBits,
      countBits,
      enableStats,
      maximumParameterCount
    )
  )
  val vExponent = Module(
    new FixedWidthFieldPacker(
      format.exponentBits,
      countBits,
      enableStats,
      maximumParameterCount
    )
  )

  val withinConfiguredMaximum = maximumParameterCount match {
    case Some(maximum) => io.parameterCount <= maximum.U
    case None => true.B
  }
  val commandValid = io.parameterCount =/= 0.U &&
    io.parameterCount(tokenIndexBits - 1, 0) === 0.U &&
    withinConfiguredMaximum
  val childStart = io.start && commandValid
  Seq(kZero, kExponent, vZero, vExponent).foreach { packer =>
    packer.io.start := childStart
    packer.io.fieldCount := io.parameterCount
  }

  val allReady = kZero.io.in.ready && kExponent.io.in.ready &&
    vZero.io.in.ready && vExponent.io.in.ready
  val active = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)
  val parameterIndex = RegInit(0.U(parameterCounterBits.W))
  val firstBlockIndexReg = RegInit(0.U(countBits.W))
  val childDone = RegInit(VecInit(Seq.fill(4)(false.B)))

  io.in.ready := active && allReady
  val joinedValid = io.in.valid && active
  kZero.io.in.valid := joinedValid && kExponent.io.in.ready &&
    vZero.io.in.ready && vExponent.io.in.ready
  kExponent.io.in.valid := joinedValid && kZero.io.in.ready &&
    vZero.io.in.ready && vExponent.io.in.ready
  vZero.io.in.valid := joinedValid && kZero.io.in.ready &&
    kExponent.io.in.ready && vExponent.io.in.ready
  vExponent.io.in.valid := joinedValid && kZero.io.in.ready &&
    kExponent.io.in.ready && vZero.io.in.ready
  kZero.io.in.bits := io.in.bits.kZeroPoint.asUInt(format.kZeroBits - 1, 0)
  kExponent.io.in.bits :=
    io.in.bits.kExponent.asUInt(format.exponentBits - 1, 0)
  vZero.io.in.bits := io.in.bits.vZeroPoint.asUInt(format.vZeroBits - 1, 0)
  vExponent.io.in.bits :=
    io.in.bits.vExponent.asUInt(format.exponentBits - 1, 0)

  io.kZeroOut <> kZero.io.out
  io.kExponentOut <> kExponent.io.out
  io.vZeroOut <> vZero.io.out
  io.vExponentOut <> vExponent.io.out

  val inputFire = io.in.valid && io.in.ready
  val expectedRoutedToken = parameterIndex(tokenIndexBits - 1, 0)
  val expectedBlockIndex = firstBlockIndexReg + (parameterIndex >> tokenIndexBits)
  val fieldsValid =
    io.in.bits.kZeroPoint >= (-(1 << (format.kZeroBits - 1))).S &&
      io.in.bits.kZeroPoint <= ((1 << (format.kZeroBits - 1)) - 1).S &&
      io.in.bits.vZeroPoint >= (-(1 << (format.vZeroBits - 1))).S &&
      io.in.bits.vZeroPoint <= ((1 << (format.vZeroBits - 1)) - 1).S &&
      io.in.bits.kExponent >= (-6).S && io.in.bits.kExponent <= 4.S &&
      io.in.bits.vExponent >= (-6).S && io.in.bits.vExponent <= 4.S

  doneReg := false.B
  when(io.start && !active) {
    active := commandValid
    doneReg := !commandValid
    errorReg := !commandValid
    parameterIndex := 0.U
    firstBlockIndexReg := io.firstBlockIndex
    childDone.foreach(_ := false.B)
  }.elsewhen(io.start) {
    errorReg := true.B
  }.elsewhen(active) {
    when(inputFire) {
      when(
        !fieldsValid ||
          io.in.bits.routedTokenIndex =/= expectedRoutedToken ||
          io.in.bits.blockIndex =/= expectedBlockIndex ||
          io.in.bits.last =/= (expectedRoutedToken === (format.blockTokens - 1).U)
      ) {
        errorReg := true.B
      }
      parameterIndex := parameterIndex + 1.U
    }
    val donePulses = VecInit(
      Seq(kZero.io.done, kExponent.io.done, vZero.io.done, vExponent.io.done)
    )
    for (index <- 0 until 4) {
      when(donePulses(index)) { childDone(index) := true.B }
    }
    val allDone = (0 until 4).map { index =>
      childDone(index) || donePulses(index)
    }.reduce(_ && _)
    when(allDone) {
      active := false.B
      doneReg := true.B
      when(parameterIndex =/= io.parameterCount) {
        errorReg := true.B
      }
    }
  }

  io.busy := active || kZero.io.busy || kExponent.io.busy ||
    vZero.io.busy || vExponent.io.busy
  io.done := doneReg
  io.error := errorReg || kZero.io.error || kExponent.io.error ||
    vZero.io.error || vExponent.io.error
  io.kZeroStats := kZero.io.stats
  io.kExponentStats := kExponent.io.stats
  io.vZeroStats := vZero.io.stats
  io.vExponentStats := vExponent.io.stats
}
